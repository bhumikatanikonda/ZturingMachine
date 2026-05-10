"""Characterise a new qubit using the trained deep ensemble.

Usage (test on a seeded VirtualQubit so we can also report true labels):
    python infer_qubit.py --model-dir models --seed 42

Usage (pre-measured features supplied as a .npy of length sum(grid_sizes)):
    python infer_qubit.py --model-dir models --features-file probe.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_ensemble import GaussianMLP  # noqa: E402


# ============================================================================
# Loading
# ============================================================================

def load_ensemble(model_dir: Path, device: str):
    """Return (members, label_mean, label_std, grids) from a trained run dir."""
    norm = np.load(model_dir / "norm_stats.npz")
    grids = np.load(model_dir / "sweep_grids.npz")

    label_mean = norm["label_mean"]
    label_std = norm["label_std"]
    input_dim = int(norm["input_dim"])

    members = []
    for ckpt in sorted(model_dir.glob("member_*.pt")):
        m = GaussianMLP(input_dim).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        members.append(m)

    if not members:
        raise FileNotFoundError(
            f"No member_*.pt checkpoints found in {model_dir} — "
            "did you run train_ensemble.py first?"
        )
    return members, label_mean, label_std, grids


# ============================================================================
# Inference
# ============================================================================

def predict(features: np.ndarray, members, label_mean, label_std, device):
    """Mixture-of-Gaussians ensemble prediction.  Returns physical-unit μ, σ
    and a separate (aleatoric, epistemic) variance decomposition.
    """
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    mus_k, vars_k = [], []
    with torch.no_grad():
        for m in members:
            mu, log_var = m(x)
            mus_k.append(mu)
            vars_k.append(log_var.exp())

    mu_stack = torch.stack(mus_k)         # (M, 1, 4) — normalised space
    var_stack = torch.stack(vars_k)       # (M, 1, 4)
    mu_ens = mu_stack.mean(0)
    var_ens = (var_stack + mu_stack.pow(2)).mean(0) - mu_ens.pow(2)
    var_ens = var_ens.clamp_min(1e-12)

    aleatoric = var_stack.mean(0)         # mean of per-member variances
    epistemic = mu_stack.var(dim=0, unbiased=False)  # variance of member means

    mu_ens_n = mu_ens.squeeze(0).cpu().numpy()
    sig_ens_n = torch.sqrt(var_ens).squeeze(0).cpu().numpy()
    aleatoric_n = torch.sqrt(aleatoric).squeeze(0).cpu().numpy()
    epistemic_n = torch.sqrt(epistemic).squeeze(0).cpu().numpy()

    # ---- denormalise ----
    mu_phys = mu_ens_n * label_std + label_mean
    sigma_phys = sig_ens_n * label_std
    aleatoric_phys = aleatoric_n * label_std
    epistemic_phys = epistemic_n * label_std
    return mu_phys, sigma_phys, aleatoric_phys, epistemic_phys


# ============================================================================
# Reporting
# ============================================================================

LABEL_NAMES = ["f_q", "T1", "T2", "amp_π"]
LABEL_UNITS = ["GHz", "µs", "µs", "  "]
# Korrektur: LABEL_SCALES nutzt 1e-6 für Zeiten, da mu_phys in Sekunden vorliegt.
LABEL_SCALES = np.array([1e9, 1e-6, 1e-6, 1.0])


def print_summary(mu_phys, sigma_phys, true_labels=None,
                  aleatoric_phys=None, epistemic_phys=None,
                  prior_bounds=None):
    """Pretty-print prediction table + optional accuracy + flags."""
    print()
    header = f"{'Parameter':<10}{'Predicted':<24}{'Uncertainty (1σ)':<22}"
    print(header)
    print("─" * 65)
    for i, n in enumerate(LABEL_NAMES):
        v = mu_phys[i] / LABEL_SCALES[i]
        s = sigma_phys[i] / LABEL_SCALES[i]
        u = LABEL_UNITS[i]
        # Höhere Präzision (.6f) um 0.0000 µs zu vermeiden
        print(f"{n:<10}{v:>12.6f} {u:<8}    ± {s:>10.6f} {u}")

    # Aleatoric / epistemic split (informational).
    if aleatoric_phys is not None and epistemic_phys is not None:
        print()
        print(f"  Uncertainty decomposition (1σ contributions):")
        print(f"  {'param':<10}{'aleatoric':<16}{'epistemic':<16}")
        for i, n in enumerate(LABEL_NAMES):
            al = aleatoric_phys[i] / LABEL_SCALES[i]
            ep = epistemic_phys[i] / LABEL_SCALES[i]
            print(f"  {n:<10}{al:>10.6f} {LABEL_UNITS[i]:<5}{ep:>10.6f} {LABEL_UNITS[i]}")

    # ---- 5% relative-uncertainty flag ----
    rel_err = sigma_phys / np.abs(mu_phys + 1e-30)
    flagged = [(LABEL_NAMES[i], rel_err[i]) for i in range(4) if rel_err[i] > 0.05]
    if flagged:
        print()
        print("[FLAG] Parameters with σ/|μ| > 5% — recommend a full sweep:")
        for n, r in flagged:
            print(f"  • {n}: relative σ = {r*100:.2f}%")

    # ---- Boundary-truncation warning ----
    if prior_bounds is not None:
        msgs = []
        for i, n in enumerate(LABEL_NAMES):
            lo, hi = prior_bounds[i]
            if lo is None or hi is None:
                continue
            if (mu_phys[i] - 1.5 * sigma_phys[i]) < lo or \
               (mu_phys[i] + 1.5 * sigma_phys[i]) > hi:
                msgs.append(
                    f"  • {n}: μ ± 1.5σ crosses prior bound "
                    f"[{lo / LABEL_SCALES[i]:.3f}, {hi / LABEL_SCALES[i]:.3f}] "
                    f"{LABEL_UNITS[i]}"
                )
        if msgs:
            print()
            print("[NOTE] Prior-boundary effects (Gaussian assumption may break):")
            for m_ in msgs:
                print(m_)

    # ---- True values + abs error if we have them ----
    if true_labels is not None:
        print()
        print(f"{'Parameter':<10}{'True':<24}{'Abs error':<22}")
        print("─" * 65)
        for i, n in enumerate(LABEL_NAMES):
            t = true_labels[i] / LABEL_SCALES[i]
            e = abs(mu_phys[i] - true_labels[i]) / LABEL_SCALES[i]
            u = LABEL_UNITS[i]
            print(f"{n:<10}{t:>12.6f} {u:<8}      {e:>10.6f} {u}")


# ============================================================================
# Probing a fresh VirtualQubit (test mode)
# ============================================================================

def probe_seeded_qubit(seed: int, grids):
    """Run the full measurement protocol on a freshly-seeded qubit."""
    from qubit import VirtualQubit
    import qubit_measurements as qm

    # Update modules with saved grids
    qm.SPEC_FREQS = np.asarray(grids["spec_freqs"])
    qm.RABI_AMPS = np.asarray(grids["rabi_amps"])
    qm.T1_DELAYS = np.asarray(grids["t1_delays"])
    qm.RAMSEY_DELAYS = np.asarray(grids["ramsey_delays"])
    qm.SHOTS = int(grids["shots"])
    qm.RAMSEY_DETUNING = float(grids["ramsey_detuning"])
    gauss_pulse_length = float(grids["gauss_pulse_length"])

    print("Compiling LabOne Q waveforms (one-time)...")
    t_spec, wave_spec, t_pulse, wave_gauss = qm.precompile_waveforms()

    print(f"Probing VirtualQubit(seed={seed}) ...")
    qubit = VirtualQubit(seed=int(seed))
    true_labels = np.array([
        qubit._fq,
        qubit._T1,
        qubit._T2,
        np.pi / (qubit._omega * gauss_pulse_length),
    ])

    features = qm.measure_qubit_features(
        qubit, t_spec, wave_spec, t_pulse, wave_gauss
    )
    return features, true_labels


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Qubit Inference")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--features-file", default=None)
    args = p.parse_args()

    if (args.seed is None) == (args.features_file is None):
        raise SystemExit("Provide exactly one of --seed or --features-file.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    members, label_mean, label_std, grids = load_ensemble(
        Path(args.model_dir), device
    )
    print(f"Loaded {len(members)} ensemble members from {args.model_dir}.")

    if args.features_file:
        features = np.load(args.features_file).astype(np.float32)
        true_labels = None
    else:
        features, true_labels = probe_seeded_qubit(args.seed, grids)

    expected_dim = (
        len(grids["spec_freqs"])
        + len(grids["rabi_amps"])
        + len(grids["t1_delays"])
        + len(grids["ramsey_delays"])
    )
    if features.shape[0] != expected_dim:
        raise SystemExit(f"Dimension mismatch: expected {expected_dim}, got {features.shape[0]}")

    mu_phys, sigma_phys, aleatoric_phys, epistemic_phys = predict(
        features, members, label_mean, label_std, device
    )

    prior_bounds = [
        (5.0e9, 6.0e9),     # f_q
        (20e-6, 40e-6),     # T1
        (15e-6, 25e-6),     # T2
        (None, None),       # amp_pi
    ]

    print_summary(
        mu_phys, sigma_phys,
        true_labels=true_labels,
        aleatoric_phys=aleatoric_phys,
        epistemic_phys=epistemic_phys,
        prior_bounds=prior_bounds,
    )


if __name__ == "__main__":
    main()