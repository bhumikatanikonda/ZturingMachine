"""Train a deep ensemble of MLPs that does amortised qubit characterisation.

Run:
    python train_ensemble.py --dataset dataset.npz --out-dir models

Each ensemble member predicts a per-parameter Gaussian (μ, log σ²).  At
inference these are combined with the mixture-of-Gaussians formula:

    μ_ens   = mean_k μ_k
    σ²_ens  = mean_k (σ²_k + μ_k²) − μ_ens²
            =          ^^^^^                ^^^^^^^^
                aleatoric (intrinsic data noise) + epistemic (member disagreement)

Why M=5 ensemble members?
  Lakshminarayanan et al. 2017 ("Simple and Scalable Predictive Uncertainty
  Estimation Using Deep Ensembles") showed empirically that 5 members give
  near-optimal calibration on regression tasks.  Beyond ~10 the calibration
  curve flattens; below 3 the epistemic-uncertainty estimate (variance of
  member means) is too noisy to trust.  5 is the standard sweet spot at 5x
  training cost — completely tractable here since each member fits a small
  MLP on a few thousand examples in < a minute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset


# ============================================================================
# Architecture
# ============================================================================

class GaussianMLP(nn.Module):
    """4-layer MLP backbone with Linear→LayerNorm→SiLU stages,
    fanning out into a mean head and a clamped log-variance head.

    Layout (input_dim → 4):
        Linear(input_dim → 256) → LN(256) → SiLU
        Linear(256 → 256)       → LN(256) → SiLU
        Linear(256 → 256)       → LN(256) → SiLU
        Linear(256 → 128)       → LN(128) → SiLU
        ├─ Linear(128 → 4)        = μ
        └─ Linear(128 → 4) clamp[-10, 6] = log σ²
    """

    HIDDEN_DIMS = (256, 256, 256, 128)
    LOG_VAR_MIN = -10.0  # σ² ≥ ~4.5e-5 — guards against div-by-zero in NLL
    LOG_VAR_MAX = 6.0    # σ² ≤ ~403 — labels are normalised to ~unit variance,
                         # so this still lets the network express "totally
                         # uncertain" without exploding gradients.

    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in self.HIDDEN_DIMS:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, output_dim)
        self.log_var_head = nn.Linear(prev, output_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mean_head(h)
        log_var = self.log_var_head(h).clamp(min=self.LOG_VAR_MIN, max=self.LOG_VAR_MAX)
        return mu, log_var


def gaussian_nll(mu, log_var, target):
    """Per-dim Gaussian NLL averaged over batch and dims.

    NLL = 0.5 * (log σ² + (y − μ)² / σ²)

    The constant 0.5*log(2π) is dropped (it's a constant offset wrt parameters).
    Both terms together force calibrated predictions: the network is penalised
    for over-confidence (small σ² when error is large) AND for under-confidence
    (large σ² when error is small).
    """
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (target - mu).pow(2) * inv_var).mean()


# ============================================================================
# Training one ensemble member
# ============================================================================

def make_split_indices(n: int, generator_seed: int = 0):
    """Deterministic 80/10/10 train/val/test split.  Returns three int arrays."""
    g = torch.Generator().manual_seed(generator_seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val :],
    )


def train_one_member(
    features: np.ndarray,
    labels_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    init_seed: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
    ckpt_path: Path,
) -> float:
    """Train a single ensemble member; save best checkpoint by val NLL."""
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    X = torch.from_numpy(features).float()
    y = torch.from_numpy(labels_norm).float()
    full = TensorDataset(X, y)
    train_set = Subset(full, train_idx.tolist())
    val_set = Subset(full, val_idx.tolist())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(init_seed),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = GaussianMLP(features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val = float("inf")
    epochs_since_best = 0

    for epoch in range(n_epochs):
        # ---- train ----
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, log_var = model(xb)
            loss = gaussian_nll(mu, log_var, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, log_var = model(xb)
                tot += gaussian_nll(mu, log_var, yb).item() * xb.size(0)
                cnt += xb.size(0)
            val_loss = tot / cnt

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(
                    f"    early stop @ epoch {epoch+1}  "
                    f"(best val NLL = {best_val:.4f})"
                )
                break

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  val NLL = {val_loss:.4f}  (best {best_val:.4f})")

    return best_val


# ============================================================================
# Ensemble evaluation
# ============================================================================

def evaluate_ensemble(
    member_ckpts,
    features: np.ndarray,
    labels: np.ndarray,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    test_idx: np.ndarray,
    device: str,
):
    """Run ensemble on test set; report MAE/RMSE in physical units and 1σ/2σ coverage."""
    input_dim = features.shape[1]
    models = []
    for ckpt in member_ckpts:
        m = GaussianMLP(input_dim).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)

    Xte = torch.from_numpy(features[test_idx]).float().to(device)
    yte_phys = labels[test_idx]

    with torch.no_grad():
        mus_k = []
        vars_k = []
        for m in models:
            mu, log_var = m(Xte)
            mus_k.append(mu)
            vars_k.append(log_var.exp())
        mu_stack = torch.stack(mus_k)        # (M, B, 4) — normalised space
        var_stack = torch.stack(vars_k)      # (M, B, 4) — normalised space
        mu_ens = mu_stack.mean(0)
        var_ens = (var_stack + mu_stack.pow(2)).mean(0) - mu_ens.pow(2)
        var_ens = var_ens.clamp_min(1e-12)

    mu_ens = mu_ens.cpu().numpy()
    var_ens = var_ens.cpu().numpy()

    # ---- denormalise to physical units ----
    mu_phys = mu_ens * label_std + label_mean
    sigma_phys = np.sqrt(var_ens) * label_std

    err = mu_phys - yte_phys
    units_scale = np.array([1e9, 1e6, 1e6, 1.0])
    units_name = ["GHz", "µs", "µs", " - "]
    label_names = ["f_q", "T1", "T2", "amp_pi"]

    print()
    print("=" * 70)
    print("  Test-set evaluation  (mixture-of-Gaussians ensemble)")
    print("=" * 70)
    print(f"  {'Param':<8}{'MAE':>14}{'RMSE':>14}{'cov(1σ)':>10}{'cov(2σ)':>10}")
    print(f"  {'-'*8}{'':>14}{'':>14}{'':>10}{'':>10}")

    miscalibrated = []
    for i, name in enumerate(label_names):
        mae = np.mean(np.abs(err[:, i])) / units_scale[i]
        rmse = np.sqrt(np.mean(err[:, i] ** 2)) / units_scale[i]
        within_1s = float(np.mean(np.abs(err[:, i]) <= sigma_phys[:, i]))
        within_2s = float(np.mean(np.abs(err[:, i]) <= 2 * sigma_phys[:, i]))
        print(
            f"  {name:<8}"
            f"{mae:>10.4f} {units_name[i]}"
            f"{rmse:>10.4f} {units_name[i]}"
            f"{within_1s:>9.1%}"
            f"{within_2s:>9.1%}"
        )
        # Calibration-check tolerance: ±10 percentage points.  Larger than
        # this means the ensemble is meaningfully over- or under-confident,
        # which usually means more data or a wider ensemble.
        if abs(within_1s - 0.68) > 0.10:
            miscalibrated.append(
                f"  [WARN] {name}: 1σ coverage {within_1s:.1%} (target 68%)"
            )
        if abs(within_2s - 0.95) > 0.05:
            miscalibrated.append(
                f"  [WARN] {name}: 2σ coverage {within_2s:.1%} (target 95%)"
            )

    if miscalibrated:
        print()
        print("  Calibration warnings:")
        for w in miscalibrated:
            print(w)
        print(
            "  (Re-fit with more training data, more ensemble members, or "
            "longer training to improve calibration.)"
        )
    else:
        print()
        print("  Calibration OK — coverage near 68% / 95% on all parameters.")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="dataset.npz")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--split-seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ...")
    data = np.load(args.dataset)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float64)
    n = features.shape[0]
    print(f"  features: {features.shape}, labels: {labels.shape}")

    # ---- 80/10/10 split (deterministic) ----
    train_idx, val_idx, test_idx = make_split_indices(n, args.split_seed)
    print(
        f"  split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test"
    )

    # ---- Normalise labels using TRAIN statistics only ----
    # (Using train-only stats prevents test-set information leaking into the
    # normalisation, which would inflate apparent coverage.)
    label_mean = labels[train_idx].mean(0)
    label_std = labels[train_idx].std(0) + 1e-12
    labels_norm = (labels - label_mean) / label_std

    # ---- Train M members ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    member_ckpts = []
    for k in range(args.n_members):
        print(f"\n=== Member {k+1}/{args.n_members} ===")
        ckpt_path = out_dir / f"member_{k}.pt"
        train_one_member(
            features=features,
            labels_norm=labels_norm,
            train_idx=train_idx,
            val_idx=val_idx,
            init_seed=1000 + k,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
            ckpt_path=ckpt_path,
        )
        member_ckpts.append(ckpt_path)

    # ---- Persist normalisation, split, and grids alongside checkpoints ----
    np.savez(
        out_dir / "norm_stats.npz",
        label_mean=label_mean,
        label_std=label_std,
        input_dim=np.int64(features.shape[1]),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        label_names=np.array(["f_q", "T1", "T2", "amp_pi"]),
        label_units=np.array(["Hz", "s", "s", "dimensionless"]),
    )
    np.savez(
        out_dir / "sweep_grids.npz",
        spec_freqs=data["spec_freqs"],
        rabi_amps=data["rabi_amps"],
        t1_delays=data["t1_delays"],
        ramsey_delays=data["ramsey_delays"],
        spec_amp=data["spec_amp"],
        spec_pulse_length=data["spec_pulse_length"],
        gauss_pulse_length=data["gauss_pulse_length"],
        shots=data["shots"],
        ramsey_detuning=data["ramsey_detuning"],
    )

    # ---- Evaluate the ensemble on the test set ----
    evaluate_ensemble(
        member_ckpts=member_ckpts,
        features=features,
        labels=labels,
        label_mean=label_mean,
        label_std=label_std,
        test_idx=test_idx,
        device=device,
    )

    print(f"\nSaved {args.n_members} members + norm_stats + sweep_grids to {out_dir}")


if __name__ == "__main__":
    main()
