"""Generate a supervised dataset of (P1 features, hidden parameters) pairs.

Run:
    python generate_dataset.py --n-qubits 2000 --workers 8 --out dataset.npz

For each seed in 0..N-1 we:
    1.  Instantiate VirtualQubit(seed=seed) and read the hidden parameters
        directly off the object (this is allowed at training time — the
        network never sees the seed itself).
    2.  Run the four measurement sweeps with the pre-compiled waveforms,
        producing a single concatenated P1 feature vector.
    3.  Store the (features, labels) pair.

Performance notes
-----------------
* `session.compile()` is called exactly TWICE in this script — once for the
  square spec pulse, once for the unit-amplitude Gaussian — and never inside
  the per-qubit loop.  See qubit_measurements.precompile_waveforms.
* Per-qubit work is dispatched to a `ProcessPoolExecutor`.  Workers receive
  ONLY plain numpy arrays (the cached envelopes); LabOne Q objects never
  cross the process boundary because they aren't fork-safe.
* On Windows multiprocessing uses spawn, so the worker re-imports
  `qubit_measurements` and `qubit` cleanly without copying the parent's
  LabOne Q state.
"""
from __future__ import annotations

# ── Thread limits — must come before NumPy/SciPy/QuTiP are imported ────────
# Each worker process inherits these before loading any BLAS-linked library.
# Without this: 56 processes × 4 internal BLAS threads = 224 threads on 64
# cores → CPU thrashing, slower than single-threaded.
# With this:    56 processes × 1 thread each = clean near-linear scaling.
import os
os.environ["OMP_NUM_THREADS"]        = "1"   # OpenMP
os.environ["OPENBLAS_NUM_THREADS"]   = "1"   # OpenBLAS (common NumPy backend)
os.environ["MKL_NUM_THREADS"]        = "1"   # Intel MKL
os.environ["NUMEXPR_NUM_THREADS"]    = "1"   # NumExpr
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # Apple Accelerate (macOS)
# ───────────────────────────────────────────────────────────────────────────


import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Make sure we can `import qubit` whether the user runs this from
# ZturingMachine/ or from the project root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qubit_measurements as qm  # noqa: E402
from qubit import VirtualQubit  # noqa: E402


# ----------------------------------------------------------------------------
# Worker function — must be top-level so it pickles cleanly on Windows spawn.
# ----------------------------------------------------------------------------
def process_one_qubit(args):
    """Simulate one qubit; return (seed, features, labels).

    args is a tuple so the executor can submit it as a single positional
    argument (cleaner than partial when there are many array params).
    """
    (seed, t_spec, wave_spec, t_pulse, wave_gauss,
     spec_freqs, rabi_amps, t1_delays, ramsey_delays,
     shots, ramsey_detuning, gauss_pulse_length) = args

    qubit = VirtualQubit(seed=int(seed))

    # Ground-truth labels — 4-vector matching the network's output order.
    labels = np.array(
        [
            qubit._fq,                                              # Hz
            qubit._T1,                                              # s
            qubit._T2,                                              # s
            np.pi / (qubit._omega * float(gauss_pulse_length)),    # dimensionless
        ],
        dtype=np.float64,
    )

    features = qm.measure_qubit_features(
        qubit,
        t_spec,
        wave_spec,
        t_pulse,
        wave_gauss,
        spec_freqs=spec_freqs,
        rabi_amps=rabi_amps,
        t1_delays=t1_delays,
        ramsey_delays=ramsey_delays,
        shots=int(shots),
        ramsey_detuning=float(ramsey_detuning),
    )

    return int(seed), features, labels


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-qubits", type=int, default=2000,
        help="Number of qubits to simulate. 2000 gives a comfortable train "
             "set for the ~200k-parameter MLP. Drop to 200 for a smoke test.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker process count. Default = os.cpu_count() - 1.",
    )
    parser.add_argument(
        "--out", type=str, default=str(_HERE / "dataset.npz"),
        help="Output .npz file path.",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="First seed (useful for generating an extension batch later).",
    )
    args = parser.parse_args()

    n = int(args.n_qubits)
    n_workers = args.workers if args.workers else max(1, (os_cpu() - 1))

    # ------------------------------------------------------------------
    # 1. Compile waveforms ONCE (main process only — LabOne Q is not
    #    fork-safe).
    # ------------------------------------------------------------------
    print(f"[1/3] Compiling LabOne Q waveforms (one-time)...")
    t0 = time.time()
    t_spec, wave_spec, t_pulse, wave_gauss = qm.precompile_waveforms()
    print(
        f"      done in {time.time()-t0:.1f}s — "
        f"spec wave len={len(wave_spec)}, gauss wave len={len(wave_gauss)}"
    )

    # ------------------------------------------------------------------
    # 2. Dispatch per-qubit simulations to workers.
    # ------------------------------------------------------------------
    print(f"[2/3] Simulating {n} qubits across {n_workers} workers...")
    feat_dim = qm.feature_dim()
    features_arr = np.empty((n, feat_dim), dtype=np.float32)
    labels_arr = np.empty((n, 4), dtype=np.float64)

    payload_template = (
        # filled in per-seed below; this is the static part shared across all
        t_spec, wave_spec, t_pulse, wave_gauss,
        qm.SPEC_FREQS, qm.RABI_AMPS, qm.T1_DELAYS, qm.RAMSEY_DELAYS,
        qm.SHOTS, qm.RAMSEY_DETUNING, qm.GAUSS_PULSE_LENGTH,
    )

    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                process_one_qubit,
                (args.seed_offset + i,) + payload_template,
            ): i
            for i in range(n)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            seed, features, labels = fut.result()
            features_arr[i] = features
            labels_arr[i] = labels
            n_done += 1
            if n_done % max(1, n // 20) == 0 or n_done == n:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (n - n_done) / rate if rate > 0 else 0.0
                print(
                    f"      {n_done:5d}/{n}   "
                    f"{rate:5.1f} qubits/s   "
                    f"ETA {eta:6.1f}s"
                )

    print(f"      done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Save dataset + grids + metadata.  Grids are stored so inference
    #    code can reconstruct the exact same probes.
    # ------------------------------------------------------------------
    print(f"[3/3] Saving to {args.out} ...")
    np.savez_compressed(
        args.out,
        features=features_arr,
        labels=labels_arr,
        spec_freqs=qm.SPEC_FREQS,
        rabi_amps=qm.RABI_AMPS,
        t1_delays=qm.T1_DELAYS,
        ramsey_delays=qm.RAMSEY_DELAYS,
        spec_amp=qm.SPEC_AMP,
        spec_pulse_length=qm.SPEC_PULSE_LENGTH,
        gauss_pulse_length=qm.GAUSS_PULSE_LENGTH,
        shots=qm.SHOTS,
        ramsey_detuning=qm.RAMSEY_DETUNING,
        label_names=np.array(["f_q", "T1", "T2", "amp_pi"]),
        label_units=np.array(["Hz", "s", "s", "dimensionless"]),
    )

    # Quick sanity report.
    print()
    print(f"  features shape: {features_arr.shape}  (dtype={features_arr.dtype})")
    print(f"  labels   shape: {labels_arr.shape}   (dtype={labels_arr.dtype})")
    print()
    print("  Label statistics (post-generation sanity check):")
    names = ["f_q [GHz]", "T1 [µs] ", "T2 [µs] ", "amp_pi  "]
    scales = [1e-9, 1e6, 1e6, 1.0]
    for i, (n_, s) in enumerate(zip(names, scales)):
        col = labels_arr[:, i] * s
        print(
            f"    {n_}  mean={col.mean():.4f}  std={col.std():.4f}  "
            f"min={col.min():.4f}  max={col.max():.4f}"
        )


def os_cpu():
    """os.cpu_count() with a safe fallback."""
    import os
    return os.cpu_count() or 4


if __name__ == "__main__":
    main()