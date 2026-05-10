"""
run_campaign.py - top-level driver. Place next to your main.py.

Imports the experiment-builder helpers from your existing main.py and
wires them into the registry. Runs N qubits x M repeats with optional
multiprocessing, saves a JSON of all results, and writes both
diagnostic plots.

Run with:
    python run_campaign.py --n_qubits 50 --n_repeats 20 --workers 4

For a smoke test before committing to a long run:
    python run_campaign.py --n_qubits 2 --n_repeats 1 --workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the parent directory (where main.py + qubit.py live) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qubit import VirtualQubit
from main import (
    make_session,
    make_spec_experiment,
    make_rabi_experiment,
    get_drive_waveform,
)

from automation import (
    build_default_registry,
    run_parallel_tuneups,
    summarise,
    to_records,
    plot_parameter_spread,
    plot_runtime_dashboard,
)


# -- factories: must be top-level for multiprocessing pickling -----------

def qubit_factory(qid: int, repeat: int) -> VirtualQubit:
    """Different qubit per ID; reproducible per (qid, repeat)."""
    return VirtualQubit(seed=10_000 * qid + repeat)


def session_factory(qid: int):
    """Build LabOne Q session per worker process.

    NOTE: this rebuilds the session for every job. In serial mode that's
    overhead per job; in multiprocessing mode it's overhead per process
    only after the first one (since each process keeps its session). If
    you find session creation dominates, refactor to use a Pool initializer.
    """
    return make_session()


# -- main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_qubits", type=int, default=50)
    p.add_argument("--n_repeats", type=int, default=20)
    p.add_argument("--max_iter", type=int, default=8)
    p.add_argument("--workers", type=int, default=1,
                   help="1 = serial; >1 launches multiprocessing pool")
    p.add_argument("--output", type=str, default="campaign_results.json")
    args = p.parse_args()

    registry = build_default_registry(
        make_spec_experiment=make_spec_experiment,
        make_rabi_experiment=make_rabi_experiment,
        get_drive_waveform=get_drive_waveform,
    )

    print(f"Campaign: {args.n_qubits} qubits x {args.n_repeats} repeats "
          f"= {args.n_qubits * args.n_repeats} tune-ups, "
          f"workers={args.workers}")

    results = run_parallel_tuneups(
        qubit_factory=qubit_factory,
        session_factory=session_factory,
        registry=registry,
        n_qubits=args.n_qubits,
        n_repeats=args.n_repeats,
        max_iter=args.max_iter,
        n_workers=args.workers,
    )

    summary = summarise(results)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, default=str))

    records = to_records(results)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "records": records}, f,
                  indent=2, default=str)
    print(f"\nWrote {args.output}")

    plot_parameter_spread(results, save_path="campaign_spread.png")
    plot_runtime_dashboard(results, save_path="campaign_dashboard.png")
    print("Wrote campaign_spread.png + campaign_dashboard.png")


if __name__ == "__main__":
    main()
