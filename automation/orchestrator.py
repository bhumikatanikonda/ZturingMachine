"""
orchestrator.py - per-qubit closed-loop tune-up + parallel scaling.

The k-agents inner loop, deterministic policy version:

    while not converged and budget remaining:
        decision = policy.decide(state, failures, budget)
        result   = registry[decision.experiment].run(...)
        state.update(result)

`run_parallel_tuneups` runs N qubits x M repeats. Use n_workers>1 to
launch a multiprocessing pool (the QuTiP solver is single-threaded
under the GIL, so plain threads do not help).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .policy import RulePolicy, PARAMS, _converged
from .registry import Registry


# -- one tune-up run --------------------------------------------------------

@dataclass
class TuneUpResult:
    qubit_id: int
    repeat:   int
    state: dict[str, dict | None] = field(default_factory=dict)
    log:   list[dict]             = field(default_factory=list)
    failures: list[dict]          = field(default_factory=list)
    iterations: int = 0
    wall_time_s: float = 0.0
    converged_params: list[str]   = field(default_factory=list)
    status: str = "pending"       # pending / ok / partial / failed


def _annotate(entry):
    """Add rel_unc to a state entry."""
    if entry is None:
        return None
    e = dict(entry)
    m = abs(e.get("mean", 0.0)) or 1.0
    e["rel_unc"] = e.get("std", float("inf")) / m
    return e


class QubitOrchestrator:
    """k-agents inner loop, scoped to one qubit."""

    def __init__(self, registry: Registry, policy=None,
                 max_iter: int = 8) -> None:
        self.registry = registry
        self.policy = policy or RulePolicy()
        self.max_iter = max_iter

    def run(self, virtual_qubit, session, device_setup, q0_loneq,
            qubit_id: int = 0, repeat: int = 0) -> TuneUpResult:
        result = TuneUpResult(qubit_id=qubit_id, repeat=repeat,
                              state={p: None for p in PARAMS})
        t0 = time.perf_counter()

        for it in range(self.max_iter):
            decision = self.policy.decide(
                state={k: _annotate(v) for k, v in result.state.items()},
                failures=result.failures,
                budget_remaining=self.max_iter - it,
            )
            if decision is None:
                break  # converged or out of options

            entry = self.registry.get(decision.experiment)
            log_entry = {
                "iter": it,
                "experiment": decision.experiment,
                "rationale": decision.rationale,
                "kwargs": dict(decision.kwargs),
            }

            try:
                produced = entry.runner(virtual_qubit, session, device_setup,
                                        q0_loneq, **decision.kwargs)
                for k, v in produced.items():
                    if k in result.state:
                        result.state[k] = v
                log_entry["result"] = produced
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                log_entry["error"] = msg
                result.failures.append({
                    "experiment": decision.experiment,
                    "kwargs": dict(decision.kwargs),
                    "error": msg,
                    "iter": it,
                })

            result.log.append(log_entry)
            result.iterations = it + 1

        result.wall_time_s = time.perf_counter() - t0

        result.converged_params = [
            p for p in PARAMS if _converged(_annotate(result.state[p]))
        ]
        if len(result.converged_params) == len(PARAMS):
            result.status = "ok"
        elif len(result.converged_params) > 0:
            result.status = "partial"
        else:
            result.status = "failed"

        return result


# -- parallel scaling -------------------------------------------------------

def run_parallel_tuneups(
    qubit_factory: Callable[[int, int], Any],     # (qid, repeat) -> qubit
    session_factory: Callable[[int], tuple],      # (qid) -> (sess, ds, q0_loneq)
    registry: Registry,
    n_qubits: int = 50,
    n_repeats: int = 20,
    max_iter: int = 8,
    n_workers: int = 1,                            # 1 = serial; >1 = processes
    progress_every: int = 5,
) -> list[TuneUpResult]:
    """Run a tune-up grid: n_qubits x n_repeats independent jobs.

    Notes:
      - QuTiP solver is single-threaded under the GIL; threading does not
        help. Use n_workers > 1 to launch separate Python processes.
      - Each process needs its own LabOne Q session: that's `session_factory`.
      - Total cost ~= n_qubits x n_repeats x time_per_tuneup.
        Budget realistically. Smoke-test with (n_qubits=2, n_repeats=1) first.
    """
    jobs = [(qid, rep) for qid in range(n_qubits) for rep in range(n_repeats)]
    results: list[TuneUpResult] = []
    t0 = time.perf_counter()

    if n_workers <= 1:
        for i, (qid, rep) in enumerate(jobs):
            r = _run_one(qid, rep, qubit_factory, session_factory,
                         registry, max_iter)
            results.append(r)
            if (i + 1) % progress_every == 0 or (i + 1) == len(jobs):
                _print_progress(i + 1, len(jobs), t0, results)
        return results

    import multiprocessing as mp
    with mp.Pool(n_workers) as pool:
        async_results = [
            pool.apply_async(
                _run_one,
                (qid, rep, qubit_factory, session_factory,
                 registry, max_iter),
            )
            for qid, rep in jobs
        ]
        for i, ar in enumerate(async_results):
            results.append(ar.get())
            if (i + 1) % progress_every == 0 or (i + 1) == len(jobs):
                _print_progress(i + 1, len(jobs), t0, results)
    return results


def _run_one(qid, rep, qubit_factory, session_factory, registry, max_iter):
    try:
        vq = qubit_factory(qid, rep)
        sess, ds, q0_loneq = session_factory(qid)
        orch = QubitOrchestrator(registry, max_iter=max_iter)
        return orch.run(vq, sess, ds, q0_loneq, qubit_id=qid, repeat=rep)
    except Exception:
        r = TuneUpResult(qubit_id=qid, repeat=rep,
                         state={p: None for p in PARAMS})
        r.status = "failed"
        r.log.append({"iter": -1, "fatal": traceback.format_exc()})
        return r


def _print_progress(done, total, t0, results):
    elapsed = time.perf_counter() - t0
    eta = elapsed / done * (total - done) if done else 0.0
    ok      = sum(1 for r in results if r.status == "ok")
    partial = sum(1 for r in results if r.status == "partial")
    failed  = sum(1 for r in results if r.status == "failed")
    print(f"[{done:>4}/{total}]  ok={ok}  partial={partial}  failed={failed}"
          f"  elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
          flush=True)


# -- summary helpers --------------------------------------------------------

def summarise(results: list[TuneUpResult]) -> dict:
    n = len(results)
    by_status: dict[str, int] = {"ok": 0, "partial": 0, "failed": 0}
    iters, walltime = [], []
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        iters.append(r.iterations)
        walltime.append(r.wall_time_s)
    return {
        "n_total": n,
        "by_status": by_status,
        "mean_iterations": (sum(iters) / n) if n else 0,
        "mean_wall_s":     (sum(walltime) / n) if n else 0,
        "total_wall_s":    sum(walltime),
    }


def to_records(results: list[TuneUpResult]) -> list[dict]:
    out = []
    for r in results:
        row = {"qubit_id": r.qubit_id, "repeat": r.repeat,
               "status": r.status, "iterations": r.iterations,
               "wall_s": r.wall_time_s}
        for p in PARAMS:
            v = r.state.get(p)
            if v is None:
                row[f"{p}_mean"] = None
                row[f"{p}_std"] = None
            else:
                row[f"{p}_mean"] = v["mean"]
                row[f"{p}_std"] = v["std"]
        out.append(row)
    return out
