"""
plots.py - Visualisation helpers for the parallel tune-up campaign.

Three plots that together tell the story for the writeup:

  1. plot_pipeline_graph(result)
       The decision graph for a single tune-up: which experiments were
       chosen, in order, with success/failure marked on each edge.

  2. plot_parameter_spread(results)
       Histograms of fitted (f_q, amp_pi, T1, T2) across all qubits and
       repeats. Bimodal distributions flag calibration that lands on
       the wrong cosine extremum etc.

  3. plot_runtime_dashboard(results)
       Status pie + iterations distribution + wall-time histogram --
       the "health dashboard".
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from .orchestrator import TuneUpResult, PARAMS


_EXP_COLORS = {
    "spectroscopy":   "#1f77b4",
    "amplitude_rabi": "#2ca02c",
    "t1":             "#d62728",
    "ramsey":         "#9467bd",
}


def plot_pipeline_graph(result: TuneUpResult, ax=None):
    """Draw the per-iteration experiment trace as a horizontal flowchart."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(-0.5, max(1, len(result.log)) + 0.5)
    ax.set_ylim(-1, 1.2)
    ax.axis("off")

    for i, entry in enumerate(result.log):
        name = entry.get("experiment", "?")
        ok = "error" not in entry
        c = _EXP_COLORS.get(name, "gray")
        ec = "black" if ok else "red"
        ax.add_patch(plt.Rectangle((i - 0.35, -0.3), 0.7, 0.6,
                                   facecolor=c, edgecolor=ec,
                                   linewidth=2 if not ok else 1, alpha=0.85))
        ax.text(i, 0.0, name, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        if not ok:
            ax.text(i, -0.55, "✗", ha="center", color="red",
                    fontsize=14, fontweight="bold")
        if i + 1 < len(result.log):
            ax.annotate("", xy=(i + 0.6, 0), xytext=(i + 0.4, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title(f"Qubit {result.qubit_id} run #{result.repeat} - "
                 f"status={result.status}, iters={result.iterations}, "
                 f"wall={result.wall_time_s:.1f}s")
    return ax


def plot_parameter_spread(results: list[TuneUpResult], save_path=None):
    """Histograms of fitted parameters across the campaign."""
    units = {"f_q": ("GHz", 1e-9), "amp_pi": ("a.u.", 1.0),
             "T1": ("us", 1e6), "T2": ("us", 1e6)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, p in zip(axes.flat, PARAMS):
        vals = [r.state[p]["mean"] for r in results if r.state.get(p)]
        if not vals:
            ax.text(0.5, 0.5, f"no {p} data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(p)
            continue
        unit, scale = units[p]
        vals = np.asarray(vals) * scale
        ax.hist(vals, bins=30, edgecolor="black", alpha=0.75)
        ax.axvline(np.median(vals), color="red", linestyle="--",
                   label=f"median = {np.median(vals):.3g} {unit}")
        ax.set_title(f"{p}  (n={len(vals)})")
        ax.set_xlabel(f"{p} [{unit}]")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
    fig.suptitle(f"Parameter spread across {len(results)} tune-ups",
                 fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_runtime_dashboard(results: list[TuneUpResult], save_path=None):
    """Three-panel dashboard: status pie / iterations / wall time."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    statuses = Counter(r.status for r in results)
    color_map = {"ok": "#2ca02c", "partial": "#ff7f0e", "failed": "#d62728"}
    axes[0].pie(statuses.values(), labels=list(statuses.keys()),
                colors=[color_map.get(k, "gray") for k in statuses],
                autopct="%1.0f%%", startangle=90)
    axes[0].set_title("Status distribution")

    iters = [r.iterations for r in results]
    if iters:
        axes[1].hist(iters, bins=range(1, max(iters) + 2),
                     align="left", rwidth=0.8, edgecolor="black")
    axes[1].set_title("Iterations to converge")
    axes[1].set_xlabel("iterations")
    axes[1].set_ylabel("count")

    walltime = [r.wall_time_s for r in results]
    if walltime:
        axes[2].hist(walltime, bins=20, edgecolor="black", alpha=0.75)
        axes[2].axvline(np.mean(walltime), color="red", linestyle="--",
                        label=f"mean = {np.mean(walltime):.1f}s")
        axes[2].legend()
    axes[2].set_title("Wall time per tune-up")
    axes[2].set_xlabel("seconds")

    fig.suptitle(f"Tune-up campaign dashboard  ({len(results)} runs)",
                 fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
