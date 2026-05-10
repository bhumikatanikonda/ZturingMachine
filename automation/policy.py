"""
policy.py - Decision policy that replaces the LLM in our k-agents loop.

For four experiments with clear dependencies, the "what to run next"
logic fits in a simple if/elif tree. This is the deterministic
counterpart of the paper's LLM agent.

Policy contract:
    decide(state, failures, budget_remaining) -> Decision | None

Decision is (experiment_name, kwargs, rationale).
None means "give up / nothing useful left to try" (loop exits).

State convention:
    state[name] is None (unknown) or dict {mean, std, rms, rel_unc}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REL_UNC_TARGET = 0.05          # 5 % relative uncertainty
RMS_TARGET = 0.06              # fit residual ceiling
PARAMS = ("f_q", "amp_pi", "T1", "T2")


@dataclass
class Decision:
    experiment: str
    kwargs: dict[str, Any]
    rationale: str


# -- helpers ---------------------------------------------------------------

def _converged(entry) -> bool:
    if entry is None:
        return False
    return (entry["rel_unc"] < REL_UNC_TARGET) and (entry["rms"] < RMS_TARGET)


def _failure_count(failures, exp_name):
    return sum(1 for f in failures if f.get("experiment") == exp_name)


def _all_converged(state) -> bool:
    return all(_converged(state.get(p)) for p in PARAMS)


# -- the policy itself -----------------------------------------------------

class RulePolicy:
    """Deterministic next-experiment picker.

    Follows the natural dependency order
        spec -> Rabi -> {T1, Ramsey}
    with three layers of recovery on failure:

      1) widen the sweep / increase shots,
      2) keep retrying the same experiment with bigger parameters,
      3) give up on a parameter (return None) after 3 attempts.
    """

    def decide(self, state, failures, budget_remaining) -> Decision | None:
        if _all_converged(state):
            return None
        if budget_remaining <= 0:
            return None

        # ---- 1. f_q ----------------------------------------------------
        if not _converged(state.get("f_q")):
            n_fail = _failure_count(failures, "spectroscopy")
            if n_fail >= 3:
                return None
            return Decision(
                experiment="spectroscopy",
                kwargs={
                    "freq_start": 5.0e9,
                    "freq_stop":  6.2e9,
                    "n_points":   81 + 40 * n_fail,
                    "shots":      2000 + 1000 * n_fail,
                    "pulse_length": 2e-6,
                },
                rationale=(
                    f"f_q unknown / unconverged; spectroscopy attempt "
                    f"#{n_fail + 1}"
                ),
            )

        # ---- 2. amp_pi -------------------------------------------------
        if not _converged(state.get("amp_pi")):
            n_fail = _failure_count(failures, "amplitude_rabi")
            if n_fail >= 3:
                return None
            return Decision(
                experiment="amplitude_rabi",
                kwargs={
                    "drive_freq": state["f_q"]["mean"],
                    # widen amplitude range on retry (in case the cosine
                    # period is longer than expected)
                    "amp_start":  0.01,
                    "amp_stop":   1.0 + 0.5 * n_fail,
                    "n_points":   51 + 20 * n_fail,
                    "shots":      2000 + 500 * n_fail,
                    "pulse_length": 100e-9,
                },
                rationale=f"amp_pi unconverged; Rabi attempt #{n_fail + 1}",
            )

        # ---- 3. T1 -----------------------------------------------------
        if not _converged(state.get("T1")):
            n_fail = _failure_count(failures, "t1")
            if n_fail >= 3:
                return None
            return Decision(
                experiment="t1",
                kwargs={
                    "drive_freq":  state["f_q"]["mean"],
                    "amp_pi":      state["amp_pi"]["mean"],
                    "delay_start": 0.5e-6,
                    # Extend wait-time window if first pass failed to
                    # capture full decay
                    "delay_stop":  80e-6 + 40e-6 * n_fail,
                    "n_points":    25 + 10 * n_fail,
                    "shots":       1500 + 500 * n_fail,
                    "pulse_length": 100e-9,
                },
                rationale=f"T1 unconverged; T1 attempt #{n_fail + 1}",
            )

        # ---- 4. T2 (Ramsey also refines f_q) ---------------------------
        if not _converged(state.get("T2")):
            n_fail = _failure_count(failures, "ramsey")
            if n_fail >= 3:
                return None
            return Decision(
                experiment="ramsey",
                kwargs={
                    "drive_freq":  state["f_q"]["mean"],
                    "amp_pi2":     state["amp_pi"]["mean"] / 2.0,
                    "detuning":    1e6,
                    "delay_start": 0.0,
                    # default 60 us covers ~3*T2; extend further on retry
                    "delay_stop":  60e-6 + 20e-6 * n_fail,
                    "n_points":    80 + 20 * n_fail,
                    "shots":       1500 + 500 * n_fail,
                    "pulse_length": 100e-9,
                },
                rationale=f"T2 unconverged; Ramsey attempt #{n_fail + 1}",
            )

        return None
