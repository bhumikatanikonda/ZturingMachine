"""
registry.py - k-agents-style experiment registry, adapted to ZturingMachine.

The ZturingMachine main.py functions (qubit_spectroscopy, amplitude_rabi,
measure_T1, ramsey) return only fitted values, not the underlying data.
For our convergence check we need (mean, std, rms), so each runner here
reproduces the data-gathering loop using the SAME experiment builders
(make_spec_experiment / make_rabi_experiment / make_ramsey_experiment)
and then fits with proper uncertainty.

This is intentional: the registry is the single place where "what to
measure + how to fit" is encapsulated, mirroring the paper's
"knowledge encapsulation" agent. The notebook fitters return point
estimates; we add the statistical layer here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import curve_fit


# -- fit models -------------------------------------------------------------

def _lorentzian(f, f0, w, A, c):
    return c + A / (1.0 + ((f - f0) / (w / 2.0)) ** 2)


def _rabi_cos(a, A, a_pi, phi, off):
    """Standard amplitude-Rabi cosine: P1(a) = off + A*(1-cos(pi*a/a_pi + phi))/2."""
    return off + 0.5 * A * (1.0 - np.cos(np.pi * a / a_pi + phi))


def _exp_decay(t, A, tau, c):
    return c + A * np.exp(-t / tau)


def _damped_cos(t, A, T2, fr, phi, c):
    return c + A * np.exp(-t / T2) * np.cos(2 * np.pi * fr * t + phi)


# -- standardised result triple --------------------------------------------

def _ms(mean, std, rms):
    return {"mean": float(mean), "std": float(std), "rms": float(rms)}


def _fit(model, x, y, p0, bounds=(-np.inf, np.inf)):
    """curve_fit -> (popt, perr, rms). Raises on non-finite output."""
    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    rms = float(np.sqrt(np.mean((y - model(x, *popt)) ** 2)))
    if not (np.all(np.isfinite(popt)) and np.all(np.isfinite(perr))):
        raise ValueError("non-finite fit result")
    return popt, perr, rms


# -- Registry entry ---------------------------------------------------------

@dataclass
class Experiment:
    name: str
    description: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    runner: Callable


class Registry:
    def __init__(self) -> None:
        self._exps: dict[str, Experiment] = {}

    def register(self, exp: Experiment) -> None:
        self._exps[exp.name] = exp

    def get(self, name: str) -> Experiment | None:
        return self._exps.get(name)

    def all(self) -> Iterable[Experiment]:
        return self._exps.values()


# ====================================================================
# Runner factories - mirror the loop from main.py so we can refit
# ====================================================================

def _make_spec_runner(make_spec_experiment, get_drive_waveform):
    """Spectroscopy: sweep + Lorentzian fit -> f_q (mean, std, rms)."""
    def run(vq, sess, ds, q0_loneq, **kw):
        f_lo = float(kw.get("freq_start", 5.0e9))
        f_hi = float(kw.get("freq_stop",  6.2e9))
        n    = int(kw.get("n_points",     81))
        shots = int(kw.get("shots",       2000))
        plen = float(kw.get("pulse_length", 2e-6))

        freqs = np.linspace(f_lo, f_hi, n)
        P1 = np.empty(n)
        for i, f in enumerate(freqs):
            exp = make_spec_experiment(q0_loneq, drive_freq=f, pulse_length=plen)
            t, wave = get_drive_waveform(sess, ds, q0_loneq, exp, plen)
            vq.reset()
            vq.evolve(t, wave, drive_freq=f)
            P1[i] = vq.measure(shots=shots).mean()

        popt, perr, rms = _fit(
            _lorentzian, freqs, P1,
            p0=[float(freqs[int(np.argmax(P1))]),
                max(1e6, (f_hi - f_lo) / 10),
                float(np.ptp(P1)),
                float(np.min(P1))],
            bounds=([f_lo, 1e5, 0.0, -0.05],
                    [f_hi, f_hi - f_lo, 1.5, 0.5]),
        )
        return {"f_q": _ms(popt[0], perr[0], rms)}
    return run


def _make_rabi_runner(make_rabi_experiment, get_drive_waveform):
    """Amplitude Rabi: sweep + cosine fit -> amp_pi (mean, std, rms)."""
    def run(vq, sess, ds, q0_loneq, **kw):
        drive_freq = float(kw["drive_freq"])
        a_lo  = float(kw.get("amp_start", 0.01))
        a_hi  = float(kw.get("amp_stop",  1.0))
        n     = int(kw.get("n_points",   51))
        shots = int(kw.get("shots",      2000))
        plen  = float(kw.get("pulse_length", 100e-9))

        amps = np.linspace(a_lo, a_hi, n)
        P1 = np.empty(n)
        for i, a in enumerate(amps):
            exp = make_rabi_experiment(q0_loneq, amp=float(a), pulse_length=plen)
            t, wave = get_drive_waveform(sess, ds, q0_loneq, exp, plen)
            vq.reset()
            vq.evolve(t, wave, drive_freq=drive_freq)
            P1[i] = vq.measure(shots=shots).mean()

        a_g = float(amps[int(np.argmax(P1))])
        if a_g <= 0:
            a_g = 0.5 * (a_lo + a_hi)
        try:
            popt, perr, rms = _fit(
                _rabi_cos, amps, P1,
                p0=[float(np.ptp(P1)), a_g, 0.0, float(np.min(P1))],
                bounds=([0.0, max(amps[1] * 0.3, 1e-3), -np.pi, -0.05],
                        [1.5, max(amps[-1] * 2.0, a_hi * 2), np.pi, 0.6]),
            )
            return {"amp_pi": _ms(popt[1], perr[1], rms)}
        except Exception as exc:
            raise ValueError(f"Rabi fit failed: {exc}")
    return run


def _make_t1_runner(make_rabi_experiment, get_drive_waveform):
    """T1: pi pulse + variable wait + measure + exponential fit."""
    def run(vq, sess, ds, q0_loneq, **kw):
        drive_freq = float(kw["drive_freq"])
        amp_pi     = float(kw["amp_pi"])
        d_lo  = float(kw.get("delay_start", 0.5e-6))
        d_hi  = float(kw.get("delay_stop",  80e-6))
        n     = int(kw.get("n_points",      25))
        shots = int(kw.get("shots",         1500))
        plen  = float(kw.get("pulse_length", 100e-9))

        # Build pi pulse once
        exp_pi = make_rabi_experiment(q0_loneq, amp=amp_pi, pulse_length=plen)
        t_pi, wave_pi = get_drive_waveform(sess, ds, q0_loneq, exp_pi, plen)

        delays = np.linspace(d_lo, d_hi, n)
        P1 = np.empty(n)
        for i, delay in enumerate(delays):
            vq.reset()
            vq.evolve(t_pi, wave_pi, drive_freq=drive_freq)
            vq.wait(0.0, float(delay))
            P1[i] = vq.measure(shots=shots).mean()

        popt, perr, rms = _fit(
            _exp_decay, delays, P1,
            p0=[float(np.ptp(P1)), 30e-6, float(np.min(P1))],
            bounds=([0.0, 1e-7, -0.05], [1.5, 5e-3, 0.6]),
        )
        return {"T1": _ms(popt[1], perr[1], rms)}
    return run


def _make_ramsey_runner(make_rabi_experiment, get_drive_waveform):
    """Ramsey: two pi/2 pulses + free precession; damped cosine -> {f_q, T2}.

    Implementation note: the ZturingMachine main.py builds a `make_ramsey_experiment`
    that splits one waveform across two pulses, but it's easier and more reliable
    to play the same pi/2 envelope twice with `vq.wait(delay)` between them
    (both methods are mathematically equivalent on the simulator).

    The default `delay_stop` is set to 60 us so the T2 envelope (~20 us)
    is clearly visible. The notebook's default 5 us is too short.
    """
    def run(vq, sess, ds, q0_loneq, **kw):
        drive_freq = float(kw["drive_freq"])
        amp_pi2    = float(kw["amp_pi2"])
        detuning   = float(kw.get("detuning", 1e6))
        d_lo  = float(kw.get("delay_start", 0.0))
        d_hi  = float(kw.get("delay_stop",  60e-6))
        n     = int(kw.get("n_points",      80))
        shots = int(kw.get("shots",         1500))
        plen  = float(kw.get("pulse_length", 100e-9))

        f_drive = drive_freq + detuning

        # Build pi/2 pulse once (driven at the DETUNED frequency)
        exp_pi2 = make_rabi_experiment(q0_loneq, amp=amp_pi2, pulse_length=plen)
        t_pi2, wave_pi2 = get_drive_waveform(sess, ds, q0_loneq, exp_pi2, plen)

        delays = np.linspace(d_lo, d_hi, n)
        P1 = np.empty(n)
        for i, delay in enumerate(delays):
            vq.reset()
            vq.evolve(t_pi2, wave_pi2, drive_freq=f_drive)   # first pi/2
            vq.wait(0.0, float(delay))                            # free precession
            vq.evolve(t_pi2, wave_pi2, drive_freq=f_drive)   # second pi/2
            P1[i] = vq.measure(shots=shots).mean()

        try:
            popt, perr, rms = _fit(
                _damped_cos, delays, P1,
                p0=[float(np.ptp(P1) / 2),
                    20e-6, detuning, 0.0, float(np.mean(P1))],
                bounds=([0.0, 1e-7, 0.0, -np.pi, -0.1],
                        [1.0, 5e-3, 5 * detuning, np.pi, 1.0]),
            )
            f_q_refined = drive_freq + (popt[2] - detuning)
            return {
                "f_q": _ms(f_q_refined, perr[2], rms),
                "T2":  _ms(popt[1], perr[1], rms),
            }
        except Exception as exc:
            raise ValueError(
                f"Ramsey damped-cosine fit failed: {exc}. "
                f"Try increasing delay_stop "
                f"(current={d_hi*1e6:.1f} us; need >= 3*T2)."
            )
    return run


# -- One-shot factory -------------------------------------------------------

def build_default_registry(
    make_spec_experiment,
    make_rabi_experiment,
    get_drive_waveform,
) -> Registry:
    """Wire the four experiments into a Registry.

    Pass references to the LabOne Q experiment-builder helpers from your
    main.py (or notebook). Each runner reproduces the data-gathering loop
    internally so we can fit with proper uncertainty.
    """
    r = Registry()
    r.register(Experiment(
        name="spectroscopy",
        description=(
            "Square-pulse drive, sweep frequency, measure P1. "
            "Lorentzian fit -> f_q (mean, std from covariance, rms residuals)."
        ),
        requires=(),
        produces=("f_q",),
        runner=_make_spec_runner(make_spec_experiment, get_drive_waveform),
    ))
    r.register(Experiment(
        name="amplitude_rabi",
        description=(
            "Gaussian drive at drive_freq, sweep amplitude, measure P1. "
            "Cosine fit -> amp_pi (and amp_pi/2 = amp_pi/2)."
        ),
        requires=("f_q",),
        produces=("amp_pi",),
        runner=_make_rabi_runner(make_rabi_experiment, get_drive_waveform),
    ))
    r.register(Experiment(
        name="t1",
        description=(
            "Pi pulse, variable wait, measure. Exponential fit -> T1."
        ),
        requires=("f_q", "amp_pi"),
        produces=("T1",),
        runner=_make_t1_runner(make_rabi_experiment, get_drive_waveform),
    ))
    r.register(Experiment(
        name="ramsey",
        description=(
            "Two pi/2 pulses with controlled detuning. Damped-cosine fit "
            "-> refined f_q + T2*. Needs amp_pi2 = amp_pi/2."
        ),
        requires=("f_q", "amp_pi"),
        produces=("f_q", "T2"),
        runner=_make_ramsey_runner(make_rabi_experiment, get_drive_waveform),
    ))
    return r
