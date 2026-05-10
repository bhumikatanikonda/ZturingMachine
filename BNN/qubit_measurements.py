"""Shared measurement utilities for the qubit-characterisation ML pipeline.

This module is the bridge between the LabOne Q world (which only the main
process touches) and the per-qubit simulation world (run in worker processes
on plain numpy arrays).

Key design choice — the bottleneck is `session.compile()` (per the project
spec).  We compile each unique pulse SHAPE (square spec, unit Gaussian) ONCE
in the main process and broadcast the resulting envelope arrays to every
worker.  All amplitude scaling is done by simple multiplication on numpy
arrays — `amp * wave_gauss_unit` is a 100-sample multiply, free compared to
re-compilation.

The `fast_*` functions therefore take ALREADY-COMPILED `(t, wave)` numpy
arrays as arguments and never touch LabOne Q.  They are pickle-safe and may
be called from `ProcessPoolExecutor` workers without issue.
"""
from __future__ import annotations

import numpy as np


# ============================================================================
# Sweep-grid configuration — IDENTICAL at training and inference time.
# Anything that changes here invalidates the saved dataset/model.
# ============================================================================

# --- Pulse shapes ----------------------------------------------------------

SPEC_AMP = 0.5            # spectroscopy drive amplitude (matches main.py reference)
SPEC_PULSE_LENGTH = 2.0e-6   # 2 µs square pulse — long enough to saturate the line
GAUSS_PULSE_LENGTH = 100e-9  # 100 ns gaussian — same convention as main.py

# --- Shot count ------------------------------------------------------------
#
# Choice: 1024 shots per sweep point.
#
# Why 1024?
#   sigma_P1 = sqrt(p(1-p)/N).  At the noisiest point p=0.5 this gives
#   sigma_P1 ≈ 0.0156 — small relative to the natural ~0.5 swing of the
#   Rabi/Ramsey signal, so the *signal* dominates the *noise* in every
#   feature.  Doubling to 2048 only shrinks sigma to 0.011 — diminishing
#   returns.  Halving to 512 doubles noise to 0.022, which the network can
#   still recover from but with noticeably more epistemic uncertainty.
#
# CRITICAL: this number must match at inference time.  The network learns
# the specific noise level that comes with 1024 shots — using 256 shots at
# inference would inject noise the network was never calibrated against and
# break the predicted-variance calibration even though point predictions
# might still look OK.
SHOTS = 1024

# --- Ramsey detuning -------------------------------------------------------
#
# Drive 1 MHz off resonance so that the Ramsey curve oscillates at ~1 MHz
# regardless of the qubit.  This guarantees the fringe-period information
# (which encodes f_q precisely) is always present in the feature vector.
RAMSEY_DETUNING = 1.0e6

# --- Spectroscopy frequency grid ------------------------------------------
#
# Choice: 15 points across [5.0, 6.0] GHz (71.4 MHz spacing).
#
# Prior on _fq: U(5.0, 6.0) GHz — we cover exactly the prior, no margin.
# With only 15 points the network will rarely see the Lorentzian peak
# sharply resolved in a single bin; instead it sees a soft bump spread
# across 1–2 neighbouring bins and learns to interpolate from that shape.
# This is intentional — we are training the network to do the interpolation
# the curve-fitter would have done explicitly.
#
# Trade-off accepted: a qubit right at the prior boundary (f_q = 5.0 or
# 6.0 GHz) will have an asymmetric feature (peak at the edge of the sweep).
# The network sees many such examples during training and learns to handle
# them, but the predicted σ_fq will be larger for boundary qubits.
# If calibration is poor at the boundaries, extend to [4.95, 6.05] GHz.
SPEC_FREQS = np.linspace(5.0e9, 6.0e9, 15)

# --- Rabi amplitude grid --------------------------------------------------
#
# Choice: 10 points across [0.0, 1.0].
#
# WARNING — upper limit 1.0 is tight.
#   The regression target amp_pi = π/(_omega·L) with L=100 ns and the prior
#   on _omega gives amp_pi ∈ [0.57, 0.70].  With a Gaussian pulse the
#   *empirical* first Rabi maximum sits higher (Gaussian area < rect area),
#   typically around amp ≈ 0.8–1.1.  With an upper limit of 1.0 the sweep
#   may not reach the first peak for some qubits — the network will see only
#   the rising edge of the sinusoid, not the full period.
#
#   This is acceptable IF the dataset is large enough for the network to
#   learn "a rising edge that reaches P1≈0.9 at amp=1.0 implies the peak
#   is just above 1.0, which maps to amp_pi ≈ X."  With millions of
#   training examples this extrapolation is learnable.  With fewer examples
#   (< 10k) consider raising the upper limit to 1.5 or 2.0.
#
# Starting at 0.0 rather than 0.05: the first point (amp=0) is always
# P1≈0 (no drive → no excitation), which is a useful anchor for the network.
RABI_AMPS = np.linspace(0.0, 1.5, 10)

# --- T1 delay grid --------------------------------------------------------
#
# Choice: 10 points log-spaced across [100 ns, 100 µs].
#
# Prior on _T1: U(20, 40) µs.  Log-spacing is strictly better than linear
# for an exponential decay: it places more points where the curve changes
# fastest (short delays) and fewer where it is flat (long delays).
#
# Starting at 100 ns (1e-7 s) rather than 0 for geomspace compatibility
# (geomspace requires a strictly positive start).  At 100 ns the qubit is
# still essentially fully excited (exp(-100ns/20µs) ≈ 0.995), so this
# point acts as the P1≈1 anchor — equivalent to the t=0 point but safe
# for log-spacing.
#
# Maximum 100 µs = 2.5 × T1_max: the qubit has decayed to e^-2.5 ≈ 0.08,
# giving the network a clear floor reference.
#
# With only 10 points the exponential is under-sampled relative to a
# classical curve-fit, but the network amortises fitting across the full
# training distribution and can recover T1 reliably.
T1_DELAYS = np.geomspace(1e-7, 100e-6, 10)

# --- Ramsey delay grid ----------------------------------------------------
#
# Choice: 10 points log-spaced across [10 ns, 10 µs].
#
# Prior on _T2: U(15, 25) µs.  Detuning 1 MHz → fringe period 1 µs.
#
# Log-spacing for Ramsey is unconventional (linear spacing is standard
# because you want uniform fringe sampling).  With only 10 points and
# log-spacing, the early delays are densely sampled (sub-period resolution)
# and the late delays are sparse (one sample per several periods).
#
# This means the network sees:
#   - Short delays: fine phase information (where in the fringe cycle)
#   - Long delays: envelope decay information (T2)
# It loses uniform fringe coverage but gains dynamic-range coverage.
# Whether this is better than linear depends on the network — with large
# training sets the network learns to use both regimes.
#
# If Ramsey calibration is poor, switch back to linear:
#   RAMSEY_DELAYS = np.linspace(0.1e-6, 10e-6, 10)
# which gives 3 points per fringe period — the Nyquist-safe choice.
#
# Starting at 10 ns (1e-8 s): this is well below one fringe period (1 µs),
# so the first few points sample the initial rise of the oscillation.
#RAMSEY_DELAYS = np.geomspace(1e-8, 10e-6, 10)
RAMSEY_DELAYS = np.linspace(0.1e-6, 5e-6, 20)


# ============================================================================
# Caveat re: Gaussian output assumption
#
# The trained ensemble outputs a per-parameter Gaussian (μ, σ²).  This is
# strictly wrong near the prior boundaries — e.g. for a qubit with true
# _fq = 5.001 GHz, the *true* posterior is asymmetric (truncated at 5.0 GHz),
# but a Gaussian must be symmetric.  In practice we detect this by:
#   - μ within ~1.5σ of a prior boundary  →  Gaussian likely truncated
#   - μ outside the prior support entirely  →  hard failure, retrain with
#     a wider feature set
# `infer_qubit.py` flags any high-relative-uncertainty parameter (sigma/μ
# > 5%); manually inspecting whether μ ± 2σ crosses the prior bounds is the
# additional check.  We extend SPEC_FREQS ±50 MHz past the prior partly to
# soften this effect at f_q's boundaries.
# ============================================================================


# ============================================================================
# One-time waveform compilation (main process only)
# ============================================================================

def precompile_waveforms():
    """Compile spec and Gaussian envelopes ONCE; return numpy arrays.

    Must be called from the MAIN process — LabOne Q sessions are not fork-safe
    and cannot be passed to worker processes.  The arrays this function
    returns are plain numpy and broadcast safely through pickle.

    Returns:
        t_spec       : (N,) float64 time grid for the 2 µs spec pulse, t[0]==0
        wave_spec    : (N,) complex128 envelope at SPEC_AMP
        t_pulse      : (M,) float64 time grid for the 100 ns Gaussian pulse
        wave_gauss_u : (M,) complex128 unit-amplitude (amp=1.0) envelope; scale
                       per-call as `amp * wave_gauss_u` for Rabi / T1 / Ramsey

    Notes:
      - The drive frequency we compile at (5.5 GHz) is irrelevant: with
        HARDWARE modulation the AWG output is at baseband, so the snippet
        is the envelope and is independent of drive_freq.  The qubit's
        rotating-frame Hamiltonian then incorporates drive_freq separately
        via the detuning term.
      - We disconnect the session before returning so it doesn't leak into
        the multiprocessing fork on Linux/macOS (irrelevant on Windows
        where spawn is used, but harmless).
    """
    from laboneq.simple import (
        Session,
        Experiment,
        ExperimentSignal,
        pulse_library,
        AcquisitionType,
        RepetitionMode,
        OutputSimulator,
        Calibration,
        SignalCalibration,
        Oscillator,
        ModulationType,
    )
    from laboneq.contrib.example_helpers.generate_device_setup import (
        generate_device_setup_qubits,
    )

    device_setup, qubits = generate_device_setup_qubits(
        number_qubits=1,
        pqsc=[{"serial": "DEV10001"}],
        hdawg=[
            {"serial": "DEV8001", "zsync": 0, "number_of_channels": 8, "options": None}
        ],
        shfqc=[
            {
                "serial": "DEV12001",
                "zsync": 1,
                "number_of_channels": 6,
                "readout_multiplex": 6,
                "options": None,
            }
        ],
        include_flux_lines=True,
        server_host="localhost",
        setup_name="hackathon_singleQubit",
    )
    session = Session(device_setup)
    session.connect(do_emulation=True)
    q0 = qubits[0]
    drive_port = device_setup.logical_signal_by_uid(q0.uid + "/drive").physical_channel

    def _build_spec(drive_freq):
        spec_pulse = pulse_library.const(
            uid="spec_pulse", length=SPEC_PULSE_LENGTH, amplitude=SPEC_AMP
        )
        exp = Experiment(uid="Spec", signals=[ExperimentSignal("drive")])
        cal = Calibration()
        cal["drive"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=drive_freq, modulation_type=ModulationType.HARDWARE
            )
        )
        exp.set_calibration(cal)
        with exp.acquire_loop_rt(
            uid="shots",
            count=1,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            repetition_mode=RepetitionMode.CONSTANT,
            repetition_time=SPEC_PULSE_LENGTH + 500e-9,
        ):
            with exp.section(uid="drive_section"):
                exp.play(signal="drive", pulse=spec_pulse)
        exp.set_signal_map({"drive": q0.signals["drive"]})
        return exp

    def _build_gauss(drive_freq, amp):
        gauss_pulse = pulse_library.gaussian(
            uid="gauss", length=GAUSS_PULSE_LENGTH, amplitude=amp
        )
        exp = Experiment(uid="Gauss", signals=[ExperimentSignal("drive")])
        cal = Calibration()
        cal["drive"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=drive_freq, modulation_type=ModulationType.HARDWARE
            )
        )
        exp.set_calibration(cal)
        with exp.acquire_loop_rt(
            uid="shots",
            count=1,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
            repetition_mode=RepetitionMode.CONSTANT,
            repetition_time=GAUSS_PULSE_LENGTH + 500e-9,
        ):
            with exp.section(uid="drive_section"):
                exp.play(signal="drive", pulse=gauss_pulse)
        exp.set_signal_map({"drive": q0.signals["drive"]})
        return exp

    def _extract_wave(exp, length):
        compiled = session.compile(exp)
        sim = OutputSimulator(compiled)
        snippet = sim.get_snippet(drive_port, start=0, output_length=length)
        t = snippet.time - snippet.time[0]
        return (
            np.asarray(t, dtype=np.float64),
            np.asarray(snippet.wave, dtype=np.complex128),
        )

    t_spec, wave_spec = _extract_wave(_build_spec(5.5e9), SPEC_PULSE_LENGTH)
    t_pulse, wave_gauss = _extract_wave(_build_gauss(5.5e9, 1.0), GAUSS_PULSE_LENGTH)

    try:
        session.disconnect()
    except Exception:
        pass

    return t_spec, wave_spec, t_pulse, wave_gauss


# ============================================================================
# Per-sweep "fast" measurement primitives (numpy + qutip only)
#
# These are what runs inside ProcessPoolExecutor workers.  They reuse the
# pre-compiled wave envelopes by reference (or scaling, for amplitude
# variation) and never call session.compile().
# ============================================================================


def fast_spec(qubit, freqs, t_spec, wave_spec, shots):
    """Spec sweep using cached envelope; return P1 vector."""
    p1 = np.empty(len(freqs), dtype=np.float32)
    for i, f in enumerate(freqs):
        qubit.reset()
        qubit.evolve(t_spec, wave_spec, drive_freq=float(f))
        bits = qubit.measure(shots=shots)
        p1[i] = bits.mean()
    return p1


def fast_rabi(qubit, amps, t_pulse, wave_gauss_unit, drive_freq, shots):
    """Rabi sweep — scale the cached unit-amplitude Gaussian per-amp."""
    p1 = np.empty(len(amps), dtype=np.float32)
    for i, a in enumerate(amps):
        qubit.reset()
        qubit.evolve(
            t_pulse, float(a) * wave_gauss_unit, drive_freq=float(drive_freq)
        )
        bits = qubit.measure(shots=shots)
        p1[i] = bits.mean()
    return p1


def fast_t1(qubit, delays, t_pulse, wave_pi, drive_freq, shots):
    """T1: π pulse (using pre-scaled envelope) + variable wait + measure."""
    p1 = np.empty(len(delays), dtype=np.float32)
    for i, d in enumerate(delays):
        qubit.reset()
        qubit.evolve(t_pulse, wave_pi, drive_freq=float(drive_freq))
        qubit.wait(float(d))
        bits = qubit.measure(shots=shots)
        p1[i] = bits.mean()
    return p1


def fast_ramsey(qubit, delays, t_pulse, wave_pi2, drive_freq, detuning, shots):
    """Ramsey: two π/2 pulses with variable wait, drive intentionally detuned."""
    f_drive = float(drive_freq) + float(detuning)
    p1 = np.empty(len(delays), dtype=np.float32)
    for i, d in enumerate(delays):
        qubit.reset()
        qubit.evolve(t_pulse, wave_pi2, drive_freq=f_drive)
        qubit.wait(float(d))
        qubit.evolve(t_pulse, wave_pi2, drive_freq=f_drive)
        bits = qubit.measure(shots=shots)
        p1[i] = bits.mean()
    return p1


# ============================================================================
# Full per-qubit protocol — the bootstrap chain
# ============================================================================


def measure_qubit_features(
    qubit,
    t_spec,
    wave_spec,
    t_pulse,
    wave_gauss_unit,
    spec_freqs=None,
    rabi_amps=None,
    t1_delays=None,
    ramsey_delays=None,
    shots=None,
    ramsey_detuning=None,
):
    """Run all four measurements and return the concatenated P1 feature vector.

    Bootstrap chain (no `curve_fit` — argmax only, just enough to set the
    sweep parameters of the next stage):

        1. Spec sweep  →  f_drive = freqs[argmax(P1_spec)]
        2. Rabi sweep at f_drive  →  amp_pi_guess = first-half argmax
        3. T1 sweep with amp = amp_pi_guess at f_drive
        4. Ramsey sweep with amp = amp_pi_guess/2 at f_drive + 1 MHz

    The argmax bootstraps are intentionally crude — they only choose where
    to drive next.  The network does the precise extraction from the raw P1
    vectors.  Grid arguments default to the module-level constants but can
    be overridden (used by inference time when the saved grids may differ
    slightly from a hot-edited module).
    """
    if spec_freqs is None:
        spec_freqs = SPEC_FREQS
    if rabi_amps is None:
        rabi_amps = RABI_AMPS
    if t1_delays is None:
        t1_delays = T1_DELAYS
    if ramsey_delays is None:
        ramsey_delays = RAMSEY_DELAYS
    if shots is None:
        shots = SHOTS
    if ramsey_detuning is None:
        ramsey_detuning = RAMSEY_DETUNING

    # Stage 1: spectroscopy
    p1_spec = fast_spec(qubit, spec_freqs, t_spec, wave_spec, shots)
    f_drive = float(spec_freqs[int(np.argmax(p1_spec))])

    # Stage 2: Rabi at the bootstrapped frequency
    p1_rabi = fast_rabi(qubit, rabi_amps, t_pulse, wave_gauss_unit, f_drive, shots)

    # Bootstrap amp_pi from the *first half* of the Rabi sweep — the first
    # peak.  We restrict to the first half because at high amplitudes the
    # sinusoid has multiple maxima and the global argmax could land on any
    # of them depending on noise.
    n_first_half = max(2, len(rabi_amps) // 2)
    amp_pi_guess = float(rabi_amps[int(np.argmax(p1_rabi[:n_first_half]))])
    # Guard: if argmax landed in the very low-amplitude region (qubit
    # essentially un-excited because off-resonance — shouldn't happen with a
    # good f_drive but can on extreme qubits), fall back to a reasonable
    # default near the centre of the prior on amp_pi.
    if amp_pi_guess < float(rabi_amps[2]):
        amp_pi_guess = float(rabi_amps[len(rabi_amps) // 4])

    wave_pi = amp_pi_guess * wave_gauss_unit
    wave_pi2 = (amp_pi_guess / 2.0) * wave_gauss_unit

    # Stage 3: T1
    p1_t1 = fast_t1(qubit, t1_delays, t_pulse, wave_pi, f_drive, shots)

    # Stage 4: Ramsey
    p1_ram = fast_ramsey(
        qubit, ramsey_delays, t_pulse, wave_pi2, f_drive, ramsey_detuning, shots
    )

    return np.concatenate([p1_spec, p1_rabi, p1_t1, p1_ram]).astype(np.float32)


def feature_dim():
    """Total feature-vector length = sum of all sweep grid sizes."""
    return len(SPEC_FREQS) + len(RABI_AMPS) + len(T1_DELAYS) + len(RAMSEY_DELAYS)
