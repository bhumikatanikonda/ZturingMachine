from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# LabOne Q imports
from laboneq.simple import (
    Session,
    Experiment,
    ExperimentSignal,
    pulse_library,
    LinearSweepParameter,
    AcquisitionType,
    RepetitionMode,
    SectionAlignment,
    OutputSimulator,
)
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup_qubits,
)

from qubit import VirtualQubit


# DEVICE SETUP

def make_session() -> tuple:
    """Create a LabOne Q device setup and session in emulation mode.

    Returns:
        (session, device_setup, q0) — the session, device descriptor, and qubit 0 object.
    """
    device_setup, qubits = generate_device_setup_qubits(
        number_qubits=2,
        pqsc=[{"serial": "DEV10001"}],
        hdawg=[{"serial": "DEV8001", "zsync": 0, "number_of_channels": 8, "options": None}],
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
        setup_name="hackathon_setup",
    )
    session = Session(device_setup)
    session.connect(do_emulation=True)
    q0 = qubits[0]
    return session, device_setup, q0


# ============================================================
# 2. WAVEFORM EXTRACTION HELPER
# ============================================================

def get_drive_waveform(
    session: Session,
    device_setup,
    q0,
    exp: Experiment,
    pulse_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compile an experiment and extract the drive waveform envelope.

    Args:
        session:      Active LabOne Q session.
        device_setup: Device descriptor.
        q0:           Qubit object (LabOne Q).
        exp:          Experiment to compile.
        pulse_length: Duration [s] of the time axis to return.

    Returns:
        (t, wave) — time array starting at 0, complex envelope array.
    """
    compiled = session.compile(exp)
    drive_port = device_setup.logical_signal_by_uid(q0.uid + "/drive").physical_channel
    sim = OutputSimulator(compiled)
    snippet = sim.get_snippet(drive_port, start=0, output_length=pulse_length)
    return snippet.time, snippet.wave


# ============================================================
# 3. EXPERIMENT BUILDERS
# ============================================================

def make_spec_experiment(q0, drive_freq: float, pulse_length: float = 2e-6) -> Experiment:
    """Square-pulse spectroscopy experiment at a single drive frequency.

    Args:
        q0:          LabOne Q qubit object (for signal map).
        drive_freq:  Drive frequency [Hz] to set on the signal calibration.
        pulse_length: Duration of the square drive pulse [s].

    Returns:
        Configured (but not compiled) LabOne Q Experiment.
    """
    from laboneq.simple import Calibration, SignalCalibration, Oscillator, ModulationType

    spec_pulse = pulse_library.const(uid="spec_pulse", length=pulse_length, amplitude=0.5)

    exp = Experiment(
        uid="Spectroscopy",
        signals=[ExperimentSignal("drive")],
    )

    # Set calibration so the drive signal oscillates at drive_freq
    cal = Calibration()
    cal["drive"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=drive_freq,
            modulation_type=ModulationType.HARDWARE,
        )
    )
    exp.set_calibration(cal)

    with exp.acquire_loop_rt(
        uid="shots",
        count=1,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
        repetition_mode=RepetitionMode.CONSTANT,
        repetition_time=pulse_length + 500e-9,
    ):
        with exp.section(uid="drive_section"):
            exp.play(signal="drive", pulse=spec_pulse)

    exp.set_signal_map({"drive": q0.signals["drive"]})
    return exp


def make_rabi_experiment(q0, amp: float, pulse_length: float = 100e-9) -> Experiment:
    """Single-amplitude Gaussian drive experiment (used in Rabi sweep loop).

    Args:
        q0:           LabOne Q qubit object.
        amp:          Pulse amplitude (0–1).
        pulse_length: Duration of the Gaussian pulse [s].

    Returns:
        Configured LabOne Q Experiment.
    """
    drive_pulse = pulse_library.gaussian(uid="rabi_pulse", length=pulse_length, amplitude=amp)

    exp = Experiment(
        uid="Rabi",
        signals=[ExperimentSignal("drive")],
    )

    with exp.acquire_loop_rt(
        uid="shots",
        count=1,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
        repetition_mode=RepetitionMode.CONSTANT,
        repetition_time=pulse_length + 500e-9,
    ):
        with exp.section(uid="excitation"):
            exp.play(signal="drive", pulse=drive_pulse)

    exp.set_signal_map({"drive": q0.signals["drive"]})
    return exp


def make_ramsey_experiment(q0, amp_pi2: float, delay: float, pulse_length: float = 100e-9) -> Experiment:
    """Two π/2 pulses separated by a free-precession delay (Ramsey).

    Args:
        q0:           LabOne Q qubit object.
        amp_pi2:      Amplitude for π/2 pulse.
        delay:        Free precession delay between the two pulses [s].
        pulse_length: Duration of each Gaussian π/2 pulse [s].

    Returns:
        Configured LabOne Q Experiment.
    """
    pi2_pulse = pulse_library.gaussian(uid="pi2", length=pulse_length, amplitude=amp_pi2)

    exp = Experiment(
        uid="Ramsey",
        signals=[ExperimentSignal("drive")],
    )

    rep_time = 2 * pulse_length + delay + 500e-9

    with exp.acquire_loop_rt(
        uid="shots",
        count=1,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
        repetition_mode=RepetitionMode.CONSTANT,
        repetition_time=rep_time,
    ):
        with exp.section(uid="first_pi2"):
            exp.play(signal="drive", pulse=pi2_pulse)
        with exp.section(uid="free_precession", play_after="first_pi2"):
            exp.delay(signal="drive", time=delay)
        with exp.section(uid="second_pi2", play_after="free_precession"):
            exp.play(signal="drive", pulse=pi2_pulse)

    exp.set_signal_map({"drive": q0.signals["drive"]})
    return exp


# ============================================================
# 4. FITTING FUNCTIONS
# ============================================================

def lorentzian(f, f0, width, amplitude, offset):
    return offset + amplitude / (1 + ((f - f0) / (width / 2)) ** 2)


def fit_lorentzian(freqs: np.ndarray, P1: np.ndarray) -> float:
    """Fit a Lorentzian peak to spectroscopy data and return the centre frequency."""
    f0_guess = freqs[np.argmax(P1)]
    width_guess = (freqs[-1] - freqs[0]) / 10
    amp_guess = np.max(P1) - np.min(P1)
    offset_guess = np.min(P1)
    try:
        popt, _ = curve_fit(
            lorentzian, freqs, P1,
            p0=[f0_guess, width_guess, amp_guess, offset_guess],
            maxfev=10000,
        )
        return popt[0]
    except RuntimeError:
        print("  [warn] Lorentzian fit failed, returning argmax estimate")
        return f0_guess


def sinusoid(x, A, omega, phi, offset):
    return offset + A * np.sin(omega * x + phi)


def fit_sinusoid(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a sinusoid and return (A, omega, phi, offset)."""
    A_guess = (np.max(y) - np.min(y)) / 2
    offset_guess = np.mean(y)
    # Rough frequency estimate via FFT
    fft = np.abs(np.fft.rfft(y - offset_guess))
    freqs = np.fft.rfftfreq(len(x), d=(x[1] - x[0]))
    omega_guess = 2 * np.pi * freqs[np.argmax(fft[1:]) + 1]
    try:
        popt, _ = curve_fit(
            sinusoid, x, y,
            p0=[A_guess, omega_guess, 0, offset_guess],
            maxfev=10000,
        )
        return popt
    except RuntimeError:
        return A_guess, omega_guess, 0.0, offset_guess


def exponential_decay(t, A, T1, offset):
    return offset + A * np.exp(-t / T1)


def fit_T1(delays: np.ndarray, P1: np.ndarray) -> float:
    """Fit exponential decay to T1 data, return T1 [s]."""
    try:
        popt, _ = curve_fit(
            exponential_decay, delays, P1,
            p0=[np.max(P1), 30e-6, np.min(P1)],
            maxfev=10000,
        )
        return popt[1]
    except RuntimeError:
        print("  [warn] T1 fit failed, returning rough estimate")
        return delays[np.argmax(P1 < 0.5 * np.max(P1))]


# ============================================================
# 5. CALIBRATION EXPERIMENTS
# ============================================================

def qubit_spectroscopy(
    virtual_qubit: VirtualQubit,
    session: Session,
    device_setup,
    q0_loneq,
    freq_start: float = 5.0e9,
    freq_stop: float = 6.2e9,
    n_points: int = 101,
    shots: int = 2000,
    pulse_length: float = 2e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep drive frequency and measure excited-state population.

    Args:
        virtual_qubit: The VirtualQubit to drive and measure.
        session:       LabOne Q session.
        device_setup:  Device descriptor.
        q0_loneq:      LabOne Q qubit object.
        freq_start:    Sweep start frequency [Hz].
        freq_stop:     Sweep stop frequency [Hz].
        n_points:      Number of frequency points.
        shots:         Measurement shots per frequency.
        pulse_length:  Square pulse duration [s].

    Returns:
        (freqs, P1) — frequency array and excited-state probabilities.
    """
    freqs = np.linspace(freq_start, freq_stop, n_points)
    P1 = []

    print(f"  Spectroscopy: sweeping {n_points} frequencies from {freq_start/1e9:.2f} to {freq_stop/1e9:.2f} GHz")

    for f in freqs:
        exp = make_spec_experiment(q0_loneq, drive_freq=f, pulse_length=pulse_length)
        t, wave = get_drive_waveform(session, device_setup, q0_loneq, exp, pulse_length)

        virtual_qubit.reset()
        virtual_qubit.evolve(t, wave, drive_freq=f)
        bits = virtual_qubit.measure(shots=shots)
        P1.append(bits.mean())

    return freqs, np.array(P1)


def amplitude_rabi(
    virtual_qubit: VirtualQubit,
    session: Session,
    device_setup,
    q0_loneq,
    drive_freq: float,
    amp_start: float = 0.01,
    amp_stop: float = 1.0,
    n_points: int = 51,
    shots: int = 2000,
    pulse_length: float = 100e-9,
) -> tuple[float, float]:
    """Sweep pulse amplitude and fit Rabi oscillations to find π and π/2 amplitudes.

    Args:
        virtual_qubit: The VirtualQubit to drive and measure.
        session:       LabOne Q session.
        device_setup:  Device descriptor.
        q0_loneq:      LabOne Q qubit object.
        drive_freq:    Resonance frequency to drive at [Hz].
        amp_start:     Sweep start amplitude.
        amp_stop:      Sweep stop amplitude.
        n_points:      Number of amplitude points.
        shots:         Measurement shots per amplitude.
        pulse_length:  Gaussian pulse duration [s].

    Returns:
        (amp_pi, amp_pi2) — π and π/2 pulse amplitudes.
    """
    amps = np.linspace(amp_start, amp_stop, n_points)
    P1 = []

    print(f"  Amplitude Rabi: sweeping {n_points} amplitudes")

    for amp in amps:
        exp = make_rabi_experiment(q0_loneq, amp=amp, pulse_length=pulse_length)
        t, wave = get_drive_waveform(session, device_setup, q0_loneq, exp, pulse_length)

        virtual_qubit.reset()
        virtual_qubit.evolve(t, wave, drive_freq=drive_freq)
        bits = virtual_qubit.measure(shots=shots)
        P1.append(bits.mean())

    P1 = np.array(P1)
    A, omega, phi, offset = fit_sinusoid(amps, P1)

    # π pulse: first maximum of the sinusoid
    amp_pi = (np.pi / 2 - phi) / omega
    # Ensure it's in the swept range
    if amp_pi < amp_start or amp_pi > amp_stop:
        amp_pi = amps[np.argmax(P1)]
    amp_pi2 = amp_pi / 2

    return float(amp_pi), float(amp_pi2)


def measure_T1(
    virtual_qubit: VirtualQubit,
    session: Session,
    device_setup,
    q0_loneq,
    drive_freq: float,
    amp_pi: float,
    delay_start: float = 0.5e-6,
    delay_stop: float = 80e-6,
    n_points: int = 40,
    shots: int = 2000,
    pulse_length: float = 100e-9,
) -> float:
    """Apply π pulse, wait variable delay, measure — fit exponential to get T1.

    Args:
        virtual_qubit: The VirtualQubit.
        session:       LabOne Q session.
        device_setup:  Device descriptor.
        q0_loneq:      LabOne Q qubit object.
        drive_freq:    Qubit resonance frequency [Hz].
        amp_pi:        Calibrated π-pulse amplitude.
        delay_start:   Shortest wait delay [s].
        delay_stop:    Longest wait delay [s].
        n_points:      Number of delay points.
        shots:         Measurement shots per delay.
        pulse_length:  Duration of the π pulse [s].

    Returns:
        Fitted T1 relaxation time [s].
    """
    delays = np.linspace(delay_start, delay_stop, n_points)
    P1 = []

    print(f"  T1 measurement: {n_points} delay points up to {delay_stop*1e6:.0f} µs")

    # Build the π-pulse waveform once (reuse across delays)
    exp_pi = make_rabi_experiment(q0_loneq, amp=amp_pi, pulse_length=pulse_length)
    t_pi, wave_pi = get_drive_waveform(session, device_setup, q0_loneq, exp_pi, pulse_length)

    for delay in delays:
        virtual_qubit.reset()
        virtual_qubit.evolve(t_pi, wave_pi, drive_freq=drive_freq)  # π pulse → |1⟩
        virtual_qubit.wait(delay)                                     # free decay
        bits = virtual_qubit.measure(shots=shots)
        P1.append(bits.mean())

    T1_fit = fit_T1(delays, np.array(P1))
    return T1_fit


def active_reset(
    virtual_qubit: VirtualQubit,
    session: Session,
    device_setup,
    q0_loneq,
    drive_freq: float,
    amp_pi: float,
    pulse_length: float = 100e-9,
    max_attempts: int = 5,
) -> int:
    """Measure-and-flip active reset: measure, apply π if in |1⟩, repeat.

    Replaces qubit.reset() with a hardware-realistic implementation.

    Args:
        virtual_qubit: The VirtualQubit.
        session:       LabOne Q session.
        device_setup:  Device descriptor.
        q0_loneq:      LabOne Q qubit object.
        drive_freq:    Qubit resonance frequency [Hz].
        amp_pi:        Calibrated π-pulse amplitude.
        pulse_length:  Duration of the π pulse [s].
        max_attempts:  Maximum flip attempts before giving up.

    Returns:
        Number of attempts taken.
    """
    exp_pi = make_rabi_experiment(q0_loneq, amp=amp_pi, pulse_length=pulse_length)
    t_pi, wave_pi = get_drive_waveform(session, device_setup, q0_loneq, exp_pi, pulse_length)

    for attempt in range(max_attempts):
        bit = virtual_qubit.measure(shots=1)[0]
        if bit == 0:
            return attempt + 1  # already in ground state
        # Apply π pulse to flip back to |0⟩
        virtual_qubit.evolve(t_pi, wave_pi, drive_freq=drive_freq)

    return max_attempts


def ramsey(
    virtual_qubit: VirtualQubit,
    session: Session,
    device_setup,
    q0_loneq,
    drive_freq: float,
    amp_pi2: float,
    detuning: float = 1e6,
    delay_start: float = 0.1e-6,
    delay_stop: float = 5e-6,
    n_points: int = 50,
    shots: int = 2000,
    pulse_length: float = 100e-9,
) -> float:
    """Ramsey interferometry for precise qubit frequency determination.

    Drive slightly off-resonance (detuning), fit oscillation frequency,
    correct back to find true f_q.

    Args:
        virtual_qubit: The VirtualQubit.
        session:       LabOne Q session.
        device_setup:  Device descriptor.
        q0_loneq:      LabOne Q qubit object.
        drive_freq:    Initial best estimate of resonance [Hz].
        amp_pi2:       Calibrated π/2-pulse amplitude.
        detuning:      Intentional offset from resonance [Hz] for visible fringes.
        delay_start:   Shortest free-precession delay [s].
        delay_stop:    Longest free-precession delay [s].
        n_points:      Number of delay points.
        shots:         Measurement shots per delay.
        pulse_length:  Duration of each π/2 pulse [s].

    Returns:
        Refined qubit frequency [Hz].
    """
    delays = np.linspace(delay_start, delay_stop, n_points)
    P1 = []

    # Drive intentionally off-resonance so fringes are visible
    f_drive = drive_freq + detuning

    print(f"  Ramsey: {n_points} delay points, detuning = {detuning/1e6:.1f} MHz")

    for delay in delays:
        exp_ram = make_ramsey_experiment(q0_loneq, amp_pi2=amp_pi2, delay=delay, pulse_length=pulse_length)
        t_ram, wave_ram = get_drive_waveform(
            session, device_setup, q0_loneq, exp_ram, 2 * pulse_length + delay
        )

        virtual_qubit.reset()

        # Split waveform into: first π/2, delay silence, second π/2
        n_pulse = int(len(t_ram) * pulse_length / (2 * pulse_length + delay))
        t1 = t_ram[:n_pulse]
        w1 = wave_ram[:n_pulse]

        # Silence gap (use wait)
        t3 = t_ram[n_pulse: 2 * n_pulse]
        w2 = wave_ram[len(t_ram) - n_pulse:]

        # Apply first π/2 pulse
        if len(t1) > 1:
            t1 = t1 - t1[0]
            virtual_qubit.evolve(t1, w1, drive_freq=f_drive)

        # Free precession
        virtual_qubit.wait(delay)

        # Apply second π/2 pulse
        if len(w2) > 1:
            t_seg = np.linspace(0, pulse_length, len(w2))
            virtual_qubit.evolve(t_seg, w2, drive_freq=f_drive)

        bits = virtual_qubit.measure(shots=shots)
        P1.append(bits.mean())

    P1 = np.array(P1)
    A, omega_fit, phi, offset = fit_sinusoid(delays, P1)

    # Measured oscillation frequency = detuning + residual error
    f_measured = omega_fit / (2 * np.pi)
    f_q_refined = drive_freq + (f_measured - detuning)

    return float(f_q_refined)


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_results(
    freqs, P1_spec,
    amps, P1_rabi,
    delays_T1, P1_T1,
    delays_ram, P1_ram,
    f_q, amp_pi, T1, f_q_precise,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Virtual Qubit Tune-Up Results", fontsize=14, fontweight="bold")

    # Spectroscopy
    ax = axes[0, 0]
    ax.plot(freqs / 1e9, P1_spec, "b.-")
    ax.axvline(f_q / 1e9, color="red", linestyle="--", label=f"f_q = {f_q/1e9:.4f} GHz")
    ax.set_xlabel("Drive frequency (GHz)")
    ax.set_ylabel("P(|1⟩)")
    ax.set_title("Qubit Spectroscopy")
    ax.legend()

    # Rabi
    ax = axes[0, 1]
    ax.plot(amps, P1_rabi, "g.-")
    ax.axvline(amp_pi, color="red", linestyle="--", label=f"π amp = {amp_pi:.3f}")
    ax.axvline(amp_pi / 2, color="orange", linestyle="--", label=f"π/2 amp = {amp_pi/2:.3f}")
    ax.set_xlabel("Drive amplitude")
    ax.set_ylabel("P(|1⟩)")
    ax.set_title("Amplitude Rabi")
    ax.legend()

    # T1
    ax = axes[1, 0]
    ax.plot(delays_T1 * 1e6, P1_T1, "r.-")
    t_fit = np.linspace(delays_T1[0], delays_T1[-1], 200)
    ax.plot(t_fit * 1e6, exponential_decay(t_fit, np.max(P1_T1), T1, np.min(P1_T1)),
            "k--", label=f"T1 = {T1*1e6:.1f} µs")
    ax.set_xlabel("Delay (µs)")
    ax.set_ylabel("P(|1⟩)")
    ax.set_title("T1 Measurement")
    ax.legend()

    # Ramsey
    ax = axes[1, 1]
    ax.plot(delays_ram * 1e6, P1_ram, "m.-")
    ax.set_xlabel("Free precession delay (µs)")
    ax.set_ylabel("P(|1⟩)")
    ax.set_title(f"Ramsey (f_q refined = {f_q_precise/1e9:.6f} GHz)")

    plt.tight_layout()
    plt.savefig("tune_up_results.png", dpi=150)
    plt.show()
    print("\n  Plot saved to tune_up_results.png")


# ============================================================
# 7. MAIN TUNE-UP SEQUENCE
# ============================================================

def main():
    print("=" * 60)
    print("  Zurich Instruments Hackathon — Virtual Qubit Tune-Up")
    print("=" * 60)

    # --- Virtual qubit (physics simulator) ---
    print("\n[INIT] Creating virtual qubit (seed=42)...")
    virtual_qubit = VirtualQubit(seed=42)

    # --- LabOne Q hardware (emulated) ---
    print("[INIT] Setting up LabOne Q session (emulation mode)...")
    session, device_setup, q0_loneq = make_session()

    # --------------------------------------------------------
    # STEP 1: Qubit Spectroscopy
    # --------------------------------------------------------
    print("\n[STEP 1] Qubit Spectroscopy")
    freqs, P1_spec = qubit_spectroscopy(
        virtual_qubit, session, device_setup, q0_loneq,
        freq_start=5.0e9, freq_stop=6.2e9, n_points=101,
        shots=2000, pulse_length=2e-6,
    )
    f_q = fit_lorentzian(freqs, P1_spec)
    print(f"  → Transition frequency: f_q = {f_q/1e9:.4f} GHz")

    # --------------------------------------------------------
    # STEP 2: Amplitude Rabi
    # --------------------------------------------------------
    print("\n[STEP 2] Amplitude Rabi")
    amps = np.linspace(0.01, 1.0, 51)
    P1_rabi = []
    for amp in amps:
        exp = make_rabi_experiment(q0_loneq, amp=amp, pulse_length=100e-9)
        t, wave = get_drive_waveform(session, device_setup, q0_loneq, exp, 100e-9)
        virtual_qubit.reset()
        virtual_qubit.evolve(t, wave, drive_freq=f_q)
        bits = virtual_qubit.measure(shots=2000)
        P1_rabi.append(bits.mean())
    P1_rabi = np.array(P1_rabi)

    A, omega_rabi, phi_rabi, offset_rabi = fit_sinusoid(amps, P1_rabi)
    amp_pi = float((np.pi / 2 - phi_rabi) / omega_rabi)
    if amp_pi < 0.01 or amp_pi > 1.0:
        amp_pi = float(amps[np.argmax(P1_rabi)])
    amp_pi2 = amp_pi / 2
    print(f"  → π-pulse amplitude:    amp_π  = {amp_pi:.4f}")
    print(f"  → π/2-pulse amplitude:  amp_π₂ = {amp_pi2:.4f}")

    # --------------------------------------------------------
    # STEP 3: T1 Measurement
    # --------------------------------------------------------
    print("\n[STEP 3] T1 Measurement")
    delays_T1 = np.linspace(0.5e-6, 80e-6, 40)
    P1_T1 = []

    exp_pi = make_rabi_experiment(q0_loneq, amp=amp_pi, pulse_length=100e-9)
    t_pi, wave_pi = get_drive_waveform(session, device_setup, q0_loneq, exp_pi, 100e-9)

    for delay in delays_T1:
        virtual_qubit.reset()
        virtual_qubit.evolve(t_pi, wave_pi, drive_freq=f_q)
        virtual_qubit.wait(delay)
        bits = virtual_qubit.measure(shots=2000)
        P1_T1.append(bits.mean())
    P1_T1 = np.array(P1_T1)

    T1 = fit_T1(delays_T1, P1_T1)
    print(f"  → T1 = {T1*1e6:.1f} µs")

    # --------------------------------------------------------
    # STEP 4: Active Reset Demo
    # --------------------------------------------------------
    print("\n[STEP 4] Active Reset")
    # First excite the qubit to |1⟩
    virtual_qubit.reset()
    virtual_qubit.evolve(t_pi, wave_pi, drive_freq=f_q)
    p1_before = virtual_qubit.measure(shots=500).mean()
    print(f"  P(|1⟩) before active reset: {p1_before:.3f}")

    attempts = active_reset(virtual_qubit, session, device_setup, q0_loneq,
                            drive_freq=f_q, amp_pi=amp_pi)
    p1_after = virtual_qubit.measure(shots=500).mean()
    print(f"  P(|1⟩) after active reset:  {p1_after:.3f} (took {attempts} attempt(s))")

    # --------------------------------------------------------
    # STEP 5: Ramsey Interferometry
    # --------------------------------------------------------
    print("\n[STEP 5] Ramsey Interferometry")
    delays_ram = np.linspace(0.1e-6, 5e-6, 50)
    P1_ram = []

    detuning = 1e6  # drive 1 MHz off-resonance for visible fringes
    f_drive_ram = f_q + detuning

    exp_pi2 = make_rabi_experiment(q0_loneq, amp=amp_pi2, pulse_length=100e-9)
    t_pi2, wave_pi2 = get_drive_waveform(session, device_setup, q0_loneq, exp_pi2, 100e-9)

    for delay in delays_ram:
        virtual_qubit.reset()
        virtual_qubit.evolve(t_pi2, wave_pi2, drive_freq=f_drive_ram)  # first π/2
        virtual_qubit.wait(delay)                                        # free precession
        virtual_qubit.evolve(t_pi2, wave_pi2, drive_freq=f_drive_ram)  # second π/2
        bits = virtual_qubit.measure(shots=2000)
        P1_ram.append(bits.mean())
    P1_ram = np.array(P1_ram)

    A_ram, omega_ram, phi_ram, _ = fit_sinusoid(delays_ram, P1_ram)
    f_measured = omega_ram / (2 * np.pi)
    f_q_precise = f_q + (f_measured - detuning)
    print(f"  → Ramsey frequency oscillation: {f_measured/1e6:.3f} MHz")
    print(f"  → Refined f_q = {f_q_precise/1e9:.6f} GHz")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TUNE-UP COMPLETE — Calibrated Parameters")
    print("=" * 60)
    print(f"  Transition frequency (spectroscopy): {f_q/1e9:.4f} GHz")
    print(f"  Transition frequency (Ramsey):       {f_q_precise/1e9:.6f} GHz")
    print(f"  π-pulse amplitude:                   {amp_pi:.4f}")
    print(f"  π/2-pulse amplitude:                 {amp_pi2:.4f}")
    print(f"  T1 (energy relaxation):              {T1*1e6:.1f} µs")
    print("=" * 60)

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    plot_results(
        freqs, P1_spec,
        amps, P1_rabi,
        delays_T1, P1_T1,
        delays_ram, P1_ram,
        f_q, amp_pi, T1, f_q_precise,
    )


if __name__ == "__main__":
    main()