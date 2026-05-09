# Tune-up of a virtual qubit with LabOne Q

In this hackaton challenge you will tune up a virtual qubit using [*LabOne Q*](https://www.zhinst.com/ch/en/quantum-computing-systems/labone-q/) through a standard superconducting-qubit calibration sequence: qubit spectroscopy, amplitude Rabi, T₁ measurement, and Ramsey interferometry. Each step is implemented as a LabOne Q experiment whose simulated pulses drive the virtual qubit. Optional stretch goals are to implement Bell inequality test on an entangled qubit pair or a code a native LabOne Q readout implementation.

## Summary

[*LabOne Q*](https://www.zhinst.com/ch/en/quantum-computing-systems/labone-q/) is the Python-based framework for quantum computing using the quantum control systems of Zurich Instruments. In this challenge you will use LabOne Q to play pulses and tune-up a virtual qubit implemented with [*QuTIP*](https://qutip.org/).

You will have access to a virtual qubit class in `qubit.py`. When initialized, the qubit starts in the ground state. You can **evolve** the state using the `evolve` method of the `VirtualQubit` class by providing a waveform envelope and the modulation frequency. To **measure** the qubit use the `measure` method that returns if the qubit is in the ground state or the excited state. The `wait` method allows you to let the qubit state decay for a given duration.

To generate the pulses you will need to define [*LabOne Q Experiments*](https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/05_experiment/concepts/index.html). The `Experiment` class allows you to define your pulse sequence using pulses, sections and real- and near-time loops, tell LabOne Q which experimental signal lines the pulses should be played on, set experiment-specific calibrations, determine how you sweep parameters, and more. Unfortunately
for us, we don't have access to a real qubit during the Hackaton. Fortunately for us, LabOne Q allows us to run in "emulation" mode and output the pulses which would otherwise be played on the actual instrument. LabOne Q can simulate the output of each channel in a sample-precise way. This feature can be used to check experiments even before they are executed on hardware.


## How it works

The virtual qubit lives in `qubit.py` as the `VirtualQubit` class. It starts in the
ground state and exposes three methods:

- `evolve(t, wave, drive_freq)` propagates the state under your pulse waveform.
- `measure(shots)` returns shot outcomes (0 = ground, 1 = excited).
- `wait(duration)` lets the state decay freely for a given time.
- `reset()` returns the qubit to the ground state.

You generate the pulses by defining a
[LabOne Q `Experiment`](https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/05_experiment/concepts/index.html),
compiling it, and feeding the resulting waveform into `evolve`. We don't have a
real qubit during the hackathon, but LabOne Q's emulation mode gives you
sample-precise simulated outputs -- exactly what would be played on hardware.

### Preparatory Tasks

1. **Find the qubit's transition frequency.** Spectroscopy: sweep the drive
   frequency and watch for an excited-state population peak.
   - Implement a LabOne Q experiment that plays a square pulse with a specified drive frequency.
   - Sweep the frequency and look for a resonance peak.
   - Fit the peak and obtain the transition frequency.
2. **Calibrate π and π/2 pulses.** Run an amplitude Rabi to find the amplitude
   that drives a full π rotation, then use it to compose π/2 pulses.
   - Implement a LabOne Q experiment that sweeps the amplitude of a Gaussian pulse modulated at the resonance frequency.
   - Fit to obtain the amplitude that is needed for π and π/2 rotations.
3. **Measure T₁.** Apply a π pulse, wait for a variable delay, then measure.
   Fit the exponential decay.
4. **Implement active reset.** Replace `qubit.reset()` with a measurement-and-flip
   scheme: measure the qubit, and apply a π pulse if it came out in |1⟩.
5. **Ramsey Spectroscopy.** The qubit spectroscopy is a rough method to find the qubit frequency.
   - Use the Ramsey method to find the precise value of the transition frequency.
   - Study how Ramsey works and implement it as a LabOne Q experiment.

### Speed up and authonomy of your tune-up sequence

You already have your tune-up workflows. Now let's see how fast you can converge to the
qubit parameters.

1. **Optimize your experiments** to use as few pulses as necessary to sweep parameters and 
   converge to the optimal ones.
   For example, speed up calibration measurements using adaptive sweeps: https://arxiv.org/abs/2506.09576
2. **Automate workflows.** LabOne Q already provides you with an automation framework.
   Incorporate your experiments into the automation framework.
   HINT: Check Workflows, Tasks and Automation in the LabOne Q documentation.
3. **Generate the automation graph** on the fly using generative AI: https://arxiv.org/abs/2412.07978
  - Prompt the LLM to generage code for adding an automation layer for
    a certain experiment (e.g. qubit spectroscopy).
  - Get results, feed them to an evaluation algorithm or the LLM.
  - Prompt the LLM for the next decision step.
    https://ethz.ch/en/the-eth-zurich/education/ai-in-education/tools.html

These points are just to give you ideas in which direciton you can push.
Feel free to go off the beaten path and propose something new and interesting.

## Stretch goals (if you are finished too quicly)
Focus on any one frist and if you are finished quickly, jump to the next problem you
find interesting.

### Bell inequality violation

You're given a `VirtualQubitPair` of two coupled qubits and a ZZ-type interaction.
The qubits start in `|00⟩`.

1. **Prepare a Bell state.** Combine your calibrated single-qubit gates with the
   built-in `cphase()` to entangle the pair.
2. **Run the CHSH test.** For each of the four CHSH measurement settings, prepare
   the Bell state, rotate into the chosen basis, and measure many shots. Compute
   the four correlators and the CHSH parameter `S = E(a,b) + E(a,b') + E(a',b) − E(a',b')`.
   Show `|S| > 2`.

If you get stuck in this problem, don't worry. Reach out and we will help you.

Tip: Keep in mind that you need **very good** knowledge of the qubit frequency and π-pulse amplitude to make this work.

### Implement measurement in LabOne Q

The current `measure` is a Python call. Build the readout as a real LabOne Q
experiment: a readout pulse, an acquisition window, and an integration kernel —
all defined through the DSL.

## Setup

This project requires Python 3.14 or newer. The instructions below use `venv`
(built into Python); `conda` or `uv` work equally well.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 2. Install the project and its dependencies
pip install --upgrade pip
pip install -e .
```

Once activated, run the challenge entry point with:

```bash
python main.py
```

## Tips

1. Start from the [*OutputSimulator*](https://docs.zhinst.com/labone_q_user_manual/core/functionality_and_concepts/10_advanced_topics/tutorials/00_output_simulator.html) tutorial. Run through the tutorial and see how the output pulses look like. Make a function that takes the output and converts it to the format used by the `VirtualQubit`. **Note that the output simulator provides only the pulse envelope and does not include the high-frequency modulation.**
3. Learn more about [*Qubit Spectroscopy*](https://docs.zhinst.com/labone_q_user_manual/applications_library/how-to-guides/sources/01_superconducting_qubits/01_workflows/03_qubit_spectroscopy.html), [*Amplitude Rabi*](https://docs.zhinst.com/labone_q_user_manual/applications_library/how-to-guides/sources/01_superconducting_qubits/01_workflows/04_amplitude_rabi.html), and [*Ramsey Interferometry*](https://docs.zhinst.com/labone_q_user_manual/applications_library/how-to-guides/sources/01_superconducting_qubits/01_workflows/05_ramsey.html) from the LabOne Q Applications library documentation.

## Example code to get you started

```python
# LabOne Q setup
device_setup, qubits = generate_device_setup_qubits(...)
session = Session(device_setup)
session.connect(do_emulation=True)

# Define your experiments
def qubit_spectroscopy(qubit):
    freqs = np.linspace(5.00e9, 6.20e9, 101)
    P1 = []
    for f in freqs:
        exp = experiments.spec_experiment(drive_freq=f)
        compiled = session.compile(exp)
        t, wf = experiments.get_waveform(compiled, pulse_length=2e-6)
        qubit.reset()
        qubit.evolve(t, wf, drive_freq=f)
        bits = qubit.measure(shots=10000)
        P1.append(bits.mean())
    return freqs, P1

def main():
    # Start the qubit with a defined seed
    # for the RNG for deterministic results
    q0 = VirtualQubit(seed=42)
    freqs, P1 = qubit_spectroscopy(q0)
    f_drive_q0 = fit_lorentzian(freqs, P1)
    amp_pi_q0 = amplitude_rabi(q0, drive_freq=f_drive_q0)
    ...
```

## Evaluation criteria
The evaluation criteria are the following (in order of importance):
1. Innovation (novel algorithm/approach) in speeding up tune-up
2. Shortest active qpu time under similar average count
3. Completion of all preparatory steps and correct identification of qubit parameters
4. Code quality

## Happy hacking!
