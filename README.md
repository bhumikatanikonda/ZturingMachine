# Qubit Characterisation Pipeline

A sequential notebook for full single-qubit characterisation using **LabOne Q** (hardware abstraction), a **VirtualQubit** physics simulator, and a **Bayesian Adaptive Estimator** for sample-efficient parameter extraction.

---

## Requirements

### Python packages

```bash
pip install numpy matplotlib scipy laboneq
```

### Local modules (must be in the same directory as the notebook)

| File | Purpose |
|---|---|
| `qubit.py` | `VirtualQubit` — physics simulator for a two-level system |
| `BayesianEstimator.py` | `BayesianAdaptiveEstimator` — adaptive Bayesian inference engine |

---

## File structure

```
your_project/
├── qubit_characterisation.ipynb   ← the notebook
├── qubit.py
├── BayesianEstimator.py
└── README.md
```

---

## How to run

Open the notebook and **run all cells in order** from top to bottom. Each step depends on results from the previous one — do not skip or reorder cells.

```
Kernel → Restart & Run All
```

Or run cell by cell with `Shift + Enter`.

---

## What each block does

### Block 1 — Imports & device setup
Imports all dependencies and defines two helper functions:
- `make_session()` — creates a LabOne Q device setup with one qubit in **emulation mode** (no real hardware needed)
- `get_drive_waveform()` — compiles an experiment and extracts the drive waveform envelope from the output simulator

Also defines the two experiment builders used throughout:
- `make_spec_experiment()` — square-pulse spectroscopy experiment
- `make_rabi_experiment()` — Gaussian-pulse drive experiment

And all curve-fitting utilities: `fit_lorentzian`, `fit_voigt`, `fit_sinusoid`, `fit_T1`, `fit_ramsey`.

### Block 2 — Initialisation
```python
virtual_qubit = VirtualQubit(seed=42)
session, device_setup, q0_loneq = make_session()
```
Instantiates the virtual qubit (fixed seed for reproducibility) and connects the LabOne Q session in emulation mode. **Must run before any measurement step.**

---

### Step 1 — Qubit Spectroscopy
```python
freqs, P1_spec, f_q, f_err = qubit_spectroscopy(...)
```
Sweeps 401 drive frequencies from 5 GHz to 6 GHz and fits a Lorentzian peak to find the qubit transition frequency `f_q`.

**Output:** `step1_spectroscopy.png` — frequency sweep with fitted peak and vertical marker at `f_q`.

**Key result:** `f_q` (Hz) — used by all subsequent steps.

---

### Step 2a — Amplitude Rabi (standard)
```python
amps, P1_rabi, amp_pi, amp_pi2 = amplitude_rabi_ramsey(...)
```
Sweeps Gaussian pulse amplitude from 0.01 to 4.0 across 51 points, fits a sinusoid, and extracts the π and π/2 pulse amplitudes from the fitted phase and frequency.

**Output:** `step2_rabi.png` — Rabi oscillation with vertical markers at π and π/2 amplitudes.

**Key results:** `amp_pi`, `amp_pi2` — used by T1 and Ramsey steps.

---

### Step 2b — Amplitude Rabi (adaptive)
```python
amp_pi, amp_pi2, counts = amplitude_rabi_ramsey_adaptive(...)
```
Same calibration goal as Step 2a, but uses the `BayesianAdaptiveEstimator` in `"sine"` mode to find `amp_pi` with far fewer measurements. The estimator chooses each probe amplitude adaptively and stops once confidence is reached.

**Key results:** `amp_pi`, `amp_pi2` — these **overwrite** the values from Step 2a and are used by the T1 and Ramsey steps below.

> **Note:** `amp_pi2` is simply set to `amp_pi / 2` in this adaptive version. If the Rabi fringe phase is non-zero, the standard Step 2a fit will give a more accurate π/2 amplitude.

---

### Step 3a — T1 Measurement (standard)
```python
evolve_times, P1, T1, P1 = measure_T1(...)
```
Applies a π pulse to prepare `|1⟩`, waits a variable delay (0.5 µs to 80 µs across 100 points), measures the survival probability, and fits an exponential decay to extract T1.

**Output:** `step3_T1.png` — exponential decay data and fit.

**Key result:** `T1` (s).

---

### Step 3b — T1 Measurement (adaptive)
```python
T1, counts = measure_T1_adaptive(...)
```
Same T1 extraction, but uses the `BayesianAdaptiveEstimator` in `"exponential"` mode. Probe times are chosen to maximise information gain at each step, and measurement stops automatically when the 95% credible interval is sufficiently narrow.

**Key results:** `T1` (s), `counts` — number of single shots used.

---

### Step 4 — Ramsey Interferometry
```python
evolve_times, P1, f_measured, f_q_precise = ramsey(...)
```
Applies two π/2 pulses with a variable free-precession delay (0.1 µs to 50 µs, 100 points) at a 1 MHz detuning. The fringe oscillation frequency is extracted and used to refine `f_q` to sub-MHz precision. A decaying sinusoid is then fitted to extract `T2`.

**Output:** `step5_ramsey.png` — Ramsey fringes with fitted decay envelope.

**Key results:** `f_q_precise` (Hz), `T2` (s), `T2_err` (s).

---

### Step 5 — Adaptive Spectroscopy
```python
f_q, f_q_err, history = adaptive_spectroscopy(...)
```
A zoom-in spectroscopy loop that starts with a coarse 50-point sweep and progressively narrows the frequency window by a factor of `zoom_factor = 4` around the fitted peak centre. Convergence requires **both**:
- Peak height `P1 ≥ 0.8` at the fitted centre
- Fit uncertainty `f_q_err < 1 MHz`

Up to 8 rounds are attempted. Raises `SpectroscopyNotConvergedError` if convergence is not reached.

**Output:** Multi-panel figure from `plot_adaptive_history()` — one panel per zoom round.

**Key result:** Refined `f_q` and `f_q_err`.

> **Note:** There is a small bug in `adaptive_spectroscopy` — `p1_ok` and `err_ok` are used inside the convergence check before they are defined. If you encounter a `NameError`, move the two lines:
> ```python
> p1_ok  = p1_final >= peak_threshold
> err_ok = np.isfinite(f_q_err) and f_q_err < freq_err_threshold
> ```
> to just before the `if p1_ok and err_ok:` check inside the loop.

---

## Expected outputs

| File | Generated by |
|---|---|
| `step1_spectroscopy.png` | Step 1 |
| `step2_rabi.png` | Step 2a |
| `step3_T1.png` | Step 3a |
| `step5_ramsey.png` | Step 4 |

---

## Key parameters to tune for real hardware

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `freq_start / freq_stop` | Step 1 | 5 – 6 GHz | Widen if qubit is not found |
| `amp_stop` | Step 2a | 4.0 | Reduce if qubit saturates early |
| `evolve_stop` | Step 3a | 80 µs | Should be ≥ 3 × expected T1 |
| `T1_prior_mean` | Steps 2b, 3b | 30 µs | Set to rough T1 estimate from Step 3a |
| `confidence_target` | Steps 2b, 3b | 0.10 | Lower = more accurate, more shots |
| `detuning` | Step 4 | 1 MHz | Must produce at least 3 visible fringes |
| `evolve_stop` | Step 4 | 50 µs | Should be ≥ 2 × expected T2 |

---

## Emulation vs real hardware

All steps run in **emulation mode** by default (`session.connect(do_emulation=True)`). To connect to real hardware, change this line in `make_session()`:

```python
session.connect(do_emulation=False)
```

and ensure the device serials in `generate_device_setup_qubits()` match your physical setup.

## ============================================================
## AUTOMATION PART (BETA)
## ============================================================
## Running the automation campaign

The ⁠ automation/ ⁠ package wires your tune-up into a closed-loop orchestrator that runs many qubits in sequence with adaptive retries on failure. No API keys required.

### Quick start

⁠ bash
# from the repo root, with .venv active
python automation/run_campaign.py --n_qubits 50 --n_repeats 20 --workers 1
 ⁠

This runs *50 qubits × 20 repeats = 1000 tune-ups* sequentially. Each tune-up measures ⁠ f_q ⁠, ⁠ amp_pi ⁠, ⁠ T1 ⁠, ⁠ T2 ⁠ to <5% relative uncertainty, with the policy retrying up to 3× per experiment if a fit fails or precision targets aren't met.

### Smoke test first

Before committing to a long run, time a single tune-up on your machine:

⁠ bash
python automation/run_campaign.py --n_qubits 1 --n_repeats 1 --workers 1
 ⁠

You should see ⁠ status=ok ⁠ for the one qubit in 15–30 seconds. If that works, scale up.

### What you can vary

| Flag | Default | What it does |
|---|---|---|
| ⁠ --n_qubits ⁠ | 50 | How many distinct virtual qubits (each gets its own seed) |
| ⁠ --n_repeats ⁠ | 20 | How many independent tune-ups per qubit (different RNG, same hidden parameters) |
| ⁠ --max_iter ⁠ | 8 | Iteration budget per tune-up; raise for harder qubits, lower for speed |
| ⁠ --workers ⁠ | 1 | 1 = serial, >1 launches a multiprocessing pool. *On Windows keep this at 1* — process pickling has issues with closure-based runners |
| ⁠ --output ⁠ | ⁠ campaign_results.json ⁠ | Where to write the per-tune-up records |

You can also tune the convergence thresholds and retry behaviour by editing ⁠ automation/policy.py ⁠:

•⁠  ⁠⁠ REL_UNC_TARGET ⁠ (default ⁠ 0.05 ⁠) — relative uncertainty for "converged"
•⁠  ⁠⁠ RMS_TARGET ⁠ (default ⁠ 0.06 ⁠) — fit residual ceiling
•⁠  ⁠Retry escalation (n_points, shots, sweep widths) per experiment — see ⁠ RulePolicy.decide ⁠

### What you get

After the run finishes, the script writes three artefacts to your working directory:

•⁠  ⁠*⁠ campaign_results.json ⁠* — full record per tune-up: status, iteration count, wall time, and (mean, std) for each parameter. Top-level summary block reports counts of ⁠ ok ⁠ / ⁠ partial ⁠ / ⁠ failed ⁠ and total wall time.
•⁠  ⁠*⁠ campaign_spread.png ⁠* — histogram of fitted ⁠ f_q ⁠, ⁠ amp_pi ⁠, ⁠ T1 ⁠, ⁠ T2 ⁠ across all tune-ups. A bimodal distribution typically signals a fit landing on the wrong cosine extremum (worth investigating).
•⁠  ⁠*⁠ campaign_dashboard.png ⁠* — three-panel health view: status pie, iterations-to-converge histogram, wall-time histogram per tune-up.

The progress line during execution looks like:


[ 124/1000]  ok=120  partial=3  failed=1  elapsed= 38.4s  ETA=271.6s


### Expected runtime

Per-tune-up wall time is dominated by the QuTiP solver inside each ⁠ vq.evolve ⁠. Typical numbers on a modern laptop CPU (~3 GHz, single core):

| Scale | Roughly |
|---|---|
| 1 tune-up | 15–30 s |
| 50 qubits × 1 repeat = 50 tune-ups | ~15–25 min |
| 50 qubits × 20 repeats = 1000 tune-ups | ~5–9 hours |

For a hackathon demo, *20 qubits × 5 repeats = 100 tune-ups (~30–50 min)* is a good balance — enough samples to make the parameter-spread histogram meaningful without requiring an overnight run.

### Notes

•⁠  ⁠The orchestrator never crashes mid-pipeline. If a fit fails, the failure is logged in ⁠ failures ⁠ and the policy decides whether to retry with widened parameters or move on. Worst-case outcome is ⁠ status="partial" ⁠ with NaN for unmeasured parameters.
•⁠  ⁠Each worker rebuilds its own LabOne Q session via ⁠ make_session() ⁠. Session creation takes ~1 s, so it's cheap relative to a tune-up.
•⁠  ⁠Reproducibility: ⁠ qubit_factory(qid, repeat) ⁠ seeds ⁠ VirtualQubit(seed=10000*qid + repeat) ⁠, so the campaign is deterministic given the same seeds. Re-run with the same flags to reproduce results bit-for-bit.

####################################
# CNN MODULE #
####################################

# Bayesian Neural Networks for 80% Efficiency Gains in Quantum Calibration

This repository contains a full pipeline for characterizing superconducting qubits using a Deep Ensemble of Neural Networks. By leveraging physics-informed sampling and Bayesian uncertainty quantification, we reduce the measurement overhead from 244 points to only 55 points per qubit.

---

## Project Structure

| File | Description |
| :--- | :--- |
| qubit.py | The simulator core. Contains the VirtualQubit class (QuTiP-based). |
| qubit_measurements.py | Measurement primitives (Spec, Rabi, T1, Ramsey) and sparse grid definitions. |
| generate_dataset.py | Multiprocessing script to generate thousands of simulated qubit datasets. |
| train_ensemble.py | Training script for the 5-member Gaussian MLP ensemble. |
| infer_qubit.py | Inference script for characterizing a new qubit and estimating uncertainty. |
| setup_euler.sh | Shell script to set up the Python environment on the ETH Euler Cluster. |
| generate_euler.sh | Slurm script to run massive dataset generation on Euler. |
| train_euler.sh | Slurm script for ensemble training on Euler. |

---

## Getting Started

### 1. Environment Setup
To run this on the ETH Euler Cluster (or a local Linux machine), initialize the environment:
$ bash setup_euler.sh

This creates a virtual environment 'qubit_env' and installs torch, qutip, laboneq, and numpy.

---

## Step-by-Step Workflow

### Phase 1: Dataset Generation
We generate a supervised dataset of (P1 features, hidden parameters) pairs. The features consist of concatenated results from 4 measurement sweeps: Spectroscopy, Rabi, T1, and Ramsey.

* Local Generation:
  $ python generate_dataset.py --n-qubits 2000 --workers 8 --out dataset.npz

* Euler Cluster (10k Qubits):
  $ sbatch generate_euler.sh
  (Note: This script uses 56 workers to simulate 10,000 qubits in parallel to avoid CPU thrashing.)

---

### Phase 2: Training the BNN
The model is a Deep Ensemble of 5 MLPs. Each member predicts a Gaussian distribution (Mean mu and Log-Variance sigma^2) for the four physical parameters (fq, T1, T2, amp_pi).

* Start Training:
  $ sbatch train_euler.sh

* Architecture Details:
  - Input: Concatenated P1 vectors (55 features).
  - Architecture: 4-layer MLP with LayerNorm and SiLU activations.
  - Loss: Gaussian Negative Log-Likelihood (NLL) to ensure calibrated uncertainty.

---

### Phase 3: Inference & Characterization
Once the model is trained, you can characterize a "new" (unseen) qubit. The BNN provides a point estimate and a Confidence Interval.

* Test on a Seeded Virtual Qubit:
  $ python infer_qubit.py --model-dir models_10k_v2 --seed 12345

* Run on Real Measurement Data:
  $ python infer_qubit.py --model-dir models_10k_v2 --features-file my_qubit_data.npy

---

## Understanding the Output

The inference script outputs a summary table with two types of uncertainty:
1. Aleatoric Uncertainty: Intrinsic noise from the measurement process (shot noise).
2. Epistemic Uncertainty: The "disagreement" between ensemble members, indicating how much the model trusts its own prediction.

### The Flag System
* SUCCESS: If relative uncertainty sigma/|mu| <= 5%, the prediction is highly reliable.
* [FLAG]: If uncertainty > 5%, the model automatically recommends a Full Sweep, as the qubit parameters likely fall outside the learned distribution.

---

## Physics-Informed Sampling
Our 80% efficiency gain is driven by the smart grids defined in qubit_measurements.py:
* Spectroscopy: 15 points linear. The network learns to interpolate the Lorentzian peak from the slopes.
* Rabi: 10 points linear.
* T1 & T2: 10 points Log-spaced (Geometric). This places more samples where the exponential decay is steepest, maximizing information per shot.

#################################
# EXECUTION GUIDE FOR THE CODES #
#################################

This document provides the exact commands required to run the various stages of the qubit characterization pipeline on a Cluster (may adapt the type of node).

---

## 1. Environment Setup
Run this once to create the virtual environment and install all dependencies.
$ bash setup_euler.sh

---

## 2. Dataset Generation
Generates a supervised dataset (features and labels) for training.

# On a local machine (e.g., 2000 qubits, 8 workers):
$ python generate_dataset.py --n-qubits 2000 --workers 8 --out dataset.npz

# On the Euler Cluster (10,000 qubits, 56 workers via Slurm):
$ sbatch generate_euler.sh

---

## 3. Model Training
Trains a Deep Ensemble of 5 MLPs using the generated dataset.

# To start training:
$ sbatch train_euler.sh

# To check training progress:
$ tail -f qubit_train-[JOB_ID].out

---

## 4. Inference (Characterization)
Uses the trained models to characterize a specific qubit and estimate uncertainty.

# Option A: Test on a specific VirtualQubit seed (e.g., seed 10001):
$ python infer_qubit.py --model-dir models_10k_v2 --seed 10001

# Option B: Run on a pre-measured feature file (.npy):
$ python infer_qubit.py --model-dir models_10k_v2 --features-file features.npy

# Option C: Run a batch of seeds using the Slurm script:
$ sbatch run_inference.sh

---

## 5. Monitoring & Maintenance
# View your active Slurm jobs:
$ squeue -u $USER

# Cancel a specific job:
$ scancel [JOB_ID]

# Verify the Python environment is active:
$ which python
# Expected: /cluster/home/[USER]/qubit_env/bin/python
