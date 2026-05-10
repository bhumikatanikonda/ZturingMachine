"""
Bayesian Adaptive Estimator for Qubit Characterization
=======================================================
Supports three experiment types:
  - "exponential" : T1 decay         P(|1>) = exp(-t / T1)
  - "gaussian"    : Amplitude Rabi   P(|1>) = sin^2(pi * A / A_pi)
  - "sine"        : Ramsey (T2*)     P(|1>) = 0.5*(1 - cos(2*pi*df*t)*exp(-t/T2))

The estimator follows the Bayesian adaptive approach from:
  Berritta et al., "Real-time adaptive tracking of fluctuating relaxation
  rates in superconducting qubits" (arXiv:2506.09576).

Key ideas:
  - Maintain a posterior P(theta) over the unknown parameter(s).
  - After each single-shot measurement m in {0,1}, update via Bayes rule:
        P(theta | m, x)  ∝  P(m | x, theta) * P(theta)
  - Choose the next probe point x adaptively to maximise information.
  - Stop when the relative uncertainty sigma/mu < confidence_target.

For T1 (exponential), the posterior is a Gamma distribution and updated
analytically via moment-matching (paper Eq. 5-6).

For Rabi and Ramsey, a discrete particle grid is used.

Usage
-----
  from bayesian_adaptive_estimator import BayesianAdaptiveEstimator

  estimator = BayesianAdaptiveEstimator(exp_type="exponential")
  estimator.reset()

  for _ in range(100):
      x = estimator.next_x()           # adaptive probe point
      m = your_experiment(x)           # single-shot: 0 or 1
      estimator.update(x, m)
      if estimator.has_converged():
          break

  result = estimator.result()
  print(result)

Or use the convenience runner:
  result = estimator.run(measurement_fn=your_experiment)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ExpType = Literal["exponential", "gaussian", "sine"]


# ---------------------------------------------------------------------------
# Physics: likelihood  P(m=1 | x, params)
# ---------------------------------------------------------------------------

def likelihood(
    x: float,
    params: np.ndarray,
    exp_type: ExpType,
    alpha: float = 0.02,
    beta: float = 0.02,
) -> float:
    """
    Return P(m=1 | x, params) including readout errors.

    Parameters
    ----------
    x        : probe point (time, amplitude, …)
    params   : parameter vector
                 exponential -> [T1]
                 gaussian    -> [A_pi]
                 sine        -> [delta_f, T2]
    alpha    : P(measure 0 | true state 1)  (mis-classification)
    beta     : P(measure 1 | true state 0)  (mis-classification)
    """
    if exp_type == "exponential":
        T1 = max(params[0], 1e-12)
        p_exc = np.exp(-x / T1)
    elif exp_type == "gaussian":
        A_pi = max(params[0], 1e-12)
        p_exc = np.sin(np.pi * x / A_pi) ** 2
    elif exp_type == "sine":
        delta_f, T2 = params[0], max(params[1], 1e-12)
        p_exc = 0.5 * (1.0 - np.cos(2.0 * np.pi * delta_f * x) * np.exp(-x / T2))
    else:
        raise ValueError(f"Unknown exp_type: {exp_type!r}")

    # Apply SPAM errors
    p1 = beta + (1.0 - alpha - beta) * p_exc
    return float(np.clip(p1, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Gamma posterior (T1 / exponential) — paper Section III
# ---------------------------------------------------------------------------

class GammaPosterior:
    """
    Gamma-distributed posterior for the decay rate Gamma1 = 1/T1.

    PARAMETERIZATION (rate form, matching the paper):
        P(Gamma1 | k, theta) ∝ Gamma1^(k-1) * exp(-theta * Gamma1)

        Here theta is the SCALE parameter of the Gamma distribution over Gamma1:
            E[Gamma1] = k / theta        (rate of decay)
            Var[Gamma1] = k / theta^2

        Therefore:
            E[T1] ≈ theta / k            (mean relaxation time)
            sigma_T1 ≈ theta / (k * sqrt(k))

    Constructor convention
    ----------------------
    Pass  theta = 1 / T1_prior_mean  so that the prior mean is correct:
        E[Gamma1] = k / theta = k * T1_prior_mean
        E[T1]     ≈ theta / k = T1_prior_mean / k  ... that's wrong.

    ACTUALLY simpler: use the SCALE parameterisation where
        theta = T1_prior_mean / k,  so E[Gamma1] = k / theta = 1/T1_prior_mean.

    To avoid confusion, accept `T1_prior_mean` directly as a constructor arg.
    """

    def __init__(self, k: float = 3.0, T1_prior_mean: float = 30e-6):
        """
        Parameters
        ----------
        k             : shape parameter (dimensionless).  Higher k → narrower prior.
        T1_prior_mean : prior best-guess for T1 [seconds].
                        Sets theta = k / T1_prior_mean  so that E[Gamma1] = 1/T1_prior_mean.
        """
        self.k = float(k)
        # E[T1] = theta/k  =>  theta = k * T1_prior_mean
        self.theta = float(k) * float(T1_prior_mean)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def mean_rate(self) -> float:
        """E[Gamma1] = k / theta  [1/s]"""
        return self.k / self.theta

    @property
    def mean_T1(self) -> float:
        """E_approx[T1] = theta / k  [s]"""
        return self.theta / self.k

    @property
    def std_rate(self) -> float:
        return np.sqrt(self.k) / self.theta

    @property
    def std_T1(self) -> float:
        """Propagated std on T1 ≈ theta / (k * sqrt(k))  [s]"""
        return self.theta / (self.k * np.sqrt(max(self.k, 1.0)))

    @property
    def rel_uncertainty(self) -> float:
        """sigma_T1 / T1_hat = 1/sqrt(k)"""
        return 1.0 / np.sqrt(max(self.k, 1e-6))

    # ------------------------------------------------------------------
    # Bayesian update — paper Eq. 5-6
    # ------------------------------------------------------------------

    def update(
        self,
        tau: float,
        m: int,
        alpha: float = 0.02,
        beta: float = 0.02,
    ) -> None:
        """
        Update (k, theta) in-place given measurement outcome m at wait time tau.

        We use the posterior moment integrals directly rather than the
        recursive formula, evaluating them numerically on a fine grid.
        This is exact and stable for all values of k, theta, tau.

        E[Gamma1 | m] = integral( Gamma1 * P(m|Gamma1,tau) * prior(Gamma1) ) / Z
        """
        k, theta = self.k, self.theta

        # Build a fine grid around the current mean
        mean_g1 = k / theta
        std_g1  = np.sqrt(k) / theta
        g1_lo   = max(1e-9, mean_g1 - 5 * std_g1)
        g1_hi   = mean_g1 + 5 * std_g1
        g1      = np.linspace(g1_lo, g1_hi, 2000)

        # Log prior: Gamma(k, theta)
        log_prior = (k * np.log(theta) - gammaln(k)
                     + (k - 1) * np.log(g1) - theta * g1)

        # Likelihood: P(m=1|tau,g1) = beta + (1-alpha-beta)*exp(-g1*tau)
        p1 = beta + (1.0 - alpha - beta) * np.exp(-g1 * tau)
        log_lk = np.log(np.clip(p1 if m == 1 else 1.0 - p1, 1e-300, 1.0))

        log_post = log_prior + log_lk
        log_post -= log_post.max()   # numerical stability
        post = np.exp(log_post)

        dg = g1[1] - g1[0]
        Z    = np.sum(post) * dg
        mu1  = np.sum(post * g1) * dg / Z
        mu2  = np.sum(post * g1 ** 2) * dg / Z

        var  = mu2 - mu1 ** 2
        if var > 1e-30 and mu1 > 1e-30:
            self.theta = mu1 / var
            self.k     = mu1 ** 2 / var

        self.k     = max(self.k, 0.5)
        self.theta = max(self.theta, 1e-15)

    # ------------------------------------------------------------------
    # PDF on a grid (for plotting / credible intervals)
    # ------------------------------------------------------------------

    def pdf_grid(self, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (T1_values, pdf_values) on a dense grid around the current estimate.
        """
        mu = self.mean_T1
        sig = self.std_T1
        t1_vals = np.linspace(max(1e-6, mu - 4 * sig), mu + 4 * sig, n)
        # Convert T1 -> Gamma1 = 1/T1 for gamma PDF, then apply Jacobian
        g1 = 1.0 / t1_vals
        log_pdf_g1 = (
            self.k * np.log(self.theta)
            - gammaln(self.k)
            + (self.k - 1) * np.log(g1)
            - self.theta * g1
        )
        pdf_g1 = np.exp(log_pdf_g1 - log_pdf_g1.max())   # normalise for display
        pdf_t1 = pdf_g1 / (t1_vals ** 2)                  # Jacobian |dGamma1/dT1|
        pdf_t1 /= np.trapezoid(pdf_t1, t1_vals)
        return t1_vals, pdf_t1

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        t1_vals, pdf_t1 = self.pdf_grid(n=2000)
        cdf = np.cumsum(pdf_t1) * (t1_vals[1] - t1_vals[0])
        cdf /= cdf[-1]
        lo = t1_vals[np.searchsorted(cdf, (1 - level) / 2)]
        hi = t1_vals[np.searchsorted(cdf, (1 + level) / 2)]
        return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Grid (particle) posterior — for Rabi and Ramsey
# ---------------------------------------------------------------------------

class GridPosterior:
    """
    Discrete grid posterior over one or two parameters.
    Each grid point carries a weight; after normalisation these are
    proportional to P(theta | data).
    """

    def __init__(
        self,
        param_ranges: list[tuple[float, float]],
        n_per_dim: int = 100,
    ):
        """
        Parameters
        ----------
        param_ranges : list of (min, max) per parameter
        n_per_dim    : grid resolution along each dimension
        """
        self.param_ranges = param_ranges
        self.n_params = len(param_ranges)
        self._build_grid(n_per_dim)
        self.weights = np.ones(len(self.grid)) / len(self.grid)

    def _build_grid(self, n: int) -> None:
        if self.n_params == 1:
            lo, hi = self.param_ranges[0]
            vals = np.linspace(lo, hi, n)
            self.grid = vals[:, None]           # shape (n, 1)
        else:
            # 2-D grid
            grids = [np.linspace(lo, hi, n) for lo, hi in self.param_ranges]
            mesh = np.meshgrid(*grids, indexing="ij")
            self.grid = np.stack([g.ravel() for g in mesh], axis=1)

    def update(
        self,
        x: float,
        m: int,
        exp_type: ExpType,
        alpha: float = 0.02,
        beta: float = 0.02,
    ) -> None:
        """Bayesian weight update for a single shot."""
        lk = np.array([
            likelihood(x, self.grid[i], exp_type, alpha, beta)
            for i in range(len(self.grid))
        ])
        lk = lk if m == 1 else (1.0 - lk)
        self.weights *= lk
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            # Reset to uniform if all weights vanish (numerical underflow)
            warnings.warn("All weights vanished — resetting to uniform.", RuntimeWarning)
            self.weights[:] = 1.0 / len(self.weights)

    def mean(self) -> np.ndarray:
        return (self.weights[:, None] * self.grid).sum(axis=0)

    def variance(self) -> np.ndarray:
        mu = self.mean()
        return (self.weights[:, None] * (self.grid - mu) ** 2).sum(axis=0)

    def std(self) -> np.ndarray:
        return np.sqrt(self.variance())

    def rel_uncertainty(self) -> float:
        """Relative uncertainty on the first (primary) parameter."""
        mu = self.mean()[0]
        sig = self.std()[0]
        return float(sig / max(abs(mu), 1e-12))

    def marginal_pdf(self, param_idx: int = 0, n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.param_ranges[param_idx]
        bins = np.linspace(lo, hi, n_bins)
        pdf = np.zeros(n_bins)
        for idx in range(len(self.grid)):
            val = self.grid[idx, param_idx]
            b = int((val - lo) / (hi - lo) * (n_bins - 1))
            b = max(0, min(n_bins - 1, b))
            pdf[b] += self.weights[idx]
        norm = np.trapezoid(pdf, bins)
        return bins, pdf / max(norm, 1e-30)

    def credible_interval(self, level: float = 0.90, param_idx: int = 0) -> tuple[float, float]:
        bins, pdf = self.marginal_pdf(param_idx, n_bins=2000)
        cdf = np.cumsum(pdf) * (bins[1] - bins[0])
        cdf /= cdf[-1]
        lo = bins[np.searchsorted(cdf, (1 - level) / 2)]
        hi = bins[np.searchsorted(cdf, (1 + level) / 2)]
        return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EstimatorResult:
    exp_type: str
    param_labels: list[str]
    estimates: list[float]
    uncertainties: list[float]
    rel_uncertainty: float
    credible_interval: tuple[float, float]   # 90% CI on primary parameter
    n_shots: int
    converged: bool
    shot_history: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"─── Bayesian Adaptive Estimator Result ───",
            f"  Experiment    : {self.exp_type}",
            f"  Shots taken   : {self.n_shots}",
            f"  Converged     : {self.converged}",
        ]
        for label, est, unc in zip(self.param_labels, self.estimates, self.uncertainties):
            lines.append(f"  {label:20s}: {est:.5g}  ±  {unc:.3g}")
        lines.append(
            f"  Rel. uncert.  : {self.rel_uncertainty * 100:.2f}%"
        )
        lo, hi = self.credible_interval
        lines.append(f"  90% CI        : [{lo:.5g}, {hi:.5g}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class BayesianAdaptiveEstimator:
    """
    Bayesian adaptive estimator for qubit characterisation experiments.

    Parameters
    ----------
    exp_type          : "exponential", "gaussian", or "sine"
    alpha             : readout error P(0|1)
    beta              : readout error P(1|0)
    confidence_target : stop when sigma/mu < this value
    max_shots         : hard upper limit on measurements
    grid_n            : grid resolution for Rabi/Ramsey (per dimension)
    gamma_k0          : initial gamma shape  (exponential only)
    gamma_theta0      : initial gamma scale  (exponential only)
    param_ranges      : override default prior ranges [(lo,hi), ...]
    """

    _DEFAULTS: dict[str, dict] = {
        "exponential": {
            "param_labels": ["T1 (s)"],
            "param_ranges": [(1e-6, 100e-6)],
            "x_range": (200e-9, 150e-6),
            "confidence_target": 0.10,
            "gamma_k0": 3.0,
            "T1_prior_mean": 30e-6,   # prior mean T1 = 30 µs (VirtualQubit range: 20–40 µs)
        },
        "gaussian": {
            "param_labels": ["A_pi"],
            "param_ranges": [(0.05, 1.5)],
            "x_range": (0.01, 1.5),
            "confidence_target": 0.03,
        },
        "sine": {
            "param_labels": ["delta_f (Hz)", "T2 (s)"],
            "param_ranges": [(0.3e6, 10e6), (2e-6, 60e-6)],
            "x_range": (50e-9, 50e-6),
            "confidence_target": 0.08,
        },
    }

    def __init__(
        self,
        exp_type: ExpType = "exponential",
        alpha: float = 0.02,
        beta: float = 0.02,
        confidence_target: Optional[float] = None,
        max_shots: int = 200,
        grid_n: int = 120,
        gamma_k0: Optional[float] = None,
        T1_prior_mean: Optional[float] = None,
        param_ranges: Optional[list[tuple[float, float]]] = None,
    ):
        """
        Parameters
        ----------
        exp_type       : "exponential", "gaussian", or "sine"
        alpha          : readout error P(0|1)
        beta           : readout error P(1|0)
        confidence_target : stop when sigma/mu < this value
        max_shots      : hard upper limit on measurements
        grid_n         : grid resolution for Rabi/Ramsey (per dimension)
        gamma_k0       : initial gamma shape for T1 (higher → narrower prior)
        T1_prior_mean  : prior mean T1 in SECONDS (exponential only).
                         Defaults to 30e-6 s.  Set to your best guess before
                         the first measurement to speed up convergence.
        param_ranges   : override prior ranges [(lo,hi), ...]
        """
        if exp_type not in self._DEFAULTS:
            raise ValueError(f"exp_type must be one of {list(self._DEFAULTS)}")
        self.exp_type = exp_type
        self.alpha = alpha
        self.beta = beta
        self.max_shots = max_shots
        self.grid_n = grid_n

        cfg = self._DEFAULTS[exp_type]
        self.param_labels: list[str] = cfg["param_labels"]
        self.x_range: tuple[float, float] = cfg["x_range"]
        self.param_ranges: list[tuple[float, float]] = (
            param_ranges if param_ranges is not None else cfg["param_ranges"]
        )
        self.confidence_target: float = (
            confidence_target if confidence_target is not None
            else cfg["confidence_target"]
        )
        self._gamma_k0: float = gamma_k0 if gamma_k0 is not None else cfg.get("gamma_k0", 3.0)
        self._T1_prior_mean: float = (
            T1_prior_mean if T1_prior_mean is not None else cfg.get("T1_prior_mean", 30e-6)
        )

        # Internal state (initialised by reset())
        self._posterior: Optional[GridPosterior] = None
        self._gamma: Optional[GammaPosterior] = None
        self._shot_history: list[dict] = []
        self._n_shots: int = 0
        self._last_x: Optional[float] = None

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return estimator to its initial state (uniform prior)."""
        if self.exp_type == "exponential":
            self._gamma = GammaPosterior(k=self._gamma_k0, T1_prior_mean=self._T1_prior_mean)
        else:
            self._posterior = GridPosterior(self.param_ranges, n_per_dim=self.grid_n)
        self._shot_history = []
        self._n_shots = 0
        self._last_x = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def next_x(self) -> float:
        """
        Return the next adaptive probe point x.

        Strategy (matches paper heuristics):
          exponential  -> tau = c * T1_hat, c ≈ 0.51
          gaussian     -> A   = A_pi_hat / 2  (steepest slope of sin^2)
          sine         -> t   = 1 / (2 * delta_f_hat)  (half-period)
        """
        est = self._get_estimate()
        xlo, xhi = self.x_range

        n = self._n_shots  # number of shots already taken
        if self.exp_type == "exponential":
            T1_hat = est[0]
            xlo_t, xhi_t = xlo, xhi
            # For the first few shots, use a log-spaced sweep across x_range
            # so the estimator can bootstrap T1 even from a bad prior.
            if n < 8:
                # Log-sweep covers the full dynamic range in a few shots
                log_probes = np.logspace(
                    np.log10(xlo_t * 2), np.log10(xhi_t * 0.8), 8
                )
                x = float(log_probes[n])
            else:
                # Paper heuristic c≈0.51 with quasi-random jitter
                c = 0.51
                jitter = 1.0 + 0.3 * np.sin(n * 1.618)
                x = c * T1_hat * jitter
        elif self.exp_type == "gaussian":
            A_pi_hat = est[0]
            # Cycle through fractions of A_pi to maximise Fisher information
            fracs = [0.25, 0.5, 0.75, 1.0, 1.25]
            x = fracs[n % len(fracs)] * A_pi_hat
        elif self.exp_type == "sine":
            delta_f_hat = abs(est[0]) or 1.0
            T2_hat      = est[1] if len(est) > 1 else 20e-6
            # Cycle over a quarter, half, and full period plus a T2-informed point
            probes = [
                1.0 / delta_f_hat,
                0.25 / delta_f_hat,
                0.75 / delta_f_hat,
                min(T2_hat * 0.5, 2.0 / delta_f_hat),
            ]
            x = probes[n % len(probes)]

        # Hard clamp: never go below x_range minimum (protects mesolve from
        # near-zero durations that cause stiff ODE / IntegratorException).
        x = float(np.clip(x, xlo, xhi))
        self._last_x = x
        return x

    def update(self, x: float, m) -> None:
        """
        Incorporate a single-shot measurement.

        Parameters
        ----------
        x : probe point used for this shot (same units as next_x() — seconds
            for exponential/sine, amplitude units for gaussian)
        m : measurement outcome — 0 or 1.  Accepts int, np.integer, or a
            length-1 numpy array (as returned by VirtualQubit.measure(shots=1)).
        """
        # Accept numpy arrays / scalars from VirtualQubit.measure(shots=1)
        if hasattr(m, '__len__'):
            m = int(m[0])
        else:
            m = int(m)

        if m not in (0, 1):
            raise ValueError(f"Measurement outcome m must be 0 or 1, got {m!r}")

        if self.exp_type == "exponential":
            self._gamma.update(x, m, self.alpha, self.beta)
        else:
            self._posterior.update(x, m, self.exp_type, self.alpha, self.beta)

        est = self._get_estimate()
        unc = self._get_uncertainty()
        self._shot_history.append({
            "shot": self._n_shots + 1,
            "x": x,
            "m": m,
            "estimate": est.copy(),
            "uncertainty": unc.copy(),
            "rel_uncertainty": self._rel_uncertainty(),
        })
        self._n_shots += 1

    def has_converged(self) -> bool:
        """True if relative uncertainty is below the confidence target."""
        if self._n_shots == 0:
            return False
        return self._rel_uncertainty() < self.confidence_target

    def result(self) -> EstimatorResult:
        """Return a structured summary of the current estimate."""
        est = self._get_estimate()
        unc = self._get_uncertainty()
        ci = self._credible_interval()
        return EstimatorResult(
            exp_type=self.exp_type,
            param_labels=self.param_labels,
            estimates=est.tolist(),
            uncertainties=unc.tolist(),
            rel_uncertainty=self._rel_uncertainty(),
            credible_interval=ci,
            n_shots=self._n_shots,
            converged=self.has_converged(),
            shot_history=list(self._shot_history),
        )

    # ------------------------------------------------------------------
    # Convenience runner
    # ------------------------------------------------------------------

    def run(
        self,
        measurement_fn: Callable[[float], int],
        verbose: bool = True,
    ) -> EstimatorResult:
        """
        Run the full adaptive estimation loop.

        Parameters
        ----------
        measurement_fn : callable x -> m in {0, 1}
                         This is your experiment.  In simulation you can
                         pass a lambda that calls `simulate_shot`.
        verbose        : print progress every 10 shots

        Returns
        -------
        EstimatorResult
        """
        self.reset()
        for i in range(self.max_shots):
            x = self.next_x()
            m = measurement_fn(x)
            self.update(x, m)
            if verbose and (i + 1) % 10 == 0:
                est = self._get_estimate()
                print(
                    f"  Shot {i+1:3d} | x={x:.4g} | m={m} | "
                    f"{self.param_labels[0]}={est[0]:.4g} | "
                    f"σ/μ={self._rel_uncertainty()*100:.1f}%"
                )
            if self.has_converged():
                if verbose:
                    print(f"\n✓ Converged after {i+1} shots  (σ/μ={self._rel_uncertainty()*100:.2f}%)")
                break
        else:
            if verbose:
                print(f"\n⚠  Reached max_shots={self.max_shots} without convergence.")
        return self.result()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_estimate(self) -> np.ndarray:
        if self.exp_type == "exponential":
            return np.array([self._gamma.mean_T1])
        return self._posterior.mean()

    def _get_uncertainty(self) -> np.ndarray:
        if self.exp_type == "exponential":
            return np.array([self._gamma.std_T1])
        return self._posterior.std()

    def _rel_uncertainty(self) -> float:
        if self.exp_type == "exponential":
            return self._gamma.rel_uncertainty
        return self._posterior.rel_uncertainty()

    def _credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        if self.exp_type == "exponential":
            return self._gamma.credible_interval(level)
        return self._posterior.credible_interval(level, param_idx=0)

    # ------------------------------------------------------------------
    # Plotting (optional — requires matplotlib)
    # ------------------------------------------------------------------

    def plot(self, true_params: Optional[list[float]] = None) -> None:
        """
        Plot posterior PDF, convergence trace, and fitted signal curve.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib is required for plotting.  pip install matplotlib")
            return

        history = self._shot_history
        if not history:
            print("No shots recorded yet.")
            return

        est = self._get_estimate()
        fig = plt.figure(figsize=(14, 9), facecolor="#0a0e1a")
        fig.suptitle(
            f"Bayesian Adaptive Estimator — {self.exp_type}",
            color="#e2e8f0", fontsize=13, y=0.98,
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ax_sig  = fig.add_subplot(gs[0, :2])   # signal model
        ax_post = fig.add_subplot(gs[0, 2])    # posterior PDF
        ax_conv = fig.add_subplot(gs[1, :2])   # convergence trace
        ax_unc  = fig.add_subplot(gs[1, 2])    # relative uncertainty

        ACCENT, ACCENT2 = "#00d4ff", "#7c3aed"
        YELLOW, GREEN   = "#f59e0b", "#10b981"
        MUTED           = "#64748b"
        BG, PANEL       = "#0a0e1a", "#0f1628"

        for ax in [ax_sig, ax_post, ax_conv, ax_unc]:
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.xaxis.label.set_color(MUTED)
            ax.yaxis.label.set_color(MUTED)
            ax.title.set_color("#e2e8f0")

        xlo, xhi = self.x_range
        x_grid = np.linspace(xlo, xhi, 300)

        # ── Signal model
        p_fit = np.array([likelihood(x, est, self.exp_type, self.alpha, self.beta)
                          for x in x_grid])
        ax_sig.plot(x_grid, p_fit, color=ACCENT, lw=2, label="Estimated")
        if true_params is not None:
            tp = np.array(true_params)
            p_true = np.array([likelihood(x, tp, self.exp_type, self.alpha, self.beta)
                               for x in x_grid])
            ax_sig.plot(x_grid, p_true, color=ACCENT2, lw=1.5, ls="--", label="True")
        xs = np.array([h["x"] for h in history])
        ms = np.array([h["m"] for h in history])
        ax_sig.scatter(xs[ms == 1], np.ones(ms.sum()) * 1.02, c=YELLOW, s=18, zorder=5, label="m=1")
        ax_sig.scatter(xs[ms == 0], np.zeros((1 - ms).sum()) - 0.02, c=ACCENT2, s=18, zorder=5, alpha=0.7, label="m=0")
        ax_sig.set_xlabel(self.param_labels[0] if self.exp_type == "gaussian" else "x")
        ax_sig.set_ylabel("P(|1⟩)")
        ax_sig.set_title("Signal model + shots")
        ax_sig.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=MUTED)

        # ── Posterior PDF
        if self.exp_type == "exponential":
            t1v, pdv = self._gamma.pdf_grid()
            ax_post.plot(t1v, pdv, color=ACCENT, lw=2)
            ax_post.fill_between(t1v, pdv, alpha=0.25, color=ACCENT)
            if true_params:
                ax_post.axvline(true_params[0], color=ACCENT2, ls="--", lw=1.5, label="True")
            ax_post.axvline(est[0], color=ACCENT, lw=1.5, label="Estimate")
            ax_post.set_xlabel(self.param_labels[0])
        else:
            bins, pdf = self._posterior.marginal_pdf(0)
            ax_post.plot(bins, pdf, color=ACCENT, lw=2)
            ax_post.fill_between(bins, pdf, alpha=0.25, color=ACCENT)
            if true_params:
                ax_post.axvline(true_params[0], color=ACCENT2, ls="--", lw=1.5, label="True")
            ax_post.axvline(est[0], color=ACCENT, lw=1.5, label="Estimate")
            ax_post.set_xlabel(self.param_labels[0])
        ax_post.set_ylabel("Posterior density")
        ax_post.set_title("Posterior P(θ)")
        ax_post.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=MUTED)

        # ── Convergence trace
        shots_n = [h["shot"] for h in history]
        param_trace = [h["estimate"][0] for h in history]
        ax_conv.plot(shots_n, param_trace, color=ACCENT, lw=1.8)
        if true_params:
            ax_conv.axhline(true_params[0], color=ACCENT2, ls="--", lw=1.5)
        ax_conv.set_xlabel("Shot #")
        ax_conv.set_ylabel(self.param_labels[0])
        ax_conv.set_title("Parameter estimate vs shots")

        # ── Relative uncertainty
        unc_trace = [h["rel_uncertainty"] * 100 for h in history]
        ax_unc.plot(shots_n, unc_trace, color=YELLOW, lw=1.8)
        ax_unc.axhline(self.confidence_target * 100, color=GREEN, ls="--", lw=1.5,
                       label=f"Target {self.confidence_target*100:.0f}%")
        ax_unc.set_xlabel("Shot #")
        ax_unc.set_ylabel("σ/μ (%)")
        ax_unc.set_title("Relative uncertainty")
        ax_unc.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=MUTED)

        plt.savefig("bayesian_estimator_result.png", dpi=150, bbox_inches="tight",
                    facecolor=BG)
        plt.show()
        print("Figure saved to bayesian_estimator_result.png")


# ---------------------------------------------------------------------------
# Simulator helper (for testing without real hardware)
# ---------------------------------------------------------------------------

def simulate_shot(
    x: float,
    true_params: list[float],
    exp_type: ExpType,
    alpha: float = 0.02,
    beta: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Simulate a single-shot measurement from a virtual experiment.

    Parameters
    ----------
    x           : probe point
    true_params : ground-truth parameters (hidden from estimator)
    exp_type    : experiment type
    alpha, beta : readout error rates
    rng         : numpy random generator (for reproducibility)

    Returns
    -------
    m : 0 or 1
    """
    p1 = likelihood(x, np.array(true_params), exp_type, alpha, beta)
    _rng = rng if rng is not None else np.random.default_rng()
    return int(_rng.random() < p1)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # headless — remove if you want an interactive window

    rng = np.random.default_rng(42)

    # ── All times in SECONDS to match VirtualQubit units ──────────────────

    print("=" * 60)
    print("  DEMO 1 — T1 decay (exponential)  [units: seconds]")
    print("=" * 60)
    TRUE_T1 = [28e-6]   # 28 µs — within VirtualQubit._T1 range [20,40] µs
    est_t1 = BayesianAdaptiveEstimator(
        exp_type="exponential",
        confidence_target=0.10,
        max_shots=150,
        # T1_prior_mean: best guess before first shot (VirtualQubit range: 20-40 µs)
        T1_prior_mean=30e-6,
    )
    result_t1 = est_t1.run(
        measurement_fn=lambda x: simulate_shot(x, TRUE_T1, "exponential", rng=rng),
        verbose=True,
    )
    print(result_t1)
    est_t1.plot(true_params=TRUE_T1)

    print()
    print("=" * 60)
    print("  DEMO 2 — Amplitude Rabi (gaussian)  [units: amplitude a.u.]")
    print("=" * 60)
    TRUE_RABI = [0.62]
    est_rabi = BayesianAdaptiveEstimator(
        exp_type="gaussian",
        confidence_target=0.03,
        max_shots=150,
        param_ranges=[(0.1, 1.4)],
    )
    result_rabi = est_rabi.run(
        measurement_fn=lambda x: simulate_shot(x, TRUE_RABI, "gaussian", rng=rng),
        verbose=True,
    )
    print(result_rabi)
    est_rabi.plot(true_params=TRUE_RABI)

    print()
    print("=" * 60)
    print("  DEMO 3 — Ramsey (sine)  [delta_f in Hz, T2 in seconds]")
    print("=" * 60)
    TRUE_RAMSEY = [2.5e6, 18e-6]   # delta_f=2.5 MHz, T2*=18 µs
    est_ramsey = BayesianAdaptiveEstimator(
        exp_type="sine",
        confidence_target=0.08,
        max_shots=200,
        param_ranges=[(0.5e6, 8e6), (3e-6, 50e-6)],
        grid_n=60,
    )
    result_ramsey = est_ramsey.run(
        measurement_fn=lambda x: simulate_shot(x, TRUE_RAMSEY, "sine", rng=rng),
        verbose=True,
    )
    print(result_ramsey)
    est_ramsey.plot(true_params=TRUE_RAMSEY)

    # ── Example integration with VirtualQubit (pseudocode) ────────────────
    print("""
─── VirtualQubit integration pattern ───────────────────────────────────────

    from qubit import VirtualQubit
    from BayesianEstimator import BayesianAdaptiveEstimator

    qubit = VirtualQubit(seed=42)

    # --- T1 measurement ---
    # Build pi-pulse once (from Rabi calibration)
    t_pi   = np.linspace(0, pulse_duration, 500)
    wave_pi = amp_pi * np.ones(len(t_pi), dtype=complex)

    est = BayesianAdaptiveEstimator("exponential", confidence_target=0.10)

    while not est.has_converged():
        tau = est.next_x()          # wait time in SECONDS

        qubit.reset()
        qubit.evolve(t_pi, wave_pi, drive_freq=f_q)   # pi pulse -> |1>
        qubit.wait(f_q, tau)                           # free decay
        bit = qubit.measure(shots=1)                   # 0 or 1

        est.update(tau, bit)        # bit can be numpy array — handled automatically

    result = est.result()
    T1_estimate = result.estimates[0]   # seconds
──────────────────────────────────────────────────────────────────────────────
""")