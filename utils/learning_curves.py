"""Learning curves over many seeds (notes, "The learning curve of one run" and "Many seeds").

One algorithm with fixed hyperparameters is run on M seeds and evaluated at n checkpoints, giving
the array gaps[j, i] = g_X(omega_j, N_i) in [0, 1]. Everything here is a function of that array
(plus, where available, the learned gains):

* the estimated curve g_hat_X(N) and its standard error sigma_X(N) / sqrt(M),
* confidence bands from Hoeffding and Chebyshev,
* the envelope min/max over seeds (needed for band separation),
* floors, the fraction of unstable runs, and the bias-variance split of the expected gap,
* running means over seeds, which illustrate the strong law of large numbers,
* the three orderings of the notes and crossing times.

The x-axis ``steps`` counts environment steps, so curves of different algorithms share an axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils import concentration


@dataclass
class LearningCurves:
    steps: np.ndarray  # (n,) environment steps at each checkpoint
    gaps: np.ndarray  # (M, n) gap of each seed at each checkpoint, in [0, 1]
    gains: np.ndarray | None = None  # (M, n, d) learned (or effective) gains, if defined
    unstable: np.ndarray | None = None  # (M, n) bool, undiscounted closed loop unstable
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps)
        self.gaps = np.asarray(self.gaps, dtype=float)
        if self.gaps.ndim != 2 or self.gaps.shape[1] != self.steps.shape[0]:
            raise ValueError("gaps must have shape (M, n) with n = len(steps)")
        if np.any(self.gaps < -1e-9) or np.any(self.gaps > 1 + 1e-9):
            raise ValueError("gaps must lie in [0, 1]")
        if self.gains is not None:
            self.gains = np.asarray(self.gains, dtype=float)
            if self.gains.ndim == 2:
                self.gains = self.gains[..., None]

    # -- the estimated curve -------------------------------------------------------------------

    @property
    def n_seeds(self) -> int:
        return self.gaps.shape[0]

    def mean(self) -> np.ndarray:
        """g_hat_X(N): the average over the M seeds run."""
        return self.gaps.mean(axis=0)

    def std(self) -> np.ndarray:
        """sigma_X(N), estimated with the unbiased sample variance."""
        return self.gaps.std(axis=0, ddof=1) if self.n_seeds > 1 else np.zeros(self.gaps.shape[1])

    def standard_error(self) -> np.ndarray:
        return self.std() / np.sqrt(self.n_seeds)

    # -- confidence bands ----------------------------------------------------------------------

    def hoeffding_band(self, delta: float = 0.05, simultaneous: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Band holding at every checkpoint with prob. >= 1 - delta (union bound if simultaneous)."""
        n = self.gaps.shape[1] if simultaneous else 1
        eps = concentration.hoeffding_eps(self.n_seeds, delta, n_checkpoints=n)
        m = self.mean()
        return np.clip(m - eps, 0.0, 1.0), np.clip(m + eps, 0.0, 1.0)

    def chebyshev_band(
        self, delta: float = 0.05, simultaneous: bool = True, estimated_variance: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chebyshev band, with the estimated variance per checkpoint or the worst case 1/4."""
        n = self.gaps.shape[1] if simultaneous else 1
        var = self.std() ** 2 if estimated_variance else np.full(self.gaps.shape[1], 0.25)
        eps = np.sqrt(var * n / (self.n_seeds * delta))
        m = self.mean()
        return np.clip(m - eps, 0.0, 1.0), np.clip(m + eps, 0.0, 1.0)

    def envelope(self) -> tuple[np.ndarray, np.ndarray]:
        """(min over seeds, max over seeds): the band used for band separation."""
        return self.gaps.min(axis=0), self.gaps.max(axis=0)

    def quantiles(self, qs=(0.25, 0.5, 0.75)) -> np.ndarray:
        return np.quantile(self.gaps, qs, axis=0)

    # -- floors, instability, bias and variance -----------------------------------------------

    def floor(self, tail_fraction: float = 0.1) -> float:
        """Estimate of lim_N g_bar_X(N): the mean of g_hat over the last fraction of checkpoints."""
        n = self.gaps.shape[1]
        k = max(1, int(round(tail_fraction * n)))
        return float(self.mean()[-k:].mean())

    def unstable_fraction(self) -> np.ndarray | None:
        return None if self.unstable is None else self.unstable.mean(axis=0)

    def bias_variance(self, K_star, curvature: float) -> dict[str, np.ndarray]:
        """Split of the expected gap near the optimum: g_bar ~ c [bias^2 + variance] (notes, Obs. 3).

        For d-dimensional gains, bias^2 = |E k - K*|^2 and variance = trace Cov k; the single
        curvature c is then only a rough stand-in for the Hessian (notes, "Limitations").
        """
        if self.gains is None:
            raise ValueError("no gains recorded for this algorithm")
        K_star = np.asarray(K_star, dtype=float).reshape(-1)
        mean_gain = self.gains.mean(axis=0)  # (n, d)
        bias2 = ((mean_gain - K_star) ** 2).sum(axis=-1)
        var = self.gains.var(axis=0, ddof=1).sum(axis=-1) if self.n_seeds > 1 else np.zeros_like(bias2)
        return {"bias2": bias2, "variance": var, "predicted_gap": curvature * (bias2 + var)}

    # -- strong law illustration ----------------------------------------------------------------

    def running_means(self, checkpoint: int = -1, n_orderings: int = 20, seed: int = 0) -> np.ndarray:
        """Running averages g_hat after 1, 2, ..., M seeds, for several random seed orderings.

        Each row is one sample path of the estimator as seeds are added. The strong law says each
        path converges to g_bar with probability one; plotting them shows the paths settling.
        """
        rng = np.random.default_rng(seed)
        y = self.gaps[:, checkpoint]
        counts = np.arange(1, self.n_seeds + 1)
        paths = [np.cumsum(y[rng.permutation(self.n_seeds)]) / counts for _ in range(n_orderings)]
        return np.asarray(paths)


# -- power laws ---------------------------------------------------------------------------------


def fit_power_law(steps: np.ndarray, values: np.ndarray, floor: float = 1e-12) -> tuple[float, float]:
    """Least-squares fit of values ~ kappa * steps^(-beta) on a log-log scale. Returns (kappa, beta)."""
    steps = np.asarray(steps, dtype=float)
    values = np.maximum(np.asarray(values, dtype=float), floor)
    keep = steps > 0
    slope, intercept = np.polyfit(np.log(steps[keep]), np.log(values[keep]), 1)
    return float(np.exp(intercept)), float(-slope)


def power_law_crossing(kappa_x: float, beta_x: float, kappa_y: float, beta_y: float) -> float | None:
    """N_XY = (kappa_x / kappa_y)^(1 / (beta_x - beta_y)), where the two power laws meet."""
    if beta_x == beta_y:
        return None
    return float((kappa_x / kappa_y) ** (1.0 / (beta_x - beta_y)))


# -- orderings across seeds (notes, "Ordering across seeds") -------------------------------------


def band_separated(a: LearningCurves, b: LearningCurves) -> np.ndarray:
    """max_omega g_A(omega, N) <= min_omega' g_B(omega', N): every run of A below every run of B."""
    return a.gaps.max(axis=0) <= b.gaps.min(axis=0)


def expected_ordered(a: LearningCurves, b: LearningCurves) -> np.ndarray:
    """g_hat_A(N) <= g_hat_B(N), the estimate of the expected-curve ordering."""
    return a.mean() <= b.mean()


def confidently_ordered(a: LearningCurves, b: LearningCurves, delta: float = 0.05) -> np.ndarray:
    """Upper Hoeffding band of A below lower band of B: g_bar_A < g_bar_B with prob. >= 1 - 2 delta."""
    _, a_hi = a.hoeffding_band(delta)
    b_lo, _ = b.hoeffding_band(delta)
    return a_hi < b_lo


def floor_ordered(a: LearningCurves, b: LearningCurves, tail_fraction: float = 0.1) -> bool:
    return a.floor(tail_fraction) <= b.floor(tail_fraction)


def crossing_time(steps: np.ndarray, ordered: np.ndarray) -> float | None:
    """Smallest checkpoint N_0 with the ordering holding at every checkpoint from N_0 on.

    ``ordered`` is a boolean array from one of the ordering functions above. Returns None when the
    ordering fails at the last checkpoint.
    """
    ordered = np.asarray(ordered, dtype=bool)
    if ordered.size == 0 or not ordered[-1]:
        return None
    failures = np.flatnonzero(~ordered)
    start = 0 if failures.size == 0 else failures[-1] + 1
    return float(np.asarray(steps)[start])
