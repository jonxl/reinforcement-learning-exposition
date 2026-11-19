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

    def bias_variance(self, K_star, curvature) -> dict[str, np.ndarray]:
        """Split of the expected gap near the optimum (notes, Observation 3).

        ``curvature`` is c_gamma (scalar case) or the matrix C = H/2, H the Hessian of the gap at K*.
        With bias b = E k - K* and covariance S of the gains across seeds,

            g_bar ~ E[e'Ce] = b'Cb + tr(C S),

        which is c_gamma (bias^2 + variance) in one dimension. Also returned: the unweighted
        bias^2 = |b|^2 and variance = tr S.
        """
        if self.gains is None:
            raise ValueError("no gains recorded for this algorithm")
        K_star = np.asarray(K_star, dtype=float).reshape(-1)
        d = K_star.size
        C = np.atleast_2d(np.asarray(curvature, dtype=float))
        C = C * np.eye(d) if C.shape == (1, 1) else C
        b = self.gains.mean(axis=0) - K_star  # (n, d)
        n = self.gains.shape[1]
        if self.n_seeds > 1:
            dev = self.gains - self.gains.mean(axis=0)
            S = np.einsum("mni,mnj->nij", dev, dev) / (self.n_seeds - 1)
        else:
            S = np.zeros((n, d, d))
        bias_term = np.einsum("ni,ij,nj->n", b, C, b)
        var_term = np.einsum("ij,nji->n", C, S)
        return {
            "bias2": (b**2).sum(axis=-1),
            "variance": np.trace(S, axis1=1, axis2=2),
            "bias_term": bias_term,
            "variance_term": var_term,
            "predicted_gap": bias_term + var_term,
        }

    def steps_to_threshold(self, threshold: float) -> np.ndarray:
        """Per seed, the first checkpoint (in environment steps) with gap <= threshold; inf if never."""
        hit = self.gaps <= threshold
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        steps = np.asarray(self.steps, dtype=float)
        return np.where(first >= 0, steps[np.maximum(first, 0)], np.inf)

    def summary(self, delta: float = 0.05, thresholds=(0.1, 0.01), tail_fraction: float = 0.1) -> dict:
        """The headline numbers at the final checkpoint, plus floors and sample efficiency."""
        lo, hi = self.hoeffding_band(delta)
        q25, q50, q75 = self.quantiles()
        out = {
            "name": self.name,
            "n_seeds": self.n_seeds,
            "final_step": float(self.steps[-1]),
            "final_mean_gap": float(self.mean()[-1]),
            "final_se": float(self.standard_error()[-1]),
            "final_hoeffding_band": [float(lo[-1]), float(hi[-1])],
            "hoeffding_half_width": float(concentration.hoeffding_eps(self.n_seeds, delta, self.gaps.shape[1])),
            "final_median_iqr": [float(q50[-1]), float(q25[-1]), float(q75[-1])],
            "final_min_max": [float(self.gaps[:, -1].min()), float(self.gaps[:, -1].max())],
            "floor_estimate": self.floor(tail_fraction),
        }
        if self.unstable is not None:
            out["final_unstable_fraction"] = float(self.unstable[:, -1].mean())
        out["final_infinite_cost_fraction"] = float((self.gaps[:, -1] >= 1.0).mean())
        for thr in thresholds:
            s = self.steps_to_threshold(thr)
            reached = np.isfinite(s)
            out[f"reached_gap_{thr}"] = float(reached.mean())
            out[f"median_steps_to_gap_{thr}"] = float(np.median(s[reached])) if reached.any() else None
        if self.gains is not None:
            out["final_mean_gain"] = self.gains[:, -1].mean(axis=0).tolist()
            out["final_gain_std"] = self.gains[:, -1].std(axis=0, ddof=1).tolist() if self.n_seeds > 1 else None
        return out

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


def align(a: LearningCurves, b: LearningCurves) -> tuple[LearningCurves, LearningCurves]:
    """Put two runs on a common step grid so the orderings below compare the same N.

    If the checkpoints already coincide, the runs are returned unchanged. Otherwise the grid is the
    union of both checkpoint sets up to the shorter run, and each seed's gap at N is that of its
    latest checkpoint at or before N (the policy one would have if training stopped at N and the
    last saved policy were used). Nothing is interpolated.
    """
    if np.array_equal(a.steps, b.steps):
        return a, b
    end = min(a.steps[-1], b.steps[-1])
    grid = np.union1d(a.steps, b.steps)
    grid = grid[(grid >= max(a.steps[0], b.steps[0])) & (grid <= end)]

    def hold(c: LearningCurves) -> LearningCurves:
        idx = np.searchsorted(c.steps, grid, side="right") - 1
        return LearningCurves(
            steps=grid,
            gaps=c.gaps[:, idx],
            gains=None if c.gains is None else c.gains[:, idx],
            unstable=None if c.unstable is None else c.unstable[:, idx],
            name=c.name,
            meta={**c.meta, "aligned": "step-hold"},
        )

    return hold(a), hold(b)


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
