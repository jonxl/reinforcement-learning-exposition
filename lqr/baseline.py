"""The exact LQR optimum: the regulation baseline of the notes (Sections 2-3 and 5).

``LQRProblem`` holds (A, B, Q, R, gamma, x0) and gives the exact optimum (P, K*, J*) and, for any
linear gain K, its discounted cost J(K), gap g(K) and undiscounted stability. The agents never see
any of this; it is used only to score them.

The scalar functions reproduce the closed forms derived in the notes for a = b = q = r = x0 = 1, so
the tests can check the general solver against the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from utils import control


@dataclass(frozen=True)
class LQRProblem:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    gamma: float
    x0: np.ndarray

    def __post_init__(self) -> None:
        for name in ("A", "B", "Q", "R"):
            object.__setattr__(self, name, control.as_matrix(getattr(self, name)))
        object.__setattr__(self, "x0", np.asarray(self.x0, dtype=float).reshape(-1))

    @property
    def n_state(self) -> int:
        return self.A.shape[0]

    @property
    def n_control(self) -> int:
        return self.B.shape[1]

    @cached_property
    def _riccati(self) -> tuple[np.ndarray, np.ndarray]:
        return control.solve_discounted_riccati(self.A, self.B, self.Q, self.R, self.gamma)

    @property
    def P(self) -> np.ndarray:
        """Cost-to-go matrix: V(x) = x'Px."""
        return self._riccati[0]

    @property
    def K_star(self) -> np.ndarray:
        return self._riccati[1]

    @property
    def J_star(self) -> float:
        return float(self.x0 @ self.P @ self.x0)

    def cost(self, K) -> float:
        """J(K): discounted cost of u = -Kx from x0, +inf outside the finite-cost set."""
        return control.gain_cost(self.A, self.B, self.Q, self.R, self._gain(K), self.gamma, self.x0)

    def gap(self, K) -> float:
        """g(K) = 1 - J*/J(K) in [0, 1]."""
        return control.gap(self.cost(K), self.J_star)

    def is_stable(self, K) -> bool:
        return control.is_stable(self.A, self.B, self._gain(K))

    def policy_cost(self, policy, horizon: int = 5000) -> float:
        """J(pi) for an arbitrary deterministic policy, by a long rollout from x0."""
        J, _, _ = control.rollout_cost(self.A, self.B, self.Q, self.R, policy, self.gamma, self.x0, horizon)
        return J

    def policy_gap(self, policy, horizon: int = 5000) -> float:
        return control.gap(self.policy_cost(policy, horizon), self.J_star)

    def _gain(self, K) -> np.ndarray:
        return np.asarray(K, dtype=float).reshape(self.n_control, self.n_state)


def scalar_lqr(gamma: float = 0.99, a: float = 1.0, b: float = 1.0, q: float = 1.0, r: float = 1.0, x0: float = 1.0) -> LQRProblem:
    """The running example of the notes: a = b = q = r = x0 = 1, gamma = 0.99 by default."""
    return LQRProblem(A=a, B=b, Q=q, R=r, gamma=gamma, x0=[x0])


# -- closed forms for a = b = q = r = x0 = 1 (notes, "The scalar case" and Section 5) -------------


def scalar_p(gamma: float) -> float:
    """Root p >= 0 of gamma p^2 + (1 - 2 gamma) p - 1 = 0; J* = p."""
    return ((2 * gamma - 1) + np.sqrt(4 * gamma**2 + 1)) / (2 * gamma)


def scalar_K_star(gamma: float) -> float:
    """Positive root of gamma k^2 + k - gamma = 0."""
    return (np.sqrt(1 + 4 * gamma**2) - 1) / (2 * gamma)


def scalar_finite_cost_interval(gamma: float) -> tuple[float, float]:
    """K_gamma = (1 - 1/sqrt(gamma), 1 + 1/sqrt(gamma)): gains with finite discounted cost."""
    s = 1.0 / np.sqrt(gamma)
    return 1.0 - s, 1.0 + s


def scalar_cost(k: float, gamma: float) -> float:
    """J(k) = (1 + k^2) / (1 - gamma (1 - k)^2) on K_gamma, +inf elsewhere."""
    denom = 1.0 - gamma * (1.0 - k) ** 2
    return float("inf") if denom <= 0 else (1.0 + k**2) / denom


def scalar_gap_exact(k: float, gamma: float) -> float:
    """g(k) = (1 + gamma J*)(k - K*)^2 / (1 + k^2) on K_gamma, 1 elsewhere."""
    lo, hi = scalar_finite_cost_interval(gamma)
    if not lo < k < hi:
        return 1.0
    p, K = scalar_p(gamma), scalar_K_star(gamma)
    return (1 + gamma * p) * (k - K) ** 2 / (1 + k**2)


def scalar_curvature(gamma: float) -> float:
    """c_gamma = lim g(K* + e) / e^2 = 1 / ((1 - K*)(1 + K*^2))."""
    K = scalar_K_star(gamma)
    return 1.0 / ((1 - K) * (1 + K**2))
