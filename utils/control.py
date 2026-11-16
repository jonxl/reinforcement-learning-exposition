"""Discounted linear-quadratic control: the math shared by the LQR and LQI baselines.

Notation follows the notes:

    x_{t+1} = A x_t + B u_t,   c(x, u) = x'Qx + u'Ru,   J(pi) = sum_t gamma^t c(x_t, u_t).

* The discounted Riccati equation is the undiscounted one for (sqrt(gamma) A, sqrt(gamma) B)
  (notes, "The discount as rescaled dynamics"), so scipy's standard solver is reused.
* The cost of a fixed linear gain u = -Kx is J(K) = x0' P_K x0, where P_K solves the discounted
  Lyapunov equation P_K = Q + K'RK + gamma (A - BK)' P_K (A - BK). It is finite exactly when
  sqrt(gamma) (A - BK) is stable, which for gamma < 1 is weaker than A - BK being stable
  (notes, "Finite cost is not stability").
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov


def as_matrix(m) -> np.ndarray:
    """Promote scalars and vectors to 2-D float arrays so scalar and matrix cases share code."""
    a = np.atleast_2d(np.asarray(m, dtype=float))
    return a


def spectral_radius(m: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(m))))


def solve_discounted_riccati(A, B, Q, R, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (P, K*) of the discounted discrete algebraic Riccati equation.

    P = Q + g A'PA - g^2 A'PB (R + g B'PB)^-1 B'PA,   K* = (R + g B'PB)^-1 g B'PA.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    A, B, Q, R = map(as_matrix, (A, B, Q, R))
    s = np.sqrt(gamma)
    P = solve_discrete_are(s * A, s * B, Q, R)
    P = 0.5 * (P + P.T)
    K = np.linalg.solve(R + gamma * B.T @ P @ B, gamma * B.T @ P @ A)
    return P, K


def gain_cost_matrix(A, B, Q, R, K, gamma: float) -> np.ndarray | None:
    """P_K for the linear gain K, or None when the discounted cost is infinite."""
    A, B, Q, R, K = map(as_matrix, (A, B, Q, R, K))
    closed = np.sqrt(gamma) * (A - B @ K)
    if spectral_radius(closed) >= 1.0:
        return None
    # solve_discrete_lyapunov(a, q) solves X = a X a' + q; we need P = q + c' P c with c = closed.
    P = solve_discrete_lyapunov(closed.T, Q + K.T @ R @ K)
    return 0.5 * (P + P.T)


def gain_cost(A, B, Q, R, K, gamma: float, x0) -> float:
    """J(K) = x0' P_K x0, or +inf outside the finite-cost set."""
    P = gain_cost_matrix(A, B, Q, R, K, gamma)
    if P is None:
        return float("inf")
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    return float(x0 @ P @ x0)


def is_stable(A, B, K) -> bool:
    """Undiscounted closed-loop stability, rho(A - BK) < 1, reported separately from the gap."""
    A, B, K = map(as_matrix, (A, B, K))
    return spectral_radius(A - B @ K) < 1.0


def gap(J: float | np.ndarray, J_star: float) -> float | np.ndarray:
    """g = 1 - J*/J in [0, 1], with a finite number divided by infinity equal to zero."""
    J = np.asarray(J, dtype=float)
    with np.errstate(divide="ignore"):
        ratio = np.where(np.isinf(J), 0.0, J_star / J)
    out = 1.0 - ratio
    return float(out) if out.ndim == 0 else out


def rollout_cost(
    A,
    B,
    Q,
    R,
    policy: Callable[[np.ndarray], np.ndarray],
    gamma: float,
    x0,
    horizon: int,
    blowup: float = 1e12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Discounted cost of an arbitrary deterministic policy by simulation.

    Used for policies that are not a single gain (tabular Monte Carlo). Use a horizon far longer
    than the training episodes so that gamma^horizon is negligible; otherwise the truncated cost
    can fall below J* and the gap turns negative. If the state exceeds ``blowup`` the run is
    declared divergent and the cost is +inf (gap one).

    Returns (cost, states, controls).
    """
    A, B, Q, R = map(as_matrix, (A, B, Q, R))
    x = np.asarray(x0, dtype=float).reshape(-1)
    xs, us = [], []
    J, discount = 0.0, 1.0
    for _ in range(horizon):
        u = np.asarray(policy(x), dtype=float).reshape(-1)
        xs.append(x)
        us.append(u)
        J += discount * float(x @ Q @ x + u @ R @ u)
        x = A @ x + B @ u
        discount *= gamma
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > blowup:
            return float("inf"), np.asarray(xs), np.asarray(us)
    return J, np.asarray(xs), np.asarray(us)
