"""Exact baselines: the discounted LQR and LQI optima that every algorithm is scored against.

Notes, Sections 2-5. The agents never see anything in this module; it is used only to evaluate them.

* ``LQRProblem`` (regulation): (A, B, Q, R, gamma, x0) -> exact (P, K*, J*), and for any linear gain
  K its discounted cost J(K), gap g(K) and undiscounted stability. Arbitrary policies (e.g. tables)
  are scored by a long rollout.
* ``LQIProblem`` (tracking): LQR on the system enlarged by the integral of the tracking error,
  written in deviation coordinates and delegated to ``LQRProblem``.
* Scalar closed forms for a = b = q = r = x0 = 1, so the tests can check the solver against the
  manuscript.

LQI scope: only the disturbance-free case d = 0 is supported. Then the integral state rests at
z = 0 for every gain, so J(K) compares all gains from the same starting deviation and the Riccati
gain is the exact optimum. With d != 0 the equilibrium z_ss = (d - K_x y_ref)/K_z depends on the
gain, which the notes have not settled yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from utils import control

# =================================================================================================
# LQR: regulation to zero
# =================================================================================================
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

    def gap_hessian(self, h: float = 1e-4) -> np.ndarray:
        """Hessian of g at K* over the flattened gain, by central differences.

        Near the optimum g(K* + e) ~ e'(H/2)e; in the scalar case H/2 is the curvature c_gamma.
        """
        k0 = self.K_star.reshape(-1)
        d = k0.size
        H = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                ei, ej = np.eye(d)[i] * h, np.eye(d)[j] * h
                H[i, j] = (
                    self.gap(k0 + ei + ej) - self.gap(k0 + ei - ej) - self.gap(k0 - ei + ej) + self.gap(k0 - ei - ej)
                ) / (4 * h * h)
        return 0.5 * (H + H.T)

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


# =================================================================================================
# LQI: tracking a constant reference with integral action (notes, "From LQR to LQI")
#
# Deviation coordinates x~ = x - x_ss, u~ = u - u_ss with x_ss = A x_ss + B u_ss, C x_ss = y_ref.
# The tracking error is -C x~, and xi = (x~, z) is an ordinary LQR system
#     xi_{t+1} = [[A, 0], [-C, I]] xi_t + [[B], [0]] u~_t,
#     cost sum_t gamma^t (x~'Q x~ + z'Q_z z + u~'R u~),
# whose optimal policy is proportional plus integral, u~ = -K_x x~ - K_z z.
# =================================================================================================


@dataclass(frozen=True)
class LQIProblem:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    Q_z: np.ndarray
    R: np.ndarray
    gamma: float
    y_ref: np.ndarray
    x0: np.ndarray
    d: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("A", "B", "C", "Q", "Q_z", "R"):
            object.__setattr__(self, name, control.as_matrix(getattr(self, name)))
        object.__setattr__(self, "y_ref", np.asarray(self.y_ref, dtype=float).reshape(-1))
        object.__setattr__(self, "x0", np.asarray(self.x0, dtype=float).reshape(-1))
        if self.d is not None and np.any(np.asarray(self.d) != 0):
            raise NotImplementedError("d != 0: the integral equilibrium depends on the gain (see module docstring)")

    @property
    def n_state(self) -> int:
        return self.A.shape[0]

    @property
    def n_output(self) -> int:
        return self.C.shape[0]

    @cached_property
    def equilibrium(self) -> tuple[np.ndarray, np.ndarray]:
        """(x_ss, u_ss) solving (A - I) x + B u = 0, C x = y_ref (least squares if not square)."""
        n, m = self.n_state, self.B.shape[1]
        lhs = np.block([[self.A - np.eye(n), self.B], [self.C, np.zeros((self.n_output, m))]])
        rhs = np.concatenate([np.zeros(n), self.y_ref])
        sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if not np.allclose(lhs @ sol, rhs, atol=1e-9):
            raise ValueError("no equilibrium reaches y_ref")
        return sol[:n], sol[n:]

    @cached_property
    def augmented(self) -> LQRProblem:
        """The LQR problem in xi = (x - x_ss, z), starting from (x0 - x_ss, 0)."""
        n, p = self.n_state, self.n_output
        A_aug = np.block([[self.A, np.zeros((n, p))], [-self.C, np.eye(p)]])
        B_aug = np.vstack([self.B, np.zeros((p, self.B.shape[1]))])
        Q_aug = np.block([[self.Q, np.zeros((n, p))], [np.zeros((p, n)), self.Q_z]])
        x_ss, _ = self.equilibrium
        xi0 = np.concatenate([self.x0 - x_ss, np.zeros(p)])
        return LQRProblem(A=A_aug, B=B_aug, Q=Q_aug, R=self.R, gamma=self.gamma, x0=xi0)

    # -- the baseline, all delegated to the augmented LQR ---------------------------------------

    @property
    def K_star(self) -> np.ndarray:
        """[K_x, K_z]."""
        return self.augmented.K_star

    @property
    def J_star(self) -> float:
        return self.augmented.J_star

    def cost(self, K) -> float:
        return self.augmented.cost(K)

    def gap(self, K) -> float:
        return self.augmented.gap(K)

    def is_stable(self, K) -> bool:
        return self.augmented.is_stable(K)

    # -- the raw system, as an agent would experience it ----------------------------------------

    def simulate(self, K, steps: int) -> dict[str, np.ndarray]:
        """Run u = u_ss - K_x (x - x_ss) - K_z z on the raw system from x0, z0 = 0."""
        K = np.asarray(K, dtype=float).reshape(self.B.shape[1], self.n_state + self.n_output)
        K_x, K_z = K[:, : self.n_state], K[:, self.n_state :]
        x_ss, u_ss = self.equilibrium
        x, z = self.x0.copy(), np.zeros(self.n_output)
        xs, zs, us = [], [], []
        for _ in range(steps):
            u = u_ss - K_x @ (x - x_ss) - K_z @ z
            xs.append(x), zs.append(z), us.append(u)
            z = z + (self.y_ref - self.C @ x)
            x = self.A @ x + self.B @ u
        return {"x": np.asarray(xs), "z": np.asarray(zs), "u": np.asarray(us), "y": np.asarray(xs) @ self.C.T}


def scalar_lqi(
    gamma: float = 0.99,
    q: float = 1.0,
    q_z: float = 1.0,
    r: float = 1.0,
    y_ref: float = 1.0,
    x0: float = 0.0,
) -> LQIProblem:
    """Scalar tracking example: a = b = c = 1, so u_ss = 0 and the raw control equals u~."""
    return LQIProblem(A=1.0, B=1.0, C=1.0, Q=q, Q_z=q_z, R=r, gamma=gamma, y_ref=[y_ref], x0=[x0])
