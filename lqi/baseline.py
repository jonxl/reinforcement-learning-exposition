"""The exact LQI optimum: LQR on the system enlarged by the integral of the tracking error.

Notes, "From LQR to LQI". The output y = Cx should follow a constant reference y_ref. The integral
state accumulates the tracking error, z_{t+1} = z_t + (y_ref - C x_t). In deviation coordinates

    x~ = x - x_ss,   u~ = u - u_ss,   with   x_ss = A x_ss + B u_ss,   C x_ss = y_ref,

the tracking error is -C x~, and xi = (x~, z) evolves as an ordinary LQR system

    xi_{t+1} = [[A, 0], [-C, I]] xi_t + [[B], [0]] u~_t,   cost  sum_t gamma^t (x~'Q x~ + z'Q_z z + u~'R u~).

The optimal policy is proportional plus integral, u~ = -K_x x~ - K_z z, and the baseline (P, K*, J*)
is the discounted Riccati solution of this augmented problem, delegated to ``LQRProblem``.

Scope: only the disturbance-free case d = 0 is supported. Then the equilibrium of the integral
state is z = 0 for every gain, so J(K) compares all gains from the same starting deviation and the
Riccati gain is the exact optimum. With d != 0 the equilibrium z_ss = (d - K_x y_ref)/K_z depends on
the gain, and that case is not yet settled in the notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from lqr.baseline import LQRProblem
from utils import control


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
