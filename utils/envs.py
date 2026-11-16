"""The environment the algorithms interact with: a discounted linear-quadratic system, batched over seeds.

Built from any ``LQRProblem``. For LQI pass ``problem.augmented``: its state is the deviation
(x - x_ss, z), which the agent can compute itself (it knows y_ref and accumulates the error), so
regulation and tracking share one environment. The agent only sees observations and rewards
r_t = -c(x_t, u_t); the matrices are used to simulate, never exposed to the algorithm's logic.

Rules agreed for the experiments:
* no early termination, and the dynamics are never clipped (ending a run early would hide future
  costs, so blowing up would look cheap);
* every seed carries its own state; the batch dimension is only for speed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from utils.baseline import LQRProblem

EVAL_HORIZON = 5000  # gamma^5000 ~ 1e-22 at gamma = 0.99: truncation is negligible
BLOWUP = 1e12  # a state this large counts as divergence: cost +inf, gap 1


class LinearQuadraticEnv:
    def __init__(self, problem: LQRProblem, n_envs: int):
        self.problem = problem
        self.A, self.B, self.Q, self.R = problem.A, problem.B, problem.Q, problem.R
        self.gamma = problem.gamma
        self.n_envs = n_envs
        self.n_state = problem.n_state
        self.n_control = problem.n_control
        self.x = np.zeros((n_envs, self.n_state))

    def reset(self) -> np.ndarray:
        self.x = np.tile(self.problem.x0, (self.n_envs, 1))
        return self.x.copy()

    def step(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply controls u (n_envs, n_control); return (next observations, rewards)."""
        u = np.asarray(u, dtype=float).reshape(self.n_envs, self.n_control)
        cost = np.einsum("ei,ij,ej->e", self.x, self.Q, self.x) + np.einsum("ei,ij,ej->e", u, self.R, u)
        self.x = self.x @ self.A.T + u @ self.B.T
        return self.x.copy(), -cost

    def evaluate(
        self, policy: Callable[[np.ndarray], np.ndarray], horizon: int = EVAL_HORIZON
    ) -> dict[str, np.ndarray]:
        """Discounted cost of a deterministic batched policy from x0, for every env at once.

        ``policy`` maps observations (n_envs, n_state) to controls (n_envs, n_control).
        Returns per env: cost J (inf if divergent), the visited states and controls (for gain fits),
        and the final state.
        """
        x = np.tile(self.problem.x0, (self.n_envs, 1))
        J = np.zeros(self.n_envs)
        alive = np.ones(self.n_envs, dtype=bool)
        discount = 1.0
        xs = np.empty((horizon, self.n_envs, self.n_state))
        us = np.empty((horizon, self.n_envs, self.n_control))
        for t in range(horizon):
            u = np.asarray(policy(x), dtype=float).reshape(self.n_envs, self.n_control)
            xs[t], us[t] = x, u
            c = np.einsum("ei,ij,ej->e", x, self.Q, x) + np.einsum("ei,ij,ej->e", u, self.R, u)
            J += discount * np.where(alive, c, 0.0)
            x = x @ self.A.T + u @ self.B.T
            discount *= self.gamma
            diverged = ~np.isfinite(x).all(axis=1) | (np.abs(x).max(axis=1) > BLOWUP)
            alive &= ~diverged
            x = np.where(alive[:, None], x, 0.0)  # freeze divergent envs; their cost is set to inf
        J[~alive] = np.inf
        return {"J": J, "states": xs, "controls": us, "final_state": x, "diverged": ~alive}


def effective_gain(states: np.ndarray, controls: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares linear gain through the origin, u ~ -K x, per env.

    states (T, E, n), controls (T, E, m) -> K (E, m, n). Places a non-linear policy (a table) on the
    same axis as the linear ones, for the bias-variance plots. Discounted weights are typical.
    """
    T = states.shape[0]
    w = np.ones(T) if weights is None else weights
    Sxx = np.einsum("t,tei,tej->eij", w, states, states)
    Sux = np.einsum("t,tei,tej->eij", w, controls, states)
    n = states.shape[2]
    Sxx = Sxx + 1e-12 * np.eye(n)
    return -np.linalg.solve(Sxx.transpose(0, 2, 1), Sux.transpose(0, 2, 1)).transpose(0, 2, 1)
