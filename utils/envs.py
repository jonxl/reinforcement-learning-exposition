"""Environments: what the algorithms interact with, how they are built by name, and how policies are scored.

Three pieces, so that any algorithm can be attached to any environment:

* ``Problem``: the specification of an environment (dynamics, stage cost, discount, start state) plus
  its exact optimum J*. ``utils.baseline`` holds the implementations: LQR, LQI (as its augmented LQR
  problem, whose state is the deviation (x - x_ss, z) the agent can compute itself) and NLQR.
* ``PROBLEMS`` / ``make_problem``: the registry. A problem is fully determined by (name, gamma,
  params), which every run saves, so the environment of any saved run can be rebuilt exactly.
  Adding an environment means adding one entry here.
* ``Env``: the batched simulator of any problem, and the scoring functions shared by all algorithms.
  The agent only sees observations and rewards r_t = -c(x_t, u_t); nothing else about the problem is
  exposed to the algorithm's logic.

Rules agreed for the experiments:
* no early termination, and the dynamics are never clipped (ending a run early would hide future
  costs, so blowing up would look cheap);
* every seed carries its own state; the batch dimension is only for speed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from utils import baseline, control

EVAL_HORIZON = 5000  # gamma^5000 ~ 1e-22 at gamma = 0.99: truncation is negligible
BLOWUP = 1e12  # a state this large counts as divergence: cost +inf, gap 1


class Problem(Protocol):
    """What an environment specification must provide. Arrays are batched: x (E, n), u (E, m)."""

    gamma: float
    x0: np.ndarray
    n_state: int
    n_control: int
    J_star: float  # exact optimal discounted cost from x0
    K_star: np.ndarray  # optimal gain if the optimum is linear, else the gain of the linearization

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...

    def optimal_policy(self, x: np.ndarray) -> np.ndarray: ...


# -- the registry ---------------------------------------------------------------------------------


def _lqi(gamma: float = 0.99, **params) -> baseline.LQRProblem:
    return baseline.scalar_lqi(gamma=gamma, **params).augmented


PROBLEMS: dict[str, Callable[..., Problem]] = {
    "lqr": baseline.scalar_lqr,  # scalar regulation, the running example of the notes
    "lqi": _lqi,  # scalar tracking with integral action
    "nlqr": baseline.nlqr,  # two-state nonlinear benchmark; params: k (nonlinearity), x0, ...
    "nlqr_scalar": baseline.scalar_nlqr,  # the scalar running example with a state-dependent gain
}


def make_problem(name: str, gamma: float = 0.99, **params) -> Problem:
    """Build a registered problem. (name, gamma, params) is all that is needed to rebuild it later."""
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; known: {sorted(PROBLEMS)}")
    return PROBLEMS[name](gamma=gamma, **params)


# -- the simulator --------------------------------------------------------------------------------


class Env:
    """``n_envs`` independent copies of one problem, stepped together."""

    def __init__(self, problem: Problem, n_envs: int):
        self.problem = problem
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
        cost = self.problem.stage_cost(self.x, u)
        self.x = self.problem.dynamics(self.x, u)
        return self.x.copy(), -cost

    def evaluate(
        self, policy: Callable[[np.ndarray], np.ndarray], horizon: int = EVAL_HORIZON
    ) -> dict[str, np.ndarray]:
        """Discounted cost of a deterministic batched policy from x0, for every env at once.

        ``policy`` maps observations (n_envs, n_state) to controls (n_envs, n_control).
        Returns per env: cost J (inf if divergent), the visited states and controls (for gain fits),
        and the final state.
        """
        problem = self.problem
        x = np.tile(problem.x0, (self.n_envs, 1))
        J = np.zeros(self.n_envs)
        alive = np.ones(self.n_envs, dtype=bool)
        discount = 1.0
        xs = np.empty((horizon, self.n_envs, self.n_state))
        us = np.empty((horizon, self.n_envs, self.n_control))
        for t in range(horizon):
            u = np.asarray(policy(x), dtype=float).reshape(self.n_envs, self.n_control)
            xs[t], us[t] = x, u
            c = problem.stage_cost(x, u)
            J += discount * np.where(alive, c, 0.0)
            x = problem.dynamics(x, u)
            discount *= self.gamma
            diverged = ~np.isfinite(x).all(axis=1) | (np.abs(x).max(axis=1) > BLOWUP)
            alive &= ~diverged
            x = np.where(alive[:, None], x, 0.0)  # freeze divergent envs; their cost is set to inf
        J[~alive] = np.inf
        return {"J": J, "states": xs, "controls": us, "final_state": x, "diverged": ~alive}


# -- scoring, shared by every algorithm -----------------------------------------------------------


def rollout_unstable(ev: dict[str, np.ndarray], problem: Problem) -> np.ndarray:
    """Per env: the evaluation rollout diverged, or its state late in the rollout exceeds the start.

    The rollout version of "closed loop unstable", for policies and problems without an eigenvalue test.
    """
    late = np.abs(ev["states"][-100:]).max(axis=(0, 2))
    return ev["diverged"] | (late > np.abs(problem.x0).max() + 1e-12)


def score_policy(problem: Problem, policy: Callable[[np.ndarray], np.ndarray], n_envs: int,
                 horizon: int = EVAL_HORIZON) -> dict[str, np.ndarray]:
    """Score any deterministic batched policy (one per env) by a long rollout from x0.

    This is the scorer for every policy that is not a plain gain: a table, a network. Returns per
    env the gap, the instability flag, the effective linear gain (discounted least-squares fit along
    the rollout, flattened to (n_envs, m * n)), the final |state| and the divergence flag.
    """
    ev = Env(problem, n_envs).evaluate(policy, horizon)
    weights = problem.gamma ** np.arange(horizon)
    return {
        "gaps": np.asarray(control.gap(ev["J"], problem.J_star)),
        "unstable": rollout_unstable(ev, problem),
        "gains": effective_gain(ev["states"], ev["controls"], weights).reshape(n_envs, -1),
        "final_abs_state": np.abs(ev["final_state"]).max(axis=1),
        "diverged": ev["diverged"],
    }


def score_gains(problem: Problem, K: np.ndarray, horizon: int = EVAL_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Gap and instability of the linear policies u = -K[j] x, for gains K of shape (M, m, n).

    On a linear problem the discounted cost of a gain is exact (Lyapunov equation) and stability is
    the eigenvalue test; on any other problem both come from a long deterministic rollout.
    """
    K = np.asarray(K, dtype=float)
    if isinstance(problem, baseline.LQRProblem):
        gaps = np.array([problem.gap(k) for k in K], dtype=float)
        unstable = np.array([not problem.is_stable(k) for k in K], dtype=bool)
        return gaps, unstable
    ev = Env(problem, K.shape[0]).evaluate(lambda x: -np.einsum("en,ejn->ej", x, K), horizon)
    return np.asarray(control.gap(ev["J"], problem.J_star)), rollout_unstable(ev, problem)


def best_linear_gap(problem: Problem, horizon: int = EVAL_HORIZON) -> tuple[float, np.ndarray]:
    """The smallest gap any linear policy u = -Kx reaches on ``problem``, and the gain that reaches it.

    Zero on a linear-quadratic problem (K* is optimal). On a nonlinear problem it is the floor of
    every algorithm whose policy is a gain: found by Nelder-Mead from K* of the linearization and two
    rescalings of it, the best of the three kept.
    """
    from scipy.optimize import minimize

    if isinstance(problem, baseline.LQRProblem):
        return 0.0, problem.K_star
    shape = problem.K_star.shape

    def gap_of(k: np.ndarray) -> float:
        return float(score_gains(problem, k.reshape(1, *shape), horizon)[0][0])

    k0 = problem.K_star.reshape(-1)
    best = min((minimize(gap_of, scale * k0, method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000}) for scale in (1.0, 0.7, 1.2)),
               key=lambda res: res.fun)
    return float(best.fun), best.x.reshape(shape)


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
