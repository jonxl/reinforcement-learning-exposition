"""Inverse Bellman construction of deterministic nonlinear quadratic benchmarks.

Diagonal P and g(x) = (0, p cos(k x_1)) make H diagonal, avoiding matrix
inverses and square-root routines in the simulation loop. Sigma is zero.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from utils.baseline import LQRProblem


@dataclass(frozen=True)
class EnergyNLQRProblem:
    gamma: float = 0.99
    p1: float = 2.0
    p2: float = 1.5
    beta: float = 0.65
    r: float = 1.0
    p: float = 0.7
    k: float = 1.0
    theta_base: float = 0.4
    alpha: float = 1.2
    kappa: float = 2.0
    x0: tuple = (1.0, -1.0)

    def __post_init__(self):
        scalars = [self.gamma, self.p1, self.p2, self.beta, self.r,
                   self.p, self.k, self.theta_base, self.alpha, self.kappa]
        if not np.isfinite(scalars).all():
            raise ValueError("parameters must be finite")
        if not 0 < self.gamma < 1 or not 0 < self.beta < 1:
            raise ValueError("gamma and beta must lie strictly between zero and one")
        if min(self.p1, self.p2, self.r) <= 0:
            raise ValueError("P and R must be positive definite")
        x0 = np.asarray(self.x0, dtype=float)
        if x0.shape != (2,) or not np.isfinite(x0).all() or not np.any(x0):
            raise ValueError("x0 must be a finite nonzero two-vector")
        object.__setattr__(self, "x0", x0)

    n_state = 2
    n_control = 1

    @cached_property
    def P(self):
        return np.diag([self.p1, self.p2])

    @cached_property
    def Q(self):
        return self.beta * self.P

    @cached_property
    def R(self):
        return np.array([[self.r]])

    @property
    def J_star(self):
        return float(self.x0 @ self.P @ self.x0)

    def value(self, x):
        return self.p1 * x[:, 0] ** 2 + self.p2 * x[:, 1] ** 2

    def input_gain(self, x):
        return self.p * np.cos(self.k * x[:, 0])

    def action_curvature(self, x):
        return self.r + self.gamma * self.p2 * self.input_gain(x) ** 2

    def metric_diagonal(self, x):
        return np.column_stack([np.full(len(x), self.gamma * self.p1),
                                self.gamma * self.p2 * self.r / self.action_curvature(x)])

    def drift(self, x):
        theta = self.theta_base + self.alpha * np.tanh(self.kappa * x[:, 0] * x[:, 1])
        c, s = np.cos(theta), np.sin(theta)
        z = x * np.sqrt((1 - self.beta) * np.array([self.p1, self.p2]))
        rotated = np.column_stack([c * z[:, 0] - s * z[:, 1], s * z[:, 0] + c * z[:, 1]])
        return rotated / np.sqrt(self.metric_diagonal(x))

    def dynamics(self, x, u):
        nxt = self.drift(x)
        nxt[:, 1] += self.input_gain(x) * u[:, 0]
        return nxt

    def stage_cost(self, x, u):
        return self.beta * self.value(x) + self.r * u[:, 0] ** 2

    def optimal_policy(self, x):
        return (-self.gamma * self.p2 * self.input_gain(x) * self.drift(x)[:, 1]
                / self.action_curvature(x))[:, None]

    @cached_property
    def linear(self):
        c, s = np.cos(self.theta_base), np.sin(self.theta_base)
        rot = np.array([[c, -s], [s, c]])
        A = (rot * np.sqrt((1 - self.beta) * np.array([self.p1, self.p2])))
        A /= np.sqrt(self.metric_diagonal(np.zeros((1, 2)))[0])[:, None]
        return LQRProblem(A, [[0], [self.p]], self.Q, self.R, self.gamma, self.x0)

    @cached_property
    def K_star(self):
        """Gain of the linearization, not the global nonlinear optimal policy."""
        return self.gamma * self.p2 * self.p * self.linear.A[1:2] / (self.r + self.gamma * self.p2 * self.p**2)


def generate_parameters(count=10, seed=0, gamma=.99, profile="moderate"):
    """Reproducible moderate instances, with a uniform open-loop P-norm contraction.

    beta exceeds 1-gamma/(1+gamma*p2*p^2/r), so even the largest input
    gain gives V(f(x)) < V(x). This is a pilot family, not an instability sweep.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if not 0 < gamma < 1:
        raise ValueError("gamma must lie strictly between zero and one")
    if profile not in ("moderate", "aggressive"):
        raise ValueError("profile must be moderate or aggressive")
    if profile == "aggressive":
        return _aggressive_parameters(count, seed, gamma)
    rng = np.random.default_rng(seed)
    parameters = [dict(p1=float(rng.uniform(1.5, 3)), p2=float(rng.uniform(1, 2)),
                 beta=float(rng.uniform(.65, .8)), r=float(rng.uniform(.8, 1.4)),
                 p=float(rng.uniform(.35, .65)), k=float(rng.uniform(.7, 1.5)),
                 theta_base=float(rng.uniform(.2, .8)), alpha=float(rng.uniform(.7, 1.5)),
                 kappa=float(rng.uniform(1, 2.5)), x0=[1.0, -1.0]) for _ in range(count)]
    for params in parameters:
        beta_min = 1 - gamma / (1 + gamma * params['p2'] * params['p']**2 / params['r'])
        params['beta'] = max(params['beta'], beta_min + .1 * (1 - beta_min))
    return parameters


def validate(problem, seed=0, count=256):
    """Exact energy, Bellman and complete-square residuals; no noise sampling."""
    rng = np.random.default_rng(seed)
    x = np.vstack([np.zeros((1, 2)), rng.uniform(-3, 3, (count, 2)),
                   [[np.pi / (2 * problem.k), 1.0]] if problem.k else [[0, 1]]])
    f = problem.drift(x)
    u = problem.optimal_policy(x)
    v = problem.value(x)
    energy = (f**2 * problem.metric_diagonal(x)).sum(axis=1) - (1 - problem.beta) * v
    bellman = problem.stage_cost(x, u) + problem.gamma * problem.value(problem.dynamics(x, u)) - v
    delta = rng.normal(size=u.shape)
    perturbed = problem.stage_cost(x, u + delta) + problem.gamma * problem.value(problem.dynamics(x, u + delta)) - v
    gap = perturbed - problem.action_curvature(x) * delta[:, 0] ** 2
    errors = dict(energy_max_abs=float(np.max(abs(energy))), bellman_max_abs=float(np.max(abs(bellman))),
                  action_gap_max_abs=float(np.max(abs(gap))))
    if max(errors.values()) > 1e-10 * max(1, float(v.max())):
        raise ValueError(f"inverse Bellman validation failed: {errors}")
    return errors


def _aggressive_parameters(count, seed, gamma):
    """Stronger coupling and prescribed local open-loop rho in [1.05, 1.25].

    Selection uses analytic parameters only, never the outcomes of learning runs.
    beta > 1-gamma also ensures global contraction of the optimal closed loop.
    """
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(1000 * count):
        params = dict(p1=float(rng.uniform(1.5, 3)), p2=float(rng.uniform(1, 2)),
                      r=float(rng.uniform(.4, .8)), p=float(rng.uniform(.9, 1.3)),
                      k=float(rng.uniform(.7, 1.5)), theta_base=float(rng.uniform(.35, .8)),
                      alpha=float(rng.uniform(1.8, 3)), kappa=float(rng.uniform(2.5, 5)),
                      x0=[1., -1.])
        reference = EnergyNLQRProblem(gamma=gamma, beta=.5, **params)
        rho_base = max(abs(np.linalg.eigvals(reference.linear.A))) / np.sqrt(.5)
        target = rng.uniform(1.05, 1.25)
        beta = float(1 - (target / rho_base)**2)
        if 1 - gamma < beta < 1:
            result.append(dict(params, beta=beta))
            if len(result) == count:
                return result
    raise ValueError("could not construct the requested instability profile at this discount")
