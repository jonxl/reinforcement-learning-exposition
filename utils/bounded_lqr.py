"""The colleague's fixed bounded scalar task, distinct from unbounded LQR/NLQR."""
import numpy as np
from utils.baseline import scalar_lqr


class BoundedScalarLQR:
    action_bound = 2.0
    tail_tolerance = 1e-6
    benchmark_specification = dict(dynamics='x_next = x + u', stage_cost='x^2 + u^2',
        x0=1., action_bound=2., gamma=.99, tail_tolerance=1e-6,
        reference='Riccati optimum is feasible from x0=1; not a globally quadratic constrained value')

    def __init__(self, gamma=.99):
        if gamma != .99:
            raise ValueError('The matched bounded protocol fixes gamma=0.99')
        self.base = scalar_lqr(gamma=gamma)
        self.gamma = gamma
        self.x0 = self.base.x0
        self.K_star = self.base.K_star
        self.J_star = self.base.J_star
        self.P = self.base.P
        self.n_state = self.n_control = 1
        # From x0=1 the unconstrained optimum is feasible at every future step.
        assert abs(self.K_star.item()) <= self.action_bound
        assert abs(1-self.K_star.item()) < 1

    def dynamics(self, x, u):
        return x + np.clip(u, -self.action_bound, self.action_bound)

    def stage_cost(self, x, u):
        return (x*x + np.clip(u, -self.action_bound, self.action_bound)**2).sum(axis=-1)

    def optimal_policy(self, x):
        return np.clip(self.base.optimal_policy(x), -self.action_bound, self.action_bound)

    def cost_tail_bound(self, x, horizon):
        v = np.abs(x[:, 0]); g = self.gamma; d = 1-g; b = self.action_bound
        return g**horizon*((v*v+b*b)/d + 2*v*b*g/d**2 + b*b*g*(1+g)/d**3)
