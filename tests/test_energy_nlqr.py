"""Verify inverse Bellman construction against independent matrix algebra and rollouts."""
import json

import numpy as np
import pytest

from utils import envs
from utils.nlqr import EnergyNLQRProblem, generate_parameters, validate


@pytest.mark.parametrize('seed', range(10))
def test_generator_matches_dense_bellman_and_action_gap(seed):
    problem = EnergyNLQRProblem(**generate_parameters(10, 7)[seed])
    assert max(validate(problem).values()) < 1e-11
    rng = np.random.default_rng(seed)
    for x in rng.normal(size=(10, 2)):
        G = np.array([[0.], [problem.p * np.cos(problem.k * x[0])]])
        B = problem.R + problem.gamma * G.T @ problem.P @ G
        H = problem.gamma * problem.P - problem.gamma**2 * problem.P @ G @ np.linalg.solve(B, G.T @ problem.P)
        drift = problem.drift(x[None])[0]
        action = -problem.gamma * np.linalg.solve(B, G.T @ problem.P @ drift)
        assert drift @ H @ drift == pytest.approx(x @ (problem.P - problem.Q) @ x, abs=1e-12)
        assert problem.optimal_policy(x[None])[0] == pytest.approx(action, abs=1e-12)
    ev = envs.Env(problem, 1).evaluate(problem.optimal_policy, horizon=200)
    assert ev['J'][0] == pytest.approx(problem.J_star, abs=1e-10)
    assert problem.linear.P == pytest.approx(problem.P, abs=1e-10)
    assert problem.linear.K_star == pytest.approx(problem.K_star, abs=1e-10)


def test_rotation_and_gain_constant_recovers_linearization():
    p = EnergyNLQRProblem(alpha=0, k=0)
    rng = np.random.default_rng(4)
    x, u = rng.normal(size=(20, 2)), rng.normal(size=(20, 1))
    assert p.dynamics(x, u) == pytest.approx(p.linear.dynamics(x, u), abs=1e-12)
    assert p.optimal_policy(x) == pytest.approx(-x @ p.K_star.T, abs=1e-12)


def test_generation_reproducible_distinct_and_rebuildable():
    params = generate_parameters(10, 12)
    assert params == generate_parameters(10, 12)
    assert params != generate_parameters(10, 13)
    assert len({json.dumps(p, sort_keys=True) for p in params}) == 10
    for p in params:
        original = envs.make_problem('nlqr_energy', **p)
        rebuilt = envs.make_problem('nlqr_energy', **json.loads(json.dumps(p)))
        assert rebuilt.drift(np.array([[1., -1.]])) == pytest.approx(original.drift(np.array([[1., -1.]])))
        bound = (1 - original.beta) / original.gamma * (1 + original.gamma * original.p2 * original.p**2 / original.r)
        assert bound < 1


@pytest.mark.parametrize('kwargs', [dict(beta=0), dict(beta=1), dict(p1=0), dict(r=-1), dict(gamma=1), dict(alpha=np.nan)])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        EnergyNLQRProblem(**kwargs)


def test_suite_runs_both_policies_and_saves_rebuildable_baselines(tmp_path):
    from utils.suite import run_suite
    root = run_suite(tmp_path / 'suite', count=1, seeds=1, steps=512, algorithms=('ppo', 'ppo_linear'))
    manifest = json.loads((root / 'problems.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    assert len(manifest['cases']) == 2 and len(summary) == 4
    assert {r['algorithm'] for r in summary} == {'ppo', 'ppo_linear'}
    assert (root / 'learning_curves.png').stat().st_size > 0
    for case in manifest['cases']:
        prob = envs.make_problem(case['problem'], manifest['gamma'], **case['problem_params'])
        assert prob.J_star == pytest.approx(case['J_star'])
    with pytest.raises(FileExistsError):
        run_suite(root, count=1, seeds=1, steps=512)


@pytest.mark.parametrize('gamma', [.05, .5, .99, .999])
def test_generated_family_is_globally_contracting_for_requested_discount(gamma):
    for params in generate_parameters(10, 3, gamma):
        problem = EnergyNLQRProblem(gamma=gamma, **params)
        bound = (1 - problem.beta) / gamma * (1 + gamma * problem.p2 * problem.p**2 / problem.r)
        assert bound < 1
        assert max(validate(problem).values()) < 1e-10


def test_aggressive_family_is_unstable_without_control_and_optimal_policy_attains_value():
    parameters = generate_parameters(10, 0, profile='aggressive')
    assert parameters == generate_parameters(10, 0, profile='aggressive')
    for params in parameters:
        p = EnergyNLQRProblem(**params)
        rho = max(abs(np.linalg.eigvals(p.linear.A)))
        assert 1.05 <= rho <= 1.25
        assert p.alpha >= 1.8 and p.kappa >= 2.5
        assert (1-p.beta)/p.gamma < 1
        assert max(validate(p).values()) < 1e-10
        ev = envs.Env(p, 1).evaluate(p.optimal_policy, horizon=1000)
        assert ev['J'][0] == pytest.approx(p.J_star, rel=1e-10)
