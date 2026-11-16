"""Environment, Monte Carlo control and the run driver."""

import numpy as np
import pytest

import rollouts
from algos import monte_carlo as mc
from utils import baseline, envs, seeds


def test_env_step_follows_dynamics_and_reward():
    prob = baseline.scalar_lqr(0.99)
    env = envs.LinearQuadraticEnv(prob, n_envs=3)
    x = env.reset()
    u = np.array([[0.5], [-0.2], [0.0]])
    x_next, r = env.step(u)
    assert x_next[:, 0] == pytest.approx(x[:, 0] + u[:, 0])
    assert r == pytest.approx(-(x[:, 0] ** 2 + u[:, 0] ** 2))


@pytest.mark.parametrize("make", [lambda: baseline.scalar_lqr(0.99), lambda: baseline.scalar_lqi(0.99).augmented])
def test_evaluating_optimal_gain_gives_J_star(make):
    prob = make()
    env = envs.LinearQuadraticEnv(prob, n_envs=2)
    K = prob.K_star
    ev = env.evaluate(lambda x: -x @ K.T)
    assert ev["J"] == pytest.approx([prob.J_star] * 2, rel=1e-9)
    assert not ev["diverged"].any()


def test_evaluate_flags_divergence():
    prob = baseline.scalar_lqr(0.99)
    env = envs.LinearQuadraticEnv(prob, n_envs=1)
    ev = env.evaluate(lambda x: 2.0 * x)  # x_{t+1} = 3 x_t
    assert ev["diverged"].all() and np.isinf(ev["J"]).all()


def test_effective_gain_recovers_linear_policy():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(100, 4, 2))
    K = np.array([[1.2, -0.4]])
    controls = -states @ K.T
    assert envs.effective_gain(states, controls)[:, 0, :] == pytest.approx(np.tile(K, (4, 1)))


def test_monte_carlo_seed_result_independent_of_batch():
    """g_X(omega, N) is a function of omega alone: same result alone or alongside other seeds."""
    prob = baseline.scalar_lqr(0.99)
    cfg = mc.Config(total_steps=5_000, horizon=100, n_checkpoints=2, eval_horizon=1000)
    s = seeds.draw_seeds(3, master_seed=11)
    together = mc.train(prob, s, cfg)
    alone = mc.train(prob, s[1:2], cfg)
    assert together["gaps"][1] == pytest.approx(alone["gaps"][0])
    assert together["gains"][1] == pytest.approx(alone["gains"][0])


def test_monte_carlo_requires_odd_bins():
    with pytest.raises(ValueError):
        mc.state_edges(mc.Config(n_bins=40))
    edges = mc.state_edges(mc.Config(n_bins=41))
    centre = len(edges) // 2
    assert edges[centre - 1] < 0 < edges[centre]  # one bin straddles zero symmetrically
    assert edges[centre - 1] == pytest.approx(-edges[centre])


def test_driver_round_trip(tmp_path):
    out = rollouts.run(
        "monte_carlo", "lqr", n_seeds=3, master_seed=5, workers=2,
        overrides={"total_steps": 2_000, "horizon": 100, "n_checkpoints": 2, "eval_horizon": 500},
        out_dir=tmp_path / "run",
    )
    arrays, meta = rollouts.load(out)
    curves = rollouts.to_curves(arrays, meta)
    assert curves.gaps.shape == (3, len(arrays["steps"]))
    assert meta["J_star"] == pytest.approx(baseline.scalar_p(0.99))
    assert meta["curvature"][0][0] == pytest.approx(baseline.scalar_curvature(0.99), rel=1e-3)
