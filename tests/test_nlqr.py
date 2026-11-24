"""The nonlinear benchmark, the problem registry and the scoring shared by the algorithms."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import rollouts
from algos import monte_carlo as mc
from algos import ppo_linear as ppo
from utils import baseline, control, envs, seeds

NLQR = [lambda **kw: baseline.nlqr(**kw), lambda **kw: baseline.scalar_nlqr(**kw)]


@pytest.mark.parametrize("gamma", [0.9, 0.99, 1.0])
def test_constructed_system_has_the_requested_diagonal_riccati_matrix(gamma):
    A, B = baseline.diagonal_riccati_system(2.0, 1.5, 0.5, b=1.0, gamma=gamma)
    P, _ = control.solve_discounted_riccati(A, B, np.eye(2), 1.0, gamma)
    assert P == pytest.approx(np.diag([2.0, 1.5]), abs=1e-9)


def test_default_instance_is_open_loop_stable_with_J_star_3_5():
    prob = baseline.nlqr(0.99)
    assert control.spectral_radius(prob.linear.A) < 1.0
    assert prob.J_star == pytest.approx(3.5)


@pytest.mark.parametrize("make", NLQR)
@pytest.mark.parametrize("k", [0.0, 1.0, 2.0, 5.0])
def test_value_function_satisfies_the_bellman_equation(make, k):
    """V(x) = min_u [c(x, u) + gamma V(f(x, u))] with V = x'Px, and the minimum is at u*(x)."""
    prob = make(gamma=0.99, k=k)
    rng = np.random.default_rng(0)

    def q(x, u):
        x, u = x[None, :], np.array([[u]])
        nxt = prob.dynamics(x, u)[0]
        return prob.stage_cost(x, u)[0] + prob.gamma * nxt @ prob.P @ nxt

    for x in rng.uniform(-3.0, 3.0, size=(25, prob.n_state)):
        best = minimize_scalar(lambda u: q(x, u), bounds=(-50.0, 50.0), method="bounded", options={"xatol": 1e-12})
        assert best.fun == pytest.approx(x @ prob.P @ x, rel=1e-9, abs=1e-12)
        assert q(x, prob.optimal_policy(x[None, :])[0, 0]) == pytest.approx(x @ prob.P @ x, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("make", NLQR)
def test_optimal_policy_attains_J_star_and_linear_gain_does_not(make):
    prob = make(gamma=0.99, k=2.0)
    ev = envs.Env(prob, 1).evaluate(prob.optimal_policy)
    assert ev["J"][0] == pytest.approx(prob.J_star, rel=1e-10)
    gaps, unstable = envs.score_gains(prob, prob.K_star[None])
    assert 1e-3 < gaps[0] < 0.1 and not unstable[0]  # the LQR gain of the linearization is suboptimal


@pytest.mark.parametrize("make", NLQR)
def test_k_zero_is_the_linear_problem(make):
    prob = make(gamma=0.99, k=0.0)
    rng = np.random.default_rng(1)
    x, u = rng.normal(size=(7, prob.n_state)), rng.normal(size=(7, 1))
    assert prob.dynamics(x, u) == pytest.approx(prob.linear.dynamics(x, u))
    assert prob.optimal_policy(x) == pytest.approx(prob.linear.optimal_policy(x))
    gaps, _ = envs.score_gains(prob, prob.K_star[None])
    assert gaps[0] == pytest.approx(0.0, abs=1e-9)


def test_construction_rejects_problems_it_does_not_cover():
    with pytest.raises(ValueError):  # LQI: the control enters the first state, not the last
        baseline.NLQRProblem(baseline.scalar_lqi(0.99).augmented)
    coupled = baseline.LQRProblem(A=[[0.5, 0.3], [0.2, 0.4]], B=[[0.0], [1.0]], Q=np.eye(2), R=1.0, gamma=0.99, x0=[1, 1])
    with pytest.raises(ValueError):  # a generic system: P is not diagonal
        baseline.NLQRProblem(coupled)


# -- registry and scoring ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(envs.PROBLEMS))
def test_every_registered_problem_builds_and_its_optimal_policy_attains_J_star(name):
    prob = envs.make_problem(name, gamma=0.99)
    ev = envs.Env(prob, 2).evaluate(prob.optimal_policy)
    assert ev["J"] == pytest.approx([prob.J_star] * 2, rel=1e-9)


def test_unknown_problem_is_an_error():
    with pytest.raises(KeyError):
        envs.make_problem("pendulum")


def test_rollout_scoring_agrees_with_exact_scoring_on_a_linear_problem():
    prob = envs.make_problem("lqi", 0.99)
    K = prob.K_star[None] + np.array([[[0.0, 0.0]], [[0.1, -0.05]], [[-0.2, 0.1]]])
    exact, _ = envs.score_gains(prob, K)
    ev = envs.Env(prob, 3).evaluate(lambda x: -np.einsum("en,ejn->ej", x, K))
    assert control.gap(ev["J"], prob.J_star) == pytest.approx(exact, abs=1e-9)


@pytest.mark.parametrize("algo, cfg", [
    ("ppo", {"total_steps": 4_000, "n_checkpoints": 2, "eval_horizon": 500}),
    ("ppo_linear", {"total_steps": 4_000, "n_checkpoints": 2, "eval_horizon": 500}),
    ("monte_carlo", {"total_steps": 2_000, "horizon": 100, "n_checkpoints": 2, "eval_horizon": 500, "n_bins": 11}),
])
def test_both_algorithms_run_on_the_nonlinear_problem_and_the_run_is_rebuildable(tmp_path, algo, cfg):
    out = rollouts.run(algo, "nlqr", n_seeds=2, master_seed=3, workers=1, overrides=cfg,
                       out_dir=tmp_path / "run", problem_params={"k": 3.0})
    arrays, meta = rollouts.load(out)
    curves = rollouts.to_curves(arrays, meta)
    assert curves.gaps.shape == (2, len(arrays["steps"]))
    assert meta["problem_params"] == {"k": 3.0} and meta["curvature"] is None
    rebuilt = rollouts.problem_of(meta)
    assert isinstance(rebuilt, baseline.NLQRProblem) and rebuilt.k == 3.0
    assert rebuilt.J_star == pytest.approx(meta["J_star"])


def test_ppo_seed_result_independent_of_batch_on_the_nonlinear_problem():
    prob = baseline.nlqr(0.99)
    cfg = ppo.Config(total_steps=2 * 250 * 8, n_checkpoints=2, eval_horizon=500)
    s = seeds.draw_seeds(3, master_seed=4)
    together = ppo.train(prob, s, cfg)
    alone = ppo.train(prob, s[2:3], cfg)
    assert together["gaps"][2] == pytest.approx(alone["gaps"][0])


def test_monte_carlo_table_handles_two_states():
    prob = baseline.nlqr(0.99)
    cfg = mc.Config(total_steps=2_000, horizon=100, n_checkpoints=1, eval_horizon=300, n_bins=11)
    out = mc.train(prob, seeds.draw_seeds(2, master_seed=1), cfg)
    assert out["n_cells"] == 11**2 and np.all((out["gaps"] >= -1e-9) & (out["gaps"] <= 1.0))
