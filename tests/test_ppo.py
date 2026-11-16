"""PPO with a linear-Gaussian policy: hand-written gradients, GAE, seed independence, learning."""

import numpy as np
import pytest

from algos import ppo
from utils import baseline, seeds


def _random_case(n=2, m=1, M=3, b=50, seed=0):
    rng = np.random.default_rng(seed)
    params = {
        "K": rng.normal(size=(M, m, n)) * 0.5,
        "log_std": rng.normal(size=(M, m)) * 0.3,
        "w": rng.normal(size=(M, ppo.n_features(n))),
    }
    x = rng.normal(size=(M, b, n))
    u = rng.normal(size=(M, b, m))
    # old log-probs near the current ones, so some ratios are clipped and some are not
    logp_old = ppo.log_prob(params, x, u) + rng.normal(scale=0.3, size=(M, b))
    batch = {"x": x, "u": u, "logp_old": logp_old, "adv": rng.normal(size=(M, b)), "ret": rng.normal(size=(M, b))}
    return params, batch


@pytest.mark.parametrize("n", [1, 2])
def test_gradients_match_finite_differences(n):
    cfg = ppo.Config(c_entropy=0.01, normalize_advantages=True)
    params, batch = _random_case(n=n)
    _, grads = ppo.objective_and_grads(params, batch, cfg)
    h = 1e-6
    for key, p in params.items():
        num = np.zeros_like(p)
        for idx in np.ndindex(*p.shape[1:]):
            for s in (+1, -1):
                q = {k: v.copy() for k, v in params.items()}
                q[key][(slice(None),) + idx] += s * h
                obj, _ = ppo.objective_and_grads(q, batch, cfg)
                num[(slice(None),) + idx] += s * obj / (2 * h)
        assert grads[key] == pytest.approx(num, rel=1e-4, abs=1e-6), key


def test_gae_matches_direct_sum():
    rng = np.random.default_rng(1)
    T, gamma, lam = 7, 0.9, 0.8
    r, v = rng.normal(size=(T, 2)), rng.normal(size=(T + 1, 2))
    adv, ret = ppo.gae(r, v, gamma, lam)
    delta = r + gamma * v[1:] - v[:-1]
    for t in range(T):
        direct = sum((gamma * lam) ** l * delta[t + l] for l in range(T - t))
        assert adv[t] == pytest.approx(direct)
    assert ret == pytest.approx(adv + v[:-1])


def test_lambda_one_gives_bootstrapped_discounted_return():
    rng = np.random.default_rng(2)
    T, gamma = 5, 0.95
    r, v = rng.normal(size=(T, 1)), rng.normal(size=(T + 1, 1))
    _, ret = ppo.gae(r, v, gamma, 1.0)
    expected = sum(gamma**l * r[l] for l in range(T)) + gamma**T * v[T]
    assert ret[0] == pytest.approx(expected)


def test_seed_result_independent_of_batch():
    prob = baseline.scalar_lqr(0.99)
    cfg = ppo.Config(total_steps=3 * 200 * 8, n_checkpoints=3)
    s = seeds.draw_seeds(3, master_seed=4)
    together = ppo.train(prob, s, cfg)
    alone = ppo.train(prob, s[2:3], cfg)
    assert together["gaps"][2] == pytest.approx(alone["gaps"][0])
    assert together["gains"][2] == pytest.approx(alone["gains"][0])


def test_short_run_moves_gain_towards_optimum():
    prob = baseline.scalar_lqr(0.99)
    cfg = ppo.Config(total_steps=100_000, n_checkpoints=2, lr=3e-3)
    out = ppo.train(prob, seeds.draw_seeds(4, master_seed=0), cfg)
    assert out["gaps"][:, -1].mean() < out["gaps"][:, 0].mean()
    assert np.abs(out["gains"][:, -1, 0] - prob.K_star.item()).mean() < np.abs(out["gains"][:, 0, 0] - prob.K_star.item()).mean()
