"""Neural PPO and the batched networks: independence of seeds, frozen seeds, learning."""

import numpy as np
import pytest
import torch

from algos import ppo
from utils import baseline, networks, seeds
from utils.seeds import rng_for


def test_networks_of_different_seeds_are_independent_of_the_batch():
    s = seeds.draw_seeds(3, master_seed=2)
    together = networks.init_mlp([rng_for(x) for x in s], (2, 8, 8, 1))
    alone = networks.init_mlp([rng_for(s[1])], (2, 8, 8, 1))
    for a, b in zip(together, alone):
        assert a[1].detach().numpy() == pytest.approx(b[0].detach().numpy())
    x = torch.randn(1, 5, 2, dtype=networks.DTYPE)
    out_together = networks.mlp(together, x.expand(3, -1, -1))[1]
    assert out_together.detach().numpy() == pytest.approx(networks.mlp(alone, x)[0].detach().numpy())


def test_hidden_layers_are_orthogonal_and_the_actor_starts_near_zero():
    params = networks.init_mlp([rng_for(1)], (2, 16, 16, 1), out_gain=0.01)
    W2 = params[2][0].detach().numpy()
    assert W2.T @ W2 == pytest.approx(2.0 * np.eye(16), abs=1e-9)  # gain sqrt(2)
    policy = networks.as_policy(params)
    assert np.abs(policy(np.array([[1.0, -1.0]]))).max() < 0.05


def test_masked_adam_freezes_a_seed_and_clipping_is_per_seed():
    params = [torch.ones((2, 3), dtype=networks.DTYPE, requires_grad=True)]
    opt = networks.Adam(params, lr=0.1)
    grads = [torch.tensor([[3.0, 4.0, 0.0], [30.0, 40.0, 0.0]], dtype=networks.DTYPE)]
    clipped = networks.clip_by_seed(grads, 1.0)
    assert clipped[0].norm(dim=1).numpy() == pytest.approx([1.0, 1.0])
    opt.step(clipped, torch.tensor([True, False]))
    assert params[0][1].detach().numpy() == pytest.approx(np.ones(3))
    assert np.all(params[0][0].detach().numpy()[:2] < 1.0)
    nan = [torch.tensor([[1.0, 1.0, 1.0], [float("nan"), 1.0, 1.0]], dtype=networks.DTYPE)]
    assert networks.finite_by_seed(nan).tolist() == [True, False]


def test_seed_result_independent_of_batch():
    prob = baseline.nlqr(0.99)
    cfg = ppo.Config(total_steps=3 * 250 * 8, n_checkpoints=3, eval_horizon=300, hidden_size=16)
    s = seeds.draw_seeds(3, master_seed=4)
    together = ppo.train(prob, s, cfg)
    alone = ppo.train(prob, s[2:3], cfg)
    assert together["gaps"][2] == pytest.approx(alone["gaps"][0], rel=1e-6)
    assert together["gains"][2] == pytest.approx(alone["gains"][0], rel=1e-6)


def test_short_run_lowers_the_gap_on_the_nonlinear_problem():
    prob = baseline.nlqr(0.99)
    cfg = ppo.Config(total_steps=60_000, n_checkpoints=2, eval_horizon=1000)
    out = ppo.train(prob, seeds.draw_seeds(3, master_seed=0), cfg)
    assert np.all(out["gaps"][:, -1] < out["gaps"][:, 0])
    assert not out["diverged"].any()
