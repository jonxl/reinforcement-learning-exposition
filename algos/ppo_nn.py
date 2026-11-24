"""Proximal Policy Optimization with neural networks (notes, Appendix "PPO").

Policy and value function are multilayer perceptrons (``utils.networks``), one independent pair per seed:

* policy  pi_theta(u | x) = N(mu_theta(x), diag(sigma^2)), with mu_theta a tanh MLP and a
  state-independent log sigma. The mean can approximate nonlinear feedback laws;
  finite network capacity and optimization can still limit its performance.
* value   V_phi(x), a second tanh MLP with its own weights.

This is the primary PPO benchmark. Both implementations use PPO-Clip, but differ in
actor and critic parameterizations, gradient implementation, clipping, and defaults.
Their comparison does not isolate actor architecture. Both collect rollouts of ``horizon`` steps in
``n_envs`` environments, generalized advantage estimation with the problem's discount gamma
(bootstrapped with V at the truncation step, since the episode end is a time limit, not a terminal
state), then ``epochs`` passes of minibatch gradient steps on

    L = E[ min(rho A, clip(rho, 1 - eps, 1 + eps) A) ] - c1 E[(V - R)^2] + c2 E[S[pi]].

Gradients come from PyTorch. The actor (with log sigma) and the critic are separate networks and
their gradients are clipped separately, per seed, so the scale of the value loss cannot crowd out the
policy gradient. Scoring uses the mean action mu_theta(x) in a long deterministic rollout
(``envs.score_policy``); the reported "gain" is the effective linear gain fitted along that rollout,
as for the Monte Carlo table.

Seeds: all random numbers of seed omega (initial weights, action noise, minibatch shuffles) come from
``rng_for(omega)``, so a seed's run does not depend on the batch. A seed whose training rollout
diverges stops learning and is scored gap 1 from then on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from algos.advantages import gae
from utils import envs, networks
from utils.seeds import rng_for
from utils.rewards import RewardWindow

NAME = "ppo-nn"
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class Config:
    device: str = "cpu"
    """Defaults follow Schulman et al. (2017), Section 6.1 and Table 3 (the MuJoCo benchmark).

    [paper] = stated there; [ours] = not stated in the paper, our choice. The discount gamma = 0.99
    of Table 3 is the problem's, not set here.
    """

    hidden_size: int = 64  # [paper] "two hidden layers of 64 units, and tanh nonlinearities"
    hidden_layers: int = 2  # [paper]
    actor_out_gain: float = 0.01  # [ours] small output layer: the untrained policy acts close to zero
    horizon: int = 2048  # [paper] T, steps each actor collects per update; gamma^2048 ~ 1e-9
    n_envs: int = 1  # [ours] number of actors N; Table 3 does not state it, one actor gives T samples
    epochs: int = 10  # [paper] K
    minibatch_size: int = 64  # [paper] M; 2048 / 64 = 32 minibatches per epoch
    lr: float = 3e-4  # [paper] Adam stepsize, constant (no annealing is mentioned)
    gae_lambda: float = 0.95  # [paper]
    clip: float = 0.2  # [paper] epsilon, the best value of their search (Table 1)
    c_value: float = 0.5  # [ours] c1; "irrelevant" without shared parameters, scales the critic loss
    c_entropy: float = 0.0  # [paper] "we don't use an entropy bonus"
    max_grad_norm: float = 0.5  # [ours] per seed, actor and critic separately
    normalize_advantages: bool = True  # [ours]
    init_std: float = 0.5  # [ours] sigma at the start; the paper says only "variable standard deviations"
    total_steps: int = 1_000_000  # [paper] "one million timesteps of training"
    n_checkpoints: int = 20  # [ours] evaluation points of the learning curve
    eval_horizon: int = envs.EVAL_HORIZON  # [ours] length of the deterministic scoring rollout


def losses(actor: networks.Params, log_std: torch.Tensor, critic: networks.Params, batch: dict, cfg: Config):
    """Per-seed actor loss and critic loss (both to be minimized) on one minibatch.

    batch: x (M, b, n), u (M, b, m), logp_old (M, b), adv (M, b), ret (M, b), as tensors.
    """
    adv = batch["adv"]
    if cfg.normalize_advantages:
        adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True, correction=0) + 1e-8)
    z = (batch["u"] - networks.mlp(actor, batch["x"])) / torch.exp(log_std)[:, None, :]
    logp = (-0.5 * z**2 - log_std[:, None, :] - 0.5 * LOG_2PI).sum(dim=-1)
    ratio = torch.exp(logp - batch["logp_old"])
    clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
    l_clip = torch.minimum(ratio * adv, clipped * adv).mean(dim=1)
    entropy = log_std.sum(dim=1) + 0.5 * log_std.shape[1] * (1.0 + LOG_2PI)
    value = networks.mlp(critic, batch["x"])[..., 0]
    l_value = ((value - batch["ret"]) ** 2).mean(dim=1)
    return -(l_clip + cfg.c_entropy * entropy), cfg.c_value * l_value


def train(problem: envs.Problem, seeds: np.ndarray, cfg: Config) -> dict:
    """Train one actor-critic pair per seed; return checkpoints and per-seed metrics (arrays (M, n_ckpt, ...))."""
    M, n, m = len(seeds), problem.n_state, problem.n_control
    T, E, gamma = cfg.horizon, cfg.n_envs, problem.gamma
    N = T * E
    if N % cfg.minibatch_size:
        raise ValueError("horizon * n_envs must be a multiple of minibatch_size")
    b = cfg.minibatch_size
    n_minibatches = N // b
    rngs = [rng_for(s) for s in seeds]
    env = envs.Env(problem, M * E)

    hidden = (cfg.hidden_size,) * cfg.hidden_layers
    actor = networks.init_mlp(rngs, (n, *hidden, m), out_gain=cfg.actor_out_gain, device=cfg.device)
    critic = networks.init_mlp(rngs, (n, *hidden, 1), out_gain=1.0, device=cfg.device)
    log_std = torch.full((M, m), float(np.log(cfg.init_std)), dtype=networks.DTYPE, device=cfg.device, requires_grad=True)
    actor_params, critic_params = [*actor, log_std], critic
    actor_opt, critic_opt = networks.Adam(actor_params, cfg.lr), networks.Adam(critic_params, cfg.lr)
    alive = np.ones(M, dtype=bool)
    mean_action = networks.as_policy(actor)

    n_iters = max(1, cfg.total_steps // N)
    checkpoint_iters = np.unique(np.linspace(0, n_iters, cfg.n_checkpoints + 1).astype(int))
    rewards = RewardWindow(M)
    out = {k: [] for k in ("training_stage_reward", "training_reward_count", "steps", "gaps", "gains", "sigma", "unstable", "final_abs_state", "diverged")}

    def score(it: int) -> None:
        reward, count = rewards.flush()
        out["training_stage_reward"].append(reward)
        out["training_reward_count"].append(count)
        s = envs.score_policy(problem, mean_action, M, cfg.eval_horizon)
        out["steps"].append(it * N)
        out["gaps"].append(np.where(alive, s["gaps"], 1.0))
        out["gains"].append(s["gains"])
        out["sigma"].append(np.exp(log_std.detach().cpu().numpy()))
        out["unstable"].append(s["unstable"] | ~alive)
        out["final_abs_state"].append(s["final_abs_state"])
        out["diverged"].append(~alive)

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=networks.DTYPE, device=cfg.device)

    score(0)
    for it in range(1, n_iters + 1):
        # 1. rollout with the current policy (theta_old); each seed's noise from its own stream
        noise = np.stack([g.standard_normal((T, E, m)) for g in rngs], axis=1)  # (T, M, E, m)
        X = np.empty((T + 1, M, E, n))
        U = np.empty((T, M, E, m))
        Rw = np.empty((T, M, E))
        sigma = np.exp(log_std.detach().cpu().numpy())[:, None, :]
        x = env.reset().reshape(M, E, n)
        with torch.no_grad(), np.errstate(over="ignore", invalid="ignore"):
            for t in range(T):
                X[t] = x
                U[t] = networks.mlp(actor, tensor(x)).cpu().numpy() + sigma * noise[t]
                obs, r = env.step(U[t].reshape(M * E, m))
                x, Rw[t] = obs.reshape(M, E, n), r.reshape(M, E)
        X[T] = x
        logp_old = (-0.5 * noise**2 - np.log(sigma)[None] - 0.5 * LOG_2PI).sum(axis=-1)

        bad = ~np.isfinite(X).all(axis=(0, 2, 3)) | (np.abs(X).max(axis=(0, 2, 3)) > envs.BLOWUP)
        alive &= ~bad
        rewards.add(Rw, failed=~alive)
        if not alive.any():
            if it in checkpoint_iters:
                score(it)
            continue
        X[:, ~alive], U[:, ~alive], Rw[:, ~alive], logp_old[:, ~alive] = 0.0, 0.0, 0.0, 0.0

        # 2. advantages and return targets
        with torch.no_grad():
            V = networks.mlp(critic, tensor(np.moveaxis(X, 1, 0).reshape(M, (T + 1) * E, n)))[..., 0].cpu().numpy()
        V = np.moveaxis(V.reshape(M, T + 1, E), 0, 1)  # (T + 1, M, E)
        adv, ret = gae(Rw, V, gamma, cfg.gae_lambda)

        # flatten time and environments: (M, N, ...)
        flat = lambda a: tensor(np.moveaxis(a, 1, 0).reshape(M, N, *a.shape[3:]))  # noqa: E731
        data = {"x": flat(X[:-1]), "u": flat(U), "logp_old": flat(logp_old), "adv": flat(adv), "ret": flat(ret)}
        alive_t = torch.as_tensor(alive, device=cfg.device)

        # 3. epochs of minibatch gradient steps on the combined objective
        for _ in range(cfg.epochs):
            perm = np.stack([g.permutation(N) for g in rngs])  # (M, N)
            for k in range(n_minibatches):
                idx = torch.as_tensor(perm[:, k * b:(k + 1) * b], device=cfg.device)
                batch = {key: torch.gather(a, 1, idx.reshape(idx.shape + (1,) * (a.dim() - 2)).expand(-1, -1, *a.shape[2:]))
                         for key, a in data.items()}
                actor_loss, critic_loss = losses(actor, log_std, critic, batch, cfg)
                g_actor = torch.autograd.grad(actor_loss.sum(), actor_params)
                g_critic = torch.autograd.grad(critic_loss.sum(), critic_params)
                actor_opt.step(networks.clip_by_seed(g_actor, cfg.max_grad_norm), alive_t & networks.finite_by_seed(g_actor))
                critic_opt.step(networks.clip_by_seed(g_critic, cfg.max_grad_norm), alive_t & networks.finite_by_seed(g_critic))

        if it in checkpoint_iters:
            score(it)

    result = {k: np.stack(v, axis=1) if k != "steps" else np.asarray(v) for k, v in out.items()}
    result["reward_metric"] = {
        "name": "training_stage_reward", "definition": "mean observed reward per transition since previous checkpoint",
        "includes_exploration": True, "step_zero": "NaN: no training transitions yet",
        "nonfinite_window": "-inf; failed seeds are not dropped", "higher_is_better": True,
    }
    result["config"] = asdict(cfg)
    result["iterations"] = n_iters
    return result
