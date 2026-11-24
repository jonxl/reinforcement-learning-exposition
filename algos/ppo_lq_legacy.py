"""Historical linear PPO used in the colleague's published bounded comparison.

Preserved separately from the primary neural implementation in ``algos.ppo``.
With action_bound enabled, deployment is clipped and the quadratic critic is an
approximation. The statements about exact linear/quadratic structure below apply
to the original unbounded LQR setting.

Policy and value function, chosen so that the optimum is representable:

* policy  pi_theta(u | x) = N(-K x, diag(sigma^2)),  theta = (K, log sigma).  The optimal LQR policy
  u = -K* x is the mean of this family, so the gap can go to zero and the learned K *is* the gain
  (no effective-gain fit, unlike the table of Monte Carlo control).
* value   V_phi(x) = phi . [1, x, upper triangle of x x'].  The value of a linear-Gaussian policy on
  a linear-quadratic system is exactly of this form.

Everything else follows the appendix: rollouts of ``horizon`` steps in ``n_envs`` environments,
temporal-difference errors and generalized advantage estimation with the problem's discount gamma
(bootstrapped with V at the truncation step, since the episode end is a time limit, not a terminal
state), then ``epochs`` passes of minibatch gradient ascent on

    L = E[ min(rho A, clip(rho, 1 - eps, 1 + eps) A) ] - c1 E[(V - R)^2] + c2 E[S[pi]].

The gradients are written out in closed form (``objective_and_grads``) rather than taken by an
automatic-differentiation library; for this policy they are a few lines. The step is Adam, as in
the reference implementations, with the gradient norm clipped per seed. Scoring uses the mean
action -K x, whose discounted cost is exact (Lyapunov equation, ``LQRProblem.gap``).

Seeds: all random numbers of seed omega (action noise, minibatch shuffles) come from
``rng_for(omega)``, so a seed's run does not depend on the batch. A seed whose training rollout
diverges stops learning and is scored gap 1 from then on (a real implementation would produce NaNs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from utils import envs
from utils.baseline import LQRProblem
from utils.seeds import rng_for

NAME = "ppo"
LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class Config:
    horizon: int = 250  # T: steps per rollout; gamma^250 ~ 0.08, the rest is bootstrapped by V
    n_envs: int = 8  # parallel environments per seed; one iteration = horizon * n_envs = 2000 steps,
    #                  so the default checkpoints (every 50k steps) coincide with Monte Carlo's
    epochs: int = 10  # E
    n_minibatches: int = 4
    lr: float = 3e-4
    gae_lambda: float = 0.95
    clip: float = 0.2  # epsilon_clip
    c_value: float = 0.5  # c1
    c_entropy: float = 0.0  # c2
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    init_gain: float = 0.0  # every entry of K at the start
    init_std: float = 0.5  # sigma at the start
    total_steps: int = 1_000_000  # environment steps per seed
    n_checkpoints: int = 20


# -- policy and value function ------------------------------------------------------------------


def features(x: np.ndarray) -> np.ndarray:
    """Quadratic features [1, x, x_i x_j (i <= j)] of states x (..., n) -> (..., 1 + n + n(n+1)/2)."""
    n = x.shape[-1]
    i, j = np.triu_indices(n)
    quad = x[..., i] * x[..., j]
    return np.concatenate([np.ones(x.shape[:-1] + (1,)), x, quad], axis=-1)


def n_features(n_state: int) -> int:
    return 1 + n_state + n_state * (n_state + 1) // 2


def log_prob(params: dict, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """log pi(u | x) per seed and sample: x (M, b, n), u (M, b, m) -> (M, b)."""
    mu = -np.einsum("mbn,mjn->mbj", x, params["K"])
    z = (u - mu) / np.exp(params["log_std"])[:, None, :]
    return (-0.5 * z**2 - params["log_std"][:, None, :] - 0.5 * LOG_2PI).sum(axis=-1)


def objective_and_grads(params: dict, batch: dict, cfg: Config) -> tuple[np.ndarray, dict]:
    """Combined objective L (per seed) on one minibatch and its gradient w.r.t. (K, log_std, w).

    batch: x (M, b, n), u (M, b, m), logp_old (M, b), adv (M, b), ret (M, b).
    """
    x, u, adv, ret = batch["x"], batch["u"], batch["adv"], batch["ret"]
    b = x.shape[1]
    if cfg.normalize_advantages:
        adv = (adv - adv.mean(axis=1, keepdims=True)) / (adv.std(axis=1, keepdims=True) + 1e-8)

    K, log_std, w = params["K"], params["log_std"], params["w"]
    sigma = np.exp(log_std)[:, None, :]  # (M, 1, m)
    mu = -np.einsum("mbn,mjn->mbj", x, K)
    z = (u - mu) / sigma  # standardized action noise
    logp = (-0.5 * z**2 - log_std[:, None, :] - 0.5 * LOG_2PI).sum(axis=-1)
    ratio = np.exp(logp - batch["logp_old"])

    # clipped objective: the gradient flows through rho only where the unclipped term is the min
    clipped = np.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
    l_clip = np.minimum(ratio * adv, clipped * adv).mean(axis=1)
    active = ~(((adv > 0) & (ratio > 1.0 + cfg.clip)) | ((adv < 0) & (ratio < 1.0 - cfg.clip)))
    coef = np.where(active, ratio * adv, 0.0) / b  # d L_clip / d log pi, per sample

    # d log pi / d K_jk = (z_j / sigma_j) (-x_k);  d log pi / d log sigma_j = z_j^2 - 1
    g_K = np.einsum("mb,mbj,mbk->mjk", coef, z / sigma, -x)
    g_log_std = np.einsum("mb,mbj->mj", coef, z**2 - 1.0)

    # entropy of a Gaussian: sum_j log sigma_j + const, the same at every state
    entropy = log_std.sum(axis=1) + 0.5 * log_std.shape[1] * (1.0 + LOG_2PI)
    g_log_std = g_log_std + cfg.c_entropy

    # value loss (V - R)^2 with V = phi . w
    phi = features(x)
    err = np.einsum("mbp,mp->mb", phi, w) - ret
    l_value = (err**2).mean(axis=1)
    g_w = -cfg.c_value * 2.0 * np.einsum("mb,mbp->mp", err, phi) / b

    objective = l_clip - cfg.c_value * l_value + cfg.c_entropy * entropy
    return objective, {"K": g_K, "log_std": g_log_std, "w": g_w}


class Adam:
    """Adam for gradient *ascent*, one independent optimizer state per seed (leading axis)."""

    def __init__(self, params: dict, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict, mask: np.ndarray) -> None:
        self.t += 1
        c1, c2 = 1.0 - self.b1**self.t, 1.0 - self.b2**self.t
        for k, g in grads.items():
            keep = mask.reshape((-1,) + (1,) * (g.ndim - 1))
            self.m[k] = np.where(keep, self.b1 * self.m[k] + (1 - self.b1) * g, self.m[k])
            self.v[k] = np.where(keep, self.b2 * self.v[k] + (1 - self.b2) * g**2, self.v[k])
            step = self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            params[k] = params[k] + np.where(keep, step, 0.0)


def _clip_grad_norm(grads: dict, max_norm: float) -> dict:
    sq = sum((g.reshape(g.shape[0], -1) ** 2).sum(axis=1) for g in grads.values())
    scale = np.minimum(1.0, max_norm / (np.sqrt(sq) + 1e-12))
    return {k: g * scale.reshape((-1,) + (1,) * (g.ndim - 1)) for k, g in grads.items()}


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and return targets. rewards (T, ...), values (T + 1, ...) incl. the bootstrap V(x_T)."""
    deltas = rewards + gamma * values[1:] - values[:-1]
    adv = np.empty_like(deltas)
    acc = np.zeros(deltas.shape[1:])
    for t in range(deltas.shape[0] - 1, -1, -1):
        acc = deltas[t] + gamma * lam * acc
        adv[t] = acc
    return adv, adv + values[:-1]


# -- training -----------------------------------------------------------------------------------


def train(problem: LQRProblem, seeds: np.ndarray, cfg: Config, *, action_bound=None, evaluator=None) -> dict:
    """Train one policy per seed; return checkpoints and per-seed metrics (arrays (M, n_ckpt, ...))."""
    M, n, m = len(seeds), problem.n_state, problem.n_control
    T, E, gamma = cfg.horizon, cfg.n_envs, problem.gamma
    N = T * E
    if N % cfg.n_minibatches:
        raise ValueError("horizon * n_envs must be divisible by n_minibatches")
    b = N // cfg.n_minibatches
    rngs = [rng_for(s) for s in seeds]
    env = envs.Env(problem, M * E)

    params = {
        "K": np.full((M, m, n), cfg.init_gain),
        "log_std": np.full((M, m), np.log(cfg.init_std)),
        "w": np.zeros((M, n_features(n))),
    }
    opt = Adam(params, cfg.lr)
    alive = np.ones(M, dtype=bool)

    n_iters = max(1, cfg.total_steps // N)
    checkpoint_iters = np.unique(np.linspace(0, n_iters, cfg.n_checkpoints + 1).astype(int))
    out = {k: [] for k in ("steps", "gaps", "gains", "sigma", "unstable", "diverged")}

    def score(it: int) -> None:
        gaps = np.ones(M)
        unstable = np.ones(M, dtype=bool)
        for j in np.flatnonzero(alive):
            gaps[j] = problem.gap(params["K"][j])
            unstable[j] = not problem.is_stable(params["K"][j])
        if evaluator is not None:
            gaps = evaluator(it * N, params, alive)
        elif action_bound is not None:
            raise ValueError("Bounded policies need an explicit deployed-policy evaluator")
        out["steps"].append(it * N)
        out["gaps"].append(gaps)
        out["gains"].append(params["K"].reshape(M, -1).copy())
        out["sigma"].append(np.exp(params["log_std"]).copy())
        out["unstable"].append(unstable)
        out["diverged"].append(~alive)

    score(0)
    for it in range(1, n_iters + 1):
        # 1. rollout with the current policy (theta_old); each seed's noise from its own stream
        noise = np.stack([g.standard_normal((T, E, m)) for g in rngs], axis=1)  # (T, M, E, m)
        X = np.empty((T + 1, M, E, n))
        U = np.empty((T, M, E, m))
        Rw = np.empty((T, M, E))
        sigma = np.exp(params["log_std"])[:, None, :]
        x = env.reset().reshape(M, E, n)
        for t in range(T):
            X[t] = x
            U[t] = -np.einsum("men,mjn->mej", x, params["K"]) + sigma * noise[t]
            # Keep the latent action and its Gaussian likelihood for PPO. Only
            # the actuator clips it; learning uses rewards from applied actions.
            applied = U[t] if action_bound is None else np.clip(U[t], -action_bound, action_bound)
            obs, r = env.step(applied.reshape(M * E, m))
            x, Rw[t] = obs.reshape(M, E, n), r.reshape(M, E)
        X[T] = x
        logp_old = (-0.5 * noise**2 - params["log_std"][None, :, None, :] - 0.5 * LOG_2PI).sum(axis=-1)

        bad = ~np.isfinite(X).all(axis=(0, 2, 3)) | (np.abs(X).max(axis=(0, 2, 3)) > envs.BLOWUP)
        alive &= ~bad
        if not alive.any():
            if it in checkpoint_iters:
                score(it)
            continue
        X[:, ~alive], U[:, ~alive], Rw[:, ~alive], logp_old[:, ~alive] = 0.0, 0.0, 0.0, 0.0

        # 2. advantages and return targets
        V = np.einsum("tmep,mp->tme", features(X), params["w"])
        adv, ret = gae(Rw, V, gamma, cfg.gae_lambda)

        # flatten time and environments: (M, N, ...)
        flat = lambda a: np.moveaxis(a, 1, 0).reshape(M, N, *a.shape[3:])  # noqa: E731
        data = {"x": flat(X[:-1]), "u": flat(U), "logp_old": flat(logp_old), "adv": flat(adv), "ret": flat(ret)}

        # 3. epochs of minibatch gradient ascent on the combined objective
        for _ in range(cfg.epochs):
            perm = np.stack([g.permutation(N) for g in rngs])  # (M, N)
            for k in range(cfg.n_minibatches):
                idx = perm[:, k * b:(k + 1) * b]
                batch = {key: np.take_along_axis(a, idx.reshape(idx.shape + (1,) * (a.ndim - 2)), axis=1)
                         for key, a in data.items()}
                _, grads = objective_and_grads(params, batch, cfg)
                opt.step(params, _clip_grad_norm(grads, cfg.max_grad_norm), alive)

        if it in checkpoint_iters:
            score(it)

    result = {k: np.stack(v, axis=1) if k != "steps" else np.asarray(v) for k, v in out.items()}
    result["config"] = asdict(cfg)
    result["iterations"] = n_iters
    return result
