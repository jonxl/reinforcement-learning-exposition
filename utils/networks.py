"""Neural networks for the algorithms: one independent network per seed, trained together.

The experiments run M seeds in one batch, and a seed's result must not depend on which other seeds
share the batch (``utils.seeds``). So a "network" here is M networks stacked on a leading seed axis:
every weight is a tensor (M, n_in, n_out), every input (M, batch, n_in). Seeds never mix: the loss is
a sum of per-seed losses, gradients are clipped per seed, and Adam acts element by element.

PyTorch is used for differentiation only. All random numbers (initial weights, exploration noise,
minibatch shuffles) come from each seed's own NumPy generator, never from PyTorch's global one.

Reused by any algorithm that needs a function approximator: PPO's actor and critic today, the
actors and Q-functions of SAC or TD3 later.
"""

from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(1)  # runs are parallel over processes (``rollouts``); threads would oversubscribe
DTYPE = torch.float64  # the same precision as the NumPy environments

Params = list[torch.Tensor]  # [W_1, b_1, ..., W_L, b_L], W_l (M, n_in, n_out), b_l (M, n_out)


def _orthogonal(rng: np.random.Generator, n_in: int, n_out: int, gain: float) -> np.ndarray:
    """A (n_in, n_out) matrix with orthonormal rows or columns, scaled by ``gain``."""
    a = rng.standard_normal((max(n_in, n_out), min(n_in, n_out)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    return gain * (q if n_in >= n_out else q.T)


def init_mlp(rngs: list[np.random.Generator], sizes: tuple[int, ...], out_gain: float = 1.0, device: str = "cpu") -> Params:
    """Weights of M tanh MLPs with layer widths ``sizes``, seed j drawn from ``rngs[j]`` alone.

    Orthogonal initialization with gain sqrt(2) on hidden layers and ``out_gain`` on the output layer
    (a small output gain makes an untrained actor act close to zero); biases start at zero.
    """
    params: Params = []
    last = len(sizes) - 2
    for layer, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        gain = out_gain if layer == last else np.sqrt(2.0)
        W = np.stack([_orthogonal(rng, n_in, n_out, gain) for rng in rngs])
        params.append(torch.tensor(W, dtype=DTYPE, device=device, requires_grad=True))
        params.append(torch.zeros((len(rngs), n_out), dtype=DTYPE, device=device, requires_grad=True))
    return params


def mlp(params: Params, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the M networks: x (M, batch, n_in) -> (M, batch, n_out), tanh between layers."""
    h = x
    for i in range(0, len(params), 2):
        h = torch.baddbmm(params[i + 1][:, None, :], h, params[i])
        if i + 2 < len(params):
            h = torch.tanh(h)
    return h


def as_policy(params: Params):
    """The deterministic policy of the M networks for scoring: observations (M, n) -> actions (M, m)."""

    def policy(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return mlp(params, torch.as_tensor(obs, dtype=DTYPE, device=params[0].device)[:, None, :])[:, 0, :].cpu().numpy()

    return policy


def clip_by_seed(grads: list[torch.Tensor], max_norm: float) -> list[torch.Tensor]:
    """Scale each seed's gradient (over all tensors given) to norm at most ``max_norm``."""
    sq = sum((g.reshape(g.shape[0], -1) ** 2).sum(dim=1) for g in grads)
    scale = torch.clamp(max_norm / (torch.sqrt(sq) + 1e-12), max=1.0)
    return [g * scale.reshape((-1,) + (1,) * (g.dim() - 1)) for g in grads]


def finite_by_seed(grads: list[torch.Tensor]) -> torch.Tensor:
    """Per seed: every gradient entry is finite. A seed that fails skips the update."""
    ok = torch.ones(grads[0].shape[0], dtype=torch.bool, device=grads[0].device)
    for g in grads:
        ok &= torch.isfinite(g.reshape(g.shape[0], -1)).all(dim=1)
    return ok


class Adam:
    """Adam (descent) over a list of per-seed tensors, with a per-seed mask.

    A masked seed keeps its parameters and its optimizer state, so a seed that has diverged or
    produced a non-finite gradient is frozen without touching the others. The step count is shared;
    that is harmless because a frozen seed's moments are frozen with it.
    """

    def __init__(self, params: Params, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[torch.Tensor], mask: torch.Tensor) -> None:
        self.t += 1
        c1, c2 = 1.0 - self.b1**self.t, 1.0 - self.b2**self.t
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(self.params, grads)):
                keep = mask.reshape((-1,) + (1,) * (g.dim() - 1))
                g = torch.where(keep, g, torch.zeros_like(g))
                self.m[i] = torch.where(keep, self.b1 * self.m[i] + (1 - self.b1) * g, self.m[i])
                self.v[i] = torch.where(keep, self.b2 * self.v[i] + (1 - self.b2) * g**2, self.v[i])
                step = self.lr * (self.m[i] / c1) / (torch.sqrt(self.v[i] / c2) + self.eps)
                p -= torch.where(keep, step, torch.zeros_like(step))
