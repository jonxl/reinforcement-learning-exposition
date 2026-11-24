"""Advantage estimation shared by the actor-critic algorithms."""

from __future__ import annotations

import numpy as np


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and return targets. rewards (T, ...), values (T + 1, ...) incl. the bootstrap V(x_T)."""
    deltas = rewards + gamma * values[1:] - values[:-1]
    adv = np.empty_like(deltas)
    acc = np.zeros(deltas.shape[1:])
    for t in range(deltas.shape[0] - 1, -1, -1):
        acc = deltas[t] + gamma * lam * acc
        adv[t] = acc
    return adv, adv + values[:-1]
