"""Seeds: the sample space Omega of the notes.

A seed fixes all the randomness of one training run (initial parameters, exploration noise,
environment noise), so a run is a deterministic function of its seed. Seeds are drawn uniformly,
without replacement, from Omega = {0, ..., 2^32 - 1}, so P(omega) = 2^-32 (notes, "Many seeds").

The draw itself is reproducible: a master seed determines the whole list, so an experiment is
identified by (master seed, number of seeds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OMEGA_SIZE = 2**32  # |Omega|


def draw_seeds(m: int, master_seed: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Draw ``m`` distinct seeds uniformly from Omega, skipping any seed in ``exclude``.

    Uniform without replacement keeps the M runs independent draws of distinct points of Omega,
    which is what the variance and concentration statements assume.
    """
    if m < 0:
        raise ValueError("m must be nonnegative")
    excluded = set() if exclude is None else {int(s) for s in exclude}
    rng = np.random.default_rng(master_seed)
    seeds: list[int] = []
    seen = set(excluded)
    while len(seeds) < m:
        batch = rng.integers(0, OMEGA_SIZE, size=2 * (m - len(seeds)) + 8, dtype=np.uint64)
        for s in batch.tolist():
            if s not in seen:
                seen.add(s)
                seeds.append(s)
                if len(seeds) == m:
                    break
    return np.asarray(seeds, dtype=np.uint64)


@dataclass(frozen=True)
class SeedSplit:
    """Disjoint tuning and evaluation seeds (notes, "A possible experiment")."""

    tune: np.ndarray
    eval: np.ndarray


def split_seeds(n_tune: int, n_eval: int, master_seed: int) -> SeedSplit:
    """Draw tuning seeds, then evaluation seeds that are guaranteed not to overlap them."""
    tune = draw_seeds(n_tune, master_seed)
    evaluation = draw_seeds(n_eval, master_seed + 1, exclude=tune)
    return SeedSplit(tune=tune, eval=evaluation)


def rng_for(seed: int | np.integer) -> np.random.Generator:
    """The random number generator of one run. Everything random in that run must come from it."""
    return np.random.default_rng(int(seed))
