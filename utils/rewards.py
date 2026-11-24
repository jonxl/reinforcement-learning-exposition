"""Training reward diagnostics, kept separate from deterministic policy evaluation."""
import numpy as np


class RewardWindow:
    """Per-seed mean stage reward since the previous checkpoint.

    Input is (time, seed[, environment]); counts are environment transitions per
    seed. Empty windows (step zero) are NaN. Failed/nonfinite rollouts are -inf,
    never silently omitted from the across-seed mean.
    """
    def __init__(self, n_seeds):
        self.total = np.zeros(n_seeds)
        self.count = 0

    def add(self, rewards, failed=None):
        rewards = np.asarray(rewards)
        by_seed = np.moveaxis(rewards, 1, 0).reshape(len(self.total), -1)
        sums = by_seed.sum(axis=1)
        invalid = ~np.isfinite(by_seed).all(axis=1)
        if failed is not None:
            invalid |= failed
        self.total += np.where(invalid, -np.inf, sums)
        self.count += by_seed.shape[1]

    def flush(self):
        mean = self.total / self.count if self.count else np.full_like(self.total, np.nan)
        count = np.full(len(self.total), self.count, dtype=np.int64)
        self.total.fill(0)
        self.count = 0
        return mean, count
