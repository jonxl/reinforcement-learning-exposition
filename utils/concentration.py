"""Weak law of large numbers for learning curves: Chebyshev and Hoeffding (notes, Sections 7-8).

At one checkpoint, the M seed runs give independent numbers Y_j = g_X(omega_j, N) in [0, 1] with
mean g_bar_X(N). The estimated curve g_hat_X(N) is their average, and

    Chebyshev:  P(|g_hat - g_bar| >= eps) <= sigma^2 / (M eps^2) <= 1 / (4 M eps^2)
    Hoeffding:  P(|g_hat - g_bar| >= eps) <= 2 exp(-2 M eps^2)

For n checkpoints at once the risks add (union bound), so delta is replaced by delta / n.
"""

from __future__ import annotations

import math

VARIANCE_BOUND = 0.25  # Var of any [0, 1] variable is at most m(1 - m) <= 1/4


def chebyshev_tail(m: int, eps: float, variance: float = VARIANCE_BOUND) -> float:
    return min(1.0, variance / (m * eps**2))


def hoeffding_tail(m: int, eps: float) -> float:
    return min(1.0, 2.0 * math.exp(-2.0 * m * eps**2))


def chebyshev_seeds(eps: float, delta: float, variance: float = VARIANCE_BOUND) -> int:
    """Smallest M with variance / (M eps^2) <= delta."""
    return math.ceil(variance / (eps**2 * delta))


def hoeffding_seeds(eps: float, delta: float, n_checkpoints: int = 1) -> int:
    """Smallest M with 2 n exp(-2 M eps^2) <= delta, i.e. M >= ln(2n/delta) / (2 eps^2)."""
    return math.ceil(math.log(2.0 * n_checkpoints / delta) / (2.0 * eps**2))


def hoeffding_eps(m: int, delta: float, n_checkpoints: int = 1) -> float:
    """Half-width of the Hoeffding band: every checkpoint within eps with probability >= 1 - delta."""
    return math.sqrt(math.log(2.0 * n_checkpoints / delta) / (2.0 * m))


def chebyshev_eps(m: int, delta: float, variance: float = VARIANCE_BOUND, n_checkpoints: int = 1) -> float:
    """Half-width of the Chebyshev band, with an estimated or worst-case variance."""
    return math.sqrt(variance * n_checkpoints / (m * delta))
