"""Seeds, concentration bounds and learning-curve statistics against the claims of the notes."""

import numpy as np
import pytest

from utils import concentration as conc
from utils import learning_curves as lc
from utils import seeds

# -- seeds ----------------------------------------------------------------------------------------


def test_seeds_distinct_in_range_and_reproducible():
    s = seeds.draw_seeds(1000, master_seed=7)
    assert len(np.unique(s)) == 1000
    assert s.min() >= 0 and s.max() < seeds.OMEGA_SIZE
    assert np.array_equal(s, seeds.draw_seeds(1000, master_seed=7))


def test_tuning_and_evaluation_seeds_disjoint():
    split = seeds.split_seeds(50, 1000, master_seed=3)
    assert not set(split.tune.tolist()) & set(split.eval.tolist())


def test_rng_for_is_deterministic():
    assert seeds.rng_for(42).normal() == seeds.rng_for(42).normal()


# -- concentration --------------------------------------------------------------------------------


def test_seed_counts_quoted_in_notes():
    assert conc.chebyshev_seeds(0.05, 0.05) == 2000
    assert conc.hoeffding_seeds(0.05, 0.05) == 738


def test_hoeffding_eps_inverts_seed_count():
    m = conc.hoeffding_seeds(0.05, 0.05)
    assert conc.hoeffding_eps(m, 0.05) <= 0.05
    assert conc.hoeffding_eps(m - 1, 0.05) > 0.05


def test_union_bound_needs_more_seeds():
    assert conc.hoeffding_seeds(0.05, 0.05, n_checkpoints=20) > conc.hoeffding_seeds(0.05, 0.05)


def test_hoeffding_band_coverage_by_simulation():
    """Draw [0, 1] variables with a known mean; the band must miss the mean at most delta of the time."""
    rng = np.random.default_rng(0)
    M, delta, trials, true_mean = 200, 0.05, 4000, 0.3
    eps = conc.hoeffding_eps(M, delta)
    draws = rng.beta(3, 7, size=(trials, M))  # mean 0.3, support [0, 1]
    miss = np.mean(np.abs(draws.mean(axis=1) - true_mean) >= eps)
    assert miss <= delta


# -- learning curves ------------------------------------------------------------------------------


def _curves(gaps, steps=None, **kw):
    gaps = np.asarray(gaps, dtype=float)
    steps = np.arange(1, gaps.shape[1] + 1) if steps is None else steps
    return lc.LearningCurves(steps=steps, gaps=gaps, **kw)


def test_mean_std_and_standard_error():
    c = _curves([[0.0, 0.2], [0.2, 0.4], [0.4, 0.6]])
    assert c.mean() == pytest.approx([0.2, 0.4])
    assert c.std() == pytest.approx([0.2, 0.2])
    assert c.standard_error() == pytest.approx([0.2 / np.sqrt(3)] * 2)


def test_rejects_gaps_outside_unit_interval():
    with pytest.raises(ValueError):
        _curves([[0.1, 1.2]])


def test_floor_ordering_does_not_imply_expected_ordering():
    """Notes: g_A = 0.1 + 0.8/N, g_B = 0.2 + 0.1/N cross at N = 7."""
    N = np.arange(1, 201)
    a = _curves([0.1 + 0.8 / N], steps=N)
    b = _curves([0.2 + 0.1 / N], steps=N)
    assert lc.floor_ordered(a, b)
    ordered = lc.expected_ordered(a, b)
    assert not ordered[:6].any() and ordered[6:].all()
    assert lc.crossing_time(N, ordered) == 7


def test_expected_ordering_does_not_imply_band_separation():
    """Notes: seeds give A = (0.01, 0.05), B = (0.02, 0.06); averages ordered, bands overlap."""
    a = _curves([[0.01], [0.05]])
    b = _curves([[0.02], [0.06]])
    assert lc.expected_ordered(a, b).all()
    assert not lc.band_separated(a, b).any()


def test_band_separation_implies_expected_ordering():
    rng = np.random.default_rng(1)
    a = _curves(rng.uniform(0.0, 0.3, size=(50, 10)))
    b = _curves(rng.uniform(0.3, 0.6, size=(50, 10)))
    assert lc.band_separated(a, b).all() and lc.expected_ordered(a, b).all()


def test_crossing_time_none_when_last_checkpoint_fails():
    assert lc.crossing_time(np.arange(5), np.array([True, True, True, True, False])) is None


def test_power_law_fit_and_crossing():
    N = np.logspace(1, 5, 50)
    kappa, beta = lc.fit_power_law(N, 3.0 * N**-0.5)
    assert kappa == pytest.approx(3.0) and beta == pytest.approx(0.5)
    # 3 N^-0.5 = 1 N^-0.25  <=>  N = 3^(1/0.25) = 81
    assert lc.power_law_crossing(3.0, 0.5, 1.0, 0.25) == pytest.approx(81.0)


def test_bias_variance_split():
    gains = np.array([[[0.5]], [[0.7]]])  # two seeds, one checkpoint, one gain
    c = _curves([[0.1], [0.1]], gains=gains)
    out = c.bias_variance(K_star=0.6, curvature=2.0)
    assert out["bias2"] == pytest.approx([0.0])
    assert out["variance"] == pytest.approx([0.02])
    assert out["predicted_gap"] == pytest.approx([0.04])


def test_running_means_converge_to_sample_mean():
    rng = np.random.default_rng(2)
    c = _curves(rng.uniform(size=(2000, 1)))
    paths = c.running_means(checkpoint=0, n_orderings=5)
    assert paths.shape == (5, 2000)
    assert np.allclose(paths[:, -1], c.mean()[0])
    assert np.max(np.abs(paths[:, -1] - paths[:, 1000])) < 0.05


def test_hoeffding_band_contains_mean_and_is_clipped():
    rng = np.random.default_rng(3)
    c = _curves(rng.uniform(0, 0.02, size=(30, 4)))
    lo, hi = c.hoeffding_band(0.05)
    assert np.all(lo >= 0) and np.all(lo <= c.mean()) and np.all(c.mean() <= hi)


def test_align_step_hold_on_union_of_checkpoints():
    from utils.learning_curves import LearningCurves, align

    a = LearningCurves(steps=np.array([0, 10, 20]), gaps=np.array([[1.0, 0.5, 0.2]]), name="a")
    b = LearningCurves(steps=np.array([0, 15, 30]), gaps=np.array([[1.0, 0.4, 0.1]]), name="b")
    a2, b2 = align(a, b)
    assert a2.steps.tolist() == [0, 10, 15, 20]  # union, up to the shorter run
    assert a2.gaps[0].tolist() == [1.0, 0.5, 0.5, 0.2]  # latest checkpoint at or before N
    assert b2.gaps[0].tolist() == [1.0, 1.0, 0.4, 0.4]
    same, other = align(a, a)
    assert same is a and other is a
