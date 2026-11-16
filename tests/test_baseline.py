"""The LQR and LQI baselines against the closed forms and claims of the notes."""

import numpy as np
import pytest

from utils import baseline
from utils import control

GAMMAS = [1.0, 0.99, 0.9]


@pytest.mark.parametrize("gamma", GAMMAS)
def test_solver_matches_scalar_closed_forms(gamma):
    prob = baseline.scalar_lqr(gamma)
    assert prob.J_star == pytest.approx(baseline.scalar_p(gamma), rel=1e-10)
    assert prob.K_star.item() == pytest.approx(baseline.scalar_K_star(gamma), rel=1e-10)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_riccati_identities(gamma):
    p, K = baseline.scalar_p(gamma), baseline.scalar_K_star(gamma)
    assert gamma * p**2 + (1 - 2 * gamma) * p - 1 == pytest.approx(0, abs=1e-12)
    assert gamma * K**2 + K - gamma == pytest.approx(0, abs=1e-12)
    assert 1 + gamma * p == pytest.approx(1 / (1 - K), rel=1e-12)


def test_undiscounted_golden_ratio():
    phi = (1 + np.sqrt(5)) / 2
    prob = baseline.scalar_lqr(1.0)
    assert prob.J_star == pytest.approx(phi)
    assert prob.K_star.item() == pytest.approx(phi - 1)
    assert baseline.scalar_curvature(1.0) == pytest.approx(1 + 2 / np.sqrt(5))


def test_numbers_quoted_at_gamma_099():
    g = 0.99
    assert baseline.scalar_K_star(g) == pytest.approx(0.6153, abs=5e-5)
    assert baseline.scalar_p(g) == pytest.approx(1.6153, abs=5e-5)
    assert 1 + g * baseline.scalar_p(g) == pytest.approx(2.599, abs=5e-4)
    assert baseline.scalar_curvature(g) == pytest.approx(1.885, abs=5e-4)
    lo, hi = baseline.scalar_finite_cost_interval(g)
    assert lo == pytest.approx(-0.005, abs=5e-4) and hi == pytest.approx(2.005, abs=5e-4)
    K = baseline.scalar_K_star(g)
    assert 1 / K == pytest.approx(1.625, abs=5e-4)  # LQR offset x_ss = d / K*


@pytest.mark.parametrize("gamma", GAMMAS)
def test_gain_cost_and_gap_match_exact_formula(gamma):
    prob = baseline.scalar_lqr(gamma)
    lo, hi = baseline.scalar_finite_cost_interval(gamma)
    for k in np.linspace(lo - 0.5, hi + 0.5, 401):
        assert prob.cost(k) == pytest.approx(baseline.scalar_cost(k, gamma), rel=1e-9)
        assert prob.gap(k) == pytest.approx(baseline.scalar_gap_exact(k, gamma), abs=1e-9)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_gap_bounded_and_one_outside_finite_cost_interval(gamma):
    prob = baseline.scalar_lqr(gamma)
    lo, hi = baseline.scalar_finite_cost_interval(gamma)
    ks = np.linspace(-3, 5, 2001)
    gaps = np.array([prob.gap(k) for k in ks])
    assert np.all((gaps >= -1e-12) & (gaps <= 1))
    assert np.all(gaps[(ks <= lo) | (ks >= hi)] == 1.0)
    assert prob.gap(prob.K_star) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_K_star_minimizes_cost_on_a_grid(gamma):
    prob = baseline.scalar_lqr(gamma)
    ks = np.linspace(0.01, 1.99, 4001)
    best = ks[np.argmin([prob.cost(k) for k in ks])]
    assert best == pytest.approx(prob.K_star.item(), abs=1e-3)


def test_quadratic_bounds_and_observations_at_gamma_099():
    g = 0.99
    K = baseline.scalar_K_star(g)
    for e in np.linspace(-0.1, 0.1, 201):
        if e == 0:
            continue
        ratio = baseline.scalar_gap_exact(K + e, g) / e**2
        assert 1.719 <= ratio <= 2.054
    # Observations 1-2: same distance 0.118 below and above, undershooting costs more.
    under, over = baseline.scalar_gap_exact(K - 0.118, g), baseline.scalar_gap_exact(K + 0.118, g)
    assert under == pytest.approx(0.029, abs=5e-4)
    assert over == pytest.approx(0.024, abs=5e-4)
    assert under > over


def test_finite_cost_is_not_stability():
    """For gamma < 1 a gain just outside (0, 2) is unstable yet has finite discounted cost."""
    prob = baseline.scalar_lqr(0.99)
    k = 2.003
    assert not prob.is_stable(k)
    assert np.isfinite(prob.cost(k)) and prob.gap(k) < 1


def test_rollout_cost_matches_lyapunov_cost():
    prob = baseline.scalar_lqr(0.99)
    K = prob.K_star
    J_roll = prob.policy_cost(lambda x: -K @ x, horizon=5000)
    assert J_roll == pytest.approx(prob.J_star, rel=1e-9)
    assert prob.policy_gap(lambda x: -K @ x) == pytest.approx(0, abs=1e-9)


def test_gap_convention_infinite_cost():
    assert control.gap(float("inf"), 1.6) == 1.0
    assert control.gap(np.array([1.6, 3.2, np.inf]), 1.6).tolist() == pytest.approx([0.0, 0.5, 1.0])


# ===================== LQI =====================


@pytest.fixture
def prob():
    return baseline.scalar_lqi(gamma=0.99)


def test_equilibrium_scalar(prob):
    x_ss, u_ss = prob.equilibrium
    assert x_ss == pytest.approx([1.0]) and u_ss == pytest.approx([0.0])


def test_augmented_structure(prob):
    aug = prob.augmented
    assert aug.A == pytest.approx(np.array([[1.0, 0.0], [-1.0, 1.0]]))
    assert aug.B == pytest.approx(np.array([[1.0], [0.0]]))
    assert aug.x0 == pytest.approx([-1.0, 0.0])  # x0 - x_ss, z0 = 0


def test_K_star_is_a_local_minimum(prob):
    K = prob.K_star.reshape(-1)
    J = prob.J_star
    assert prob.cost(K) == pytest.approx(J, rel=1e-10)
    rng = np.random.default_rng(0)
    for _ in range(200):
        dK = rng.normal(scale=0.02, size=2)
        assert prob.cost(K + dK) >= J - 1e-12
    assert prob.gap(K) == pytest.approx(0, abs=1e-12)


def test_gain_has_proportional_and_integral_parts(prob):
    K_x, K_z = prob.K_star.reshape(-1)
    assert K_x > 0 and K_z < 0  # u = -K_x x~ - K_z z with z accumulating y_ref - y
    assert prob.is_stable(prob.K_star)


def test_zero_steady_state_error(prob):
    run = prob.simulate(prob.K_star, steps=400)
    assert run["y"][-1] == pytest.approx(prob.y_ref, abs=1e-8)
    assert run["z"][-1] == pytest.approx([0.0], abs=1e-8)


def test_rollout_of_raw_system_matches_augmented_cost(prob):
    """The discounted cost of the simulated deviations equals J(K*) from the Lyapunov equation."""
    run = prob.simulate(prob.K_star, steps=5000)
    x_ss, u_ss = prob.equilibrium
    dx, z, du = run["x"] - x_ss, run["z"], run["u"] - u_ss
    disc = prob.gamma ** np.arange(len(dx))
    J = np.sum(disc * (dx[:, 0] ** 2 * prob.Q.item() + z[:, 0] ** 2 * prob.Q_z.item() + du[:, 0] ** 2 * prob.R.item()))
    assert J == pytest.approx(prob.J_star, rel=1e-9)


def test_disturbance_not_supported_yet():
    with pytest.raises(NotImplementedError):
        baseline.LQIProblem(A=1, B=1, C=1, Q=1, Q_z=1, R=1, gamma=0.99, y_ref=[1], x0=[0], d=[0.1])
