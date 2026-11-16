"""The LQI baseline: augmented structure, optimality of K*, zero steady-state error."""

import numpy as np
import pytest

from lqi import baseline as lqi


@pytest.fixture
def prob():
    return lqi.scalar_lqi(gamma=0.99)


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
        lqi.LQIProblem(A=1, B=1, C=1, Q=1, Q_z=1, R=1, gamma=0.99, y_ref=[1], x0=[0], d=[0.1])
