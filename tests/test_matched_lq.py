import json
import numpy as np
import pytest
from utils import envs
from utils.bounded_lqr import BoundedScalarLQR
from utils.suite import run_suite
from utils.matched_lq import import_published


def test_bounded_physics_and_certified_mean_controller():
    problem = BoundedScalarLQR()
    env = envs.Env(problem, 1)
    env.reset()
    x, reward = env.step(np.array([[50.]]))
    assert x.item() == 3
    assert reward.item() == -5
    optimal = env.evaluate(problem.optimal_policy)
    assert optimal['J'][0] == pytest.approx(problem.J_star, abs=1e-10)
    zero = env.evaluate(lambda x: np.zeros_like(x))
    assert zero['J'][0] == pytest.approx(100, abs=1e-10)
    clipped = env.evaluate(lambda x: np.full_like(x, 20.))
    assert np.max(clipped['controls']) == 2
    with pytest.raises(ValueError, match='horizon too short'):
        env.evaluate(problem.optimal_policy, horizon=10)


def test_td_not_silently_applied_to_nlqr(tmp_path):
    with pytest.raises(ValueError, match='bounded scalar'):
        run_suite(tmp_path/'no_run', algorithms=['tdmpc2_lq'])
    assert not (tmp_path/'no_run').exists()


def test_published_results_keep_linear_identity_and_bounds(tmp_path):
    for method in ('ppo', 'td'):
        folder = tmp_path/'source'/method
        folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(action_bound=2., gamma=.99, seeds=[7, 9])))
        row = dict(step=0, gap_upper=[.2,.4], gap_lower=[.199,.399], cost_lower=[2.,3.],
                   cost_upper=[2.001,3.001], alive=[True, False])
        (folder/'result.json').write_text(json.dumps(dict(history=[row], elapsed=1)))
    output = tmp_path/'converted'
    import_published(tmp_path/'source', output)
    saved = json.loads((output/'problems.json').read_text())
    assert saved['algorithms'] == ['ppo_linear_legacy', 'tdmpc2_lq']
    with np.load(output/'lqr_bounded'/'tdmpc2_lq'/'arrays.npz') as data:
        np.testing.assert_array_equal(data['seeds'], [7,9])
        np.testing.assert_array_equal(data['gaps'][:,0], [.2,.4])
        np.testing.assert_array_equal(data['diverged'][:,0], [False,True])


def test_td_adapter_preserves_steps_seeds_and_rewards():
    pytest.importorskip('numba')
    from algos.tdmpc2_benchmark import Config, train
    result = train(BoundedScalarLQR(), np.array([7,9]), Config(total_steps=4, n_checkpoints=2))
    np.testing.assert_array_equal(result['steps'], [0,2,4])
    assert result['gaps'].shape == (2,3)
    assert result['parameters'].shape == (2,3,16)
    np.testing.assert_array_equal(result['training_reward_count'], [[0,2,2],[0,2,2]])
    assert np.isnan(result['training_stage_reward'][:,0]).all()
    assert np.isfinite(result['training_stage_reward'][:,1:]).all()
    assert np.all(result['cost_upper'] >= result['cost_lower'])
