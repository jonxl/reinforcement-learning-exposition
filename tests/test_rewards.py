"""Training rewards must reflect the transitions actually seen, including exploration."""
import numpy as np
import pytest

from utils.rewards import RewardWindow
from utils import envs
from algos import ppo, ppo_linear, monte_carlo


def test_reward_window_weights_transitions_and_keeps_failures():
    window = RewardWindow(2)
    initial, count = window.flush()
    assert np.isnan(initial).all() and not count.any()
    window.add(np.array([[-1., -2.], [-3., -4.]]))
    window.add(np.array([[-8., -9.]]))
    reward, count = window.flush()
    assert reward == pytest.approx([-4., -5.])
    assert count.tolist() == [3, 3]
    window.add(np.array([[-1., np.nan]]))
    reward, _ = window.flush()
    assert reward[0] == -1 and reward[1] == -np.inf


@pytest.mark.parametrize('module', [ppo, ppo_linear, monte_carlo])
def test_logged_rewards_equal_actual_environment_rewards(monkeypatch, module):
    recorded = []
    original = envs.Env.step
    def step(self, action):
        obs, reward = original(self, action)
        recorded.append(reward.copy())
        return obs, reward
    monkeypatch.setattr(envs.Env, 'step', step)
    cfg = dict(total_steps=64, horizon=8, n_checkpoints=2, eval_horizon=30)
    E = 1 if module is monte_carlo else 2
    if module is ppo:
        cfg.update(n_envs=E, epochs=1, minibatch_size=16, hidden_size=8)
    elif module is ppo_linear:
        cfg.update(n_envs=E, epochs=1, n_minibatches=2)
    result = module.train(envs.make_problem('nlqr_energy'), np.array([7, 9]), module.Config(**cfg))
    raw = np.asarray(recorded).reshape(-1, 2, E)
    assert np.isnan(result['training_stage_reward'][:, 0]).all()
    for j in range(1, len(result['steps'])):
        start, stop = result['steps'][j-1:j+1]
        expected = raw[start//E:stop//E].mean(axis=(0, 2))
        assert result['training_stage_reward'][:, j] == pytest.approx(expected)
        assert (result['training_reward_count'][:, j] == stop-start).all()


def test_three_algorithm_suite_persists_rewards_and_keeps_old_runs_readable(tmp_path):
    import json
    import rollouts
    from utils.suite import run_suite, plot_suite_rewards
    root = run_suite(tmp_path / 'suite', count=1, seeds=1, steps=512,
                     algorithms=['ppo_linear', 'ppo', 'monte_carlo'])
    assert len(json.loads((root/'summary.json').read_text())) == 6
    assert (root/'training_rewards_lqr.png').is_file()
    assert (root/'training_rewards_nlqr_01_01.png').is_file()
    for case in ['lqr', 'nlqr_00']:
        for algo in ['ppo_linear', 'ppo', 'monte_carlo']:
            directory = root/case/algo
            arrays, meta = rollouts.load(directory)
            assert arrays['training_reward_count'][0].sum() == 512
            assert meta['reward_metric']['includes_exploration'] is True
            # Simulate old saved data: replotting must skip unavailable diagnostics.
            arrays.pop('training_stage_reward')
            arrays.pop('training_reward_count')
            np.savez_compressed(directory/'arrays.npz', **arrays)
    assert plot_suite_rewards(root) == []
