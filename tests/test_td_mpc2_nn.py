import copy
import numpy as np
import torch
from algos.td_mpc2_nn import Config, Learner, Replay, train, two_hot
from utils import envs


def config():
    return Config(hidden_size=8, latent_dim=4, num_critics=2, num_bins=21,
                  horizon=2, candidates=4, elites=2, prior_candidates=1, iterations=1,
                  batch_size=2, warmup=4, update_every=2, episode_length=8,
                  total_steps=8, n_checkpoints=1, eval_horizon=64, eval_streams=1)


def test_neural_components_learn_and_target_tracks():
    learner = Learner(2, 1, .5, 3, config())
    before = copy.deepcopy(learner.model.state_dict())
    rng = np.random.default_rng(5)
    tensors = [torch.tensor(a, dtype=torch.float64) for a in
               (rng.normal(size=(2,3,2)), rng.normal(size=(2,2,1)), np.ones((2,2)))]
    losses = learner.update(*tensors)
    assert np.isfinite(losses).all()
    after = learner.model.state_dict()
    for component in ('encoder','dynamics','actor','cost','critics.0','critics.1'):
        assert any(not torch.equal(v, after[k]) for k,v in before.items() if k.startswith(component+'.'))
    for k,v in after.items():
        if k.startswith('target_critics.'):
            online = after[k.replace('target_critics.', 'critics.')]
            torch.testing.assert_close(v, before[k]*(1-config().tau)+online*config().tau)


def test_replay_respects_resets_and_wraparound():
    replay = Replay(5,1,1)
    for i in range(12): replay.add([i],[0],i,[i+1],i//3)
    x,u,c = replay.sample(np.random.default_rng(1),100,2)
    assert (c[:,0] >= 7).all()
    assert ((c[:,0].long()//3)==(c[:,1].long()//3)).all()
    torch.testing.assert_close(x[:,1:,0],c+1)


def test_seed_results_independent_of_batching():
    problem=envs.make_problem('lqr',gamma=.5)
    together=train(problem,[3,7],config())
    for i,seed in enumerate([3,7]):
        alone=train(problem,[seed],config())
        for key in ('gaps','model_parameters','training_stage_reward','updates'):
            np.testing.assert_allclose(together[key][i],alone[key][0],rtol=0,atol=0,equal_nan=True)
    assert (together['updates'][:,-1] == 3).all()
    assert (together['training_reward_count'][:,-1] == 8).all()


def test_bounded_planner_and_distribution():
    learner=Learner(2,1,.5,2,config(),bound=2.)
    for _ in range(5): assert np.abs(learner.plan([1.,2.],training=True)).max() <= 2
    target=two_hot(torch.tensor([-1e20,0.,1e20],dtype=torch.float64),21,20.)
    torch.testing.assert_close(target.sum(-1),torch.ones(3,dtype=torch.float64))
    assert (target >= 0).all()


def test_neural_suite_reporting_without_linear_stability_metric(tmp_path, monkeypatch):
    import json
    import rollouts
    from dataclasses import asdict
    from utils.suite import run_suite
    original = rollouts.run

    def short_run(*args, **kwargs):
        args = list(args)
        args[6] = asdict(config())
        args[6].pop('device')
        return original(*args, **kwargs)

    monkeypatch.setattr(rollouts, 'run', short_run)
    root = run_suite(tmp_path/'suite', count=1, seeds=1, steps=512,
                     gamma=.5, algorithms=['td-mpc2-nn'])
    rows = json.loads((root/'summary.json').read_text())
    assert len(rows) == 2
    assert all(row['final_unstable_fraction'] is None for row in rows)
    assert list(root.rglob('mean_learning_curve.png'))
