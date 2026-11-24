import inspect
import json
import numpy as np
import pytest
import rollouts
from algos.families import PPO, PPONN, PPOLG, TDMPC2, TDMPC2LG, TDMPC2NN, canonical_algorithms
from utils import envs
from utils.graphs import label, color


def test_polymorphic_families_and_aliases():
    assert inspect.isabstract(PPO) and inspect.isabstract(TDMPC2)
    assert isinstance(rollouts.get_algo('ppo-nn'), PPONN)
    assert isinstance(rollouts.get_algo('ppo-lg'), PPOLG)
    assert isinstance(rollouts.get_algo('td-mpc2-lg'), TDMPC2LG)
    for old, new in [('ppo','ppo-nn'), ('ppo_linear','ppo-lg'), ('ppo-lq','ppo-lg'), ('tdmpc2_lq','td-mpc2-lg')]:
        assert type(rollouts.get_algo(old)) is type(rollouts.get_algo(new))
        assert label(old) == label(new)
        assert color(old) == color(new)
    with pytest.raises(ValueError, match='distinct'):
        canonical_algorithms(['ppo', 'ppo-nn'])


def test_neural_variant_available():
    algo = rollouts.get_algo('td-mpc2-nn')
    assert isinstance(algo, TDMPC2NN)
    assert algo.Config().num_critics >= 2
    algo.validate_problem('nlqr_energy')


@pytest.mark.parametrize('name', ['ppo-nn','ppo-lg'])
def test_ppo_variant_preserves_backend_numerics(name):
    from importlib import import_module
    algo=rollouts.get_algo(name)
    overrides=dict(total_steps=16, horizon=8, n_envs=2, epochs=1, n_checkpoints=1, eval_horizon=30)
    overrides['minibatch_size' if name=='ppo-nn' else 'n_minibatches']=4
    cfg=rollouts.make_config(algo, overrides)
    problem=envs.make_problem('lqr')
    seeds=np.array([4,7])
    a=algo.train(problem,seeds,cfg)
    b=import_module(algo.backend).train(problem,seeds,cfg)
    for key in a:
        if isinstance(a[key],np.ndarray):
            np.testing.assert_array_equal(a[key],b[key])


def test_alias_run_saves_canonical_metadata(tmp_path):
    out=rollouts.run('ppo_linear',n_seeds=1,workers=1,out_dir=tmp_path/'run',
        overrides=dict(total_steps=16,horizon=8,n_envs=2,epochs=1,n_minibatches=2,n_checkpoints=1))
    meta=json.loads((out/'meta.json').read_text())
    assert (meta['algo'],meta['algorithm_family'],meta['algorithm_variant'])==('ppo-lg','ppo','lg')
