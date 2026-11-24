from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
import rollouts
from main import build_parser
from utils.devices import select_devices
from algos import ppo_nn, td_mpc2_nn
from utils import envs


def test_flags_and_cuda_discovery(monkeypatch):
    parser = build_parser()
    for command in (['run','--algo','ppo-nn'], ['crude'], ['suite'], ['matched-lq']):
        assert parser.parse_args(command+['--gpu']).device == 'gpu'
        assert parser.parse_args(command+['--cpu']).device == 'cpu'
        with pytest.raises(SystemExit): parser.parse_args(command+['--cpu','--gpu'])
    monkeypatch.setattr(torch.cuda,'is_available',lambda: False)
    with pytest.warns(RuntimeWarning,match='falling back'):
        assert select_devices('gpu','ppo-nn') == ['cpu']
    monkeypatch.setattr(torch.cuda,'is_available',lambda: True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda: 2)
    assert select_devices('gpu','ppo-nn') == ['cuda:0','cuda:1']
    assert select_devices('gpu','ppo-lg') == ['cpu']
    assert select_devices('cpu','ppo-nn') == ['cpu']


def test_gpu_queue_preserves_order_and_assigns_devices(monkeypatch):
    monkeypatch.setattr(rollouts,'ProcessPoolExecutor',lambda **kw: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(rollouts,'_run_chunk',lambda a,p,g,params,chunk,cfg: (chunk.tolist(),cfg.device))
    chunks=[np.array([i]) for i in range(7)]
    parts=rollouts._gpu_batches('ppo-nn','lqr',.5,{},chunks,ppo_nn.Config(),['cuda:0','cuda:1'])
    assert [p[0] for p in parts] == [[i] for i in range(7)]
    assert {p[1] for p in parts} == {'cuda:0','cuda:1'}


def test_fallback_batches_match_cpu_run(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda: False)
    overrides=dict(total_steps=16,horizon=8,n_envs=2,epochs=1,minibatch_size=4,n_checkpoints=1,eval_horizon=64)
    cpu=rollouts.run('ppo-nn',n_seeds=3,workers=1,gamma=.5,overrides=overrides,out_dir=tmp_path/'cpu')
    with pytest.warns(RuntimeWarning,match='falling back'):
        gpu=rollouts.run('ppo-nn',n_seeds=3,workers=1,gamma=.5,overrides=overrides,out_dir=tmp_path/'gpu',device='gpu',seed_batch_size=2)
    a,_=rollouts.load(cpu); b,meta=rollouts.load(gpu)
    for key in a: np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12,equal_nan=True)
    assert meta['execution'] == dict(requested='gpu',devices=['cpu'],seed_batch_size=2,batches=2)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA hardware')
def test_cuda_neural_backends():
    problem=envs.make_problem('lqr',gamma=.5)
    cfg=ppo_nn.Config(total_steps=16,horizon=8,n_envs=2,epochs=1,minibatch_size=4,n_checkpoints=1,eval_horizon=64)
    cpu=ppo_nn.train(problem,np.array([3,7]),cfg)
    gpu=ppo_nn.train(problem,np.array([3,7]),replace(cfg,device='cuda:0'))
    np.testing.assert_allclose(cpu['gaps'],gpu['gaps'],rtol=1e-7,atol=1e-8)
    cfg=td_mpc2_nn.Config(device='cuda:0',hidden_size=8,latent_dim=4,num_critics=2,num_bins=21,
        total_steps=8,horizon=2,warmup=4,update_every=2,episode_length=8,batch_size=2,
        candidates=4,elites=2,prior_candidates=1,iterations=1,n_checkpoints=1,eval_horizon=64,eval_streams=1)
    result=td_mpc2_nn.train(problem,[3],cfg)
    assert result['updates'][0,-1] == 3
    assert np.isfinite(result['model_parameters']).all()
