import copy
import numpy as np
import pytest
pytest.importorskip('numba',reason='Install the optional lq extra for compiled-planner tests')
from numpy.testing import assert_allclose
from algos.tdmpc2_lq import Config,Learner,model_loss_grad,actor_loss_grad
from algos.lq_planner import kernel,plan
from experiments.lq_comparison import Replay,evaluate,tail,score_td
from utils.baseline import scalar_lqr


def test_joint_multistep_gradient():
    rng=np.random.default_rng(73);p=rng.normal(0,.2,(2,16))
    x=rng.normal(size=(2,5,4));u=rng.normal(size=(2,5,3))
    c=rng.normal(size=u.shape);target=rng.normal(size=u.shape);cfg=Config()
    _,g,_=model_loss_grad(p,x,u,c,target,cfg)
    for j in range(16):
        plus=p.copy();minus=p.copy();plus[:,j]+=1e-6;minus[:,j]-=1e-6
        numerical=(model_loss_grad(plus,x,u,c,target,cfg)[0]-model_loss_grad(minus,x,u,c,target,cfg)[0])/2e-6
        assert_allclose(g[:,j],numerical,rtol=2e-5,atol=1e-7)


def test_actor_gradient_with_saturated_actions():
    rng=np.random.default_rng(4);p=rng.normal(0,.3,(2,16));p[:,15]=.5
    z=rng.normal(size=(2,7,4));eps=rng.normal(size=z.shape);scale=np.array([1.,3.])
    _,g=actor_loss_grad(p,z,eps,scale,Config())
    for j in (14,15):
        plus=p.copy();minus=p.copy();plus[:,j]+=1e-6;minus[:,j]-=1e-6
        numerical=(actor_loss_grad(plus,z,eps,scale,Config())[0]-actor_loss_grad(minus,z,eps,scale,Config())[0])/2e-6
        assert_allclose(g[:,j],numerical,rtol=1e-5,atol=1e-8)


def test_compiled_arithmetic_and_known_quadratic():
    # One-step objective u^2 + gamma*(x+u)^2 has known optimum.
    p=np.zeros(16);p[:2]=1;p[4]=1;p[6]=p[10]=1;p[15]=-20
    rg=np.random.default_rng(8);normals=rg.standard_normal(6*4096*2+1)
    args=(p,np.zeros(1),1.,normals,.5,1,4096,16,0,6,.99,2.,False)
    a,m=kernel(*args);aa,mm=kernel.py_func(*args)
    assert_allclose([a,*m],[aa,*mm],rtol=1e-12,atol=1e-12)
    assert abs(a+.99/1.99)<.08


def test_actor_proposals_drive_all_prior_candidates():
    l=Learner([7],Config(candidates=8,elites=8,prior_candidates=8))
    l.p[:]=0;l.p[:,14]=1;l.p[:,15]=-25
    assert_allclose(plan(l,np.ones(1)),[-1],atol=1e-8)


def test_seed_isolation_and_evaluation_restoration():
    a=Learner([9]);b=Learner([7,9]);x=np.ones(1)
    for _ in range(4):assert_allclose(plan(a,x),plan(b,np.ones(2))[1:],atol=1e-14)
    before=copy.deepcopy(a.rng);warm=a.mean.copy()
    score_td(a)
    assert_allclose(a.mean,warm)
    for ra,rb in zip(a.rng[0],before[0]):assert ra.bit_generator.state==rb.bit_generator.state


def test_replay_alignment_and_partial_episode():
    replay=Replay(2,1000)
    for t in range(510):
        x=np.array([t%500,1000+t%500],float);u=np.ones(2)
        replay.add(x,u,x*x+1,x+1)
    xx,uu,cc=replay.sample([np.random.default_rng(1),np.random.default_rng(2)],20000)
    assert_allclose(np.diff(xx,axis=-1),uu)
    assert_allclose(cc,xx[:,:,:-1]**2+1)
    assert np.max(xx[0])<=500 and np.min(xx[1])>=1000
    # A reset-crossing sequence would contain a negative state difference.
    assert np.all(np.diff(xx,axis=-1)==1)


def test_certified_cost_contains_exact_riccati_and_zero_policy():
    problem=scalar_lqr();k=problem.K_star.item()
    r=evaluate(lambda x:np.clip(-k*x,-2,2),1)
    assert r['cost_lower'][0]-1e-14<=problem.J_star<=r['cost_upper'][0]+1e-14
    r=evaluate(lambda x:np.zeros_like(x),1)
    assert r['cost_lower'][0]<=100<=r['cost_upper'][0]
    # Worst state-growth path lies inside the analytic envelope.
    h=100;state=7.;j=np.arange(20000)
    actual=.99**h*np.sum(.99**j*((state+2*j)**2+4))
    assert_allclose(tail(state,h),actual,rtol=1e-13)


def test_update_batch_isolation_and_target_schedule():
    cfg=Config(batch=8);a=Learner([17],cfg);b=Learner([10,17],cfg)
    rng=np.random.default_rng(99);x=rng.normal(size=(2,8,4));u=rng.normal(size=(2,8,3));c=x[:,:,:-1]**2+u*u
    old=a.target.copy()
    a.update(x[1:],u[1:],c[1:]);b.update(x,u,c)
    assert_allclose(a.p,b.p[1:],atol=1e-14)
    assert_allclose(a.target,(1-cfg.tau)*old+cfg.tau*a.p[:,6:14])


def test_bounded_ppo_keeps_latent_samples_but_applies_clipped_actions(monkeypatch):
    from algos import ppo_lq_legacy as ppo
    from utils import envs
    original_step=envs.Env.step
    original_objective=ppo.objective_and_grads
    seen={'physical':0,'latent_outside':False}
    def step(self,u):
        assert np.max(np.abs(u))<=.01
        seen['physical']+=1
        return original_step(self,u)
    def objective(params,batch,cfg):
        seen['latent_outside']|=bool(np.any(np.abs(batch['u'])>.01))
        return original_objective(params,batch,cfg)
    monkeypatch.setattr(envs.Env,'step',step)
    monkeypatch.setattr(ppo,'objective_and_grads',objective)
    ppo.train(scalar_lqr(),np.array([42]),ppo.Config(total_steps=2000,n_checkpoints=1),
              action_bound=.01,evaluator=lambda step,p,alive:np.zeros(1))
    assert seen['physical']==250 and seen['latent_outside']
