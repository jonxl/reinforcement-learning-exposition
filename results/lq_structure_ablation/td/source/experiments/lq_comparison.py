"""Matched bounded PPO / TD-MPC2-LQ experiment. Run from repository root.

Example: python -m experiments.lq_comparison --method td --seeds 40 --steps 1000000
Evaluation is conditional on the three predeclared planner streams.
"""
import argparse
import copy
import hashlib
import json
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from algos import ppo
from algos.tdmpc2_lq import Learner, Config
from algos.lq_planner import plan
from utils.baseline import scalar_lqr
from utils.seeds import split_seeds

EVAL_SEEDS = (910000, 910001, 910002)
GAMMA = .99
BOUND = 2.
TAIL_TOL = 1e-6


def tail(x, h, gamma=GAMMA, bound=BOUND):
    """Pathwise upper bound: |x_{H+j}| <= |x_H| + bound*j."""
    v = np.abs(x); d = 1-gamma
    return gamma**h*((v*v+bound*bound)/d+2*v*bound*gamma/d**2+
                     bound*bound*gamma*(1+gamma)/d**3)


def evaluate(policy, count):
    x = np.ones(count); J = np.zeros(count); sxx=np.zeros(count); sxu=np.zeros(count)
    for h in range(5000):
        u = np.asarray(policy(x))
        if not np.isfinite(u).all() or np.max(np.abs(u)) > BOUND+1e-12:
            raise FloatingPointError("Invalid evaluation action")
        J += GAMMA**h*(x*x+u*u)
        sxx += GAMMA**h*x*x; sxu += GAMMA**h*x*u
        x += u
        remainder = tail(x,h+1)
        if np.max(remainder) < TAIL_TOL: break
    else: raise RuntimeError("Tail certificate not reached")
    return dict(cost_lower=J, cost_upper=J+remainder, tail=remainder,
                horizon=h+1, final_state=x, effective_gain=-sxu/sxx)


def score_td(learner):
    before=learner.mean.copy()
    try:
        results=[]
        for seed in EVAL_SEEDS:
            learner.mean[:]=0
            # Same stream suite for every controller; disjoint from training.
            rngs=[np.random.default_rng(seed) for _ in learner.seeds]
            results.append(evaluate(lambda x:plan(learner,x,rngs=rngs),len(learner.seeds)))
        return {k:np.mean([r[k] for r in results],axis=0) for k in results[0]}
    finally: learner.mean[:]=before


class Replay:
    """Uniform valid H-transition starts; never sample across a 500-step reset."""
    def __init__(self, count, steps, horizon=3):
        episodes=(steps+499)//500
        self.x=np.empty((count,episodes,501))
        self.u=np.empty((count,episodes,500));self.c=np.empty_like(self.u)
        self.horizon=horizon;self.count=count;self.size=0

    def add(self,x,u,cost,xp):
        e,t=divmod(self.size,500)
        self.x[:,e,t]=x;self.x[:,e,t+1]=xp
        self.u[:,e,t]=u;self.c[:,e,t]=cost;self.size+=1

    def sample(self,rngs,batch):
        H=self.horizon
        # Each complete episode has 501-H valid starts; the partial episode
        # contributes max(0, partial-H+1). Map a uniform index to this union.
        complete,partial=divmod(self.size,500);per=501-H
        total=complete*per+max(0,partial-H+1)
        ind=np.stack([r.integers(total,size=batch) for r in rngs])
        ep=ind//per;start=ind%per
        m=np.arange(self.count)[:,None,None]
        xx=self.x[m,ep[:,:,None],start[:,:,None]+np.arange(H+1)]
        uu=self.u[m,ep[:,:,None],start[:,:,None]+np.arange(H)]
        cc=self.c[m,ep[:,:,None],start[:,:,None]+np.arange(H)]
        return xx,uu,cc


def jsonable(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    return v


def write(path,data):path.write_text(json.dumps(jsonable(data),indent=2),encoding='utf-8')


def run_td(seeds,steps,folder,cfg,n_checkpoints=20):
    learner=Learner(seeds,cfg);replay=Replay(len(seeds),steps,cfg.horizon)
    alive=np.ones(len(seeds),bool);failures=[];history=[];updates=0
    x=np.ones(len(seeds));clock=time.perf_counter();eval_seconds=0.
    checkpoints=set(np.linspace(0,steps,n_checkpoints+1,dtype=int))
    rngs=[r[2] for r in learner.rng]
    def score(step):
        nonlocal eval_seconds
        tic=time.perf_counter();metrics=score_td(learner);eval_seconds+=time.perf_counter()-tic
        # A failure penalty is explicitly separate from a certified policy cost.
        jl=metrics['cost_lower'];ju=metrics['cost_upper'];jstar=scalar_lqr().J_star
        metrics.update(step=step,alive=alive.copy(),updates=updates,
                       gap_lower=np.where(alive,np.maximum(0,1-jstar/jl),1.),
                       gap_upper=np.where(alive,np.maximum(0,1-jstar/ju),1.),
                       elapsed=time.perf_counter()-clock,eval_seconds=eval_seconds)
        history.append(metrics)
        np.savez_compressed(folder/f'checkpoint_{step:07d}.npz',p=learner.p,target=learner.target,
                            m=learner.m,v=learner.v,scale=learner.scale,alive=alive,
                            mean=learner.mean,adam_steps=learner.t)
        write(folder/'metrics.json',dict(history=history,failures=failures))
        print(json.dumps(dict(method='td',step=step,updates=updates,
                              mean_gap=float(np.mean(metrics['gap_upper'])),
                              elapsed=time.perf_counter()-clock)),flush=True)
    def update():
        nonlocal updates
        loss=learner.update(*replay.sample(rngs,cfg.batch));updates+=1
        good=np.isfinite(learner.p).all(1)&np.isfinite(learner.target).all(1)&np.isfinite(loss)
        for j in np.flatnonzero(alive&~good):
            failures.append(dict(seed=int(seeds[j]),step=replay.size,update=updates))
            np.savez(folder/f'failure_{int(seeds[j])}.npz',p=learner.p[j],target=learner.target[j])
        alive[:]&=good
        # Keep failed entries numerically harmless for batching; never count
        # them as successful runs or substitute their fallback rollout costs.
        learner.p[~alive]=0;learner.p[~alive,15]=np.log(.5)
        learner.target[~alive]=0;learner.m[~alive]=0;learner.v[~alive]=0
    score(0)
    for step in range(1,steps+1):
        if step<=2500:
            u=np.array([r[1].uniform(-BOUND,BOUND) for r in learner.rng])
        else:
            u=plan(learner,x,training=True,reset=(step-1)%500==0)
        cost=x*x+u*u;xp=x+u;replay.add(x,u,cost,xp);x=xp
        if step%500==0:x=np.ones(len(seeds))
        if step==2500:
            for _ in range(int(np.ceil(2500/cfg.update_every))):update()
        elif step>2500 and (step-2500)%cfg.update_every==0:update()
        if step in checkpoints:score(step)
        elif step%5000==0:
            write(folder/'progress.json',dict(step=step,updates=updates,elapsed=time.perf_counter()-clock))
            print(json.dumps(dict(method='td',step=step,elapsed=time.perf_counter()-clock)),flush=True)
    write(folder/'rng-final.json',[[r.bit_generator.state for r in group] for group in learner.rng])
    return dict(history=history,failures=failures,elapsed=time.perf_counter()-clock,eval_seconds=eval_seconds)


def run_ppo(seeds,steps,folder,n_checkpoints=20):
    history=[];clock=time.perf_counter();eval_seconds=0.
    def evaluator(step,params,alive):
        nonlocal eval_seconds
        tic=time.perf_counter();k=params['K'][:,0,0].copy()
        metrics=evaluate(lambda x:np.clip(-k*x,-BOUND,BOUND),len(seeds))
        eval_seconds+=time.perf_counter()-tic
        jstar=scalar_lqr().J_star
        metrics.update(step=step,alive=alive.copy(),gain=k,
                       gap_lower=np.where(alive,np.maximum(0,1-jstar/metrics['cost_lower']),1.),
                       gap_upper=np.where(alive,np.maximum(0,1-jstar/metrics['cost_upper']),1.),
                       elapsed=time.perf_counter()-clock)
        history.append(metrics)
        np.savez_compressed(folder/f'checkpoint_{step:07d}.npz',**params,alive=alive)
        write(folder/'metrics.json',dict(history=history))
        print(json.dumps(dict(method='ppo',step=step,mean_gap=float(np.mean(metrics['gap_upper'])),
                              elapsed=time.perf_counter()-clock)),flush=True)
        return metrics['gap_upper']
    result=ppo.train(scalar_lqr(),seeds,ppo.Config(total_steps=steps,n_checkpoints=n_checkpoints),
                     action_bound=BOUND,evaluator=evaluator)
    np.savez_compressed(folder/'ppo-arrays.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    return dict(history=history,elapsed=time.perf_counter()-clock,eval_seconds=eval_seconds,
                config=result['config'])


def parallel_td(seeds,steps,folder,cfg,n_checkpoints,workers):
    """Independent seed groups only; concatenate in the original draw order."""
    clock=time.perf_counter();groups=np.array_split(seeds,workers)
    dirs=[folder/f'shard_{i}' for i in range(workers)]
    for d in dirs:d.mkdir()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_td,s,steps,d,cfg,n_checkpoints) for s,d in zip(groups,dirs)]
        results=[f.result() for f in futures]
    history=[]
    for i in range(len(results[0]['history'])):
        rows=[r['history'][i] for r in results]
        history.append({k:np.concatenate([row[k] for row in rows]) if isinstance(rows[0][k],np.ndarray)
                        else max(row[k] for row in rows) for k in rows[0]})
        step=history[-1]['step'];files=[np.load(d/f'checkpoint_{step:07d}.npz') for d in dirs]
        combined={k:(files[0][k] if k=='adam_steps' else np.concatenate([f[k] for f in files])) for k in files[0].files}
        np.savez_compressed(folder/f'checkpoint_{step:07d}.npz',**combined)
        for f in files:f.close()
    failures=sum([r['failures'] for r in results],[])
    write(folder/'metrics.json',dict(history=history,failures=failures))
    return dict(history=history,failures=failures,elapsed=time.perf_counter()-clock,
                eval_seconds=max(r['eval_seconds'] for r in results),
                worker_elapsed=[r['elapsed'] for r in results],workers=workers)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['td','ppo'],required=True)
    parser.add_argument('--seeds',type=int,default=40)
    parser.add_argument('--steps',type=int,default=1_000_000)
    parser.add_argument('--checkpoints',type=int,default=20)
    parser.add_argument('--structure-only',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--out',default='results/lq_comparison')
    args=parser.parse_args();folder=Path(args.out)/args.method;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():raise FileExistsError('Preserve existing experiment; choose a new output folder')
    seeds=split_seeds(0,args.seeds,0).eval
    cfg=Config()
    if args.structure_only:
        cfg=Config(candidates=512,elites=64,prior_candidates=24,iterations=6,batch=256,update_every=1)
    sources=['algos/tdmpc2_lq.py','algos/lq_planner.py','algos/ppo.py','experiments/lq_comparison.py']
    manifest=dict(arguments=vars(args),seeds=seeds,td_config=asdict(cfg),
                  action_bound=BOUND,gamma=GAMMA,eval_seeds=EVAL_SEEDS,tail_tolerance=TAIL_TOL,
                  python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                  source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    write(folder/'manifest.json',manifest)
    for source in sources:
        dest=folder/'source'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if args.method=='ppo':result=run_ppo(seeds,args.steps,folder,args.checkpoints)
    elif args.workers==1:result=run_td(seeds,args.steps,folder,cfg,args.checkpoints)
    else:result=parallel_td(seeds,args.steps,folder,cfg,args.checkpoints,args.workers)
    write(folder/'result.json',result)
    print('COMPLETE',args.method,result['elapsed'],flush=True)


if __name__=='__main__':main()
