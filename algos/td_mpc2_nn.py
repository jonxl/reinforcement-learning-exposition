"""Neural TD-MPC2 variant for the repository's state-observation benchmarks.

Neural encoder/dynamics with simplex normalization; neural distributional cost
and Q ensemble; neural Gaussian actor prior; receding-horizon sampling planner;
multistep consistency, two-hot symlog losses, actor updates and target critics.

This is a CPU/CUDA float64 benchmark implementation, not an upstream reproduction.
It uses the problem discount, tanh MLPs, all-critic means for planning, and cost
minimization. On unbounded tasks proposals are unsquashed (no actuator bound is
added to the problem); bounded tasks use a squashed/scaled prior and planner.
Only observations, physical actions and costs enter learning. Exact references
are used exclusively for evaluation. Each seed has independent training streams.
"""
from dataclasses import dataclass, asdict
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import envs, networks
from utils.rewards import RewardWindow

NAME = 'td-mpc2-nn'
DTYPE = networks.DTYPE


@dataclass
class Config:
    device: str = "cpu"
    hidden_size: int = 32
    latent_dim: int = 16
    num_critics: int = 5
    num_bins: int = 101
    value_limit: float = 20.
    horizon: int = 3
    candidates: int = 32
    elites: int = 8
    prior_candidates: int = 4
    iterations: int = 3
    min_std: float = .05
    max_std: float = 2.
    temperature: float = .5
    batch_size: int = 32
    replay_capacity: int = 100000
    warmup: int = 256
    update_every: int = 16
    episode_length: int = 500
    lr: float = 3e-4
    tau: float = .01
    rho: float = .5
    entropy: float = 1e-4
    grad_clip: float = 20.
    consistency_weight: float = 20.
    cost_weight: float = .1
    value_weight: float = .1
    total_steps: int = 10000
    n_checkpoints: int = 10
    eval_horizon: int = envs.EVAL_HORIZON
    eval_streams: int = 3

    def validate(self):
        positive = (self.hidden_size, self.latent_dim, self.horizon, self.candidates, self.iterations,
                    self.batch_size, self.replay_capacity, self.update_every, self.episode_length,
                    self.total_steps, self.n_checkpoints, self.eval_horizon, self.eval_streams)
        if min(positive) < 1 or self.latent_dim % 4 or self.num_critics < 2 or self.num_bins < 3:
            raise ValueError('Positive sizes, latent_dim divisible by 4, >=2 critics and >=3 bins required')
        if not (1 <= self.elites <= self.candidates and 0 <= self.prior_candidates <= self.candidates):
            raise ValueError('Invalid planner candidate counts')
        if self.horizon > min(self.replay_capacity, self.episode_length) or self.warmup < self.horizon:
            raise ValueError('Replay, episodes and warmup must accommodate the planning horizon')
        if not (0 < self.tau <= 1 and 0 < self.rho <= 1 and 0 < self.min_std <= self.max_std):
            raise ValueError('Invalid target update, temporal weight, or proposal standard deviation')
        if min(self.lr, self.value_limit, self.grad_clip, self.temperature) <= 0:
            raise ValueError('Optimizer, value support and temperature must be positive')


def symlog(x):
    return x.sign()*torch.log1p(x.abs())


def symexp(x):
    return x.sign()*torch.expm1(x.abs())


def two_hot(x, bins, limit):
    position = (symlog(x).clamp(-limit, limit)+limit)*(bins-1)/(2*limit)
    lower = position.floor().long()
    upper = (lower+1).clamp(max=bins-1)
    fraction = position-lower
    target = torch.zeros(*x.shape, bins, dtype=x.dtype, device=x.device)
    target.scatter_add_(-1, lower.unsqueeze(-1), (1-fraction).unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return target


def decode(logits, limit):
    support = torch.linspace(-limit, limit, logits.shape[-1], dtype=logits.dtype, device=logits.device)
    return symexp((logits.softmax(-1)*support).sum(-1))


def soft_ce(logits, target, cfg):
    return -(two_hot(target, cfg.num_bins, cfg.value_limit)*logits.log_softmax(-1)).sum(-1)


class SimNorm(nn.Module):
    def forward(self, x):
        return x.reshape(*x.shape[:-1], -1, 4).softmax(-1).reshape_as(x)


def mlp(n_in, n_out, cfg, rng, normalized=False, zero_output=False):
    # Torch initialization is isolated; all retained weights come from this seed's NumPy stream.
    with torch.random.fork_rng(devices=[]):
        layers = nn.Sequential(nn.Linear(n_in,cfg.hidden_size), nn.Tanh(),
            nn.Linear(cfg.hidden_size,cfg.hidden_size), nn.Tanh(), nn.Linear(cfg.hidden_size,n_out))
    layers = layers.to(dtype=DTYPE)
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(torch.tensor(rng.normal(0, 1/np.sqrt(layer.in_features), layer.weight.shape), dtype=DTYPE))
                layer.bias.zero_()
        if zero_output:
            layers[-1].weight.zero_()
    return nn.Sequential(layers, SimNorm()) if normalized else layers


class WorldModel(nn.Module):
    def __init__(self, n_state, n_action, cfg, rng):
        super().__init__()
        self.encoder = mlp(n_state,cfg.latent_dim,cfg,rng,normalized=True)
        self.dynamics = mlp(cfg.latent_dim+n_action,cfg.latent_dim,cfg,rng,normalized=True)
        self.cost = mlp(cfg.latent_dim+n_action,cfg.num_bins,cfg,rng,zero_output=True)
        self.actor = mlp(cfg.latent_dim,2*n_action,cfg,rng)
        self.critics = nn.ModuleList([mlp(cfg.latent_dim+n_action,cfg.num_bins,cfg,rng,zero_output=True)
                                      for _ in range(cfg.num_critics)])
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)

    def model_parameters(self):
        return [p for module in (self.encoder,self.dynamics,self.cost,self.critics) for p in module.parameters()]


class Replay:
    """Ring buffer; sampled multi-step sequences never cross a reset or overwrite."""
    def __init__(self, capacity, n_state, n_action):
        self.capacity=capacity; self.total=0
        self.x=np.empty((capacity,n_state)); self.xp=np.empty_like(self.x)
        self.u=np.empty((capacity,n_action)); self.cost=np.empty(capacity)
        self.episode=np.empty(capacity,dtype=np.int64)

    def add(self, x, u, cost, xp, episode):
        i=self.total%self.capacity
        self.x[i],self.u[i],self.cost[i],self.xp[i],self.episode[i]=x,u,cost,xp,episode
        self.total+=1

    def sample(self, rng, batch, horizon):
        starts=np.arange(max(0,self.total-self.capacity),self.total-horizon+1)
        starts=starts[self.episode[starts%self.capacity]==self.episode[(starts+horizon-1)%self.capacity]]
        if not len(starts):
            raise ValueError('No complete within-episode replay sequences')
        indices=(rng.choice(starts,size=batch)[:,None]+np.arange(horizon))%self.capacity
        x=np.concatenate([self.x[indices[:,:1]],self.xp[indices]],axis=1)
        return tuple(torch.tensor(a,dtype=DTYPE) for a in (x,self.u[indices],self.cost[indices]))


class Learner:
    def __init__(self, n_state, n_action, gamma, seed, cfg, bound=None):
        cfg.validate(); self.cfg=cfg; self.gamma=gamma; self.bound=bound; self.n_action=n_action
        streams=np.random.SeedSequence(int(seed)).spawn(4)
        self.init_rng,self.collect_rng,self.replay_rng,self.update_rng=[np.random.default_rng(s) for s in streams]
        self.device=torch.device(cfg.device)
        self.model=WorldModel(n_state,n_action,cfg,self.init_rng).to(self.device)
        self.model_opt=torch.optim.Adam(self.model.model_parameters(),lr=cfg.lr)
        self.actor_opt=torch.optim.Adam(self.model.actor.parameters(),lr=cfg.lr)
        self.mean=np.zeros((cfg.horizon,n_action)); self.scale=1.; self.updates=0

    def prior(self,z,rng):
        mean,raw_std=self.model.actor(z).chunk(2,-1)
        log_std=-5+7*torch.sigmoid(raw_std)
        eps=torch.tensor(rng.standard_normal(mean.shape),dtype=DTYPE,device=self.device)
        raw=mean+log_std.exp()*eps
        logp=(-.5*eps.square()-log_std-.5*np.log(2*np.pi)).sum(-1)
        if self.bound is None:
            return raw,logp
        squashed=raw.tanh()
        logp=logp-(2*(np.log(2)-raw-F.softplus(-2*raw))).sum(-1)-self.n_action*np.log(self.bound)
        return self.bound*squashed,logp

    def q(self,z,u,target=False):
        features=torch.cat([z,u],-1)
        modules=self.model.target_critics if target else self.model.critics
        return torch.stack([decode(net(features),self.cfg.value_limit) for net in modules])

    def update(self,x,u,cost):
        c=self.cfg; H=c.horizon
        x,u,cost=(t.to(self.device) for t in (x,u,cost))
        with torch.no_grad():
            next_z=self.model.encoder(x[:,1:])
            next_u,_=self.prior(next_z,self.update_rng)
            pair=self.update_rng.choice(c.num_critics,size=2,replace=False)
            targets=cost+self.gamma*self.q(next_z,next_u,target=True)[pair].max(0).values
        z=self.model.encoder(x[:,0]); zs=[z]; loss=torch.zeros((),dtype=DTYPE,device=self.device)
        for h in range(H):
            features=torch.cat([z,u[:,h]],-1)
            prediction=self.model.dynamics(features)
            consistency=F.mse_loss(prediction,next_z[:,h])
            cost_loss=soft_ce(self.model.cost(features),cost[:,h],c).mean()
            value_loss=torch.stack([soft_ce(net(features),targets[:,h],c).mean() for net in self.model.critics]).mean()
            loss=loss+c.rho**h*(c.consistency_weight*consistency+c.cost_weight*cost_loss+c.value_weight*value_loss)/H
            z=prediction; zs.append(z)
        self.model_opt.zero_grad(); loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.model_parameters()):
            raise FloatingPointError('Nonfinite neural model/critic update')
        nn.utils.clip_grad_norm_(self.model.model_parameters(),c.grad_clip); self.model_opt.step()
        z=torch.stack(zs,dim=1).detach()
        self.model.critics.requires_grad_(False)
        try:
            action,logp=self.prior(z,self.update_rng)
            q=self.q(z,action).mean(0)
            with torch.no_grad():
                quantiles=torch.quantile(q[:,0],torch.tensor([.05,.95],dtype=DTYPE,device=self.device))
                self.scale=.99*self.scale+.01*max(float(quantiles[1]-quantiles[0]),1.)
            weights=torch.tensor(c.rho**np.arange(H+1),dtype=DTYPE,device=self.device)
            actor_loss=((q/self.scale+c.entropy*logp)*weights).mean()
            self.actor_opt.zero_grad(); actor_loss.backward()
            if not torch.isfinite(actor_loss) or any(not torch.isfinite(p.grad).all() for p in self.model.actor.parameters()):
                raise FloatingPointError('Nonfinite neural actor update')
            nn.utils.clip_grad_norm_(self.model.actor.parameters(),c.grad_clip); self.actor_opt.step()
        finally:
            self.model.critics.requires_grad_(True)
        with torch.no_grad():
            for target,online in zip(self.model.target_critics.parameters(),self.model.critics.parameters()):
                target.lerp_(online,c.tau)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError('Nonfinite neural parameters')
        self.updates+=1
        return float(loss.detach()),float(actor_loss.detach())

    @torch.no_grad()
    def plan(self, x, rng=None, training=False, reset=False):
        c=self.cfg; rng=self.collect_rng if rng is None else rng
        if reset: self.mean.fill(0)
        mean=np.zeros_like(self.mean); mean[:-1]=self.mean[1:]
        std=np.full_like(mean,c.max_std)
        initial=self.model.encoder(torch.tensor(x,dtype=DTYPE,device=self.device).reshape(1,-1))
        prior=np.empty((c.prior_candidates,c.horizon,self.n_action))
        z=initial.expand(c.prior_candidates,-1)
        for h in range(c.horizon):
            u,_=self.prior(z,rng); prior[:,h]=u.cpu().numpy()
            z=self.model.dynamics(torch.cat([z,u],-1))
        for _ in range(c.iterations):
            actions=mean[None]+std[None]*rng.standard_normal((c.candidates,c.horizon,self.n_action))
            if self.bound is not None: actions=np.clip(actions,-self.bound,self.bound)
            actions[:c.prior_candidates]=prior
            z=initial.expand(c.candidates,-1); score=torch.zeros(c.candidates,dtype=DTYPE,device=self.device)
            for h in range(c.horizon):
                u=torch.tensor(actions[:,h],dtype=DTYPE,device=self.device); features=torch.cat([z,u],-1)
                score+=self.gamma**h*decode(self.model.cost(features),c.value_limit)
                z=self.model.dynamics(features)
            u,_=self.prior(z,rng); score+=self.gamma**c.horizon*self.q(z,u).mean(0)
            score=score.cpu().numpy()
            if not np.isfinite(score).all(): raise FloatingPointError('Nonfinite planner scores')
            indices=np.argsort(score)[:c.elites]; elite=actions[indices]
            weights=np.exp(-c.temperature*(score[indices]-score[indices].min())); weights/=weights.sum()
            mean=(weights[:,None,None]*elite).sum(0)
            std=np.clip(np.sqrt((weights[:,None,None]*(elite-mean)**2).sum(0)),c.min_std,c.max_std)
        action=elite[rng.choice(c.elites,p=weights),0].copy()
        if training: action+=std[0]*rng.standard_normal(self.n_action)
        self.mean[:]=mean
        return np.clip(action,-self.bound,self.bound) if self.bound is not None else action


def evaluate(learner,problem,cfg):
    """Fixed independent planner streams; average costs before transforming to gap."""
    warm=learner.mean.copy(); costs=[]; terminal=[]; failures=[]
    try:
        for stream in range(cfg.eval_streams):
            rng=np.random.default_rng(910000+stream); learner.mean.fill(0)
            def policy(x):
                return learner.plan(x[0],rng=rng)[None,:]
            result=envs.Env(problem,1).evaluate(policy,cfg.eval_horizon)
            costs.append(result['J'][0]); terminal.append(np.abs(result['final_state']).max())
            failures.append(result['diverged'][0])
    finally:
        learner.mean[:]=warm
    mean_cost=float(np.mean(costs))
    if mean_cost < problem.J_star*(1-1e-6):
        raise ValueError('Evaluation cost below exact optimum; increase eval_horizon')
    return max(0.,1-problem.J_star/mean_cost),float(np.mean(terminal)),bool(np.any(failures))


def train(problem,seeds,cfg):
    cfg.validate(); rows=[]
    for seed in seeds:
        learner=Learner(problem.n_state,problem.n_control,problem.gamma,seed,cfg,
                        bound=getattr(problem,'action_bound',None))
        replay=Replay(min(cfg.replay_capacity,cfg.total_steps),problem.n_state,problem.n_control)
        checkpoints=set(np.linspace(0,cfg.total_steps,cfg.n_checkpoints+1,dtype=int))
        x=np.array(problem.x0,copy=True); episode=0; alive=True
        reward_window=RewardWindow(1); history=[]
        def score(step):
            reward,count=reward_window.flush()
            if alive:
                try: gap,final,diverged=evaluate(learner,problem,cfg)
                except FloatingPointError: gap,final,diverged=1.,float('inf'),True
            else: gap,final,diverged=1.,float('inf'),True
            history.append(dict(step=step,gaps=gap,final_abs_state=final,diverged=not alive,
                evaluation_diverged=diverged,training_stage_reward=reward[0],training_reward_count=count[0],
                updates=learner.updates))
        score(0)
        for step in range(1,cfg.total_steps+1):
            if alive:
                try:
                    if step<=cfg.warmup:
                        bound=learner.bound or cfg.max_std
                        action=learner.collect_rng.uniform(-bound,bound,problem.n_control)
                    else: action=learner.plan(x,training=True,reset=(step-1)%cfg.episode_length==0)
                    cost=float(problem.stage_cost(x[None],action[None])[0])
                    xp=problem.dynamics(x[None],action[None])[0]
                    if not np.isfinite(cost) or not np.isfinite(xp).all() or np.max(np.abs(xp))>envs.BLOWUP:
                        raise FloatingPointError('Training rollout diverged')
                    replay.add(x,action,cost,xp,episode); x=xp
                    reward_window.add(np.array([[-cost]]))
                    if step>=cfg.warmup and (step-cfg.warmup)%cfg.update_every==0:
                        learner.update(*replay.sample(learner.replay_rng,cfg.batch_size,cfg.horizon))
                except FloatingPointError:
                    alive=False
                    # Mark the current reward window failed, without discarding the seed.
                    reward_window.total[:]=-np.inf
                    if reward_window.count < step-(history[-1]['step']):
                        reward_window.add(np.array([[-np.inf]]))
            else: reward_window.add(np.array([[-np.inf]]))
            if step%cfg.episode_length==0:
                x=np.array(problem.x0,copy=True); episode+=1
            if step in checkpoints: score(step)
        arrays={key:np.asarray([r[key] for r in history]) for key in history[0] if key!='step'}
        arrays['steps']=np.asarray([r['step'] for r in history])
        arrays['model_parameters']=torch.cat([p.detach().ravel() for p in learner.model.parameters()]).cpu().numpy()
        rows.append(arrays)
    result={key:np.stack([r[key] for r in rows]) for key in rows[0] if key!='steps'}
    result['steps']=rows[0]['steps']; result['config']=asdict(cfg)
    result['evaluation_protocol']=dict(planner_streams=list(range(910000,910000+cfg.eval_streams)),
        score='gap of mean cost over fixed planner streams',action_bound=getattr(problem,'action_bound',None),
        neural_components=['encoder','dynamics','cost','actor','critics'],
        implementation='Neural benchmark variant; tanh MLPs, problem discount, unsquashed proposals on unbounded tasks')
    return result
