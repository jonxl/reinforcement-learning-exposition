"""Experimental bounded scalar TD-MPC2-LQ; not standard neural TD-MPC2.

Cost convention: pessimistic max target, optimistic mean in planning.
All learners use observations only; this module contains no plant/Riccati solver.
"""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Config:
    horizon:int=3
    candidates:int=64
    elites:int=8
    prior_candidates:int=8
    iterations:int=3
    batch:int=64
    update_every:int=16
    lr:float=3e-4
    tau:float=.01
    rho:float=.5
    gamma:float=.99
    bound:float=2.
    entropy:float=1e-4

def features(x,u):
    return np.stack([x*x,2*x*u,u*u,np.ones_like(x)],axis=-1)

def values(w,x,u):return np.einsum('mk,mbk->mb',w,features(x,u))

def model_loss_grad(p,x,u,c,targets,cfg):
    """Joint loss, exact reverse-mode derivatives through H model predictions.

    p=(M,16), x=(M,B,H+1), u,c,targets=(M,B,H).
    Targets fixed by caller. Actor columns have zero gradient here.
    """
    M,B,H=u.shape; z=np.empty_like(x); z[:,:,0]=x[:,:,0]
    for h in range(H):z[:,:,h+1]=p[:,0,None]*z[:,:,h]+p[:,1,None]*u[:,:,h]
    g=np.zeros_like(p); dz=np.zeros_like(z); loss=np.zeros(M)
    for h in range(H):
        a=z[:,:,h];b=u[:,:,h];phi=features(a,b);w=cfg.rho**h/(B*H)
        em=z[:,:,h+1]-x[:,:,h+1]
        ec=values(p[:,2:6],a,b)-c[:,:,h]
        loss+=w*(20*em**2+.1*ec**2).sum(1)
        dz[:,:,h+1]+=40*w*em
        g[:,2:6]+=.2*w*np.einsum('mb,mbk->mk',ec,phi)
        dz[:,:,h]+=.2*w*ec*(2*p[:,2,None]*a+2*p[:,3,None]*b)
        for start in (6,10):
            ev=values(p[:,start:start+4],a,b)-targets[:,:,h]
            loss+=.05*w*(ev**2).sum(1)
            g[:,start:start+4]+=.1*w*np.einsum('mb,mbk->mk',ev,phi)
            dz[:,:,h]+=.1*w*ev*(2*p[:,start,None]*a+2*p[:,start+1,None]*b)
    for h in range(H-1,-1,-1):
        adj=dz[:,:,h+1]
        g[:,0]+=(adj*z[:,:,h]).sum(1);g[:,1]+=(adj*u[:,:,h]).sum(1)
        dz[:,:,h]+=p[:,0,None]*adj
    return loss,g,z

def actor_loss_grad(p,z,eps,scale,cfg):
    # Entropy is that of the latent proposal, not of the clipped physical action.
    M,B,H=z.shape;sigma=np.exp(p[:,15,None,None]);raw=-p[:,14,None,None]*z+sigma*eps
    u=np.clip(raw,-cfg.bound,cfg.bound);mask=np.abs(raw)<cfg.bound
    w=(cfg.rho**np.arange(H))[None,None,:]/(B*H)
    q=.5*(p[:,6:10]+p[:,10:14]);phi=features(z,u)
    cost=np.einsum('mk,mbhk->mbh',q,phi)
    loss=(w*cost/scale[:,None,None]).sum((1,2))-cfg.entropy*(p[:,15]+.5*np.log(2*np.pi*np.e))*(cfg.rho**np.arange(H)).mean()
    du=(2*q[:,1,None,None]*z+2*q[:,2,None,None]*u)/scale[:,None,None]*mask
    g=np.zeros_like(p);g[:,14]=(w*du*(-z)).sum((1,2))
    g[:,15]=(w*du*sigma*eps).sum((1,2))-cfg.entropy*(cfg.rho**np.arange(H)).mean()
    return loss,g

class Learner:
    def __init__(self,seeds,cfg=Config()):
        self.cfg=cfg;self.seeds=np.asarray(seeds);M=len(seeds)
        self.rng=[]
        for seed in seeds:self.rng.append([np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(4)])
        self.p=np.zeros((M,16));self.p[:,2:14]=np.stack([r[0].normal(0,.01,12) for r in self.rng]);self.p[:,15]=np.log(.5)
        self.target=self.p[:,6:14].copy();self.mean=np.zeros((M,cfg.horizon));self.scale=np.ones(M)
        self.m=np.zeros_like(self.p);self.v=np.zeros_like(self.p);self.t=[0,0]

    def noise(self,stream,shape):return np.stack([r[stream].standard_normal(shape) for r in self.rng])

    def prior(self,x,noise):
        return np.clip(-self.p[:,14,None]*x+np.exp(self.p[:,15,None])*noise,-self.cfg.bound,self.cfg.bound)

    def plan(self,x,rngs=None,reset=False,training=False):
        c=self.cfg;M=len(x);rg=rngs if rngs is not None else [r[1] for r in self.rng]
        normal=lambda shape:np.stack([r.standard_normal(shape) for r in rg])
        if reset:self.mean[:]=0
        mean=np.zeros_like(self.mean);mean[:,:-1]=self.mean[:,1:];std=np.full_like(mean,2.)
        prior=np.empty((M,c.prior_candidates,c.horizon));z=np.broadcast_to(x[:,None],(M,c.prior_candidates)).copy()
        for h in range(c.horizon):
            prior[:,:,h]=self.prior(z,normal((c.prior_candidates,)))
            z=self.p[:,0,None]*z+self.p[:,1,None]*prior[:,:,h]
        for _ in range(c.iterations):
            actions=np.clip(mean[:,None,:]+std[:,None,:]*normal((c.candidates,c.horizon)),-c.bound,c.bound)
            actions[:,:c.prior_candidates]=prior
            z=np.broadcast_to(x[:,None],(M,c.candidates)).copy();score=np.zeros_like(z)
            for h in range(c.horizon):
                a=actions[:,:,h];score+=c.gamma**h*values(self.p[:,2:6],z,a)
                z=self.p[:,0,None]*z+self.p[:,1,None]*a
            a=self.prior(z,normal((c.candidates,)))
            score+=c.gamma**c.horizon*values(.5*(self.p[:,6:10]+self.p[:,10:14]),z,a)
            idx=np.argsort(score,axis=1)[:,:c.elites]
            elite=np.take_along_axis(actions,idx[:,:,None],axis=1);s=np.take_along_axis(score,idx,axis=1)
            weights=np.exp(-.5*(s-s.min(1,keepdims=True)));weights/=weights.sum(1,keepdims=True)
            mean=(elite*weights[:,:,None]).sum(1)
            std=np.sqrt(((elite-mean[:,None,:])**2*weights[:,:,None]).sum(1));std=np.clip(std,.05,2.)
        chosen=np.array([r.choice(c.elites,p=w) for r,w in zip(rg,weights)])
        a=elite[np.arange(M),chosen,0]
        if training:a=a+std[:,0]*normal(())
        self.mean[:]=mean
        return np.clip(a,-c.bound,c.bound)

    def adam(self,g,actor=False):
        c=self.cfg;sl=slice(14,16) if actor else slice(0,14);i=int(actor);self.t[i]+=1
        g=g[:,sl];norm=np.linalg.norm(g,axis=1);g*=np.minimum(1,20/(norm+1e-12))[:,None]
        self.m[:,sl]=.9*self.m[:,sl]+.1*g;self.v[:,sl]=.999*self.v[:,sl]+.001*g*g
        self.p[:,sl]-=c.lr*(self.m[:,sl]/(1-.9**self.t[i]))/(np.sqrt(self.v[:,sl]/(1-.999**self.t[i]))+1e-8)

    def update(self,x,u,cost):
        cfg=self.cfg;M,B,H=u.shape
        targets=np.empty_like(cost)
        for h in range(H):
            xp=x[:,:,h+1];up=self.prior(xp,self.noise(2,(B,)))
            targets[:,:,h]=cost[:,:,h]+cfg.gamma*np.maximum(values(self.target[:,:4],xp,up),values(self.target[:,4:],xp,up))
        loss,g,z=model_loss_grad(self.p,x,u,cost,targets,cfg);self.adam(g)
        eps=self.noise(2,(B,H+1));zs=z
        probe=self.prior(zs[:,:,0],eps[:,:,0]);q=values(.5*(self.p[:,6:10]+self.p[:,10:14]),zs[:,:,0],probe)
        spread=np.maximum(np.percentile(q,95,axis=1)-np.percentile(q,5,axis=1),1.)
        self.scale=.99*self.scale+.01*spread
        _,g=actor_loss_grad(self.p,zs,eps,self.scale,cfg);self.adam(g,actor=True)
        self.target=(1-cfg.tau)*self.target+cfg.tau*self.p[:,6:14]
        return loss
