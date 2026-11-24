"""Build a reproducible analysis after both full runs have completed."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from algos.tdmpc2_lq import Learner
from algos.lq_planner import plan
from utils.baseline import scalar_lqr
from experiments.lq_comparison import write

ROOT=Path('results/lq_comparison')
OUT=Path('output/latex/ppo-tdmpc2-lq-comparison')
FIG=OUT/'figures'
LABELS={'ppo':'Bounded linear PPO','td':'TD-MPC2-LQ (fast)'}
COLORS={'ppo':'#2066a8','td':'#bd4c26'}


def save(name):
    plt.tight_layout();plt.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    plt.savefig(FIG/f'{name}.png',dpi=160,bbox_inches='tight');plt.close()


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    runs={m:json.loads((ROOT/m/'result.json').read_text()) for m in ('ppo','td')}
    manifests={m:json.loads((ROOT/m/'manifest.json').read_text()) for m in runs}
    assert manifests['ppo']['seeds']==manifests['td']['seeds']
    n=len(manifests['td']['seeds']);assert n==40
    gaps={m:np.array([r['gap_upper'] for r in runs[m]['history']]).T for m in runs}
    lo={m:np.array([r['gap_lower'] for r in runs[m]['history']]).T for m in runs}
    steps=np.array([r['step'] for r in runs['td']['history']]);assert np.array_equal(steps,np.arange(0,1000001,50000))
    summaries={}
    for m in runs:
        assert [r['step'] for r in runs[m]['history']]==steps.tolist()
        g=gaps[m][:,-1];last=runs[m]['history'][-1]
        summaries[m]=dict(mean=float(g.mean()),se=float(g.std(ddof=1)/np.sqrt(n)),median=float(np.median(g)),
            minimum=float(g.min()),maximum=float(g.max()),success=int((g<=.1).sum()),
            mean_cost=float(np.mean(last['cost_upper'])),failures=int(np.sum(~np.asarray(last['alive'],bool))),
            elapsed=runs[m]['elapsed'],max_gap_width=float(np.max(gaps[m]-lo[m])),
            max_tail=float(max(np.max(r['tail']) for r in runs[m]['history'])))
        means=gaps[m].mean(0);best=int(np.argmin(means))
        summaries[m].update(best_mean_gap=float(means[best]),best_step=int(steps[best]),
                            mean_at_100k=float(means[2]),mean_at_500k=float(means[10]))
    d=gaps['td'][:,-1]-gaps['ppo'][:,-1];rng=np.random.default_rng(20261123)
    boot=d[rng.integers(0,n,size=(50000,n))].mean(1)
    # Uniform allowance for both score errors, including unsampled seeds.
    h=float(np.sqrt(2*np.log(40)/n));error=2e-6
    comparison=dict(mean_difference=float(d.mean()),paired_se=float(d.std(ddof=1)/np.sqrt(n)),
        bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
        hoeffding_95=[max(-1,float(d.mean()-h-error)),min(1,float(d.mean()+h+error))],
        hoeffding_half_width=h,td_higher_count=int((d>0).sum()),n=n)
    p=np.load(ROOT/'td'/'checkpoint_1000000.npz')['p']
    summaries['td']['median_a']=float(np.median(p[:,0]));summaries['td']['median_b']=float(np.median(p[:,1]))
    summaries['td']['median_sigma']=float(np.median(np.exp(p[:,15])))
    summaries['td']['updates_per_seed']=runs['td']['history'][-1]['updates']
    ablations={}
    for label,folder in [('fast','lq_fast_ablation'),('structure_only','lq_structure_ablation')]:
        r=json.loads((Path('results')/folder/'td'/'result.json').read_text())
        ablations[label]=dict(final_mean_gap=float(np.mean(r['history'][-1]['gap_upper'])),
                              gaps=r['history'][-1]['gap_upper'],elapsed=r['elapsed'],
                              updates=r['history'][-1]['updates'])
    summary=dict(methods=summaries,comparison=comparison,ablations=ablations,
                 j_star=scalar_lqr().J_star,k_star=scalar_lqr().K_star.item())
    write(OUT/'summary.json',summary)
    with (OUT/'checkpoint-summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','steps','mean_gap','sample_se','min_gap','max_gap'])
        for m,g in gaps.items():
            for i,s in enumerate(steps):writer.writerow([m,s,g[:,i].mean(),g[:,i].std(ddof=1)/np.sqrt(n),g[:,i].min(),g[:,i].max()])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.8))
    for m,g in gaps.items():
        mu=g.mean(0);se=g.std(0,ddof=1)/np.sqrt(n)
        for ax in axs:
            ax.plot(steps/1000,mu,label=LABELS[m],color=COLORS[m])
            ax.fill_between(steps/1000,np.maximum(0,mu-1.96*se),np.minimum(1,mu+1.96*se),color=COLORS[m],alpha=.16)
    axs[0].set_ylim(0,1);axs[1].set_ylim(0,min(1.,max(.02,max(v[:,-1].mean() for v in gaps.values())*1.4)))
    for ax in axs:ax.set_xlabel('Training interactions per seed (thousands)');ax.set_ylabel('Mean normalized cost gap');ax.grid(alpha=.2)
    axs[0].legend();axs[0].set_title('Whole scale');axs[1].set_title('Late-performance magnification')
    save('learning_curves')
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.8))
    for ax,m in zip(axs,('ppo','td')):
        ax.plot(steps/1000,gaps[m].T,color=COLORS[m],alpha=.3,linewidth=.7)
        ax.plot(steps/1000,gaps[m].mean(0),color='black',linewidth=1.8)
        ax.set(xlabel='Interactions per seed (thousands)',ylabel='Normalized cost gap',title=LABELS[m],ylim=(0,1));ax.grid(alpha=.2)
    save('individual_runs')
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.8))
    for m,g in gaps.items():
        v=np.sort(g[:,-1]);axs[0].step(v,np.arange(1,n+1)/n,where='post',label=LABELS[m],color=COLORS[m])
        axs[1].plot(np.arange(1,n+1),np.cumsum(g[:,-1])/np.arange(1,n+1),label=LABELS[m],color=COLORS[m])
    axs[0].set(xlabel='Final normalized cost gap',ylabel='Empirical fraction at or below gap',ylim=(0,1.02))
    axs[1].set(xlabel='Number of training draws in fixed order',ylabel='Final-gap prefix average')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    save('final_distribution')
    td_weights=np.stack([np.load(ROOT/'td'/f'checkpoint_{s:07d}.npz')['p'] for s in steps])
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.8))
    for col,label in [(0,'a estimate'),(1,'b estimate')]:
        axs[0].plot(steps/1000,np.median(td_weights[:,:,col],1),label=label)
    axs[0].axhline(1,color='black',ls='--',label='True coefficients (evaluation only)')
    axs[0].set(xlabel='Interactions per seed (thousands)',ylabel='Median model coefficient');axs[0].legend()
    axs[1].plot(steps/1000,np.median(np.exp(td_weights[:,:,15]),1),color=COLORS['td'])
    axs[1].set(xlabel='Interactions per seed (thousands)',ylabel='Median actor proposal standard deviation',yscale='log')
    for ax in axs:ax.grid(alpha=.2)
    save('model_diagnostics')
    # Fixed first training draw and first primary planner stream, not best seed.
    learner=Learner(manifests['td']['seeds'][:1]);learner.p[:]=p[:1];rg=[np.random.default_rng(910000)]
    k=np.load(ROOT/'ppo'/'checkpoint_1000000.npz')['K'][0,0,0]
    xt=xp=xo=1.;trajectory=[]
    for t in range(61):
        trajectory.append([t,xt,xp,xo]);xt+=plan(learner,np.array([xt]),rngs=rg)[0]
        xp+=np.clip(-k*xp,-2,2);xo*=1-summary['k_star']
    v=np.array(trajectory);np.save(OUT/'first-seed-trajectory.npy',v)
    plt.figure(figsize=(9,3.5))
    for i,label,color in [(1,LABELS['td'],COLORS['td']),(2,LABELS['ppo'],COLORS['ppo']),(3,'Exact optimum','black')]:
        plt.plot(v[:,0],v[:,i],label=label,color=color)
    plt.xlabel('Evaluation time step');plt.ylabel('Physical state');plt.legend();plt.grid(alpha=.2);save('trajectory')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
