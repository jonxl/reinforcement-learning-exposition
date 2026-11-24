"""Reuse the common runner/report for the fixed bounded scalar comparison."""
import json
from pathlib import Path
import numpy as np
import rollouts
from utils.bounded_lqr import BoundedScalarLQR
from utils.benchmark_report import suite_report

ALGORITHMS = ('ppo', 'monte_carlo', 'tdmpc2_lq')
PROTOCOL = dict(action_bound=2., gamma=.99, x0=1., dynamics='x_next=x+u',
    cost='x^2+u^2', evaluation='upper infinite-horizon gap bound; tail <= 1e-6',
    planner_streams=[910000, 910001, 910002],
    interpretation='Equal physical interaction budgets; optimizer work and reset schedules differ. '
                   'TD is the scalar structural adaptation, not neural TD-MPC2.')


def manifest(algorithms):
    problem = BoundedScalarLQR()
    return dict(profile='matched-bounded-lqr', algorithms=list(algorithms), protocol=PROTOCOL,
        cases=[dict(id='lqr_bounded', problem='lqr_bounded', problem_params={},
                    J_star=problem.J_star, P=problem.P.tolist(), K_linearization=problem.K_star.tolist())])


def run_matched(out, algorithms=ALGORITHMS, seeds=3, steps=10000, checkpoints=5,
                workers=1, master_seed=0):
    if min(seeds, steps, checkpoints, workers) < 1 or steps % 2000:
        raise ValueError('Positive counts required; steps must be a multiple of 2000 for PPO rollouts')
    if not algorithms or len(set(algorithms)) != len(algorithms) or set(algorithms)-set((*ALGORITHMS, 'ppo_linear')):
        raise ValueError('Choose distinct algorithms from ppo, monte_carlo, tdmpc2_lq, ppo_linear')
    if 'tdmpc2_lq' in algorithms:
        import algos.lq_planner  # fail before training if optional dependencies are missing
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    info = manifest(algorithms)
    info.update(seeds=seeds, steps_per_seed=steps, master_seed=master_seed, checkpoints=checkpoints)
    (out/'problems.json').write_text(json.dumps(info, indent=2))
    for algorithm in algorithms:
        config = dict(total_steps=steps, n_checkpoints=checkpoints)
        if algorithm in ('ppo', 'ppo_linear'):
            config.update(horizon=250, n_envs=8, epochs=10)
            if algorithm == 'ppo':
                config['minibatch_size'] = 500
        rollouts.run(algorithm, 'lqr_bounded', seeds, master_seed, .99, workers,
                     config, out_dir=out/'lqr_bounded'/algorithm)
    return suite_report(out)


def import_published(source, out=None):
    """Convert saved published metrics, preserving seeds/protocol; no retraining."""
    source = Path(source)
    out = Path(out) if out else source/'unified'
    mapping = {'ppo': 'ppo_linear_legacy', 'td': 'tdmpc2_lq'}
    info = manifest(mapping.values())
    info['source'] = str(source.resolve())
    out.mkdir(parents=True, exist_ok=True)
    for original, algorithm in mapping.items():
        saved = json.loads((source/original/'manifest.json').read_text())
        result = json.loads((source/original/'result.json').read_text())
        if saved['action_bound'] != 2 or saved['gamma'] != .99:
            raise ValueError('Published results do not match the fixed bounded protocol')
        history = result['history']
        stack = lambda key: np.asarray([r[key] for r in history]).T
        arrays = dict(steps=np.array([r['step'] for r in history]), seeds=np.array(saved['seeds']),
                      gaps=stack('gap_upper'), gap_lower=stack('gap_lower'),
                      cost_lower=stack('cost_lower'), cost_upper=stack('cost_upper'),
                      diverged=~stack('alive').astype(bool))
        meta = dict(algo=algorithm, problem='lqr_bounded', problem_params={}, gamma=.99,
                    J_star=BoundedScalarLQR().J_star, K_star=BoundedScalarLQR().K_star.ravel().tolist(),
                    curvature=None, n_seeds=len(saved['seeds']), config=result.get('config', saved.get('td_config', {})),
                    problem_specification=BoundedScalarLQR.benchmark_specification,
                    source_manifest=saved, evaluation_protocol=PROTOCOL, wall_seconds=result['elapsed'])
        folder = out/'lqr_bounded'/algorithm
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'arrays.npz', **arrays)
        (folder/'meta.json').write_text(json.dumps(meta, indent=2))
    (out/'problems.json').write_text(json.dumps(info, indent=2))
    return suite_report(out)
