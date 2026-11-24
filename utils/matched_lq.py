"""Reuse the common runner/report for the fixed bounded scalar comparison."""
import json
from pathlib import Path
import numpy as np
import rollouts
from algos.families import canonical_algorithms
from utils.bounded_lqr import BoundedScalarLQR
from utils.benchmark_report import suite_report

ALGORITHMS = ('ppo-nn', 'monte_carlo', 'td-mpc2-lg')
PROTOCOL = dict(action_bound=2., gamma=.99, x0=1., dynamics='x_next=x+u',
    cost='x^2+u^2', evaluation='upper infinite-horizon gap bound; tail <= 1e-6',
    planner_streams=[910000, 910001, 910002],
    interpretation='Equal physical interaction budgets; optimizer work and reset schedules differ. '
                   'LG uses a scalar structured model; NN uses neural models and critics.')


def manifest(algorithms):
    problem = BoundedScalarLQR()
    return dict(profile='matched-bounded-lqr', algorithms=list(algorithms), protocol=PROTOCOL,
        cases=[dict(id='lqr_bounded', problem='lqr_bounded', problem_params={},
                    J_star=problem.J_star, P=problem.P.tolist(), K_linearization=problem.K_star.tolist())])


def run_matched(out, algorithms=ALGORITHMS, seeds=3, steps=10000, checkpoints=5,
                workers=1, master_seed=0, device="cpu", seed_batch_size=None):
    algorithms = canonical_algorithms(algorithms)
    if min(seeds, steps, checkpoints, workers) < 1 or steps % 2000:
        raise ValueError('Positive counts required; steps must be a multiple of 2000 for PPO rollouts')
    if not algorithms or len(set(algorithms)) != len(algorithms) or set(algorithms)-set((*ALGORITHMS, 'ppo-lg', 'td-mpc2-nn')):
        raise ValueError('Choose distinct variants from ppo-nn, monte_carlo, td-mpc2-lg, td-mpc2-nn, ppo-lg')
    if 'td-mpc2-lg' in algorithms:
        import algos.lq_planner  # fail before training if optional dependencies are missing
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    info = manifest(algorithms)
    info.update(seeds=seeds, steps_per_seed=steps, master_seed=master_seed, checkpoints=checkpoints)
    (out/'problems.json').write_text(json.dumps(info, indent=2))
    for algorithm in algorithms:
        config = dict(total_steps=steps, n_checkpoints=checkpoints)
        if algorithm in ('ppo-nn', 'ppo-lg'):
            config.update(horizon=250, n_envs=8, epochs=10)
            if algorithm == 'ppo-nn':
                config['minibatch_size'] = 500
        rollouts.run(algorithm, 'lqr_bounded', seeds, master_seed, .99, workers,
                     config, out_dir=out/'lqr_bounded'/algorithm, device=device, seed_batch_size=seed_batch_size)
    return suite_report(out)


def import_published(source, out=None):
    """Convert saved published metrics, preserving seeds/protocol; no retraining."""
    source = Path(source)
    out = Path(out) if out else source/'unified'
    mapping = {'ppo': 'ppo-lg-published', 'td': 'td-mpc2-lg'}
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
