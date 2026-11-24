"""Runner adapter for the colleague's bounded scalar TD-MPC2-LQ implementation.

This is not neural TD-MPC2 and does not support the generated two-state NLQR family.
The original learner, planner, replay, warmup, and certified evaluator are reused.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from algos.tdmpc2_lq import Config as PlannerConfig
from utils.bounded_lqr import BoundedScalarLQR

NAME = 'td-mpc2-lg'
SUPPORTED_PROBLEMS = ('lqr_bounded',)

@dataclass
class Config(PlannerConfig):
    total_steps: int = 1_000_000
    n_checkpoints: int = 20


def train(problem, seeds, cfg):
    if not isinstance(problem, BoundedScalarLQR):
        raise ValueError('TD-MPC2-LQ supports only lqr_bounded; use main.py matched-lq')
    if cfg.gamma != problem.gamma or cfg.bound != problem.action_bound:
        raise ValueError('Planner discount and action bound must match the fixed protocol')
    if cfg.total_steps < 1 or cfg.n_checkpoints < 1:
        raise ValueError('Positive training steps and checkpoints required')
    if not (1 <= cfg.horizon <= 500 and 1 <= cfg.elites <= cfg.candidates and
            0 <= cfg.prior_candidates <= cfg.candidates and cfg.update_every > 0 and
            cfg.batch > 0 and cfg.iterations > 0):
        raise ValueError('Invalid planner/replay configuration')
    from experiments.lq_comparison import run_td, EVAL_SEEDS
    with TemporaryDirectory(prefix='tdmpc2-benchmark-') as temporary:
        folder = Path(temporary)
        result = run_td(seeds, cfg.total_steps, folder, cfg, cfg.n_checkpoints)
        history = result['history']
        stack = lambda key: np.stack([row[key] for row in history], axis=1)
        out = dict(steps=np.array([row['step'] for row in history]),
                   gaps=stack('gap_upper'), gap_lower=stack('gap_lower'),
                   cost_lower=stack('cost_lower'), cost_upper=stack('cost_upper'),
                   gains=stack('effective_gain')[..., None],
                   final_abs_state=np.abs(stack('final_state')),
                   diverged=~stack('alive'),
                   training_stage_reward=stack('training_stage_reward'),
                   training_reward_count=stack('training_reward_count'),
                   config=asdict(cfg),
                   evaluation_protocol=dict(action_bound=2., tail_tolerance=1e-6,
                       planner_streams=list(EVAL_SEEDS), score='upper gap bound; mean cost over fixed planner streams'),
                   failure_records=result['failures'])
        weights = []
        for row in history:
            with np.load(folder/f"checkpoint_{row['step']:07d}.npz") as checkpoint:
                weights.append(checkpoint['p'])
        out['parameters'] = np.stack(weights, axis=1)
    return out
