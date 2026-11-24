import json
import numpy as np
import pytest
from utils.learning_curves import LearningCurves
from utils.benchmark_report import problem_report


def run(name, steps=(0, 10), gaps=((1., .2), (.5, .4)), seeds=(9, 3), **params):
    curve = LearningCurves(np.array(steps), np.array(gaps), name=name)
    return curve, {'seeds': np.array(seeds)}, dict(problem='nlqr_energy', gamma=.99,
        J_star=2., problem_params=params)


def test_report_seed_selection_averaging_and_shared_limits(tmp_path):
    report = problem_report([run('ppo'), run('custom_algorithm', steps=(0, 8, 12),
        gaps=((1., .6, .1), (.8, .8, .3)))], tmp_path)
    assert report['algorithms']['ppo']['single_seed'] == 3
    assert report['algorithms']['ppo']['running_mean'] == pytest.approx([.4, .3])
    assert report['algorithms']['custom_algorithm']['averaging_checkpoint'] == 8
    assert report['algorithms']['custom_algorithm']['running_mean'] == pytest.approx([.8, .7])
    assert report['training_y_limits'][0] <= .1
    assert report['training_y_limits'][1] >= 1
    assert len(report['figures']) == 4
    for f in report['figures']:
        assert (tmp_path/(f['stem']+'.pdf')).exists()
    assert 'custom_algorithm' in (tmp_path/'index.html').read_text()
    assert json.loads((tmp_path/'report.json').read_text())['schema_version'] == 1
    assert 'PPO neural variant' in report['algorithm_roles']['ppo']


def test_primary_defaults_and_optional_linear_baseline():
    import inspect
    from main import build_parser
    from rollouts import DEFAULT_BENCHMARK_ALGOS
    from utils.suite import run_suite
    parser = build_parser()
    for command in ('suite', 'crude'):
        assert tuple(parser.parse_args([command]).algos) == DEFAULT_BENCHMARK_ALGOS
        assert parser.parse_args([command, '--algos', 'ppo', 'monte_carlo', 'ppo_linear']).algos == [
            'ppo', 'monte_carlo', 'ppo_linear']
    assert inspect.signature(run_suite).parameters['algorithms'].default == DEFAULT_BENCHMARK_ALGOS


def test_no_pooling_different_problems(tmp_path):
    with pytest.raises(ValueError, match='one fixed problem'):
        problem_report([run('ppo', beta=.2), run('ppo_linear', beta=.3)], tmp_path)


def test_single_seed_zero_and_failed_curves(tmp_path):
    report = problem_report([run('ppo', gaps=((0., 0.),), seeds=(3,)),
                             run('ppo_linear', gaps=((1., 1.),), seeds=(3,))], tmp_path)
    assert report['training_y_limits'][0] == 0
    assert report['algorithms']['ppo']['final_se_gap'] == 0
    assert report['algorithms']['ppo_linear']['final_gap_one_fraction'] == 1
