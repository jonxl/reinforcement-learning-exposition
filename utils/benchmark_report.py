"""Uniform reports for fixed problems, algorithms, and independent training seeds.

Rebuild from saved runs; no training, problem generation, or algorithm-specific plots.
"""
from pathlib import Path
import html
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

from utils.graphs import color, label
from utils.plot_scales import adaptive_y
from algos.families import canonical_name

ALGORITHM_ROLES = {
    'tdmpc2_lq': 'Bounded scalar TD-MPC2-LQ adaptation; not standard neural TD-MPC2. Evaluation averages costs over three fixed planner streams.',
    'ppo_linear_legacy': 'Published bounded linear PPO with quadratic critic; preserved separately from neural PPO.',
    'ppo': 'PPO neural variant: Gaussian neural actor and neural critic.',
    'ppo_linear': 'PPO linear-Gaussian variant: linear-Gaussian actor and quadratic-feature critic.',
    'monte_carlo': 'Tabular Monte Carlo control baseline.',
}
ALGORITHM_ROLES['td-mpc2-nn'] = 'Neural encoder, dynamics, cost model, Gaussian actor prior and distributional critic ensemble; sampling planner.'
ALGORITHM_ROLES['ppo-lg-published'] = ALGORITHM_ROLES['ppo_linear_legacy']
ALGORITHM_ROLES.update({canonical_name(name): role for name, role in list(ALGORITHM_ROLES.items())})


def tex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&',
                    '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def _axis(ax, values, xlabel, limits=None):
    adaptive_y(ax, values, scale='linear')
    if limits is not None:
        ax.set_ylim(limits)
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(0, lo), min(1.02, hi))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set(xlabel=xlabel, ylabel='Optimality gap (%)')
    ax.grid(axis='y', alpha=.2)
    if ax.get_ylim()[0] == 0:
        ax.axhline(0, color='black', ls=':', lw=1)
    ax.text(.02, .02, 'Optimum: 0%' + (' (below view)' if lo > 0 else ''),
            transform=ax.transAxes, fontsize=8)


def problem_report(runs, out, problem_id=None):
    """Use existing (LearningCurves, arrays, metadata) records for one fixed problem."""
    if not runs:
        raise ValueError('At least one algorithm is required')
    identities = {(m['problem'], m['gamma'], json.dumps(m.get('problem_params', {}), sort_keys=True),
                   m['J_star']) for _, _, m in runs}
    if len(identities) != 1:
        raise ValueError('A problem report must contain one fixed problem and optimal cost')
    names = [c.name for c, _, _ in runs]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate algorithms in one problem report')
    for c, arrays, _ in runs:
        if not c.n_seeds or not np.isfinite(c.gaps).all():
            raise ValueError('Expected nonempty finite scored gaps; retain failures as gap one')
        if len(arrays['seeds']) != c.n_seeds or len(set(arrays['seeds'])) != c.n_seeds:
            raise ValueError('Expected one distinct seed identifier per trajectory')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = runs[0][2]
    identity = problem_id or meta['problem']
    figures = []
    def save(fig, stem, caption):
        fig.suptitle(f'{identity} | {caption}', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94))
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{stem}.{ext}', bbox_inches='tight', dpi=150)
        plt.close(fig)
        figures.append(dict(stem=stem, caption=caption))

    # Fit all seed observations once; use these limits for all training views.
    all_gaps = np.concatenate([c.gaps.ravel() for c, _, _ in runs])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _axis(ax, all_gaps, 'Training environment steps')
    limits = ax.get_ylim()
    selected = {}
    for c, arrays, _ in runs:
        index = int(np.argmin(arrays['seeds']))  # deterministic, independent of performance
        seed = int(arrays['seeds'][index])
        selected[c.name] = seed
        ax.plot(c.steps, c.gaps[index], marker='.', color=color(c.name),
                label=f'{label(c.name)} | seed {seed}')
    ax.legend(fontsize=8)
    save(fig, 'single_seed', 'One predetermined seed per algorithm')

    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 3*len(runs)), squeeze=False)
    for ax, (c, _, _) in zip(axes.flat, runs):
        for gap in c.gaps:
            ax.plot(c.steps, gap, color=color(c.name), alpha=.2, lw=.8)
        ax.plot(c.steps, c.mean(), color=color(c.name), lw=2.5, label='Mean learning curve')
        _axis(ax, all_gaps, 'Training environment steps', limits)
        ax.set_title(f'{label(c.name)} | {c.n_seeds} seeds; thin: individual runs; thick: mean')
    save(fig, 'seeds_and_mean', 'All training seeds and their mean learning curves')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c, _, _ in runs:
        ax.plot(c.steps, c.mean(), marker='.', color=color(c.name),
                label=f'{label(c.name)} | final {100*c.mean()[-1]:.2f}%')
    _axis(ax, all_gaps, 'Training environment steps', limits)
    ax.legend(fontsize=8)
    save(fig, 'mean_learning_curve', 'Mean learning curves; same scale as individual runs')

    # One common training budget; latest available checkpoint at or before it.
    budget = min(c.steps[-1] for c, _, _ in runs)
    if budget < max(c.steps[0] for c, _, _ in runs):
        raise ValueError('Algorithms have no overlapping training budgets')
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 3*len(runs)), squeeze=False)
    statistics = {}
    for ax, (c, arrays, m) in zip(axes.flat, runs):
        checkpoint = int(np.searchsorted(c.steps, budget, side='right')-1)
        order = np.argsort(arrays['seeds'])
        values = c.gaps[order, checkpoint]
        counts = np.arange(1, c.n_seeds+1)
        running = np.cumsum(values)/counts
        # Permutations reuse the same finite sample; they are not additional seeds.
        rng = np.random.default_rng(0)
        paths = np.array([np.cumsum(rng.permutation(values))/counts for _ in range(20)])
        for path in paths:
            ax.plot(counts, path, color=color(c.name), alpha=.15, lw=.8)
        ax.plot(counts, running, marker='.', color=color(c.name), lw=2)
        ax.axhline(values.mean(), color='black', ls='--', label='Full-sample mean (not known expectation)')
        _axis(ax, paths, 'Number of training seeds included')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f'{label(c.name)} | fixed checkpoint: {int(c.steps[checkpoint]):,} steps')
        ax.legend(fontsize=8)
        statistics[c.name] = dict(n_seeds=c.n_seeds, single_seed=selected[c.name],
            final_step=int(c.steps[-1]), final_mean_gap=float(c.mean()[-1]),
            final_std_gap=float(c.std()[-1]), final_se_gap=float(c.standard_error()[-1]),
            final_gap_one_fraction=float(np.mean(c.gaps[:, -1] >= 1)),
            averaging_checkpoint=int(c.steps[checkpoint]), running_mean=running.tolist(),
            training_config=m.get('config', {}), seeds=[int(s) for s in arrays['seeds']])
    save(fig, 'seed_averaging', 'Averaging at a fixed training checkpoint')
    report = dict(schema_version=1, problem_id=identity, problem=meta['problem'],
        problem_params=meta.get('problem_params', {}), gamma=meta['gamma'], J_star=meta['J_star'],
        training_y_limits=list(limits), algorithms=statistics, figures=figures,
        interpretation='Seed averaging illustrates finite-sample behavior, not proof of convergence. '
        'Permutations reuse the same seeds and all end at the same empirical mean. '
        'Problems are never pooled. Gap one includes failed evaluations.')
    report['run_metadata'] = {c.name: m for c, _, m in runs}
    report['algorithm_roles'] = {c.name: ALGORITHM_ROLES.get(c.name, 'Additional configured algorithm.')
                               for c, _, _ in runs}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    intro = f"Fixed problem: {identity}; gamma={meta['gamma']}; exact optimal cost J*={meta['J_star']:.6g}."
    if meta.get('problem_specification'):
        spec = meta['problem_specification']
        intro += (f" Applied actions are clipped to +/-{spec['action_bound']}; "
                  f"x0={spec['x0']}. Scores include an omitted-cost bound <= {spec['tail_tolerance']}. "
                  'The optimal reference is valid from this initial state.')
    body = f'<h1>{html.escape(identity)}</h1><p>{html.escape(intro)}</p>'
    body += '<pre>'+html.escape(json.dumps(report['problem_params'], indent=2))+'</pre>'
    body += '<p>'+html.escape(report['interpretation'])+'</p>'
    roles = ' '.join(report['algorithm_roles'].values())
    roles += (' Linear and neural PPO differ in actor, critic, and implementation; '
              'this is not an isolated actor-architecture comparison. '
              'PPO optimality gaps evaluate deterministic mean controllers; Gaussian action noise is used during PPO training. '
              'TD-MPC2-LQ deployment retains planner randomness; its reported gap is conditional on the fixed evaluation streams.')
    body += '<p>'+html.escape(roles)+'</p>'
    latex = [r'\section{'+tex_escape(identity)+'}', tex_escape(intro),
             tex_escape(report['interpretation']), tex_escape(roles), r'\begin{verbatim}',
             json.dumps(report['problem_params'], indent=2), r'\end{verbatim}']
    body += '<table><tr><th>Algorithm</th><th>Seeds</th><th>Final mean gap</th><th>SD</th><th>Gap-one fraction</th></tr>'
    latex += [r'\begin{tabular}{lrrrr} Algorithm & Seeds & Mean gap & SD & Gap-one fraction \\ \hline']
    for name, stat in statistics.items():
        vals = [label(name), str(stat['n_seeds'])]+[f"{100*stat[k]:.2f}%" for k in
                    ('final_mean_gap', 'final_std_gap', 'final_gap_one_fraction')]
        body += '<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>'
        latex += [' & '.join(tex_escape(v) for v in vals)+r' \\']
    body += '</table>'
    latex += [r'\end{tabular}']
    for f in figures:
        body += f'<h2>{html.escape(f["caption"])}</h2><img src="{f["stem"]}.png" style="max-width:100%">'
        latex += [r'\begin{figure}[p]\centering',
                  r'\includegraphics[width=\textwidth,height=.82\textheight,keepaspectratio]{'+
                  (out/f'{f["stem"]}.pdf').resolve().as_posix()+'}',
                  r'\caption{'+tex_escape(identity+': '+f['caption'])+'}', r'\end{figure}']
    latex += [r'\clearpage']
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(identity)+'</title>'+body)
    (out/'report.tex').write_text('\n'.join(latex))
    return report


def suite_report(root, out=None):
    """Render every fixed instance in a saved suite, with its declared algorithms."""
    import rollouts
    root = Path(root)
    out = Path(out) if out else root/'benchmark_report'
    manifest = json.loads((root/'problems.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in manifest['cases']:
        runs = []
        for name in manifest['algorithms']:
            arrays, meta = rollouts.load(root/case['id']/name)
            runs.append((rollouts.to_curves(arrays, meta), arrays, meta))
        reports.append(problem_report(runs, out/case['id'], case['id']))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Fixed-problem benchmark</h1>'
        '<p>Each problem has its own exact reference. Training seeds are averaged within each problem.</p><ul>'+
        ''.join(f'<li><a href="{html.escape(r["problem_id"])}/index.html">{html.escape(r["problem_id"])}'
                f' ({html.escape(r["problem"])})</a></li>' for r in reports)+'</ul>')
    (out/'benchmark.tex').write_text('\n'.join([
        r'\documentclass{article}\usepackage[margin=1in]{geometry}\usepackage{graphicx}',
        r'\usepackage[colorlinks=true]{hyperref}\usepackage{bookmark}\begin{document}',
        r'\tableofcontents\clearpage']+
        [r'\input{'+(out/r['problem_id']/'report.tex').resolve().as_posix()+'}' for r in reports]+
        [r'\end{document}']))
    (out/'manifest.json').write_text(json.dumps(dict(source=str(root.resolve()),
        generator_seed=manifest.get('generator_seed'), profile=manifest.get('profile'),
        fixed_problems=manifest['cases'],
        problem_ids=[r['problem_id'] for r in reports], algorithms=manifest['algorithms']), indent=2))
    return out
