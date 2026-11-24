"""One-command pilot comparing existing PPO policies on exact control benchmarks."""
import csv
import json
from pathlib import Path

import numpy as np

import rollouts
from algos.families import canonical_algorithms
from utils import envs
from utils.nlqr import EnergyNLQRProblem, generate_parameters, validate


def run_suite(out, count=10, seeds=2, steps=4096, workers=1, generator_seed=0,
              master_seed=0, gamma=.99, algorithms=rollouts.DEFAULT_BENCHMARK_ALGOS, profile="moderate", checkpoints=4, device="cpu", seed_batch_size=None):
    algorithms = canonical_algorithms(algorithms)
    if min(count, seeds, steps, workers, checkpoints) < 1 or steps % 512:
        raise ValueError("count, seeds and workers must be positive; steps must be a positive multiple of 512")
    if not 0 < gamma < 1:
        raise ValueError("gamma must lie strictly between zero and one")
    if not algorithms or len(set(algorithms)) != len(algorithms) or set(algorithms) - set(rollouts.ALGOS):
        raise ValueError("algorithms must be distinct registered names")
    root = Path(out)
    if 'td-mpc2-lg' in algorithms:
        raise ValueError('TD-MPC2-LQ is a bounded scalar adaptation, not a generated NLQR learner; use matched-lq')
    root.mkdir(parents=True, exist_ok=False)
    parameters = generate_parameters(count, generator_seed, gamma, profile)
    cases = [("lqr", "lqr", {})] + [(f"nlqr_{i:02d}", "nlqr_energy", p) for i, p in enumerate(parameters)]
    manifest = dict(generator_seed=generator_seed, master_seed=master_seed, gamma=gamma,
                    count=count, seeds=seeds, steps_per_seed=steps, workers=workers, algorithms=list(algorithms),
                    profile=profile, checkpoints=checkpoints,
                    purpose="Pilot learning curves; no convergence or confidence claim", Sigma=[[0, 0], [0, 0]],
                    cases=[])
    for name, kind, params in cases:
        problem = envs.make_problem(kind, gamma, **params)
        entry = dict(id=name, problem=kind, problem_params=params, P=problem.P.tolist(),
                     J_star=problem.J_star, K_linearization=problem.K_star.tolist())
        if isinstance(problem, EnergyNLQRProblem):
            entry['validation'] = validate(problem, generator_seed)
            entry['open_loop_spectral_radius'] = float(max(abs(np.linalg.eigvals(problem.linear.A))))
            entry['open_loop_P_contraction_bound'] = ((1 - problem.beta) / gamma *
                (1 + gamma * problem.p2 * problem.p**2 / problem.r))
        ev = envs.Env(problem, 1).evaluate(problem.optimal_policy)
        entry['optimal_rollout_cost'] = float(ev['J'][0])
        entry['optimal_rollout_error'] = abs(float(ev['J'][0]) - problem.J_star)
        if not np.isclose(ev['J'][0], problem.J_star, atol=1e-9, rtol=1e-9):
            raise ValueError(f"optimal rollout failed for {name}")
        manifest['cases'].append(entry)
    (root / 'problems.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    common = dict(total_steps=steps, horizon=128, n_envs=4, epochs=4,
                  n_checkpoints=checkpoints, eval_horizon=envs.EVAL_HORIZON)
    for name, kind, params in cases:
        for algo in algorithms:
            config = dict(common)
            if algo == 'ppo-nn':
                config.update(hidden_size=32, minibatch_size=128)
            elif algo == 'ppo-lg':
                config.update(n_minibatches=4)
            elif algo == 'td-mpc2-nn':
                config = dict(total_steps=steps, n_checkpoints=checkpoints, eval_horizon=envs.EVAL_HORIZON)
            else:
                config = dict(total_steps=steps, horizon=128, n_checkpoints=checkpoints,
                              eval_horizon=envs.EVAL_HORIZON)
            print(f"{name}: {algo}, {seeds} seeds x {steps} steps", flush=True)
            directory = rollouts.run(algo, kind, seeds, master_seed, gamma, workers,
                                     config, out_dir=root / name / algo, problem_params=params, device=device, seed_batch_size=seed_batch_size)
            arrays, meta = rollouts.load(directory)
            gaps = rollouts.to_curves(arrays, meta).gaps
            row = dict(problem=name, algorithm=algo, initial_mean_gap=float(gaps[:, 0].mean()),
                       final_mean_gap=float(gaps[:, -1].mean()), final_min_gap=float(gaps[:, -1].min()),
                       final_max_gap=float(gaps[:, -1].max()),
                       final_unstable_fraction=(float(arrays['unstable'][:, -1].mean())
                                                if 'unstable' in arrays else None),
                       training_diverged_fraction=(None if algo == 'monte_carlo' else float(arrays['diverged'][:, -1].mean())),
                       wall_seconds=meta['wall_seconds'], run=str(directory))
            rows.append(row)
            (root / 'summary.json').write_text(json.dumps(rows, indent=2))
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_suite(root)
    plot_suite_rewards(root)
    from utils.benchmark_report import suite_report
    suite_report(root)
    if profile == "aggressive":
        plot_aggressive_learning_curves(root)
    print(f"Saved {len(cases)} exact baselines and {len(rows)} training runs to {root}")
    return root


def plot_suite(root, yscale="auto"):
    """Plot the LQR reference separately from the generated nonlinear problems."""
    root = Path(root)
    manifest = json.loads((root / 'problems.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from utils.plot_scales import adaptive_y

    algorithms = manifest.get('algorithms', ['ppo_linear', 'ppo'])
    nonlinear = [c for c in manifest['cases'] if c['problem'] == 'nlqr_energy']
    linear = [c for c in manifest['cases'] if c['problem'] == 'lqr']
    curves = {}
    for case in manifest['cases']:
        for algo in algorithms:
            arrays, meta = rollouts.load(root / case['id'] / algo)
            curves[case['id'], algo] = (arrays['steps'], rollouts.to_curves(arrays, meta).gaps.mean(axis=0))

    def draw(cases, destination, title, nonlinear_group):
        cols = 2 if len(cases) > 1 else 1
        rows = int(np.ceil(len(cases) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows + 1.1), squeeze=False)
        for ax, case in zip(axes.flat, cases):
            from utils.graphs import label, color
            for algo in algorithms:
                steps, gap = curves[case['id'], algo]
                ax.plot(steps, gap, marker='o', markersize=3, label=label(algo),
                        color=color(algo))
            if nonlinear_group:
                index = nonlinear.index(case) + 1
                p = case['problem_params']
                panel = (f"NLQR {index:02d}  |  saved ID: {case['id']}  |  J* = {case['J_star']:.4f}"
                         f"\nP = diag({p['p1']:.3f}, {p['p2']:.3f}), beta = {p['beta']:.3f}, "
                         f"h = {p['p']:.3f}, alpha = {p['alpha']:.3f}")
            else:
                panel = 'Scalar LQR reference: x(t+1) = x(t) + u(t), cost = x² + u²\nx(0) = 1; J* = ' + f"{case['J_star']:.6f}"
            values = np.concatenate([curves[case['id'], algo][1] for algo in algorithms])
            scale = adaptive_y(ax, values, scale=yscale)
            ax.set(title=panel, xlabel='Training steps per seed', ylabel=f'Mean cost gap (%, {scale})')
            lo, hi = ax.get_ylim()
            if lo <= 0 <= hi and scale != 'log':
                ax.axhline(0, color='black', linewidth=1, linestyle=':', label='Known optimal policy: gap = 0')
            else:
                ax.text(.98, .96, '0% optimum below view', transform=ax.transAxes,
                        ha='right', va='top', fontsize=8, color='#52514e',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.grid(axis='y', alpha=.2)
            ax.title.set_fontsize(10)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10)
        subtitle = (f"{manifest['seeds']} training seeds per problem; gamma = {manifest['gamma']}; no process noise"
                    + ('; x(0) = (1, -1)' if nonlinear_group else ''))
        fig.suptitle(title + '\n' + subtitle + '\nIndependent y limits fitted to each panel; compare numerical ticks', fontsize=12)
        fig.tight_layout(rect=(0, .05, 1, .93))
        fig.savefig(destination, dpi=180)
        plt.close(fig)

    if linear:
        draw(linear, root / 'lqr_reference.png', 'REFERENCE ONLY — scalar linear dynamics', False)
    if nonlinear:
        draw(nonlinear, root / 'learning_curves.png', 'GENERATED NLQR SUITE — each panel is a different nonlinear problem', True)
        for start in range(0, len(nonlinear), 5):
            stop = min(start + 5, len(nonlinear))
            draw(nonlinear[start:stop], root / f'nlqr_{start+1:02d}_{stop:02d}.png',
                 f'GENERATED NLQR {start+1:02d}–{stop:02d} — different dynamics; same training protocol', True)
    return root / 'learning_curves.png'


def plot_suite_rewards(root):
    """Separate LQR and five-case NLQR pages, using actual recorded training rewards."""
    from utils.graphs import plot_training_rewards
    import matplotlib.pyplot as plt
    root = Path(root)
    manifest = json.loads((root / 'problems.json').read_text())
    algorithms = manifest.get('algorithms', ['ppo_linear', 'ppo'])
    nl = [c for c in manifest['cases'] if c['problem'] == 'nlqr_energy']
    groups = [('lqr', [c for c in manifest['cases'] if c['problem'] == 'lqr'])]
    groups += [(f'nlqr_{i+1:02d}_{min(i+5, len(nl)):02d}', nl[i:i+5]) for i in range(0, len(nl), 5)]
    destinations = []
    for name, cases in groups:
        if not cases:
            continue
        fig, axes = plt.subplots(int(np.ceil(len(cases)/2)), 2 if len(cases)>1 else 1,
                                 figsize=(12, 3.5 * int(np.ceil(len(cases)/2)) + 1), squeeze=False)
        any_rewards = False
        for ax, case in zip(axes.flat, cases):
            runs = []
            for algo in algorithms:
                arrays, meta = rollouts.load(root / case['id'] / algo)
                if 'training_stage_reward' in arrays:
                    any_rewards = True
                    runs.append((None, arrays, meta))
            if runs:
                plot_training_rewards(runs, ax)
            else:
                ax.text(.5, .5, 'Training rewards were not recorded', ha='center', transform=ax.transAxes)
            title = 'Scalar LQR reference' if case['problem']=='lqr' else f"NLQR {nl.index(case)+1:02d} (saved ID: {case['id']})"
            ax.set_title(title + f" | J* = {case['J_star']:.4f}")

        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        fig.suptitle('Observed training stage reward — higher (less negative) is better\n'
                     f"{manifest['seeds']} seeds; mean since previous checkpoint; includes exploration; 128-step resets\nIndependent y limits; scale shown on each axis", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .92))
        if any_rewards:
            dest = root / f'training_rewards_{name}.png'
            fig.savefig(dest, dpi=180)
            destinations.append(dest)
        plt.close(fig)
    return destinations


def plot_aggressive_learning_curves(root):
    """A.1-style policy-gap comparisons; empirical IQR is not a confidence band."""
    from utils.graphs import color, label
    from utils.plot_scales import adaptive_y
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    root = Path(root)
    manifest = json.loads((root/'problems.json').read_text())
    cases = [c for c in manifest['cases'] if c['problem'] == 'nlqr_energy']
    algorithms = manifest['algorithms']

    def draw(group, destination):
        cols = 1 if len(group) == 1 else 2
        rows = int(np.ceil(len(group)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8*rows+1), squeeze=False)
        for ax, case in zip(axes.flat, group):
            visible = []
            for algo in algorithms:
                arrays, meta = rollouts.load(root/case['id']/algo)
                gap = rollouts.to_curves(arrays, meta).gaps
                q25, q75 = np.quantile(gap, [.25, .75], axis=0)
                mean = gap.mean(axis=0)
                ax.plot(arrays['steps'], np.maximum(mean, 1e-8), color=color(algo), label=label(algo))
                ax.fill_between(arrays['steps'], np.maximum(q25, 1e-8), np.maximum(q75, 1e-8),
                                color=color(algo), alpha=.17)
                visible.extend([mean, q25, q75])
            displayed = np.maximum(np.concatenate(visible), 1e-8)
            adaptive_y(ax, displayed, scale='linear')
            if np.ptp(displayed) < 1e-8:
                ax.set_ylim(max(1e-9, displayed[0]*.8), displayed[0]*1.05)
            if np.all(np.concatenate(visible) == 1):
                ax.text(.5, .55, 'Every seed at 100% evaluated gap', transform=ax.transAxes,
                        ha='center', fontsize=9)
            index = cases.index(case)+1
            params = case['problem_params']
            ax.set_title(f"Hard NLQR H{index:02d} | rho(A0) = {case['open_loop_spectral_radius']:.3f} | J* = {case['J_star']:.3f}"
                         f"\nbeta = {params['beta']:.3f}, alpha = {params['alpha']:.3f}, kappa = {params['kappa']:.3f}", fontsize=10)
            ax.set_xlabel('Training environment steps per seed')
            ax.set_ylabel('Evaluated cost gap (%)')
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, min(hi, 1.05))
            ax.yaxis.set_major_locator(FixedLocator(np.linspace(max(0, lo), min(hi, 1.0), 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.3g}%"))
            ax.minorticks_off()
            ax.grid(axis='y', which='both', alpha=.2)
            ax.text(.98, .03, 'Exact optimum: 0%; fitted linear axis', transform=ax.transAxes,
                    ha='right', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        for ax in list(axes.flat)[len(group):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3)
        fig.suptitle('Harder nonlinear environments — learning curves against exact optimal cost\n'
                     f"Lines: means of {manifest['seeds']} seeds; shading: middle 50% of seeds (not confidence intervals)\n"
                     'Independent fitted linear axes; no-control origin is unstable; lower is better', fontsize=12)
        fig.tight_layout(rect=(0, .045, 1, .91))
        fig.savefig(destination, dpi=180)
        plt.close(fig)
    for i, case in enumerate(cases, 1):
        draw([case], root/f'hard_H{i:02d}_learning_curve.png')
        # Keep the historical gap definition, but fit the axis to the means.
        curves = []
        for algo in algorithms:
            arrays, meta = rollouts.load(root/case['id']/algo)
            curves.append(rollouts.to_curves(arrays, meta))
        fig, ax = plt.subplots(figsize=(12, 6))
        for curve in curves:
            mean = curve.gaps.mean(axis=0)
            ax.plot(curve.steps, mean, marker='o', markersize=4,
                    color=color(curve.name),
                    label=f'{label(curve.name)} — final {100*mean[-1]:.1f}%')
        means = np.concatenate([curve.gaps.mean(axis=0) for curve in curves])
        adaptive_y(ax, means, scale='linear')
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(0, lo), min(1.02, hi))
        from matplotlib.ticker import MaxNLocator, PercentFormatter
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        ax.set(xlabel='Training environment steps per seed', ylabel='Mean optimality gap (%)')
        ax.grid(axis='y', alpha=.25)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=1, frameon=False)
        if lo <= 0:
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.text(.02, .04, 'Exact optimum: 0%' + (' (below view)' if lo > 0 else ''),
                transform=ax.transAxes, fontsize=9)
        if np.ptp(means) < 1e-10:
            ax.text(.5, .5, 'All three mean curves coincide at 100%',
                    transform=ax.transAxes, ha='center')
        fig.suptitle(f"Hard NLQR H{i:02d} — optimality gap during training\n"
                     f"J* = {case['J_star']:.3f}; {manifest['seeds']} seeds per algorithm", fontsize=12)
        fig.subplots_adjust(top=.84, bottom=.28)
        fig.savefig(root/f'hard_H{i:02d}_historical_gap.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    for start in range(0, len(cases), 5):
        stop = min(start+5, len(cases))
        draw(cases[start:stop], root/f'hard_H{start+1:02d}_H{stop:02d}_learning_curves.png')
