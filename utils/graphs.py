"""Figures for learning curves, with one consistent look across every run and every algorithm.

Conventions (fixed here so they never drift between figures):
* every gap axis is logarithmic, with one fixed range and floor (``GAP_FLOOR``, ``_gap_axis``): the
  gap lives in [0, 1] and the differences that matter are near zero, where a linear axis shows
  nothing. No figure sets its own gap scale;
* each algorithm owns one colour, assigned by name in a fixed order, never by rank;
* the exact optimum (the gap floor, K*) is drawn in ink as a dashed reference, never in a series colour;
* text is ink, never the series colour; gridlines are hairlines; one y-axis per chart;
* every chart has a legend (``_legend``) naming each line, band and reference, including the optimum;
  with two or more algorithms each line also gets a direct label at its end. The exceptions are the
  individual-runs and running-mean charts, where a legend would only repeat the subtitle over the data;
  their subtitle names every line.

Every function takes ``LearningCurves`` (``utils.learning_curves``) and an optional axis, so the same
pieces build single figures and the full report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils import concentration  # noqa: E402
from utils import learning_curves as lc  # noqa: E402

# categorical slots in fixed order (validated light palette); algorithms are pinned to slots by name
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALGO_SLOT = {"monte_carlo": 0, "ppo": 1, "ppo_linear": 2}
LABELS = {"monte_carlo": "Monte Carlo control", "ppo": "PPO (neural policy)", "ppo_linear": "PPO (linear policy)"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# The gap axis. A log axis cannot show zero, so every gap at or below GAP_FLOOR, including an exact
# optimum (gap 0), is drawn on the floor line. Diverged runs sit at gap 1, the top gridline.
GAP_FLOOR = 1e-8
GAP_LIMITS = (GAP_FLOOR / 3.0, 3.0)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_2,
            "font.size": 9,
            "figure.dpi": 150,
        }
    )


def color(name: str) -> str:
    if name not in ALGO_SLOT:
        ALGO_SLOT[name] = len(ALGO_SLOT)
    return SLOTS[ALGO_SLOT[name] % len(SLOTS)]


def label(name: str) -> str:
    return LABELS.get(name, name)


def _ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))
    return ax


def _steps_label(x, _pos=None) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def _title(ax, main: str, sub: str = "") -> None:
    """Short bold headline, with the explanatory note as a smaller line beneath it."""
    ax.set_title(main, loc="left", pad=18 if sub else 6)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=7.5, color=INK_2, va="bottom", ha="left")


def _steps_axis(ax, xlabel="environment steps") -> None:
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_steps_label))


def _legend(ax, below: bool = False) -> None:
    """The legend every chart carries: inside on a light backing, or below the axes for long legends."""
    if below:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    else:
        ax.legend(loc="best", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.85)


def _finish(ax, curves: list[lc.LearningCurves], title: str, sub: str = "", xlabel="environment steps", ylabel="",
            legend_below: bool = False, legend: bool = True):
    _title(ax, title, sub)
    if xlabel == "environment steps":
        _steps_axis(ax)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        _legend(ax, below=legend_below)


def _direct_label(ax, x, y, text, col):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.plot([x], [y], "o", ms=4, color=col, mec=SURFACE, mew=1.2)


def _on_gap_axis(values) -> np.ndarray:
    """Gap values as drawn: anything below the floor is placed on it."""
    return np.maximum(np.asarray(values, dtype=float), GAP_FLOOR)


def _gap_axis(ax, which: str = "y") -> None:
    """Make one axis of ``ax`` the standard gap axis: log scale, fixed limits, the optimum on the floor."""
    note = "optimum (gap ≤ $10^{%d}$)" % int(np.log10(GAP_FLOOR))
    if which == "y":
        ax.set_yscale("log")
        ax.set_ylim(*GAP_LIMITS)
        ax.axhline(GAP_FLOOR, color=INK, ls="--", lw=1.0, label=note)
    else:
        ax.set_xscale("log")
        ax.set_xlim(*GAP_LIMITS)
        ax.axvline(GAP_FLOOR, color=INK, ls="--", lw=1.0, label=note)


# -- figures --------------------------------------------------------------------------------------


def _linear_floor(ax, floor: float | None, which: str = "y") -> None:
    """The best gap any linear policy u = -Kx can reach (nonlinear problems only), as a reference line."""
    if floor is None:
        return
    text = f"best linear policy (gap {floor:.2g})"
    if which == "y":
        ax.axhline(floor, color=INK_2, ls="-.", lw=1.0, label=text)
    else:
        ax.axvline(floor, color=INK_2, ls="-.", lw=1.0, label=text)


def plot_learning_curves(curves, ax=None, delta=0.05, linear_floor=None):
    """Estimated curve g_hat with its simultaneous Hoeffding band, one line per algorithm.

    ``linear_floor``: on a nonlinear problem, the best gap of a linear policy, drawn as a reference.
    """
    ax = _ax(ax)
    for c in curves:
        col = color(c.name)
        lo, hi = c.hoeffding_band(delta)
        ax.plot(c.steps, _on_gap_axis(c.mean()), color=col, label=f"{label(c.name)}: mean gap (M = {c.n_seeds})")
        ax.fill_between(c.steps, _on_gap_axis(lo), _on_gap_axis(hi), color=col, alpha=0.15, lw=0,
                        label=f"{label(c.name)}: {int(100 * (1 - delta))}% Hoeffding band")
        if len(curves) >= 2:
            _direct_label(ax, c.steps[-1], _on_gap_axis(c.mean()[-1]), label(c.name), col)
    _gap_axis(ax)
    _linear_floor(ax, linear_floor)
    title = "Learning curve" if len(curves) >= 2 else f"Learning curve: {label(curves[0].name)}"
    sub = f"mean gap over seeds; shaded: {int(100 * (1 - delta))}% Hoeffding band, simultaneous over checkpoints"
    _finish(ax, curves, title, sub, ylabel="gap  g = 1 − J*/J  (log scale)", legend_below=True)
    return ax


def plot_runs(curve, ax=None, max_runs=None):
    """Every seed's curve, faint, with the min/max envelope and the mean. ``max_runs=None`` draws all seeds."""
    ax = _ax(ax)
    col = color(curve.name)
    for g in curve.gaps[:max_runs]:
        ax.plot(curve.steps, _on_gap_axis(g), color=col, alpha=0.12, lw=0.7)
    lo, hi = curve.envelope()
    ax.plot(curve.steps, _on_gap_axis(lo), color=INK_2, lw=0.9, ls=":")
    ax.plot(curve.steps, _on_gap_axis(hi), color=INK_2, lw=0.9, ls=":")
    ax.plot(curve.steps, _on_gap_axis(curve.mean()), color=col, lw=2.0)
    _gap_axis(ax)
    shown = curve.n_seeds if max_runs is None else min(max_runs, curve.n_seeds)
    _finish(ax, [curve], f"Individual runs: {label(curve.name)}",
            f"one faint line per seed ({shown} of {curve.n_seeds}); dotted: min and max over seeds; thick: mean; "
            "dashed: optimum", ylabel="gap  (log scale)", legend=False)
    return ax


def plot_unstable_fraction(curves, ax=None):
    ax = _ax(ax)
    for c in curves:
        frac = c.unstable_fraction()
        if frac is not None:
            ax.plot(c.steps, frac, color=color(c.name), label=label(c.name))
    ax.set_ylim(-0.02, 1.02)
    title = "Fraction of unstable runs" + ("" if len(curves) >= 2 else f": {label(curves[0].name)}")
    _finish(ax, curves, title, ylabel="fraction of seeds")
    return ax


def plot_gains(curve, K_star, ax=None, component_names=None, optimal=True):
    """Mean learned (or effective) gain +/- one standard deviation across seeds, against K*.

    ``optimal`` says whether K* is the optimum (linear problems) or only the gain of the linearization
    at the origin (nonlinear problems, where no linear gain is optimal).
    """
    ax = _ax(ax)
    col = color(curve.name)
    K_star = np.asarray(K_star, dtype=float).reshape(-1)
    names = component_names or ([r"$k$"] if K_star.size == 1 else [f"$K_{{{i}}}$" for i in range(K_star.size)])
    styles = ["-", "--", "-.", ":"]
    ref ="optimum" if optimal else "linearization, not optimal"
    for i, ks in enumerate(K_star):
        mean = curve.gains[:, :, i].mean(axis=0)
        sd = curve.gains[:, :, i].std(axis=0)
        ls = styles[i % len(styles)]
        ax.plot(curve.steps, mean, color=col, ls=ls, label=f"{names[i]}: mean over seeds")
        ax.fill_between(curve.steps, mean - sd, mean + sd, color=col, alpha=0.12, lw=0, label=f"{names[i]}: ± one sd")
        ax.axhline(ks, color=INK, ls=ls, lw=1.0, label=f"{names[i]}* = {ks:.3f} ({ref})")
    _legend(ax)
    _title(ax, f"Gains: {label(curve.name)}", "learned gain, or for a table or network the effective linear gain")
    _steps_axis(ax)
    ax.set_ylabel("gain")
    return ax


def plot_bias_variance(curve, K_star, curvature, ax=None):
    """Actual g_hat against the near-optimum prediction b'Cb + tr(C S), and its two parts."""
    ax = _ax(ax)
    col = color(curve.name)
    bv = curve.bias_variance(K_star, curvature)
    ax.plot(curve.steps, _on_gap_axis(curve.mean()), color=col, lw=2.0, label="actual mean gap")
    ax.plot(curve.steps, _on_gap_axis(bv["predicted_gap"]), color=INK, lw=1.2, ls="--", label="predicted: bias + variance")
    ax.plot(curve.steps, _on_gap_axis(bv["bias_term"]), color=INK_2, lw=1.0, ls="-.", label="bias term")
    ax.plot(curve.steps, _on_gap_axis(bv["variance_term"]), color=INK_2, lw=1.0, ls=":", label="variance term")
    _gap_axis(ax)
    _legend(ax)
    _title(ax, f"Bias and variance: {label(curve.name)}", "near-optimum prediction b'Cb + tr(CS) against the measured mean gap")
    _steps_axis(ax)
    ax.set_ylabel("gap  (log scale)")
    return ax


def plot_running_means(curve, ax=None, checkpoint=-1, delta=0.05, n_orderings=20):
    """Strong law: running averages over seeds, inside the Hoeffding funnel around the final mean."""
    ax = _ax(ax)
    col = color(curve.name)
    paths = curve.running_means(checkpoint, n_orderings)
    m = np.arange(1, curve.n_seeds + 1)
    for p in paths:
        ax.plot(m, _on_gap_axis(p), color=col, alpha=0.35, lw=0.8)
    center = curve.mean()[checkpoint]
    eps = np.array([concentration.hoeffding_eps(k, delta) for k in m])
    ax.plot(m, _on_gap_axis(np.clip(center + eps, 0, 1)), color=INK_2, lw=0.9, ls=":")
    ax.plot(m, _on_gap_axis(np.clip(center - eps, 0, 1)), color=INK_2, lw=0.9, ls=":")
    ax.axhline(_on_gap_axis(center), color=INK_2, ls="-", lw=1.0)
    _gap_axis(ax)
    step = curve.steps[checkpoint]
    _finish(
        ax,
        [curve],
        f"Running mean over seeds: {label(curve.name)}",
        f"at step {_steps_label(step)}, {paths.shape[0]} random seed orders; solid: all-seed mean; "
        "dotted: Hoeffding ±ε(M); dashed: optimum",
        xlabel="number of seeds M",
        ylabel="running mean of the gap  (log scale)",
        legend=False,
    )
    return ax


def plot_final_ecdf(curves, ax=None, checkpoint=-1, linear_floor=None):
    """Distribution of the gap across seeds at one checkpoint (a performance profile)."""
    ax = _ax(ax)
    for c in curves:
        g = _on_gap_axis(np.sort(c.gaps[:, checkpoint]))
        ax.step(g, np.arange(1, g.size + 1) / g.size, where="post", color=color(c.name), label=label(c.name))
    _gap_axis(ax, "x")
    _linear_floor(ax, linear_floor, "x")
    ax.set_ylim(0, 1.02)
    title = "Final gap across seeds (ECDF)" + ("" if len(curves) >= 2 else f": {label(curves[0].name)}")
    _finish(ax, curves, title, xlabel="gap at final checkpoint  (log scale)", ylabel="fraction of seeds ≤ gap")
    return ax


def plot_seed_metric(steps, values, name, title, ylabel, ax=None, logy=False):
    """Any per-seed diagnostic (n_seeds, n_checkpoints): median and interquartile band."""
    ax = _ax(ax)
    col = color(name)
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    ax.plot(steps, q50, color=col, label="median over seeds")
    ax.fill_between(steps, q25, q75, color=col, alpha=0.15, lw=0, label="interquartile range")
    if logy:
        ax.set_yscale("log")
    _legend(ax)
    _title(ax, f"{title}: {label(name)}", "median across seeds; shaded: interquartile range")
    _steps_axis(ax)
    ax.set_ylabel(ylabel)
    return ax


def plot_training_rewards(runs, ax=None):
    """Mean observed stage reward over each checkpoint interval, including exploration."""
    ax = _ax(ax)
    from utils.plot_scales import adaptive_y
    failures = []
    plotted = []
    for _, arrays, meta in runs:
        if "training_stage_reward" not in arrays:
            continue  # Older runs did not record rewards; do not fabricate them from gaps.
        values = arrays["training_stage_reward"]
        counts = arrays["training_reward_count"]
        mask = np.all(counts > 0, axis=0)
        means = values[:, mask].mean(axis=0)
        plotted.append(means)
        bad = ~np.isfinite(means)
        if bad.any():
            failures.append(f"{label(meta['algo'])}: {int(bad.sum())} nonfinite windows")
        ax.plot(arrays["steps"][mask], np.where(bad, np.nan, means), marker="o", ms=3,
                color=color(meta["algo"]), label=label(meta["algo"]))
    if failures:
        ax.text(.02, .03, "\n".join(failures), transform=ax.transAxes, fontsize=7)
    ax.set_title("Training stage reward (higher / less negative is better)")
    ax.set_xlabel("Environment steps per seed (end of averaging window)")
    scale = adaptive_y(ax, np.concatenate(plotted)) if plotted else "linear"
    ax.set_ylabel(f"Mean reward per transition ({scale})")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    return ax


# -- the full report ------------------------------------------------------------------------------


def _save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")  # tight: keeps a legend placed below the axes
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_report(runs: list[tuple[lc.LearningCurves, dict, dict]], out: Path, delta: float = 0.05,
                linear_floor: float | None = None) -> dict:
    """All figures and metrics for a set of runs on the same problem.

    ``runs`` holds (curves, arrays, meta) per algorithm, as returned by ``rollouts.load``.
    Writes PNG + PDF figures, summary.json and summary.md to ``out``; returns the summary.
    """
    apply_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from utils.benchmark_report import problem_report
    problem_report(runs, out / 'benchmark')
    curves = [c for c, _, _ in runs]
    if any("training_stage_reward" in arrays for _, arrays, _ in runs):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        plot_training_rewards(runs, ax)
        _save(fig, out, "training_stage_reward")


    fig, ax = plt.subplots(figsize=(6.4, 5.0))  # taller: its legend sits below the axes
    plot_learning_curves(curves, ax, delta, linear_floor)
    _save(fig, out, "learning_curves")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_unstable_fraction(curves, ax)
    _save(fig, out, "unstable_fraction")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_final_ecdf(curves, ax, linear_floor=linear_floor)
    _save(fig, out, "final_gap_ecdf")

    summary = {"problem": runs[0][2]["problem"], "problem_params": runs[0][2].get("problem_params", {}),
               "gamma": runs[0][2]["gamma"], "J_star": runs[0][2]["J_star"], "K_star": runs[0][2]["K_star"],
               "best_linear_gap": linear_floor, "algorithms": {}}
    for c, arrays, meta in runs:
        stem = c.name
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        plot_runs(c, ax)
        _save(fig, out, f"{stem}_runs")

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        plot_running_means(c, ax, delta=delta)
        _save(fig, out, f"{stem}_running_means")

        s = c.summary(delta)
        if c.gains is not None:
            # the bias-variance split expands the gap around an optimal *gain*; it exists only where
            # the run saved a curvature, i.e. where the optimal policy is linear
            linear_optimum = meta.get("curvature") is not None
            names = [r"$K_x$", r"$K_z$"] if meta["problem"] == "lqi" else None
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            plot_gains(c, meta["K_star"], ax, names, optimal=linear_optimum)
            _save(fig, out, f"{stem}_gains")

            if linear_optimum:
                fig, ax = plt.subplots(figsize=(6.4, 3.8))
                plot_bias_variance(c, meta["K_star"], meta["curvature"], ax)
                _save(fig, out, f"{stem}_bias_variance")
                bv = c.bias_variance(meta["K_star"], meta["curvature"])
                s["final_bias_term"] = float(bv["bias_term"][-1])
                s["final_variance_term"] = float(bv["variance_term"][-1])
                s["final_predicted_gap"] = float(bv["predicted_gap"][-1])

        if "final_abs_state" in arrays:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            plot_seed_metric(c.steps, np.maximum(arrays["final_abs_state"], 1e-16), c.name,
                             "Residual state", "|x| after the 5000-step evaluation", ax, logy=True)
            _save(fig, out, f"{stem}_residual_state")
            s["final_median_residual_state"] = float(np.median(arrays["final_abs_state"][:, -1]))
        if "visited_frac" in arrays:
            s["final_mean_visited_state_cells"] = float(arrays["visited_frac"][:, -1].mean())
        s["wall_seconds"] = meta.get("wall_seconds")
        summary["algorithms"][c.name] = s

    if len(curves) >= 2:
        summary["orderings"] = {}
        for i, c1 in enumerate(curves):
            for c2 in curves[i + 1 :]:
                same_grid = np.array_equal(c1.steps, c2.steps)
                x, y = lc.align(c1, c2)
                # name the pair so that the claim tested is "lower final mean gap <= higher"
                a, b = (x, y) if x.mean()[-1] <= y.mean()[-1] else (y, x)
                exp = lc.expected_ordered(a, b)
                conf = lc.confidently_ordered(a, b, delta)
                band = lc.band_separated(a, b)
                summary["orderings"][f"{a.name} <= {b.name}"] = {
                    "checkpoints": "identical" if same_grid else "step-hold on the union of checkpoints",
                    "band_separated_final": bool(band[-1]),
                    "expected_ordered_final": bool(exp[-1]),
                    "confidently_ordered_final": bool(conf[-1]),
                    "floor_ordered": bool(lc.floor_ordered(a, b)),
                    "crossing_step_band": lc.crossing_time(a.steps, band),
                    "crossing_step_expected": lc.crossing_time(a.steps, exp),
                    "crossing_step_confident": lc.crossing_time(a.steps, conf),
                }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "summary.md").write_text(_summary_markdown(summary))
    return summary


def _summary_markdown(summary: dict) -> str:
    params = summary.get("problem_params") or {}
    spec = "".join(f", {k} = {v}" for k, v in params.items())
    lines = [f"# Run summary: {summary['problem'].upper()}{spec}, gamma = {summary['gamma']}", "",
             f"J* = {summary['J_star']:.6f}, K* = {np.round(summary['K_star'], 6).tolist()}", ""]
    if summary.get("best_linear_gap") is not None:
        lines[-1:-1] = [f"best gap of any linear policy u = -Kx: {summary['best_linear_gap']:.4g}"]
    for name, s in summary["algorithms"].items():
        lines.append(f"## {label(name)}  (M = {s['n_seeds']}, {s['final_step']:,.0f} steps per seed)")
        lo, hi = s["final_hoeffding_band"]
        lines.append(f"- final mean gap {s['final_mean_gap']:.4g} ± {s['final_se']:.2g} (SE); "
                     f"Hoeffding band [{lo:.4f}, {hi:.4f}] (half-width {s['hoeffding_half_width']:.4f})")
        med, q25, q75 = s["final_median_iqr"]
        mn, mx = s["final_min_max"]
        lines.append(f"- median {med:.4g} (IQR {q25:.4g}–{q75:.4g}); min {mn:.4g}, max {mx:.4g}")
        lines.append(f"- floor estimate {s['floor_estimate']:.4g}; infinite-cost fraction {s['final_infinite_cost_fraction']:.3f}"
                     + (f"; unstable fraction {s['final_unstable_fraction']:.3f}" if "final_unstable_fraction" in s else ""))
        for k, v in s.items():
            if k.startswith("reached_gap_"):
                thr = k.removeprefix("reached_gap_")
                med_steps = s.get(f"median_steps_to_gap_{thr}")
                lines.append(f"- reached gap ≤ {thr}: {v:.2%} of seeds" + (f", median {med_steps:,.0f} steps" if med_steps else ""))
        if "final_mean_gain" in s:
            lines.append(f"- final gain mean {np.round(s['final_mean_gain'], 4).tolist()}, sd {np.round(s['final_gain_std'], 4).tolist() if s['final_gain_std'] else None}")
        if "final_predicted_gap" in s:
            lines.append(f"- predicted gap (bias {s['final_bias_term']:.4g} + variance {s['final_variance_term']:.4g}) = {s['final_predicted_gap']:.4g}")
        if "final_median_residual_state" in s:
            cells = s.get("final_mean_visited_state_cells")
            lines.append(f"- median residual |state| {s['final_median_residual_state']:.3g}"
                         + (f"; visited state cells {cells:.2%}" if cells is not None else ""))
        lines.append(f"- wall time {s['wall_seconds']:.1f} s")
        lines.append("")
    for pair, o in summary.get("orderings", {}).items():
        lines.append(f"## Ordering {pair}")
        lines.extend(f"- {k}: {v}" for k, v in o.items())
        lines.append("")
    return "\n".join(lines)
