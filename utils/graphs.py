"""Figures for learning curves, with one consistent look across every run and every algorithm.

Conventions (fixed here so they never drift between figures):
* each algorithm owns one colour, assigned by name in a fixed order, never by rank;
* the exact optimum (gap 0, K*) is drawn in ink as a dashed reference, never in a series colour;
* text is ink, never the series colour; gridlines are hairlines; one y-axis per chart;
* two or more series get a legend plus a direct label at the line end; one series gets neither,
  its title names it.

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
ALGO_SLOT = {"monte_carlo": 0, "ppo": 1}
LABELS = {"monte_carlo": "Monte Carlo control", "ppo": "PPO"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


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


def _finish(ax, curves: list[lc.LearningCurves], title: str, sub: str = "", xlabel="environment steps", ylabel=""):
    _title(ax, title, sub)
    if xlabel == "environment steps":
        _steps_axis(ax)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(curves) >= 2:
        ax.legend(loc="best")


def _direct_label(ax, x, y, text, col):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.plot([x], [y], "o", ms=4, color=col, mec=SURFACE, mew=1.2)


# -- figures --------------------------------------------------------------------------------------


def plot_learning_curves(curves, ax=None, delta=0.05, logy=False):
    """Estimated curve g_hat with its simultaneous Hoeffding band, one line per algorithm."""
    ax = _ax(ax)
    for c in curves:
        col = color(c.name)
        lo, hi = c.hoeffding_band(delta)
        ax.fill_between(c.steps, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(c.steps, c.mean(), color=col, label=f"{label(c.name)} (M = {c.n_seeds})")
        if len(curves) >= 2:
            _direct_label(ax, c.steps[-1], c.mean()[-1], label(c.name), col)
    ax.axhline(0.0, color=INK, ls="--", lw=1.0)
    ax.annotate("optimum (gap 0)", (0, 0), xytext=(2, 3), textcoords="offset points", fontsize=7, color=INK_2)
    if logy:
        ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(-0.02 if not logy else 0, 1.02)
    title = "Learning curve" if len(curves) >= 2 else f"Learning curve: {label(curves[0].name)}"
    sub = f"mean gap over seeds; shaded: {int(100 * (1 - delta))}% Hoeffding band, simultaneous over checkpoints"
    _finish(ax, curves, title, sub, ylabel="gap  g = 1 − J*/J")
    return ax


def plot_runs(curve, ax=None, max_runs=200):
    """Every seed's curve, faint, with the min/max envelope and the mean."""
    ax = _ax(ax)
    col = color(curve.name)
    for g in curve.gaps[:max_runs]:
        ax.plot(curve.steps, g, color=col, alpha=0.12, lw=0.7)
    lo, hi = curve.envelope()
    ax.plot(curve.steps, lo, color=INK_2, lw=0.9, ls=":")
    ax.plot(curve.steps, hi, color=INK_2, lw=0.9, ls=":")
    ax.plot(curve.steps, curve.mean(), color=col, lw=2.0)
    ax.axhline(0.0, color=INK, ls="--", lw=1.0)
    ax.set_ylim(-0.02, 1.02)
    _finish(ax, [curve], f"Individual runs: {label(curve.name)}", "one faint line per seed; dotted: min and max over seeds", ylabel="gap")
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


def plot_gains(curve, K_star, ax=None, component_names=None):
    """Mean learned (or effective) gain +/- one standard deviation across seeds, against K*."""
    ax = _ax(ax)
    col = color(curve.name)
    K_star = np.asarray(K_star, dtype=float).reshape(-1)
    names = component_names or ([r"$k$"] if K_star.size == 1 else [f"$K_{{{i}}}$" for i in range(K_star.size)])
    styles = ["-", "--", "-.", ":"]
    for i, ks in enumerate(K_star):
        mean = curve.gains[:, :, i].mean(axis=0)
        sd = curve.gains[:, :, i].std(axis=0)
        ls = styles[i % len(styles)]
        ax.fill_between(curve.steps, mean - sd, mean + sd, color=col, alpha=0.12, lw=0)
        ax.plot(curve.steps, mean, color=col, ls=ls, label=f"{names[i]} (mean ± sd)")
        ax.axhline(ks, color=INK, ls="--", lw=1.0)
        ax.annotate(f"{names[i]}* = {ks:.3f}", (curve.steps[0], ks), xytext=(2, 3), textcoords="offset points", fontsize=7, color=INK_2)
    if K_star.size >= 2:
        ax.legend(loc="best")
    _title(ax, f"Gains: {label(curve.name)}", "mean across seeds; shaded: ± one standard deviation; dashed: optimum")
    _steps_axis(ax)
    ax.set_ylabel("gain")
    return ax


def plot_bias_variance(curve, K_star, curvature, ax=None):
    """Actual g_hat against the near-optimum prediction b'Cb + tr(C S), and its two parts."""
    ax = _ax(ax)
    col = color(curve.name)
    bv = curve.bias_variance(K_star, curvature)
    ax.plot(curve.steps, curve.mean(), color=col, lw=2.0, label="actual mean gap")
    ax.plot(curve.steps, bv["predicted_gap"], color=INK, lw=1.2, ls="--", label="predicted: bias + variance")
    ax.plot(curve.steps, bv["bias_term"], color=INK_2, lw=1.0, ls="-.", label="bias term")
    ax.plot(curve.steps, bv["variance_term"], color=INK_2, lw=1.0, ls=":", label="variance term")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(loc="best")
    _title(ax, f"Bias and variance: {label(curve.name)}", "near-optimum prediction b'Cb + tr(CS) against the measured mean gap")
    _steps_axis(ax)
    ax.set_ylabel("gap (symlog)")
    return ax


def plot_running_means(curve, ax=None, checkpoint=-1, delta=0.05, n_orderings=20):
    """Strong law: running averages over seeds, inside the Hoeffding funnel around the final mean."""
    ax = _ax(ax)
    col = color(curve.name)
    paths = curve.running_means(checkpoint, n_orderings)
    m = np.arange(1, curve.n_seeds + 1)
    for p in paths:
        ax.plot(m, p, color=col, alpha=0.35, lw=0.8)
    center = curve.mean()[checkpoint]
    eps = np.array([concentration.hoeffding_eps(k, delta) for k in m])
    ax.plot(m, np.clip(center + eps, 0, 1), color=INK_2, lw=0.9, ls=":")
    ax.plot(m, np.clip(center - eps, 0, 1), color=INK_2, lw=0.9, ls=":")
    ax.axhline(center, color=INK, ls="--", lw=1.0)
    ax.set_ylim(-0.02, 1.02)
    step = curve.steps[checkpoint]
    _finish(
        ax,
        [curve],
        f"Running mean over seeds: {label(curve.name)}",
        f"gap at step {_steps_label(step)}, seeds added in {paths.shape[0]} random orders; dotted: Hoeffding ±ε(M)",
        xlabel="number of seeds M",
        ylabel="running mean of the gap",
    )
    return ax


def plot_final_ecdf(curves, ax=None, checkpoint=-1):
    """Distribution of the gap across seeds at one checkpoint (a performance profile)."""
    ax = _ax(ax)
    for c in curves:
        g = np.sort(c.gaps[:, checkpoint])
        ax.step(g, np.arange(1, g.size + 1) / g.size, where="post", color=color(c.name), label=label(c.name))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    title = "Final gap across seeds (ECDF)" + ("" if len(curves) >= 2 else f": {label(curves[0].name)}")
    _finish(ax, curves, title, xlabel="gap at final checkpoint", ylabel="fraction of seeds ≤ gap")
    return ax


def plot_seed_metric(steps, values, name, title, ylabel, ax=None, logy=False):
    """Any per-seed diagnostic (n_seeds, n_checkpoints): median and interquartile band."""
    ax = _ax(ax)
    col = color(name)
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    ax.fill_between(steps, q25, q75, color=col, alpha=0.15, lw=0)
    ax.plot(steps, q50, color=col)
    if logy:
        ax.set_yscale("log")
    _title(ax, f"{title}: {label(name)}", "median across seeds; shaded: interquartile range")
    _steps_axis(ax)
    ax.set_ylabel(ylabel)
    return ax


# -- the full report ------------------------------------------------------------------------------


def _save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def make_report(runs: list[tuple[lc.LearningCurves, dict, dict]], out: Path, delta: float = 0.05) -> dict:
    """All figures and metrics for a set of runs on the same problem.

    ``runs`` holds (curves, arrays, meta) per algorithm, as returned by ``rollouts.load``.
    Writes PNG + PDF figures, summary.json and summary.md to ``out``; returns the summary.
    """
    apply_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    curves = [c for c, _, _ in runs]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_learning_curves(curves, ax, delta)
    _save(fig, out, "learning_curves")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_learning_curves(curves, ax, delta, logy=True)
    _save(fig, out, "learning_curves_log")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_unstable_fraction(curves, ax)
    _save(fig, out, "unstable_fraction")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_final_ecdf(curves, ax)
    _save(fig, out, "final_gap_ecdf")

    summary = {"problem": runs[0][2]["problem"], "gamma": runs[0][2]["gamma"], "J_star": runs[0][2]["J_star"],
               "K_star": runs[0][2]["K_star"], "algorithms": {}}
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
            names = [r"$K_x$", r"$K_z$"] if meta["problem"] == "lqi" else None
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            plot_gains(c, meta["K_star"], ax, names)
            _save(fig, out, f"{stem}_gains")

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
    lines = [f"# Run summary: {summary['problem'].upper()}, gamma = {summary['gamma']}", "",
             f"J* = {summary['J_star']:.6f}, K* = {np.round(summary['K_star'], 6).tolist()}", ""]
    for name, s in summary["algorithms"].items():
        lines.append(f"## {label(name)}  (M = {s['n_seeds']}, {s['final_step']:,.0f} steps per seed)")
        lo, hi = s["final_hoeffding_band"]
        lines.append(f"- final mean gap {s['final_mean_gap']:.4f} ± {s['final_se']:.4f} (SE); "
                     f"Hoeffding band [{lo:.4f}, {hi:.4f}] (half-width {s['hoeffding_half_width']:.4f})")
        med, q25, q75 = s["final_median_iqr"]
        mn, mx = s["final_min_max"]
        lines.append(f"- median {med:.4f} (IQR {q25:.4f}–{q75:.4f}); min {mn:.4f}, max {mx:.4f}")
        lines.append(f"- floor estimate {s['floor_estimate']:.4f}; infinite-cost fraction {s['final_infinite_cost_fraction']:.3f}"
                     + (f"; unstable fraction {s['final_unstable_fraction']:.3f}" if "final_unstable_fraction" in s else ""))
        for k, v in s.items():
            if k.startswith("reached_gap_"):
                thr = k.removeprefix("reached_gap_")
                med_steps = s.get(f"median_steps_to_gap_{thr}")
                lines.append(f"- reached gap ≤ {thr}: {v:.2%} of seeds" + (f", median {med_steps:,.0f} steps" if med_steps else ""))
        if "final_mean_gain" in s:
            lines.append(f"- final gain mean {np.round(s['final_mean_gain'], 4).tolist()}, sd {np.round(s['final_gain_std'], 4).tolist() if s['final_gain_std'] else None}")
            lines.append(f"- predicted gap (bias {s['final_bias_term']:.4g} + variance {s['final_variance_term']:.4g}) = {s['final_predicted_gap']:.4g}")
        if "final_median_residual_state" in s:
            lines.append(f"- median residual |state| {s['final_median_residual_state']:.3g}; visited state cells {s['final_mean_visited_state_cells']:.2%}")
        lines.append(f"- wall time {s['wall_seconds']:.1f} s")
        lines.append("")
    for pair, o in summary.get("orderings", {}).items():
        lines.append(f"## Ordering {pair}")
        lines.extend(f"- {k}: {v}" for k, v in o.items())
        lines.append("")
    return "\n".join(lines)
