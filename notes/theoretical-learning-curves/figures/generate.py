# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
"""Regenerate the explanatory figures with: uv run figures/generate.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEST = Path(__file__).resolve().parent
K = (np.sqrt(5) - 1) / 2
C = 1 + 2 / np.sqrt(5)
BLUE, ORANGE = "#2166ac", "#b45309"


def gap(k):
    # Algebraically equivalent to J(k)/J* - 1; stable near the optimum.
    return (1 + K) * (k - K) ** 2 / (k * (2 - k))


plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def finish(fig, name):
    for ax in fig.axes:
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
    fig.savefig(DEST / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(DEST / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


fig, (ax, zoom) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
k = np.linspace(0.015, 1.985, 1500)
ax.plot(k, gap(k), color=BLUE)
ax.axvline(K, color="0.5", ls=":")
ax.scatter([K], [0], color=BLUE, zorder=4)
sample = 1.2
height = gap(sample)
ax.plot([sample, sample], [0, height], color=ORANGE, ls="--")
ax.scatter([sample], [height], color=ORANGE, zorder=4)
ax.annotate("", (sample, 0.13), (K, 0.13),
            arrowprops={"arrowstyle": "<->", "color": ORANGE})
ax.text((K + sample) / 2, 0.22, "$e=k-K^*$", ha="center", color=ORANGE)
ax.text(sample + 0.04, height / 2, "$g(k)$", color=ORANGE)
ax.text(K, -0.43, "$K^*$", ha="center")
ax.set(xlim=(0, 2), ylim=(-0.05, 5), xlabel="Controller gain $k$",
       ylabel="Relative cost gap $g(k)$", title="(a) Horizontal error, vertical cost gap")
ax.text(0.04, 4.55, "Gap diverges at both stability boundaries", fontsize=8)

e = np.linspace(-0.25, 0.25, 600)
zoom.plot(e, gap(K + e), color=BLUE, label="Exact gap")
zoom.plot(e, C * e**2, color=ORANGE, ls="--", label="$1.894 e^2$ (local approximation)")
for signed_e in [-0.118, 0.118]:
    y = gap(K + signed_e)
    zoom.plot([signed_e, signed_e], [0, y], color="0.6", ls=":")
    zoom.scatter([signed_e], [y], color=BLUE)
    zoom.annotate(f"{100*y:.1f}%", (signed_e, y), xytext=(0, 9),
                  textcoords="offset points", ha="center")
zoom.set(xlabel="Signed perturbation $e=k-K^*$", ylabel="Relative cost gap",
         title="(b) Near the optimum: almost quadratic", ylim=(0, 0.17))
zoom.legend(fontsize=8, loc="upper center")
finish(fig, "gap-geometry")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
n = np.arange(1, 1001)
for beta, color, style in [(0.5, BLUE, "-"), (1.0, ORANGE, "--")]:
    perturbation = 0.2 * n ** (-beta)
    axes[0].loglog(n, perturbation, color=color, ls=style,
                   label=rf"$e(N)=0.2N^{{-{beta:g}}}$")
    axes[1].loglog(n, gap(K + perturbation), color=color, ls=style,
                   label=rf"Exact gap for $\beta={beta:g}$")
axes[0].set(xlabel="Iteration $N$", ylabel="Perturbation $e(N)$",
            title="(a) Illustrative gain errors")
axes[1].set(xlabel="Iteration $N$", ylabel="Relative cost gap $g(K^*+e(N))$",
            title="(b) Corresponding learning curves")
for ax in axes:
    ax.legend(fontsize=8)
finish(fig, "perturbation-learning-curves")
