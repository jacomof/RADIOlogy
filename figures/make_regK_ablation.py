"""Single region count K (MVTec AD), swept over {3, 5, 8, 10, 12} in place
of the paper's {8, 32}. Each point averages that K's raw and refined
region fields."""
import os

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
K = [3, 5, 8, 10, 12]
p_auroc = [0.9114, 0.9184, 0.9204, 0.9190, 0.9192]
aupro   = [0.7735, 0.7982, 0.8073, 0.8110, 0.8128]
p_f1    = [0.3520, 0.3792, 0.3770, 0.3827, 0.3829]

BLUE   = "#2a78d6"   # pixel AUROC
ORANGE = "#eb6834"   # AUPRO
AQUA   = "#1baf7a"   # pixel max-F1 (dark marker edge + value labels for contrast)
INK = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECOND,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# Stagger label heights so adjacent points' value labels don't collide.
LABEL_DY = [9, 20, 9, 20, 9]


def panel(ax, y, color, title, ylabel, ylim, fmt):
    ax.plot(K, y, color=color, linewidth=2.0, zorder=3)
    ax.scatter(K, y, s=64, color=color, edgecolors=INK, linewidths=1.0,
               zorder=4)
    for x, v, dy in zip(K, y, LABEL_DY):
        ax.annotate(fmt.format(v), (x, v), xytext=(0, dy),
                   textcoords="offset points", ha="center", va="bottom",
                   fontsize=9, color=INK_SECOND)

    ax.set_xlim(2, 13)
    ax.set_xticks(K)
    ax.set_ylim(*ylim)
    ax.set_xlabel("SAM region count $K$", fontsize=12, labelpad=7)
    ax.set_ylabel(ylabel + "  ↑ better", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=12)

    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8,
           zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(labelsize=11, length=3, width=0.8)


fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
panel(axes[0], p_auroc, BLUE,   "Pixel AUROC", "Pixel AUROC",
     (0.905, 0.925), "{:.3f}")
panel(axes[1], aupro,   ORANGE, "AUPRO",       "AUPRO",
     (0.76, 0.82),   "{:.3f}")
panel(axes[2], p_f1,    AQUA,   "Pixel max-F1", "Pixel max-F1",
     (0.335, 0.395), "{:.3f}")

fig.tight_layout(w_pad=3.0)
fig.savefig(os.path.join(_HERE, "regK_ablation.pdf"),
           bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "regK_ablation.png"),
           dpi=300, bbox_inches="tight")
print("wrote figures/regK_ablation.{pdf,png}")
