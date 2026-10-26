"""Region-pooling K-set (MVTec AD): every non-empty subset of {8, 32, 128},
everything else at the paper configuration. {8, 32} is highlighted."""
import os

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data (order as specified: singles, then pairs, then the triple)
# ---------------------------------------------------------------------------
labels   = ["{8}", "{32}", "{128}", "{8,32}", "{32,128}", "{8,128}", "{8,32,128}"]
declared = [False, False, False, True, False, False, False]
# x-positions: wider spacing within the "pairs" group (longer labels) and a
# gap between groups (singles | pairs | triple).
xpos     = [0, 1, 2, 4, 6.0, 7.7, 10.2]

p_auroc = [0.9209, 0.9175, 0.9121, 0.9215, 0.9165, 0.9196, 0.9200]
aupro   = [0.8143, 0.8278, 0.8234, 0.8289, 0.8315, 0.8305, 0.8331]
p_f1    = [0.3825, 0.3819, 0.3664, 0.3898, 0.3787, 0.3833, 0.3860]

ACCENT = "#e34948"       # accent color for the default {8,32} pair
ACCENT_HALO = "#e3494833"
BASE_BLUE = "#6da7ec"    # shared color for alternative K-sets
BASE_EDGE = "#2a78d6"
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

LABEL_DY = [9, 20, 9, 20, 9, 20, 9]   # stagger heights so close labels don't collide


def panel(ax, y, title, ylabel, ylim, fmt):
    for x, v, is_decl, dy in zip(xpos, y, declared, LABEL_DY):
        if is_decl:
            ax.scatter([x], [v], s=560, color=ACCENT_HALO, zorder=2,
                      linewidths=0)
        ax.scatter([x], [v], s=140 if is_decl else 78,
                  marker="*" if is_decl else "o",
                  color=ACCENT if is_decl else BASE_BLUE,
                  edgecolors=SURFACE if is_decl else BASE_EDGE,
                  linewidths=1.2 if is_decl else 1.0,
                  zorder=4 if is_decl else 3)
        ax.annotate(fmt.format(v), (x, v), xytext=(0, dy),
                   textcoords="offset points", ha="center", va="bottom",
                   fontsize=9,
                   fontweight="bold" if is_decl else "normal",
                   color=ACCENT if is_decl else INK_SECOND)

    ax.set_xlim(-0.8, 11.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Region-pooling $K$-set", fontsize=12, labelpad=7)
    ax.set_ylabel(ylabel + "  ↑ better", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=12)

    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8,
           zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(labelsize=10.5, length=3, width=0.8)


fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
panel(axes[0], p_auroc, "Pixel AUROC",  "Pixel AUROC",  (0.907, 0.9235), "{:.3f}")
panel(axes[1], aupro,   "AUPRO",        "AUPRO",        (0.805, 0.840),  "{:.3f}")
panel(axes[2], p_f1,    "Pixel max-F1", "Pixel max-F1", (0.354, 0.393),  "{:.3f}")

handles = [
    plt.Line2D([0], [0], marker="*", linestyle="none", markersize=13,
              markerfacecolor=ACCENT, markeredgecolor=SURFACE,
              label="Declared ({8,32})"),
    plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
              markerfacecolor=BASE_BLUE, markeredgecolor=BASE_EDGE,
              label="Alternative $K$-set"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
          bbox_to_anchor=(0.5, 1.06), fontsize=11)

fig.tight_layout(w_pad=3.0)
fig.savefig(os.path.join(_HERE, "regKset_ablation.pdf"),
           bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "regKset_ablation.png"),
           dpi=300, bbox_inches="tight")
print("wrote figures/regKset_ablation.{pdf,png}")
