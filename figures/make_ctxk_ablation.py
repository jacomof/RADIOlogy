"""Context-pooling window size k (MVTec AD), swept over {3, 5, 8, 10, 12}
with everything else at the paper configuration. k=5 is highlighted."""
import os
from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 fonts

_HERE = os.path.dirname(os.path.abspath(__file__))


def pct(v):
    """Fraction (4 decimals) -> percent at 1 decimal, rounded half-up like the
    paper's tables (0.9215 reads 92.2)."""
    return float((Decimal(str(v)) * 100).quantize(Decimal("0.1"), ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
k       = [3, 5, 8, 10, 12]
DECL_I  = 1   # index of k=5, the default context size

p_auroc = [pct(v) for v in [0.9149, 0.9215, 0.9203, 0.9175, 0.9150]]
aupro   = [pct(v) for v in [0.8244, 0.8289, 0.8114, 0.7960, 0.7844]]
p_f1    = [pct(v) for v in [0.3716, 0.3898, 0.3858, 0.3767, 0.3695]]

ACCENT = "#e34948"       # accent color for the default k=5
ACCENT_HALO = "#e3494833"
BASE_BLUE = "#6da7ec"    # shared color for swept values
BASE_EDGE = "#2a78d6"
INK = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
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
LABEL_DY = [10, 26, 10, 26, 10]


def panel(ax, y, title, ylim, is_bottom, fmt):
    ax.plot(k, y, color=BASE_EDGE, linewidth=2.0, zorder=2)
    for x, v, dy in zip(k, y, LABEL_DY):
        is_decl = (x == k[DECL_I])
        if is_decl:
            ax.scatter([x], [v], s=460, color=ACCENT_HALO, zorder=3,
                      linewidths=0)
        ax.scatter([x], [v], s=130 if is_decl else 66,
                  marker="*" if is_decl else "o",
                  color=ACCENT if is_decl else BASE_BLUE,
                  edgecolors=SURFACE if is_decl else BASE_EDGE,
                  linewidths=1.1 if is_decl else 0.9,
                  zorder=5 if is_decl else 4)
        ax.annotate(fmt.format(v), (x, v), xytext=(0, dy),
                   textcoords="offset points", ha="center", va="bottom",
                   fontsize=9.5,
                   fontweight="bold" if is_decl else "normal",
                   color=ACCENT if is_decl else INK_SECOND)

    ax.set_xlim(1.5, 13.5)
    ax.set_xticks(k)
    ax.set_ylim(*ylim)
    if is_bottom:
        ax.set_xticklabels(k, fontsize=12)
        ax.set_xlabel("Context window $k$", fontsize=13, labelpad=7)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(title, fontsize=13)

    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8,
           zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(labelsize=12, length=3, width=0.8)


fig, axes = plt.subplots(3, 1, figsize=(3.3, 7.6), sharex=True)
panel(axes[0], p_auroc, "Pixel AUROC (%)",  (85, 95), False, "{:.1f}")
panel(axes[1], aupro,   "AUPRO (%)",        (70, 90), False, "{:.1f}")
panel(axes[2], p_f1,    "Pixel max-F1 (%)", (25, 50), True,  "{:.1f}")

handles = [
    plt.Line2D([0], [0], marker="*", linestyle="none", markersize=13,
              markerfacecolor=ACCENT, markeredgecolor=SURFACE,
              label="Declared ($k$=5)"),
    plt.Line2D([0], [0], marker="o", color=BASE_EDGE, linestyle="-",
              linewidth=2, markersize=8,
              markerfacecolor=BASE_BLUE, markeredgecolor=BASE_EDGE,
              label="Swept $k$"),
]
fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
          bbox_to_anchor=(0.5, 1.045), fontsize=11.5)
fig.suptitle("Context-pooling $k$", fontsize=15, fontweight="bold",
             color=INK, y=1.11)

fig.tight_layout(h_pad=1.3)
fig.savefig(os.path.join(_HERE, "ctxk_ablation.pdf"),
           bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "ctxk_ablation.png"),
           dpi=300, bbox_inches="tight")
print("wrote figures/ctxk_ablation.{pdf,png}")
