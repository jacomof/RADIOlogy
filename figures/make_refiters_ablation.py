"""Affinity-diffusion iterations (``--refine-iters``, MVTec AD), swept over
{1, 2, 5, 10} with everything else at the paper configuration. iters=1 is
highlighted."""
import os

import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
iters   = [1, 2, 5, 10]
DECL_I  = 0   # index of iters=1, the default diffusion step count

p_auroc = [0.9215, 0.9231, 0.9247, 0.9241]
aupro   = [0.8289, 0.8286, 0.8222, 0.8105]
p_f1    = [0.3898, 0.3954, 0.3991, 0.3922]

ACCENT = "#e34948"       # accent color for the default iters=1
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

# Offsets keep the crowded iters=1/iters=2 labels from overlapping.
LABEL_DY = [-22, 10, 10, 10]
LABEL_DX = [5, 0, 0, 0]
LABEL_HA = ["center", "center", "center", "center"]


def panel(ax, y, title, ylim, is_bottom, fmt):
    ax.plot(iters, y, color=BASE_EDGE, linewidth=2.0, zorder=2)
    for x, v, dy, dx, ha in zip(iters, y, LABEL_DY, LABEL_DX, LABEL_HA):
        is_decl = (x == iters[DECL_I])
        if is_decl:
            ax.scatter([x], [v], s=460, color=ACCENT_HALO, zorder=3,
                      linewidths=0)
        ax.scatter([x], [v], s=130 if is_decl else 66,
                  marker="*" if is_decl else "o",
                  color=ACCENT if is_decl else BASE_BLUE,
                  edgecolors=SURFACE if is_decl else BASE_EDGE,
                  linewidths=1.1 if is_decl else 0.9,
                  zorder=5 if is_decl else 4)
        ax.annotate(fmt.format(v), (x, v), xytext=(dx, dy),
                   textcoords="offset points", ha=ha, va="bottom",
                   fontsize=9.5,
                   fontweight="bold" if is_decl else "normal",
                   color=ACCENT if is_decl else INK_SECOND)

    ax.set_xlim(0, 11)
    ax.set_xticks(iters)
    ax.set_ylim(*ylim)
    if is_bottom:
        ax.set_xticklabels(iters, fontsize=12)
        ax.set_xlabel("Diffusion iterations", fontsize=13, labelpad=7)
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
# Same y-ranges as the companion context-window (k) figure, for comparability
# and so this sweep's smaller spread is not visually exaggerated.
panel(axes[0], p_auroc, "Pixel AUROC",  (0.85, 0.95), False, "{:.3f}")
panel(axes[1], aupro,   "AUPRO",        (0.70, 0.90), False, "{:.3f}")
panel(axes[2], p_f1,    "Pixel max-F1", (0.25, 0.50), True,  "{:.3f}")

handles = [
    plt.Line2D([0], [0], marker="*", linestyle="none", markersize=13,
              markerfacecolor=ACCENT, markeredgecolor=SURFACE,
              label="Declared (iters=1)"),
    plt.Line2D([0], [0], marker="o", color=BASE_EDGE, linestyle="-",
              linewidth=2, markersize=8,
              markerfacecolor=BASE_BLUE, markeredgecolor=BASE_EDGE,
              label="Swept iterations"),
]
fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
          bbox_to_anchor=(0.5, 1.045), fontsize=11.5)
fig.suptitle("Affinity diffusion iterations", fontsize=15, fontweight="bold",
             color=INK, y=1.11)

fig.tight_layout(h_pad=1.3)
fig.savefig(os.path.join(_HERE, "refiters_ablation.pdf"),
           bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "refiters_ablation.png"),
           dpi=300, bbox_inches="tight")
print("wrote figures/refiters_ablation.{pdf,png}")
