"""Region-pooling K-set (MVTec AD), raw-value view of the
make_regKset_ablation.py sweep with the [min, max] spread per metric."""
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
# Data (identical to make_regKset_ablation.py)
# ---------------------------------------------------------------------------
labels   = ["{8}", "{32}", "{128}", "{8,32}", "{32,128}", "{8,128}", "{8,32,128}"]
declared = [False, False, False, True, False, False, False]
# x-positions tuned to the narrower single-column axis width, with extra
# spacing at the group boundaries.
xpos     = [0, 0.24, 0.50, 0.93, 1.25, 1.57, 2.08]

p_auroc = [pct(v) for v in [0.9209, 0.9175, 0.9121, 0.9215, 0.9165, 0.9196, 0.9200]]
aupro   = [pct(v) for v in [0.8143, 0.8278, 0.8234, 0.8289, 0.8315, 0.8305, 0.8331]]
p_f1    = [pct(v) for v in [0.3825, 0.3819, 0.3664, 0.3898, 0.3787, 0.3833, 0.3860]]

ACCENT = "#e34948"       # accent color for the default {8,32} pair
ACCENT_HALO = "#e3494833"
BASE_BLUE = "#6da7ec"    # shared color for alternative K-sets
BASE_EDGE = "#2a78d6"
BAND = "#dedcd3"
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

LABEL_DY = [9, 22, 9, 22, 9, 22, 9]   # stagger heights so close labels don't collide


def panel(ax, y, ylabel, ylim, is_bottom, fmt):
    lo, hi = min(y), max(y)
    ax.axhspan(lo, hi, color=BAND, alpha=0.6, zorder=0)

    for x, v, is_decl, dy in zip(xpos, y, declared, LABEL_DY):
        if is_decl:
            ax.scatter([x], [v], s=440, color=ACCENT_HALO, zorder=2,
                      linewidths=0)
        ax.scatter([x], [v], s=120 if is_decl else 66,
                  marker="*" if is_decl else "o",
                  color=ACCENT if is_decl else BASE_BLUE,
                  edgecolors=SURFACE if is_decl else BASE_EDGE,
                  linewidths=1.1 if is_decl else 0.9,
                  zorder=4 if is_decl else 3)
        ax.annotate(fmt.format(v), (x, v), xytext=(0, dy),
                   textcoords="offset points", ha="center", va="bottom",
                   fontsize=9.5,
                   fontweight="bold" if is_decl else "normal",
                   color=ACCENT if is_decl else INK_SECOND)

    ax.text(0.98, 0.045, f"range {fmt.format(hi - lo)} pt",
           transform=ax.transAxes, ha="right", va="bottom",
           fontsize=9.5, color=INK_MUTED, style="italic")

    ax.set_xlim(-0.25, 2.23)
    ax.set_xticks(xpos)
    if is_bottom:
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=9.5)
        ax.set_xlabel("Region-pooling $K$-set", fontsize=13, labelpad=7)
    else:
        ax.set_xticklabels([])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=13)

    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8,
           zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(labelsize=11.5, length=3, width=0.8)


# Axis ranges are wide relative to the data spread, so small differences
# between configurations do not appear visually exaggerated.
fig, axes = plt.subplots(3, 1, figsize=(3.3, 9.4), sharex=True)
panel(axes[0], p_auroc, "Pixel AUROC (%)",  (85, 95), False, "{:.1f}")
panel(axes[1], aupro,   "AUPRO (%)",        (70, 90), False, "{:.1f}")
panel(axes[2], p_f1,    "Pixel max-F1 (%)", (25, 50), True,  "{:.1f}")

handles = [
    plt.Line2D([0], [0], marker="*", linestyle="none", markersize=13,
              markerfacecolor=ACCENT, markeredgecolor=SURFACE,
              label="Declared ({8,32})"),
    plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
              markerfacecolor=BASE_BLUE, markeredgecolor=BASE_EDGE,
              label="Alternative $K$-set"),
    plt.Rectangle((0, 0), 1, 1, facecolor=BAND, edgecolor="none",
                 label="full [min, max] span"),
]
fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
          bbox_to_anchor=(0.5, 1.05), fontsize=11.5)
fig.suptitle("Region-pooling $K$-set", fontsize=15, fontweight="bold",
             color=INK, y=1.135)

fig.tight_layout(h_pad=1.3)
fig.savefig(os.path.join(_HERE, "regKset_stability.pdf"),
           bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "regKset_stability.png"),
           dpi=300, bbox_inches="tight")
print("wrote figures/regKset_stability.{pdf,png}")
