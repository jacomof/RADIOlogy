"""Teaser: mean industrial image/pixel AUROC vs. latency (ms/image, log
scale) for every training-free method. Latency uses the same warmup-excluded,
CUDA-synchronized timing as main.py."""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
methods = ["CLIP", "SAA", "WinCLIP", "FADE", "DIVAD", "MRAD-TF", "Ours"]
i_auroc = [84.8, 66.1, 82.8, 80.5, 72.5, 79.5, 92.6]   # image-level, industrial mean
p_auroc = [36.5, 71.1, 80.7, 91.7, 91.4, 88.4, 95.2]   # pixel-level, industrial mean
latency = [10.9, 588.5, 249.4, 266.5, 4490.0, 83.4, 35.7]   # ms / image

ACCENT = "#e34948"       # accent color for "Ours"
ACCENT_HALO = "#e3494833"
BASE_BLUE = "#6da7ec"    # shared color for baseline methods
BASE_EDGE = "#2a78d6"
INK = "#000000"
INK_SECOND = "#000000"
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

# Per-method label offsets (dx, dy in points, alignment).
LABEL_OFFSETS_I = {
    "CLIP": (0, 14, "center"), "SAA": (0, -14, "center"),
    "WinCLIP": (12, 10, "left"), "FADE": (0, -18, "center"),
    "DIVAD": (0, 14, "center"), "MRAD-TF": (4, 10, "right"),
    "Ours": (0, 16, "center"),
}
LABEL_OFFSETS_P = {
    "CLIP": (0, 14, "center"), "SAA": (0, -14, "center"),
    "WinCLIP": (12, 0, "left"), "FADE": (0, 14, "center"),
    "DIVAD": (0, 14, "center"), "MRAD-TF": (-12, -10, "right"),
    "Ours": (0, 15, "center"),
}
LEADER_LINE = {"WinCLIP", "FADE", "MRAD-TF"}


def panel(ax, y, offsets, title, ylabel, ylim):
    for m, x, yv in zip(methods, latency, y):
        is_ours = m == "Ours"
        if is_ours:
            ax.scatter([x], [yv], s=620, color=ACCENT_HALO, zorder=2,
                      linewidths=0)
        ax.scatter([x], [yv], s=150 if is_ours else 88,
                  marker="*" if is_ours else "o",
                  color=ACCENT if is_ours else BASE_BLUE,
                  edgecolors=SURFACE if is_ours else BASE_EDGE,
                  linewidths=1.3 if is_ours else 1.0,
                  zorder=4 if is_ours else 3)
        dx, dy, ha = offsets[m]
        arrowprops = (dict(arrowstyle="-", color=INK_MUTED, lw=0.7,
                           shrinkA=0, shrinkB=5, alpha=0.8)
                     if m in LEADER_LINE else None)
        ax.annotate(m, (x, yv), xytext=(dx, dy), textcoords="offset points",
                   ha=ha, va="center",
                   fontsize=11.5 if is_ours else 10.5,
                   fontweight="bold" if is_ours else "normal",
                   color=ACCENT if is_ours else INK_SECOND, zorder=5,
                   arrowprops=arrowprops)

    ax.set_xscale("log")
    ax.set_xlim(6, 9000)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Latency (ms / image, log scale)  ↓ better",
                 fontsize=12, labelpad=7)
    ax.set_ylabel(ylabel + "  ↑ better", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=12)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8,
           zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(labelsize=11, length=3, width=0.8, labelcolor=INK_SECOND)
    ax.tick_params(which="minor", length=1.5, width=0.6)


fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.85))  # two-column width
panel(axes[0], i_auroc, LABEL_OFFSETS_I, "Detection", "Image AUROC (%)",
     (60, 100))
panel(axes[1], p_auroc, LABEL_OFFSETS_P, "Localization", "Pixel AUROC (%)",
     (30, 100))

fig.suptitle("", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 1), w_pad=4.5)
fig.savefig(os.path.join(_HERE, "teaser_speed_accuracy.pdf"),
           bbox_inches="tight")
fig.savefig(os.path.join(_HERE, "teaser_speed_accuracy.png"),
           dpi=300, bbox_inches="tight")
print("wrote figures/teaser_speed_accuracy.{pdf,png}")
