"""Failure cases: Image | GT | Raw (unmasked) | Final (masked).

Rows 1-2: the foreground mask removes part of a correctly localized defect.
Rows 3-4: the mask keeps the defect, but the raw score misses it. The cache
only stores the masked map, so ``compute_raw_final`` rebuilds the raw one.
"""
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 fonts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "figures"))

from src.data_mvtec import MVTecAD  # noqa: E402
from src.pixel_metrics import upsample_map  # noqa: E402
from zs_combine import _colour_fg, _text_fg  # noqa: E402
import make_qualitative_comparison as mq  # noqa: E402 (palette, overlay, pad_to_square, sel_root, dataset_idx, OURS_ROOT)

_COMBO = ["class_reg8mix", "class_reg32mix", "class_reg8mix_refined", "class_reg32mix_refined"]

COLUMNS = ["Image", "GT", "Raw", "Final"]
N_MASK_ROWS = 2  # first N_MASK_ROWS rows are the mask-failure group, rest are VL-failure

SELECTIONS = [
    dict(domain="mvtec", cat="pill", defect_type="crack", stem="012",
         row_label="MVTec-AD\npill / crack"),
    dict(domain="visa", cat="fryum", defect_type="bad", stem="018",
         row_label="VisA\nfryum / bad"),
    dict(domain="visa", cat="chewinggum", defect_type="bad", stem="032",
         row_label="VisA\nchewinggum / bad"),
    dict(domain="medical", cat="colondb", defect_type="defect", stem="107",
         row_label="Medical\ncolondb / defect"),
]


def declared_config(sel):
    """Score-cache / text-gate configuration for one domain."""
    if sel["domain"] == "mvtec":
        return dict(score_dir=os.path.join(REPO, "results_zs_regS"),
                    text_dir=os.path.join(REPO, "results_zs_textmask", "mvtec"),
                    vocab="objsurf", gate=0.016)
    if sel["domain"] == "visa":
        return dict(score_dir=os.path.join(REPO, "results_zs_visa"),
                    text_dir=os.path.join(REPO, "results_zs_textmask", "visa"),
                    vocab="objsurf", gate=0.016)
    return dict(score_dir=os.path.join(REPO, "results_zs_medpix_clinical"),
                text_dir=os.path.join(REPO, "results_zs_textmask", sel["cat"]),
                vocab="anatex", gate=0.014)


def load_ours_npz(sel):
    path = os.path.join(mq.OURS_ROOT, sel["domain"] if sel["domain"] != "medical" else "medical",
                         sel["cat"], f"{sel['defect_type']}__{sel['stem']}.npz")
    d = np.load(path, allow_pickle=True)
    img_path = str(d["path"])
    img = np.array(Image.open(img_path).convert("RGB"))
    return img, d["map"].astype(np.float32), img_path


def compute_raw_final(sel, img_path, cached_final_map, size=518, sigma=4.0):
    cfg = declared_config(sel)
    score_path = os.path.join(cfg["score_dir"], "scores", f"{sel['cat']}.npz")
    d = np.load(score_path, allow_pickle=True)
    grid = tuple(int(v) for v in d["grid"])
    stack = np.mean([d[f"field_{f}"] for f in _COMBO], axis=0)

    cache_rows = {}
    for i, p in enumerate(d["paths"]):
        cache_rows.setdefault(os.path.basename(str(p)), []).append(i)
    bn = os.path.basename(img_path)
    cand = cache_rows.get(bn, [])
    if not cand:
        raise SystemExit(f"no score-cache row match for {bn} in {score_path}")
    idx = cand[0]
    if len(cand) > 1:
        for c in cand:
            if sel["defect_type"] in str(d["paths"][c]):
                idx = c
                break

    raw_patch = stack[idx]
    raw_map = upsample_map(raw_patch, grid, size, sigma=sigma)

    _, tglob = _text_fg(cfg["text_dir"], sel["cat"], d["paths"], grid, cfg["vocab"], "fgctx", "zero")
    exempt = tglob < cfg["gate"]
    mask_i, _ = _colour_fg(img_path, grid)
    mask_i = mask_i | bool(exempt[idx])
    fill = float(raw_patch.min())
    final_recompute = np.where(mask_i, raw_patch, fill)
    final_map = upsample_map(final_recompute, grid, size, sigma=sigma)

    diff = float(np.abs(final_map - cached_final_map).max())
    if diff > 1e-3:
        raise SystemExit(f"{sel['domain']}/{sel['cat']}/{sel['stem']}: raw/final "
                          f"reconstruction diverges from cached final map by {diff:.5f}")
    return raw_map, final_map


def main():
    n_rows = len(SELECTIONS)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols + 0.5, 1.6 * n_rows + 1.0))

    for c, name in enumerate(COLUMNS):
        axes[0, c].set_title(name, fontsize=16, fontweight="bold", color=mq.INK, pad=10)

    for r, sel in enumerate(SELECTIONS):
        img, final_map, img_path = load_ours_npz(sel)
        raw_map, _ = compute_raw_final(sel, img_path, final_map)

        ds = MVTecAD(mq.sel_root(sel), sel["cat"], train=False)
        idx = mq.dataset_idx(sel, ds=ds)
        gt = ds.load_mask(idx, size=None).astype(np.float32)

        panels = {
            "Image": np.clip(img.astype(np.float32) / 255.0, 0, 1),
            "GT": np.stack([gt] * 3, axis=-1),
            "Raw": mq.overlay(img, raw_map),
            "Final": mq.overlay(img, final_map),
        }

        for c, name in enumerate(COLUMNS):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(mq.GRID)
            ax.imshow(mq.pad_to_square(panels[name]))
            if name == "Final":
                for s in ax.spines.values():
                    s.set_edgecolor(mq.ACCENT)
                    s.set_linewidth(2.0)
        axes[r, 0].set_ylabel(sel["row_label"], fontsize=12, fontweight="bold",
                               color=mq.INK_SECOND, rotation=0, ha="right",
                               va="center", labelpad=10)

    fig.tight_layout(h_pad=0.8, w_pad=0.3)

    # Open a gap between the two row-groups for a divider and labels.
    gap = 0.05
    for r in range(N_MASK_ROWS, n_rows):
        for c in range(n_cols):
            pos = axes[r, c].get_position()
            axes[r, c].set_position([pos.x0, pos.y0 - gap, pos.width, pos.height])

    pos_top = axes[N_MASK_ROWS - 1, 0].get_position()
    pos_bot = axes[N_MASK_ROWS, 0].get_position()
    y_div = (pos_top.y0 + pos_bot.y1) / 2
    fig.add_artist(plt.Line2D([0.03, 0.99], [y_div, y_div], transform=fig.transFigure,
                               color=mq.INK_MUTED, linewidth=1.2))
    fig.text(0.51, y_div + 0.010, "Foreground-mask failure", fontsize=12, fontweight="bold",
              color=mq.INK_SECOND, ha="center", va="bottom")
    fig.text(0.51, y_div - 0.010, "Vision-language alignment failure", fontsize=12, fontweight="bold",
              color=mq.INK_SECOND, ha="center", va="top")

    out_pdf = os.path.join(REPO, "figures/failure_cases.pdf")
    out_png = os.path.join(REPO, "figures/failure_cases.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
