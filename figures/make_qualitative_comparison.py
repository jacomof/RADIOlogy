"""Qualitative localization comparison on MVTec AD, VisA and the medical
suite: Image | GT | SAA+ | WinCLIP | FADE | DIVAD | MRAD-TF | Ours.

FADE is not evaluated on the medical suite and is left blank there.
"""
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 fonts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data_mvtec import MVTecAD

OURS_ROOT = "/scratch-shared/<user>/results_ours_qualitative"
SAA_MVTEC = "/scratch-shared/<user>/results_saa_plus/mvtec"
SAA_VISA = "/scratch-shared/<user>/results_saa_plus/visa"
SAA_MED = "/scratch-shared/<user>/results_saa_plus/medical"  # industrial vocabulary
WINCLIP_MVTEC_OFFICIAL = "/scratch-shared/<user>/results_winclip_industrial_official/mvtec/imgs"
WINCLIP_VISA_OFFICIAL = "/scratch-shared/<user>/results_winclip_industrial_official/visa/imgs"
WINCLIP_MED = "/scratch-shared/<user>/results_winclip_medical_v2_industrial_vocab"
FADE_ROOT = "/home/<user>/BMVC-FADE/experiments"
DIVAD_ROOT = "/scratch-shared/<user>/divad_runs"
MRAD_ROOT = "/scratch-shared/<user>/MRAD/results/mrad_qualdump_28661315"

MVTEC_ROOT = "/home/<user>/datasets/mvtecad"
VISA_ROOT = "/home/<user>/datasets/visa_mvtec/1cls"

# ---------------------------------------------------------------------------
# Palette: matches figures/make_teaser_speed_accuracy.py
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECOND = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
ACCENT = "#e34948"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

COLUMNS = ["Image", "GT", "SAA+", "WinCLIP", "FADE", "DIVAD", "MRAD-TF", "Ours"]

SELECTIONS = [
    dict(domain="mvtec", cat="hazelnut", defect_type="hole", stem="015",
         label="MVTec-AD\nhazelnut / hole"),
    dict(domain="mvtec", cat="tile", defect_type="rough", stem="009",
         label="MVTec-AD\ntile / rough"),
    dict(domain="mvtec", cat="carpet", defect_type="cut", stem="012",
         label="MVTec-AD\ncarpet / cut"),
    dict(domain="mvtec", cat="cable", defect_type="cut_inner_insulation", stem="004",
         label="MVTec-AD\ncable / insulation cut"),
    dict(domain="visa", cat="pcb2", defect_type="bad", stem="074",
         label="VisA\npcb2"),
    dict(domain="visa", cat="pcb4", defect_type="bad", stem="023",
         label="VisA\npcb4"),
    dict(domain="medical", cat="tn3k", defect_type="defect", stem="0233",
         label="Medical (TN3K)\nthyroid nodule"),
    dict(domain="medical", cat="clinicdb", defect_type="defect", stem="611",
         label="Medical (ClinicDB)\ncolon polyp"),
]


def load_ours(sel):
    path = os.path.join(OURS_ROOT, sel["domain"] if sel["domain"] != "medical" else "medical",
                         sel["cat"], f"{sel['defect_type']}__{sel['stem']}.npz")
    d = np.load(path, allow_pickle=True)
    img = np.array(Image.open(str(d["path"])).convert("RGB"))
    return img, d["map"]


def sel_root(sel):
    if sel["domain"] == "mvtec":
        return MVTEC_ROOT
    if sel["domain"] == "visa":
        return VISA_ROOT
    return "/scratch-shared/<user>/datasets/adaclip_med_pixel"


def dataset_idx(sel, ds=None):
    ds = ds or MVTecAD(sel_root(sel), sel["cat"], train=False)
    for i, s in enumerate(ds.samples):
        if s["defect_type"] == sel["defect_type"] and \
           os.path.splitext(os.path.basename(s["path"]))[0] == sel["stem"]:
            return i
    raise SystemExit(f"no dataset idx match for {sel}")


def load_npz_map(baseline_dir, idx):
    path = os.path.join(baseline_dir, f"{idx:06d}.npz")
    d = np.load(path)
    return d["map"]


def find_one(root, must_contain, must_not_contain=()):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if all(s in full for s in must_contain) and not any(s in full for s in must_not_contain):
                hits.append(full)
    if len(hits) != 1:
        raise SystemExit(f"expected 1 match under {root} for {must_contain}, got {len(hits)}: {hits}")
    return hits[0]


def load_winclip_industrial_png(sel, native_size):
    """Load a pre-rendered WinCLIP overlay and resize it to the image's true
    aspect ratio (the cached overlay is a fixed square)."""
    root = WINCLIP_MVTEC_OFFICIAL if sel["domain"] == "mvtec" else WINCLIP_VISA_OFFICIAL
    root = os.path.join(root, sel["cat"])
    must = [f"/{sel['stem']}."]
    if sel["domain"] == "mvtec":
        must.append(f"/{sel['defect_type']}/")
    path = find_one(root, must, must_not_contain=["good", "Normal"])
    return Image.open(path).convert("RGB").resize(native_size, Image.BILINEAR)


def load_divad_png(sel, native_size):
    """Load a cached DIVAD heatmap and resize it to the image's true aspect
    ratio (``native_size``: width, height); the cached heatmap is square."""
    ds = sel["domain"] if sel["domain"] != "medical" else sel["cat"]
    root = os.path.join(DIVAD_ROOT, f"qualdump_{ds}_ss40_is100_nis50", sel["cat"])
    path = find_one(root, [f"{sel['defect_type']}_{sel['stem']}.", "heatmap.png"])
    return Image.open(path).convert("RGB").resize(native_size, Image.BILINEAR)


def load_mrad_png(sel, native_size):
    """Load a cached MRAD-TF panel and resize it to the image's true aspect
    ratio; the cached panel is square."""
    if sel["domain"] == "mvtec":
        root = os.path.join(MRAD_ROOT, "mvtec", "visualization", "mvtec")
        fname = f"{sel['cat']}-test-{sel['defect_type']}-{sel['stem']}_MRAD.png"
        img = Image.open(os.path.join(root, fname)).convert("RGB")
    elif sel["domain"] == "visa":
        root = os.path.join(MRAD_ROOT, "visa", "visualization", "visa")
        path = find_one(root, [sel["cat"], "Anomaly", f"-{sel['stem']}.JPG_MRAD.png"])
        img = Image.open(path).convert("RGB")
    else:
        ds_name = {"tn3k": "tn3k", "clinicdb": "ClinicDB"}[sel["cat"]]
        root = os.path.join(MRAD_ROOT, ds_name, "visualization", ds_name)
        path = find_one(root, [f"defect-{sel['stem']}", "_MRAD.png"])
        img = Image.open(path).convert("RGB")
    return img.resize(native_size, Image.BILINEAR)


def load_fade_map(sel):
    # FADE is not evaluated on the medical suite.
    root = os.path.join(FADE_ROOT, sel["domain"], "zeroshot_qualdump",
                         "cm_language_sm_language_clip_gem", "segmentations")
    must = [f"/{sel['cat']}/", f"/{sel['stem']}.png"]
    if sel["domain"] == "mvtec":
        must.append(f"/{sel['defect_type']}/")
    path = find_one(root, must, must_not_contain=["good", "Normal"])
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def overlay(gray_img, score_map, alpha=0.55, cmap_range=(0.08, 0.92)):
    """Alpha-blend a per-image min-max normalized score map (jet colormap)
    over a grayscale version of the input image.

    ``cmap_range`` remaps the normalized score into a narrower sub-band of
    jet before colouring, avoiding jet's near-black ends; purely cosmetic,
    ordering and normalization are unaffected."""
    m = score_map.astype(np.float32)
    m = (m - m.min()) / max(m.max() - m.min(), 1e-8)
    lo, hi = cmap_range
    m = lo + m * (hi - lo)
    cmap = plt.get_cmap("jet")
    heat = cmap(m)[..., :3]
    if gray_img.ndim == 3:
        gray = np.array(Image.fromarray(gray_img).convert("L").convert("RGB"), dtype=np.float32) / 255.0
    else:
        gray = np.stack([gray_img] * 3, axis=-1) / 255.0
    if heat.shape[:2] != gray.shape[:2]:
        heat = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (gray.shape[1], gray.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    return np.clip((1 - alpha) * gray + alpha * heat, 0, 1)


def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


SURFACE_RGB = _hex_to_rgb01(SURFACE)


def pad_to_square(arr, bg=SURFACE_RGB):
    """Letterbox an (H, W, 3) float image in [0, 1] onto a centered square
    canvas, so every panel renders at its true aspect ratio."""
    h, w = arr.shape[:2]
    s = max(h, w)
    canvas = np.empty((s, s, 3), dtype=np.float32)
    canvas[:] = bg
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    return canvas


def na_panel(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    n_rows = len(SELECTIONS)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.08 * n_cols + 0.5, 1.08 * n_rows + 0.5))

    for c, name in enumerate(COLUMNS):
        axes[0, c].set_title(name, fontsize=15, fontweight="bold", color=INK, pad=10)

    for r, sel in enumerate(SELECTIONS):
        img, ours_map = load_ours(sel)
        industrial = sel["domain"] in ("mvtec", "visa")

        ds = MVTecAD(sel_root(sel), sel["cat"], train=False)
        idx = dataset_idx(sel, ds=ds)
        native_size = (img.shape[1], img.shape[0])  # (width, height)

        panels = {}
        panels["Image"] = np.clip(img.astype(np.float32) / 255.0, 0, 1)
        gt = ds.load_mask(idx, size=None).astype(np.float32)
        panels["GT"] = np.stack([gt] * 3, axis=-1)

        if industrial:
            saa_dir = SAA_MVTEC if sel["domain"] == "mvtec" else SAA_VISA
            saa_map = load_npz_map(os.path.join(saa_dir, sel["cat"]), idx)
            panels["SAA+"] = overlay(img, saa_map)
            winclip_png = load_winclip_industrial_png(sel, native_size)
            panels["WinCLIP"] = np.array(winclip_png, dtype=np.float32) / 255.0
            fade_map = load_fade_map(sel)
            panels["FADE"] = overlay(img, fade_map)
        else:
            saa_map = load_npz_map(os.path.join(SAA_MED, sel["cat"]), idx)
            winclip_map = load_npz_map(os.path.join(WINCLIP_MED, sel["cat"]), idx)
            panels["SAA+"] = overlay(img, saa_map)
            panels["WinCLIP"] = overlay(img, winclip_map)
            panels["FADE"] = None  # N/A: excluded from medical suite by design

        panels["DIVAD"] = np.array(load_divad_png(sel, native_size), dtype=np.float32) / 255.0
        panels["MRAD-TF"] = np.array(load_mrad_png(sel, native_size), dtype=np.float32) / 255.0
        panels["Ours"] = overlay(img, ours_map)

        for c, name in enumerate(COLUMNS):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(GRID)
            p = panels[name]
            if p is None:
                na_panel(ax)
            else:
                ax.imshow(pad_to_square(p))
            if name == "Ours":
                for s in ax.spines.values():
                    s.set_edgecolor(ACCENT)
                    s.set_linewidth(2.0)
        axes[r, 0].set_ylabel(str(r + 1), fontsize=16, fontweight="bold",
                               color=INK_SECOND, rotation=0, ha="right",
                               va="center", labelpad=10)

    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    out_pdf = os.path.join(REPO, "figures/qualitative_comparison.pdf")
    out_png = os.path.join(REPO, "figures/qualitative_comparison.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
