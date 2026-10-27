"""Recombine KSDD bands scored by main_zeroshot.py and evaluate on the original frames.

Pixel map: each band's field average is upsampled to its share of the output
rows and the bands are stacked. Image score: the maximum global margin over a
frame's bands. Metrics use the untiled dataset's masks. No foreground mask is
applied, because the language gate exempts every KSDD band anyway.

Usage: python scripts/ksdd_stitch.py <stage-1 output dir> --orig-root /path/to/ksdd
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_mvtec import MVTecAD
from src.metrics import compute_image_metrics
from src.pixel_metrics import PixelAccumulator

_BAND_RE = re.compile(r"^(.*)_b(\d+)$")
DEFAULT_FIELDS = ["class_reg8mix", "class_reg32mix",
                  "class_reg8mix_refined", "class_reg32mix_refined"]


def upsample_band(scores: np.ndarray, grid, out_h: int, out_w: int, sigma: float) -> np.ndarray:
    """Bilinear upsampling plus Gaussian blur to a non-square target."""
    x = torch.from_numpy(np.asarray(scores, dtype=np.float32)).reshape(1, 1, *grid)
    out = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0].numpy()
    return gaussian_filter(out, sigma=sigma) if sigma > 0 else out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score_dir", help="main_zeroshot.py --output-dir of the banded run.")
    ap.add_argument("--orig-root", required=True, help="Untiled KSDD in MVTec layout.")
    ap.add_argument("--category", default="ksdd")
    ap.add_argument("--pixel-size", type=int, default=518)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    args = ap.parse_args()

    d = np.load(os.path.join(args.score_dir, "scores", f"{args.category}.npz"))
    if "global_margin" not in d:
        raise SystemExit("No global_margin in the cache; rerun stage 1 with this version "
                         "of main_zeroshot.py.")
    grid = tuple(int(v) for v in d["grid"])
    combo = np.mean([d[f"field_{f}"] for f in args.fields], axis=0)
    global_margin = d["global_margin"]

    groups = {}
    for i, p in enumerate(d["paths"]):
        m = _BAND_RE.match(os.path.splitext(os.path.basename(str(p)))[0])
        if not m:
            raise SystemExit(f"Not a band file (<stem>_b<k>): {p}")
        groups.setdefault(m.group(1), {})[int(m.group(2))] = i

    ds = MVTecAD(args.orig_root, args.category, train=False, require_masks=True)
    stem_to_idx = {os.path.splitext(os.path.basename(s["path"]))[0]: i
                   for i, s in enumerate(ds.samples)}
    unmatched = set(groups) - set(stem_to_idx)
    if unmatched:
        raise SystemExit(f"{len(unmatched)} band stems not in the original dataset, "
                         f"e.g. {sorted(unmatched)[:3]}")

    size = args.pixel_size
    acc = PixelAccumulator()
    scores, labels, defects = [], [], []
    for stem, bands in sorted(groups.items(), key=lambda kv: stem_to_idx[kv[0]]):
        idx = stem_to_idx[stem]
        order = [bands[k] for k in sorted(bands)]
        edges = [round(k * size / len(order)) for k in range(len(order) + 1)]
        stitched = np.concatenate(
            [upsample_band(combo[b].reshape(grid), grid, edges[k + 1] - edges[k], size, args.sigma)
             for k, b in enumerate(order)], axis=0)
        acc.add(stitched, ds.load_mask(idx, size=size))
        scores.append(max(float(global_margin[b]) for b in order))
        labels.append(int(ds.samples[idx]["label"]))
        defects.append(ds.samples[idx]["defect_type"])

    img = compute_image_metrics(scores, labels, defects)["overall"]
    pix = acc.compute()
    print(f"{len(groups)} frames, {len(combo)} bands, fields={'+'.join(args.fields)}")
    print(f"[image] AUROC={img['auroc']*100:.2f}  AP={img['ap']*100:.2f}  "
          f"max-F1={img['f1']*100:.2f}")
    print(f"[pixel] AUROC={pix['pixel_auroc']*100:.2f}  AUPRO={pix['aupro']*100:.2f}  "
          f"AP={pix['pixel_ap']*100:.2f}  max-F1={pix['pixel_f1']*100:.2f}")


if __name__ == "__main__":
    main()
