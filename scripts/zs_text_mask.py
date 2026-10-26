"""Stage 2: cache the foreground/background text margins for the language gate.

Per category and vocabulary: the per-patch margin (``fg_*``), the same on
5x5 context-pooled patches (``fgctx_*``) and the MAP-embedding margin
(``glob_*``).
"""

import argparse
import json
import os
import socket
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backbones import build_backbone
from src.data_mvtec import MVTEC_AD_CLASS_NAMES, build_test_loader
from src.utils import neighborhood_aggregate
from src.zeroshot import FGBG_VOCABULARIES, fgbg_anchors, patch_margin

DEFAULT_OUT = "results_zs_textmask"

# The default stage-1 context window.
CTX = 5


@torch.inference_mode()
def encode_category(backbone, loader, anchors, ctx: int) -> dict:
    """One pass over a category's test split -> fg/bg margins, patch and pooled."""
    out: dict = {v: {"fg": [], "fgctx": [], "glob": []} for v in anchors}
    cos: dict = {"fgcos": [], "bgcos": []}
    labels, paths, grid = [], [], None

    for batch in tqdm(loader, desc="encode", leave=False):
        feats = backbone.encode(batch["image"])
        grid = feats.grid
        tp = feats.clip_text_patch
        tp_ctx = neighborhood_aggregate(tp, grid, ctx) if ctx > 1 else tp
        summ = feats.clip_summary
        for vocab, (fg, bg) in anchors.items():
            o = out[vocab]
            o["fg"].append(patch_margin(tp, bg, fg).cpu().numpy().astype(np.float32))
            o["fgctx"].append(patch_margin(tp_ctx, bg, fg).cpu().numpy().astype(np.float32))
            o["glob"].append(patch_margin(summ, bg, fg).cpu().numpy().astype(np.float32))
        # Diagnostic only: no scoring path reads these.
        fg, bg = anchors["keyword"]
        p = F.normalize(tp.float(), dim=-1)
        cos["fgcos"].append((p @ fg.to(p.device)).cpu().numpy().astype(np.float32))
        cos["bgcos"].append((p @ bg.to(p.device)).cpu().numpy().astype(np.float32))
        labels.extend(int(x) for x in batch["label"])
        paths.extend(batch["path"])

    arrays = {f"{k}_{v}": np.concatenate(o[k], 0)
              for v, o in out.items() for k in ("fg", "fgctx", "glob")}
    arrays.update({k: np.concatenate(v, 0) for k, v in cos.items()})
    arrays.update(labels=np.asarray(labels, dtype=np.int64),
                  paths=np.array(paths), grid=np.asarray(grid))
    return arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="Root of an MVTec-AD-layout dataset.")
    ap.add_argument("--name", default=None,
                    help="Subdirectory of --out (default: the data-root's basename).")
    ap.add_argument("--categories", nargs="+", default=MVTEC_AD_CLASS_NAMES)
    ap.add_argument("--backbone", default="c-radio_v4-h")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--patch-context", type=int, default=CTX)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-masks", action="store_true",
                    help="The category ships no ground-truth masks. Nothing "
                         "here reads a mask; this only stops the loader from "
                         "demanding one.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    name = a.name or os.path.basename(a.data_root.rstrip("/"))
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)

    backbone = build_backbone(a.backbone, img_size=a.img_size, want_text=True,
                              model_name=a.model_name)
    if not backbone.has_text:
        raise SystemExit(f"'{a.backbone}' has no text head; the language mask is "
                         f"nothing but a text head.")
    print(f"[textmask] {name}: backbone={backbone.name} grid={backbone.grid} "
          f"img_size={backbone.img_size} ctx={a.patch_context} "
          f"vocabularies={sorted(FGBG_VOCABULARIES)}", flush=True)

    with open(os.path.join(out, "cache_config.json"), "w") as fh:
        json.dump({"argv": sys.argv, "name": name,
                   "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "hostname": socket.gethostname(),
                   "backbone": a.backbone, "img_size": a.img_size,
                   "grid": list(backbone.grid), "patch_context": a.patch_context,
                   "data_root": a.data_root, "categories": list(a.categories),
                   # Stored expanded so the cache records the exact words.
                   "vocabularies": {k: [list(v[0]), list(v[1])]
                                    for k, v in FGBG_VOCABULARIES.items()}}, fh, indent=2)

    for category in a.categories:
        loader = build_test_loader(a.data_root, category,
                                   img_size=backbone.img_size,
                                   batch_size=a.batch_size,
                                   num_workers=a.num_workers,
                                   require_masks=not a.no_masks)
        anchors = {v: fgbg_anchors(backbone, category, vocab=v)
                   for v in FGBG_VOCABULARIES}
        arrays = encode_category(backbone, loader, anchors, a.patch_context)
        np.savez(os.path.join(out, f"{category}.npz"), **arrays)
        # Coverage at the zero split; 0% or 100% signals a vocabulary failure.
        fg = arrays["fg_keyword"]
        print(f"  {category:<16} {len(fg):>4} imgs  "
              f"fg>0 coverage patch {float((fg > 0).mean()):.3f}  "
              f"ctx {float((arrays['fgctx_keyword'] > 0).mean()):.3f}  "
              f"| MAP margin p10/p50/p90 "
              f"{np.percentile(arrays['glob_keyword'], 10):+.4f}/"
              f"{np.median(arrays['glob_keyword']):+.4f}/"
              f"{np.percentile(arrays['glob_keyword'], 90):+.4f}", flush=True)

    print(f"[textmask] wrote {len(a.categories)} categories to {out}")


if __name__ == "__main__":
    main()
