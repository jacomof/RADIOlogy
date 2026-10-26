"""Throughput of the raw-CLIP baseline (CLIP ViT-L/14@336), timed like main.py.

Only the raw patch-text margin and the pooled-embedding image score: no
context or region pooling, no affinity refinement, no foreground mask. The
map is still upsampled and blurred to 518px inside the timed region.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.backbones import build_backbone
from src.basis import blur
from src.data_mvtec import build_test_loader
from src.zeroshot import patch_margin, text_anchors, STATE_VOCABULARIES


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="Root of an MVTec-AD-layout dataset.")
    ap.add_argument("--categories", nargs="+",
                    default=["bottle", "cable", "capsule", "carpet"])
    ap.add_argument("--model-name", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--img-size", type=int, default=336)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--pixel-size", type=int, default=518)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--prompt-name", default=None)
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone("clip", model_name=args.model_name, img_size=args.img_size,
                              want_text=True, want_dino=False, want_sam=False)
    print(f"[clip-raw] backbone={backbone.name} model={args.model_name} "
          f"grid={backbone.grid} img_size={args.img_size}", flush=True)

    vocab_n, vocab_a = STATE_VOCABULARIES["industrial"]
    timing = {"seconds": 0.0, "images": 0}

    for cat in args.categories:
        loader = build_test_loader(args.data_root, cat, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers, require_masks=True)
        t_n, t_a = text_anchors(backbone, cat, name=args.prompt_name,
                                normal_states=vocab_n, anomaly_states=vocab_a)
        n_batches = 0
        for batch in loader:
            n_batches += 1
            warm = n_batches <= args.warmup_batches
            images = batch["image"].to(device, non_blocking=True)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            feats = backbone.encode(images)
            grid = feats.grid
            m = patch_margin(feats.clip_text_patch, t_n, t_a)          # (B, N) raw margin
            g = F.normalize(feats.clip_summary.float(), dim=-1)
            im_s = g @ (t_a - t_n).float()
            up = F.interpolate(m.reshape(-1, 1, *grid).float(), size=(args.pixel_size,) * 2,
                               mode="bilinear", align_corners=False).squeeze(1)
            out_map = blur(up, args.sigma)

            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if not warm:
                timing["seconds"] += dt
                timing["images"] += images.shape[0]
            _ = (out_map, im_s)   # produced but not scored: throughput only

        print(f"[{cat}] done, {n_batches} batches", flush=True)

    ips = timing["images"] / max(timing["seconds"], 1e-9)
    print(f"\n[throughput] {timing['images']} images (post-warmup) in "
          f"{timing['seconds']:.3f}s -> {ips:.2f} images/sec "
          f"({1000.0 / max(ips, 1e-9):.2f} ms/image)", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "throughput_images_per_sec": ips}, fh, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
