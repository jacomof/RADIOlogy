"""Single-pass zero-shot anomaly detection with throughput measurement.

Runs the paper configuration (src.basis.BasisConfig defaults) in memory, one
forward pass per image, and matches the cached pipeline (main_zeroshot.py +
scripts/zs_combine.py). Timing excludes the warmup batches and --verify.
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
from src.basis import BasisConfig, blur, deployed_basis
from src.data_mvtec import MVTEC_AD_CLASS_NAMES, build_test_loader
from src.metrics import average_over_categories, compute_image_metrics
from src.pixel_metrics import PixelAccumulator, format_pixel_metrics
from src.zeroshot import REFINE_MODES, fgbg_anchors, patch_margin, text_anchors

# deployed_basis's own head dict only wires {none, dino, sam, trunk, self} --
# "native" is main_zeroshot.py-only.
_E2E_REFINE_CHOICES = [m for m in REFINE_MODES if m != "native"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from zs_combine import _colour_fg

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="Root of an MVTec-AD-layout dataset.")
    ap.add_argument("--categories", nargs="+", default=MVTEC_AD_CLASS_NAMES)
    ap.add_argument("--backbone", default="c-radio_v4-h")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--states", default="industrial", choices=["industrial", "clinical"])
    ap.add_argument("--prompt-name", default=None)
    ap.add_argument("--text-vocab", default="objsurf")
    ap.add_argument("--text-gate", type=float, default=0.016)
    ap.add_argument("--pixel-size", type=int, default=518)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--refine", default="trunk", choices=_E2E_REFINE_CHOICES,
                    help="Feature space for affinity refinement.")
    ap.add_argument("--no-pixel-upsample", action="store_true",
                    help="Stop at the patch-grid map (skip upsampling and blur).")
    ap.add_argument("--exact-mask", action="store_true",
                    help="Use the slower disk-reading foreground mask from "
                         "scripts/zs_combine.py.")
    ap.add_argument("--dtype", default="fp32", choices=list(_DTYPES))
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--verify", default=None,
                    help="main_zeroshot.py output dir to compare the first "
                         "category's map against (not timed).")
    ap.add_argument("--out", default=None,
                    help="Write per-category metrics and throughput to this JSON.")
    return ap.parse_args()


def colour_mask_batched(images: torch.Tensor, grid, border_frac: float = 0.15 / 2,
                        border_px: int = 12) -> torch.Tensor:
    """Batched, in-memory version of scripts/zs_combine.py's ``_colour_fg``.

    ``images`` is (B, 3, H, W) in [0, 1]. Each image is resized to (gh*8, gw*8),
    thresholded with Otsu on the distance to its median border colour, and
    max-pooled to the patch grid.
    """
    gh, gw = grid
    b = images.shape[0]
    im = F.interpolate(images, size=(gh * 8, gw * 8), mode="bilinear",
                       align_corners=False) * 255.0            # (B, 3, H, W)
    im = im.permute(0, 2, 3, 1)                                 # (B, H, W, 3)
    bpx = border_px
    border = torch.cat([im[:, :bpx].reshape(b, -1, 3), im[:, -bpx:].reshape(b, -1, 3),
                        im[:, :, :bpx].reshape(b, -1, 3), im[:, :, -bpx:].reshape(b, -1, 3)],
                       dim=1)                                   # (B, n_border, 3)
    med = border.median(dim=1).values                          # (B, 3)
    d = (im - med[:, None, None, :]).norm(dim=-1)               # (B, H, W)

    masks = torch.empty(b, gh * 8, gw * 8, dtype=torch.bool, device=images.device)
    for i in range(b):
        di = d[i]
        hist = torch.histc(di, bins=128, min=float(di.min()), max=float(di.max()))
        p = hist / hist.sum().clamp_min(1e-12)
        c = torch.linspace(float(di.min()), float(di.max()), 128, device=di.device)
        w0 = torch.cumsum(p, 0)
        w1 = 1.0 - w0
        m0 = torch.cumsum(p * c, 0) / w0.clamp_min(1e-12)
        m1 = ((p * c).sum() - torch.cumsum(p * c, 0)) / w1.clamp_min(1e-12)
        t = c[int(torch.nan_to_num(w0 * w1 * (m0 - m1) ** 2, nan=0.0).argmax())]
        masks[i] = di > t
    return masks.reshape(b, gh, 8, gw, 8).amax(dim=(2, 4)).reshape(b, gh * gw)


@torch.inference_mode()
def run_category(backbone, loader, t_class, t_gate_bg, t_gate_fg, args, device, dtype,
                 timing: dict):
    """Run the pipeline on one category and add its time to ``timing``.

    Returns per-image pixel maps, image scores, labels, defect types, paths
    and the patch grid.
    """
    cfg = BasisConfig(refine=args.refine)
    pixel_maps, im_scores, labels, defect_types, paths = [], [], [], [], []
    grid = None
    n_batches = 0

    for batch in loader:
        n_batches += 1
        images = batch["image"].to(device, non_blocking=True)
        warm = n_batches <= args.warmup_batches
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=dtype, enabled=(dtype != torch.float32)):
            feats = backbone.encode(images)
            grid = feats.grid
            basis = deployed_basis(feats, cfg)                       # (B, N, D)
            margin = torch.einsum("bnd,d->bn", basis.float(), t_class.float())
            g = F.normalize(feats.clip_summary.float(), dim=-1)
            im_s = g @ t_class.float()
            gate = g @ (t_gate_fg.float() - t_gate_bg.float())

            if args.exact_mask:
                fg = torch.stack([torch.from_numpy(
                        _colour_fg(p, grid)[0]).to(device)
                    for p in batch["path"]])
            else:
                fg = colour_mask_batched(images, grid).to(device)
            exempt = gate < args.text_gate
            fg = fg | exempt[:, None]

            filled = torch.where(fg, margin, margin.min(dim=1, keepdim=True).values)
            if not args.no_pixel_upsample:
                up = F.interpolate(filled.reshape(-1, 1, *grid), size=(args.pixel_size,) * 2,
                                   mode="bilinear", align_corners=False).squeeze(1)
                out_map = blur(up, args.sigma)
            else:
                out_map = filled

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if not warm:
            timing["seconds"] += dt
            timing["images"] += images.shape[0]

        pixel_maps.extend(out_map.float().cpu().numpy())
        im_scores.extend(im_s.float().cpu().numpy())
        labels.extend(int(x) for x in batch["label"])
        defect_types.extend(batch["defect_type"])
        paths.extend(batch["path"])

    return pixel_maps, np.asarray(im_scores), np.asarray(labels), defect_types, paths, grid


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[args.dtype]

    backbone = build_backbone(args.backbone, img_size=args.img_size,
                              want_text=True, want_dino=(args.refine == "dino"),
                              want_sam=True)
    print(f"[e2e] backbone={backbone.name} grid={backbone.grid} img_size={args.img_size} "
          f"refine={args.refine} dtype={args.dtype} "
          f"pixel_upsample={not args.no_pixel_upsample} "
          f"mask={'exact(disk)' if args.exact_mask else 'in-memory-approx'}", flush=True)

    from src.zeroshot import STATE_VOCABULARIES
    vocab_n, vocab_a = STATE_VOCABULARIES[args.states]

    timing = {"seconds": 0.0, "images": 0}
    per_category = {}
    per_category_image = {}
    for cat in args.categories:
        loader = build_test_loader(args.data_root, cat, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers, require_masks=True)
        t_n, t_a = text_anchors(backbone, cat, name=args.prompt_name,
                                normal_states=vocab_n, anomaly_states=vocab_a)
        t_class = t_a - t_n
        t_fg, t_bg = fgbg_anchors(backbone, cat, vocab=args.text_vocab)

        pixel_maps, im_scores, labels, defect_types, paths, grid = run_category(
            backbone, loader, t_class, t_bg, t_fg, args, device, dtype, timing)

        per_category_image[cat] = compute_image_metrics(im_scores, labels, defect_types)
        entry = {"n_images": len(labels),
                 "image": per_category_image[cat]["overall"]}
        if not args.no_pixel_upsample:
            ds = loader.dataset
            acc = PixelAccumulator()
            for i, m in enumerate(pixel_maps):
                acc.add(m, ds.load_mask(i, size=args.pixel_size))
            pm = acc.compute()
            entry["pixel"] = pm
            im = entry["image"]
            print(f"[{cat:<16}] n={len(labels):4d}  {format_pixel_metrics(pm)}  "
                  f"image_auroc={im['auroc']:.4f} image_f1={im['f1']:.4f}", flush=True)
        else:
            im = entry["image"]
            print(f"[{cat:<16}] n={len(labels):4d}  image_auroc={im['auroc']:.4f} "
                  f"image_f1={im['f1']:.4f}", flush=True)
        per_category[cat] = entry

    ips = timing["images"] / max(timing["seconds"], 1e-9)
    print(f"\n[throughput] {timing['images']} images (post-warmup) in "
          f"{timing['seconds']:.3f}s -> {ips:.2f} images/sec "
          f"({1000.0 / max(ips, 1e-9):.2f} ms/image)", flush=True)

    avg_image = average_over_categories(per_category_image)["overall"]
    print(f"\n[image, mean over {len(per_category_image)} categories] "
          f"auroc={avg_image['auroc']:.4f} ap={avg_image['ap']:.4f} f1={avg_image['f1']:.4f}",
          flush=True)
    if not args.no_pixel_upsample:
        pixel_keys = next(iter(per_category.values()))["pixel"].keys()
        avg_pixel = {k: float(np.mean([per_category[c]["pixel"][k] for c in per_category]))
                     for k in pixel_keys}
        print(f"[pixel, mean over {len(per_category)} categories] "
              + "  ".join(f"{k}={v:.4f}" for k, v in avg_pixel.items()), flush=True)

    if args.verify:
        _run_verify(args, backbone)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "throughput_images_per_sec": ips,
                       "image_average": avg_image, "per_category": per_category},
                      fh, indent=2, default=str)
        print(f"wrote {args.out}", flush=True)


def _run_verify(args, backbone):
    """Compare the in-memory map with a main_zeroshot.py cache on one category."""
    cache_dir = args.verify
    cat = args.categories[0]
    npz_path = os.path.join(cache_dir, "scores", f"{cat}.npz")
    if not os.path.exists(npz_path):
        print(f"[verify] no cache at {npz_path}, skipping.")
        return
    d = np.load(npz_path, allow_pickle=True)
    need = ["field_class_reg8mix", "field_class_reg32mix",
            "field_class_reg8mix_refined", "field_class_reg32mix_refined"]
    if not all(k in d for k in need):
        print(f"[verify] cache missing {need}, skipping.")
        return
    cached = np.mean([d[k] for k in need], axis=0)            # (n, N)

    loader = build_test_loader(args.data_root, cat, img_size=args.img_size,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers, require_masks=False)
    from src.zeroshot import STATE_VOCABULARIES
    vocab_n, vocab_a = STATE_VOCABULARIES[args.states]
    t_n, t_a = text_anchors(backbone, cat, name=args.prompt_name,
                            normal_states=vocab_n, anomaly_states=vocab_a)
    t_class = t_a - t_n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BasisConfig(refine=args.refine)
    ours, paths = [], []
    with torch.inference_mode():
        for batch in loader:
            feats = backbone.encode(batch["image"].to(device))
            basis = deployed_basis(feats, cfg)
            m = torch.einsum("bnd,d->bn", basis.float(), t_class.float())
            ours.extend(m.cpu().numpy())
            paths.extend(batch["path"])
    ours = np.stack(ours)
    order = {str(p): i for i, p in enumerate(d["paths"])}
    idx = [order[p] for p in paths]
    ref = cached[idx]
    diff = np.abs(ours - ref)
    print(f"[verify:{cat}] combo field vs cache: max|diff|={diff.max():.2e} "
          f"mean|diff|={diff.mean():.2e} (should be ~1e-5, float rounding only)")


if __name__ == "__main__":
    main()
