"""Stage 1: encode each test image once and cache its patch-margin fields.

Also reports image-level metrics and the pixel metrics of every cached field.
scripts/zs_combine.py combines the fields; see reproduce_mvtec.sh.
"""

import argparse
import os
import socket
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from src.backbones import build_backbone
from src.data_mvtec import MVTEC_AD_CLASS_NAMES, MVTEC_AD_TEXTURES, MVTecAD, build_test_loader
from src.metrics import average_over_categories, compute_image_metrics
from src.pixel_metrics import (PixelAccumulator, average_pixel_metrics,
                               format_pixel_metrics, upsample_map)
from src.utils import neighborhood_aggregate, save_json, set_seed
from src.zeroshot import (REFINE_MODES, REGION_SPACES, STATE_VOCABULARIES,
                          affinity_refine, foreground_mask, masked_neighborhood_aggregate,
                          patch_margin, region_margin, sam_objectness, text_anchors)


def _git_commit() -> str:
    """Commit the run was launched from, or "unknown"."""
    try:
        rev = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                              "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"

# Cohen's "negligible effect" cutoff for whether an image has a background at
# all, before restricting context pooling / region clustering to it.
_COHEN_NEGLIGIBLE = 0.2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Root of an MVTec-AD-layout dataset "
                        "(<root>/<category>/{train,test}/...).")
    p.add_argument("--categories", nargs="+", default=MVTEC_AD_CLASS_NAMES)
    p.add_argument("--backbone", default="c-radio_v4-h",
                   help="c-radio_v4-h (SigLIP2 head, default) | radio_v2.5-b (CLIP head)")
    p.add_argument("--model-name", default=None)
    p.add_argument("--img-size", type=int, default=448, help="Multiple of 16.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--pixel", action="store_true", default=True,
                   help="Compute pixel AUROC/AP/AUPRO from the patch maps.")
    p.add_argument("--no-pixel", dest="pixel", action="store_false")
    p.add_argument("--pixel-size", type=int, default=518,
                   help="Working resolution for pixel metrics.")
    p.add_argument("--pixel-sigma", type=float, default=4.0,
                   help="Gaussian blur (pixels) on the upsampled map, for pixel "
                        "metrics only; image scores stay unsmoothed.")
    # Localization levers (see src.zeroshot.affinity_refine).
    p.add_argument("--refine", default="trunk", choices=list(REFINE_MODES),
                   help="Diffuse the patch-text map over a patch-affinity graph: "
                        "'trunk' = RADIO's shared backbone output (default, "
                        "free, same forward pass), 'dino'/'sam' = an adaptor "
                        "head, 'self' = the text-aligned patches themselves "
                        "(control), 'none' = off.")
    p.add_argument("--refine-temp", type=float, default=0.1,
                   help="Affinity softmax temperature (lower = sharper graph).")
    p.add_argument("--refine-iters", type=int, default=1)
    p.add_argument("--refine-keep", type=float, default=0.5,
                   help="Weight a patch keeps on its own evidence (0..1).")
    p.add_argument("--prompt-variants", nargs="+", default=["class"],
                   choices=["class", "object"],
                   help="Which prompt variants to produce.")
    p.add_argument("--patch-context", type=int, nargs="+", default=[5],
                   help="k x k average-pool of patch embeddings before the text dot "
                        "product (1 = off). Several values produce one field each "
                        "plus a multi-scale '_ctxms' field averaging them.")
    p.add_argument("--regions", type=int, nargs="+", default=[8, 32],
                   help="Segment each image's patch grid into K clusters and score "
                        "each *region* against text (see zeroshot.region_margin). "
                        "Several K produce one field each.")
    p.add_argument("--region-space", default="sam", choices=list(REGION_SPACES),
                   help="Feature space the regions are cut in: RADIO's 'sam' head "
                        "(default), 'dino', 'trunk', or 'text' as a control.")
    p.add_argument("--region-kmeans-iters", type=int, default=15,
                   help="Iterations for the spherical k-means that cuts regions "
                        "(src.zeroshot._kmeans).")
    p.add_argument("--states", default="industrial",
                   choices=sorted(STATE_VOCABULARIES),
                   help="Named state vocabulary. 'industrial' (default, WinCLIP's "
                        "words); 'clinical' substitutes lesion/pathology words for "
                        "medical benchmarks.")
    p.add_argument("--normal-states", nargs="+", default=[],
                   help="Override the normal state vocabulary (each must contain "
                        "'{}'). Default: whatever --states selects.")
    p.add_argument("--anomaly-states", nargs="+", default=[],
                   help="Override the anomaly state vocabulary.")
    p.add_argument("--prompt-name", default=None,
                   help="Override the class name in the prompts for every "
                        "category (default: src.zeroshot.PROMPT_NAMES).")
    p.add_argument("--fg-select", default="none", choices=["none", "otsu"],
                   help="Restrict context pooling and region clustering to the\n"
                        "SAM-objectness foreground (Otsu, per image). Background\n"
                        "patches form one region of their own.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results_zeroshot")
    return p.parse_args()


@torch.inference_mode()
def encode_category(backbone, loader, anchors, args):
    """One pass over a category's test split, returning per-image patch margins.

    Only the ``(N,)`` margins are kept, not the features, so every field can
    be recombined offline. ``anchors`` maps a prompt variant ('class',
    'object') to its ``(normal, anomaly)`` text embeddings.
    """
    out: dict = {}
    global_margin, labels, defect_types, paths, sam_z = [], [], [], [], []
    fg_d: list = []
    fg_mask: list = []
    grid = None

    def _add(name, tensor):
        out.setdefault(name, []).append(tensor.cpu().numpy().astype(np.float32))

    for batch in tqdm(loader, desc="encode", leave=False):
        feats = backbone.encode(batch["image"])
        grid = feats.grid
        tp = feats.clip_text_patch
        b = tp.shape[0]
        # Context pooling: k x k neighbourhood average before the text dot
        # product. Foreground selection (if enabled) happens before any pooling.
        if args.fg_select == "otsu":
            fgm, fgd = zip(*(foreground_mask(sam_objectness(feats.sam_patch[i]), grid)
                             for i in range(feats.sam_patch.shape[0])))
            fg_batch = torch.stack(fgm)                                   # (B, N)
            fg_d.extend(float(x) for x in fgd)
            # Keep the full grid when there is no clear background (|d| < 0.2).
            keep_all = torch.tensor([abs(float(x)) < _COHEN_NEGLIGIBLE for x in fgd],
                                    device=fg_batch.device)
            fg_batch = fg_batch | keep_all.unsqueeze(1)
            fg_mask.extend(fg_batch[i].cpu().numpy() for i in range(b))
        else:
            fg_batch = None

        ctx = {f"_ctx{k}": (neighborhood_aggregate(tp, grid, k) if fg_batch is None
                            else masked_neighborhood_aggregate(tp, grid, k, fg_batch))
               for k in args.patch_context if k > 1}
        # Substrates for affinity refinement / region clustering, from heads
        # RADIO already produced. 'trunk' (feats.patch_tokens) is always present.
        head = {"dino": feats.dino_patch, "sam": feats.sam_patch, "text": tp,
                "native": feats.clip_patch, "trunk": feats.patch_tokens, "self": tp}
        aff = head[args.refine] if args.refine != "none" else None
        reg_src = head[args.region_space]

        for variant, (n_txt, a_txt) in anchors.items():
            margins = {tag: patch_margin(src, n_txt, a_txt)            # (B, N)
                       for tag, src in [("", tp)] + list(ctx.items())}
            if len(ctx) > 1:      # multi-scale: the k-free option
                margins["_ctxms"] = torch.stack(
                    [margins[t] for t in margins], 0).mean(0)
            # Region-pooled margins, plus a blend with the best patch-level
            # field to keep the gradient within a region.
            base = margins.get("_ctxms", margins.get(list(ctx)[0] if ctx else ""))
            for K in args.regions:
                r = torch.stack([region_margin(tp[i], reg_src[i], n_txt, a_txt,
                                               K, seed=args.seed,
                                               fg=None if fg_batch is None
                                                  else fg_batch[i],
                                               kmeans_iters=args.region_kmeans_iters)
                                 for i in range(b)], 0)
                margins[f"_reg{K}"] = r
                if base is not None:
                    margins[f"_reg{K}mix"] = 0.5 * (r + base)

            for tag, m in margins.items():
                for i in range(b):
                    _add(f"{variant}{tag}", m[i])
                    if aff is not None:
                        _add(f"{variant}{tag}_refined",
                             affinity_refine(m[i], aff[i], iters=args.refine_iters,
                                             temp=args.refine_temp,
                                             keep=args.refine_keep))

        # SAM-head objectness, cached as a signed z-score (sign/squashing stay
        # CPU-side choices).
        if feats.sam_patch is not None:
            for i in range(b):
                sam_z.append(sam_objectness(feats.sam_patch[i]).cpu().numpy()
                             .astype(np.float32))

        # The global (image-level) score always uses the class-name prompts.
        g = patch_margin(feats.clip_summary, *anchors["class"])            # (B,)
        global_margin.extend(g.cpu().numpy().astype(np.float64))
        labels.extend(int(x) for x in batch["label"])
        defect_types.extend(batch["defect_type"])
        paths.extend(batch["path"])

    return dict(fields=out, global_margin=np.asarray(global_margin),
                labels=np.asarray(labels), defect_types=defect_types,
                paths=paths, grid=grid,
                sam_z=(np.stack(sam_z) if sam_z
                       else np.zeros((0, 0), dtype=np.float32)),
                fg_d=np.asarray(fg_d, dtype=np.float32),
                fg_mask=np.stack(fg_mask) if fg_mask else np.zeros((0, 0), bool))


def pixel_metrics_for(dataset: MVTecAD, zmaps, grid, args):
    """Pixel AUROC/AP/AUPRO from the per-image z-maps against GT masks."""
    acc = PixelAccumulator()
    for idx, z in enumerate(tqdm(zmaps, desc="pixel", leave=False)):
        smap = upsample_map(z, grid, args.pixel_size, sigma=args.pixel_sigma)
        acc.add(smap, dataset.load_mask(idx, size=args.pixel_size))
    return acc.compute()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = None if args.device == "auto" else args.device

    # 'sam' is always requested: it backs the objectness gate (cached as sam_z).
    spaces = {args.refine} | ({args.region_space} if args.regions else set()) | {"sam"}
    backbone = build_backbone(args.backbone, device=device, img_size=args.img_size,
                              want_text=True, model_name=args.model_name,
                              want_dino="dino" in spaces, want_sam="sam" in spaces)
    if "dino" in spaces and not getattr(backbone, "has_dino", False):
        print(f"[zeroshot] {args.backbone} exposes no DINO adaptor; falling back to "
              f"the text-aligned patches as substrate.")
        if args.refine == "dino":
            args.refine = "self"
        if args.region_space == "dino":
            args.region_space = "text"
    if "sam" in spaces and not getattr(backbone, "has_sam", False):
        print(f"[zeroshot] {args.backbone} exposes no SAM adaptor; falling back to "
              f"the text-aligned patches as substrate. No 'sam_z' will be cached, "
              f"so stage-2 background suppression must run with --fg-source colour "
              f"(scripts/zs_combine.py).")
        if args.refine == "sam":
            args.refine = "self"
        if args.region_space == "sam":
            args.region_space = "text"
        if args.fg_select != "none":
            raise SystemExit(
                f"--fg-select {args.fg_select} needs SAM-head objectness, which "
                f"'{args.backbone}' does not provide.")
    if not backbone.has_text:
        raise SystemExit(f"Backbone '{args.backbone}' has no text head; "
                         f"zero-shot patch-text scoring needs one.")
    print(f"[zeroshot] backbone={backbone.name} grid={backbone.grid} "
          f"img_size={backbone.img_size}")

    # Provenance, written before the run so it exists even if the run dies.
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_n, vocab_a = STATE_VOCABULARIES[args.states]
    save_json({"argv": sys.argv,
               "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
               "git_commit": _git_commit(),
               "hostname": socket.gethostname(),
               "args": {k: v for k, v in sorted(vars(args).items())},
               "resolved": {
                   "grid": list(backbone.grid),
                   "img_size": backbone.img_size,
                   "normal_states": list(args.normal_states or vocab_n),
                   "anomaly_states": list(args.anomaly_states or vocab_a),
               }},
              os.path.join(args.output_dir, "run_config.json"))

    per_category, per_category_pixel = {}, {}
    for category in args.categories:
        print(f"\n{'='*70}\n{category}\n{'='*70}")
        loader = build_test_loader(args.data_root, category,
                                   img_size=backbone.img_size,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   require_masks=args.pixel)
        vocab_n, vocab_a = STATE_VOCABULARIES[args.states]
        anchors = {v: text_anchors(backbone, category,
                                   name=("object" if v == "object" else args.prompt_name),
                                   normal_states=args.normal_states or vocab_n,
                                   anomaly_states=args.anomaly_states or vocab_a)
                   for v in args.prompt_variants}
        enc = encode_category(backbone, loader, anchors, args)
        fields = enc["fields"]

        # The default image-level score: the margin on the MAP-pooled global
        # embedding, no patches involved.
        cat_metrics = compute_image_metrics(enc["global_margin"], enc["labels"],
                                            enc["defect_types"])
        per_category[category] = cat_metrics
        m = cat_metrics["overall"]
        print(f"[image] AUROC {m['auroc']:.4f}  AP {m['ap']:.4f}  F1max {m['f1']:.4f}")

        if args.pixel:
            # Localization is scored for every cached margin field.
            for name, field in fields.items():
                pm = pixel_metrics_for(loader.dataset, field, enc["grid"], args)
                per_category_pixel.setdefault(name, {})[category] = pm
                print(f"[pixel:{name:<20}] {format_pixel_metrics(pm)}")

        sdir = os.path.join(args.output_dir, "scores")
        os.makedirs(sdir, exist_ok=True)
        np.savez(os.path.join(sdir, f"{category}.npz"),
                 labels=enc["labels"], defect_types=np.array(enc["defect_types"]),
                 paths=np.array(enc["paths"]), grid=np.array(enc["grid"]),
                 global_margin=enc["global_margin"], sam_z=enc["sam_z"], fg_d=enc["fg_d"], fg_mask=enc["fg_mask"],
                 **{f"field_{k}": np.stack(v) for k, v in fields.items()})

    # Summary over categories.
    print(f"\n{'='*70}\nMVTec AD ZERO-SHOT ({args.backbone}, {args.img_size}px) "
          f"-- mean over {len(per_category)} categories\n{'='*70}")
    averaged = average_over_categories(per_category)
    m = averaged["overall"]
    print(f"[image] AUROC {m['auroc']:.4f}  AP {m['ap']:.4f}  F1max {m['f1']:.4f}")

    # Objects vs textures: zero-shot prompts behave very differently on the two.
    for group, cats in [("textures", MVTEC_AD_TEXTURES),
                        ("objects", set(MVTEC_AD_CLASS_NAMES) - MVTEC_AD_TEXTURES)]:
        sel = [c for c in per_category if c in cats]
        if not sel:
            continue
        g = average_over_categories({c: per_category[c] for c in sel})
        print(f"  image AUROC on {group:<9} ({len(sel):2d} cats): "
              f"{g['overall']['auroc']:.4f}")

    pixel_avg = {}
    if per_category_pixel:
        pixel_avg = {name: average_pixel_metrics(per_cat)
                     for name, per_cat in per_category_pixel.items()}
        print(f"\n-- pixel metrics, mean over categories --")
        for name, m in pixel_avg.items():
            print(f"  {name:<22} {format_pixel_metrics(m)}")

    save_json({"backbone": args.backbone, "img_size": args.img_size,
               "per_category": per_category, "average": averaged,
               "per_category_pixel": per_category_pixel, "pixel_average": pixel_avg},
              os.path.join(args.output_dir, f"summary_{args.backbone}.json"))


if __name__ == "__main__":
    main()
