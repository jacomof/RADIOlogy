"""Stage 3: combine cached margin fields into a map and score localization.

All cached fields are cosine margins on the same scale, so any average of
them is a valid map. See reproduce_mvtec.sh for the paper's flags.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_mvtec import MVTecAD
from src.pixel_metrics import PixelAccumulator, average_pixel_metrics, upsample_map
from src.zeroshot import foreground_mask

_COHEN_NEGLIGIBLE = 0.2

# Lower bound of a difference of two cosines, used by --fg-fill floor.
_MARGIN_FLOOR = -2.0


def apply_gate(stack: np.ndarray, sam_z: np.ndarray, fn: str,
               sign: str, strength: float) -> np.ndarray:
    """Attenuate background patches using the cached SAM-head objectness z-score.

    ``sam_z`` (see ``sam_objectness``) is squashed to a gate ``g`` in [0, 1]
    and applied per image as ``m' = m_min + (m - m_min) * (1 - s + s * g)``,
    so ``s=0`` is a no-op and ``s=1`` collapses gated patches to the image's
    minimum margin.
    """
    if fn == "none" or strength == 0:
        return stack
    z = sam_z if sign == "pos" else -sam_z
    if fn == "sigmoid":
        g = 1.0 / (1.0 + np.exp(-z))
    elif fn == "tanh":
        g = np.tanh(np.maximum(z, 0.0))
    elif fn == "relu":
        g = np.clip(np.maximum(z, 0.0), 0.0, 1.0)
    else:
        raise SystemExit(f"Unknown --gate '{fn}' (sigmoid, tanh, relu, none).")
    lo = stack.min(axis=1, keepdims=True)
    return lo + (stack - lo) * (1.0 - strength + strength * g)


def _colour_fg(path: str, grid, border: int = 12, border_frac: float = 0.15,
               dilate: int = 1, probe: int = 0, probe_frac: float = 0.5,
               border_scale: float = None):
    """Patch-grid foreground mask from the distance to the median border colour.

    Returns ``(mask, d)``: the ``(N,)`` bool mask and Cohen's d between border
    and interior of the distance field. Model-free, but blind to defects that
    show background through the object (e.g. holes).
    """
    from PIL import Image
    gh, gw = grid
    im = np.asarray(Image.open(path).convert("RGB").resize((gw * 8, gh * 8)), np.float32)
    # `border` sets the reference-colour band (fixed pixel count unless
    # border_scale is given); `border_frac` below is only for the effect size.
    b = border if border_scale is None else max(1, int(round(border_scale * min(im.shape[:2]))))
    fr = np.concatenate([im[:b].reshape(-1, 3), im[-b:].reshape(-1, 3),
                         im[:, :b].reshape(-1, 3), im[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(im - np.median(fr, 0), axis=-1)
    h, e = np.histogram(d, bins=128)
    pr = h / h.sum(); c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(pr); w1 = 1 - w0
    m0 = np.cumsum(pr * c) / np.maximum(w0, 1e-12)
    m1 = ((pr * c).sum() - np.cumsum(pr * c)) / np.maximum(w1, 1e-12)
    t = c[np.nanargmax(w0 * w1 * (m0 - m1) ** 2)]
    fg = base = (d > t).reshape(gh, 8, gw, 8).max(axis=(1, 3))
    if probe > 1:
        # If the dilated mask covers most of the image, skip masking it.
        from scipy.ndimage import maximum_filter
        if maximum_filter(base, size=probe).mean() >= probe_frac:
            return np.ones(gh * gw, bool), 0.0
    if dilate > 1:
        # Widen the mask's pooling block from 8px to 8*dilate px, aligned to
        # the patch grid.
        from scipy.ndimage import maximum_filter
        fg = maximum_filter(fg, size=dilate)

    # Border/interior effect size on the pixel-resolution distance field.
    bh = max(1, int(round(border_frac * min(d.shape))))
    is_border = np.zeros(d.shape, bool)
    is_border[:bh] = is_border[-bh:] = True
    is_border[:, :bh] = is_border[:, -bh:] = True
    db, di = d[is_border], d[~is_border]
    pooled = np.sqrt(0.5 * (db.var() + di.var()))
    eff = (db.mean() - di.mean()) / max(float(pooled), 1e-6)
    return fg.reshape(-1), float(eff)


def _rebase_paths(paths, path_rebase):
    """Rewrite OLD_PREFIX->NEW_PREFIX on paths that start with it; pass through
    unchanged otherwise. Idempotent. Directory-boundary-aware (matches on
    OLD_PREFIX + '/', so OLD_PREFIX being a string-prefix of NEW_PREFIX can't
    cause a double rewrite).
    """
    if not path_rebase:
        return list(paths)
    old, new = path_rebase
    old_slash = old.rstrip("/") + "/"
    return [new + p[len(old_slash) - 1:] if p.startswith(old_slash) else p
            for p in paths]


def _text_fg(text_dir: str, cat: str, paths: np.ndarray, grid, vocab: str,
             field: str, split: str, path_rebase=None):
    """Cached language foreground mask and its image-level margin.

    Returns ``(mask, glob)``: the ``(B, N)`` patch mask and the ``(B,)``
    foreground-minus-background margin of the MAP-pooled embedding.
    """
    import torch

    from src.zeroshot import text_foreground_mask

    path = os.path.join(text_dir, f"{cat}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{cat}: no language-mask cache at {path}. Produce it "
                         f"with scripts/zs_text_mask.py (one encode pass), or "
                         f"use --fg-source colour.")
    t = np.load(path, allow_pickle=True)
    key = f"{field}_{vocab}"
    if key not in t:
        raise SystemExit(f"{cat}: {path} has no '{key}'. Available: "
                         f"{sorted(k for k in t.files if k.startswith(('fg', 'glob')))}")
    # Both caches must agree on image order (rebase is a no-op unless a path
    # actually has the old prefix).
    t_paths = _rebase_paths(t["paths"], path_rebase)
    s_paths = _rebase_paths(paths, path_rebase)
    if len(t_paths) != len(s_paths) or t_paths != s_paths:
        raise SystemExit(f"{cat}: language-mask cache and score cache disagree on "
                         f"image order/contents ({len(t['paths'])} vs {len(paths)})")
    if tuple(int(v) for v in t["grid"]) != tuple(grid):
        raise SystemExit(f"{cat}: language-mask grid {tuple(t['grid'])} != {tuple(grid)}")
    mask = text_foreground_mask(t[key], split).numpy()
    return mask, t[f"glob_{vocab}"].astype(np.float64)


def edge_frac(mask: np.ndarray, grid) -> np.ndarray:
    """Fraction of 4-adjacent patch pairs straddling ``mask``'s boundary, per image.

    A coherence statistic: low when the mask is one blob, high when it's
    fragmented. ``mask`` is ``(B, N)`` boolean over the patch grid; returns
    ``(B,)`` float.
    """
    m = mask.reshape(len(mask), *grid)
    dh = m[:, :, 1:] != m[:, :, :-1]
    dv = m[:, 1:, :] != m[:, :-1, :]
    return ((dh.sum(axis=(1, 2)) + dv.sum(axis=(1, 2)))
            / float(dh[0].size + dv[0].size))


def _region_labels(field: np.ndarray, expect: int = 0) -> np.ndarray:
    """Recover the stage-1 k-means partition from a cached pure-region field.

    A pure ``class_regK`` field is piecewise constant over exactly the
    k-means regions, so grouping patches by equal value recovers the region
    assignment with no encoder pass. ``expect`` (the field's K) triggers a
    warning if two regions collide onto the same value.
    """
    out = np.empty(field.shape, dtype=np.int32)
    short = 0
    for i, f in enumerate(field):
        _, inv = np.unique(f, return_inverse=True)
        out[i] = inv
        short += int(expect and inv.max() + 1 < expect)
    if short:
        print(f"[warn] {short}/{len(field)} images recovered fewer than {expect} "
              f"regions (identical pooled margins); partition is coarser there.")
    return out


def _region_vote(mask: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Make a patch mask spatially coherent: each k-means region takes its
    majority label.
    """
    out = np.zeros_like(mask)
    for i in range(len(mask)):
        l = lab[i]
        k = int(l.max()) + 1
        frac = (np.bincount(l, weights=mask[i].astype(np.float64), minlength=k)
                / np.maximum(np.bincount(l, minlength=k), 1))
        out[i] = (frac >= 0.5)[l]
    return out


def _upsample_mask(mask: np.ndarray, grid: tuple, size: int) -> np.ndarray:
    """Nearest-neighbour lift of a patch-grid bool mask to ``size x size`` pixels.

    Nearest (not bilinear) so the fill never blends with a real score.
    Index arithmetic matches F.interpolate(mode="nearest").
    """
    gh, gw = grid
    ys = np.arange(size) * gh // size
    xs = np.arange(size) * gw // size
    return mask.reshape(gh, gw)[np.ix_(ys, xs)]


def score(score_dir: str, root: str, fields, size: int, sigma: float,
          gate: str = "none", gate_sign: str = "pos", gate_strength: float = 1.0,
          suppress_bg: bool = False, fg_fill: str = "min",
          fg_source: str = "colour", suppress_mode: str = "floor",
          lam: float = 1.0, guard: str = "binary", categories=None,
          guard_source: str = "sam", fg_apply: str = "patch",
          fg_vote: str = "none", fg_dilate: int = 1, fg_probe: int = 0,
          fg_probe_frac: float = 0.5, coherence_gate: float = 0.0,
          text_dir: str = None, text_vocab: str = "keyword",
          text_field: str = "fgctx", text_split: str = "zero",
          text_gate: float = None, path_rebase=None,
          fg_border_scale: float = None):
    """Mean pixel metrics for one field combination (a plain average of maps)."""
    def _rebase(paths):
        return _rebase_paths(paths, path_rebase)

    per_cat = {}
    for path in sorted(glob.glob(os.path.join(score_dir, "scores", "*.npz"))):
        cat = os.path.splitext(os.path.basename(path))[0]
        if categories and cat not in categories:
            continue
        d = np.load(path, allow_pickle=True)
        missing = [f for f in fields if f"field_{f}" not in d]
        if missing:
            raise SystemExit(f"{cat}: no cached field(s) {missing}. "
                             f"Available: {sorted(k[6:] for k in d if k.startswith('field_'))}")
        if gate != "none" and not ("sam_z" in d and d["sam_z"].size):
            raise SystemExit(f"{cat}: no cached 'sam_z'; --gate is the SAM-head "
                             f"objectness gate, so it needs a backbone that has "
                             f"that head (re-run main_zeroshot.py to cache it).")
        grid = tuple(int(v) for v in d["grid"])
        stack = np.mean([d[f"field_{f}"] for f in fields], axis=0)
        deferred = None      # set by --fg-apply pixel; consumed after upsampling
        if suppress_bg:
            has_mask = "fg_mask" in d and d["fg_mask"].size
            has_sam = "sam_z" in d and d["sam_z"].size
            # 'colour' (default) and 'text' bring their own mask and don't
            # need cached objectness.
            if fg_source not in ("colour", "text") and not (has_mask or has_sam):
                raise SystemExit(f"{cat}: needs a cached 'fg_mask' or 'sam_z' for "
                                 f"--fg-source {fg_source}; re-run main_zeroshot.py "
                                 f"to cache objectness, or use --fg-source colour.")
            if not (has_mask or has_sam) and guard == "continuous":
                raise SystemExit(f"{cat}: --guard continuous scales suppression by "
                                 f"the border/interior objectness effect size, "
                                 f"which is not cached here. Use --guard binary.")
            # Background patches are filled rather than scored; the default
            # fill is 'min' (this image's own minimum).
            import torch
            if has_sam:
                fm = [foreground_mask(torch.from_numpy(z), grid) for z in d["sam_z"]]
                raw = np.stack([m.numpy() for m, _ in fm])
                dv = np.array([abs(float(dd)) for _, dd in fm])
            elif has_mask:
                raw = d["fg_mask"]
                dv = np.abs(d["fg_d"]) if "fg_d" in d else np.ones(len(raw))
            else:
                # Colour-only: no objectness cached, so start from an empty mask.
                raw = np.zeros(stack.shape, dtype=bool)
                dv = np.zeros(len(stack), dtype=np.float32)
            # 'binary': full suppression above Cohen's negligible-effect
            # cutoff, none below. 'continuous': strength = tanh(|d|).
            col = cdv = None
            if fg_source in ("colour", "union"):
                cols = [_colour_fg(x, grid, dilate=fg_dilate, probe=fg_probe,
                                   probe_frac=fg_probe_frac,
                                   border_scale=fg_border_scale)
                        for x in _rebase(d["paths"])]
                col = np.stack([c for c, _ in cols])
                cdv = np.abs(np.array([e for _, e in cols], dtype=np.float32))
                if fg_probe > 1:
                    n_ex = int(col.all(axis=1).sum())
                    print(f"[probe N={fg_probe}] {cat}: {n_ex}/{len(col)} images "
                          f"({100 * n_ex / len(col):.0f}%) exempted from masking",
                          flush=True)
            # --guard-source must match --fg-source or the guard gates nothing.
            gdv = dv if (guard_source == "sam" or cdv is None) else cdv
            if guard == "binary":
                mask = raw | (gdv < _COHEN_NEGLIGIBLE)[:, None]
                strength = np.ones((len(raw), 1), dtype=np.float32)
            else:
                mask, strength = raw, np.tanh(gdv)[:, None].astype(np.float32)
            if col is not None:
                if fg_source == "union":
                    # Suppress only where both estimators agree it's background.
                    mask = mask | col
                elif guard_source == "colour":
                    # Colour mask, guard sourced from the colour field.
                    mask = col | (gdv < _COHEN_NEGLIGIBLE)[:, None]
                else:
                    # Default: the colour mask, applied to every image.
                    mask = col
            if fg_source == "text":
                # The language mask replaces the mask entirely.
                mask, _ = _text_fg(text_dir, cat, d["paths"], grid,
                                   text_vocab, text_field, text_split, path_rebase)
            if text_gate is not None:
                # Outside the branch above so --fg-source colour --text-gate
                # is reachable.
                _, tglob = _text_fg(text_dir, cat, d["paths"], grid,
                                    text_vocab, text_field, text_split, path_rebase)
                # Exempt images whose MAP margin reads surface/texture.
                exempt = tglob < text_gate
                mask = mask | exempt[:, None]
                print(f"[text-gate {text_gate:g} {text_vocab}] {cat}: "
                      f"{int(exempt.sum())}/{len(exempt)} images "
                      f"({100 * exempt.mean():.0f}%) exempted   "
                      f"MAP margin p10/p50/p90 = {np.percentile(tglob, 10):+.4f}/"
                      f"{np.median(tglob):+.4f}/{np.percentile(tglob, 90):+.4f}",
                      flush=True)
            if coherence_gate > 0:
                # Abstain where the mask itself is fragmented (high
                # edge_frac), not where the image looks anomalous.
                ef = edge_frac(mask, grid)
                exempt = ef > coherence_gate
                mask = mask | exempt[:, None]
                print(f"[coherence-gate {coherence_gate:g}] {cat}: "
                      f"{int(exempt.sum())}/{len(exempt)} images "
                      f"({100 * exempt.mean():.0f}%) exempted   "
                      f"edge_frac p10/p50/p90 = {np.percentile(ef, 10):.3f}/"
                      f"{np.median(ef):.3f}/{np.percentile(ef, 90):.3f}",
                      flush=True)
            if fg_vote != "none":
                # Regularize the mask by the stage-1 region partition.
                key = f"field_{fg_vote}"
                if key not in d:
                    raise SystemExit(
                        f"{cat}: --fg-vote {fg_vote} needs the cached PURE region "
                        f"field '{key}' (a 'mix'/'refined' field is not piecewise "
                        f"constant, so no partition can be read from it). "
                        f"Available: {sorted(k[6:] for k in d if k.startswith('field_'))}")
                expect = int("".join(ch for ch in fg_vote if ch.isdigit()) or 0)
                mask = _region_vote(mask, _region_labels(d[key], expect))
            if suppress_mode == "floor":
                fill = {"min": stack.min(axis=1, keepdims=True),
                        "zero": 0.0,
                        "catmin": float(stack.min()),
                        "floor": _MARGIN_FLOOR}[fg_fill]
                if fg_apply == "patch":
                    stack = np.where(mask, stack, (1 - strength) * stack + strength * fill)
                else:
                    # Defer fill to pixel resolution, after the blur, so the
                    # fill value doesn't bleed into mask boundaries.
                    deferred = (mask, strength, fill)
            else:
                # Additive, bounded penalty instead of a veto.
                rng = (stack.max(axis=1, keepdims=True)
                       - stack.min(axis=1, keepdims=True))
                stack = stack - lam * strength * (~mask) * rng
        if gate != "none":
            stack = apply_gate(stack, d["sam_z"], gate, gate_sign, gate_strength)
        ds = MVTecAD(root, cat, train=False)
        acc = PixelAccumulator()
        for i, m in enumerate(stack):
            px = upsample_map(m, grid, size, sigma=sigma)
            if deferred is not None:
                mk, st, fl = deferred
                f = float(fl[i, 0]) if isinstance(fl, np.ndarray) else float(fl)
                s = float(st[i, 0])
                px = np.where(_upsample_mask(mk[i], grid, size), px,
                              (1 - s) * px + s * f)
            acc.add(px, ds.load_mask(i, size=size))
        per_cat[cat] = acc.compute()
    return average_pixel_metrics(per_cat), per_cat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score_dir")
    ap.add_argument("--data-root", required=True,
                    help="Root of an MVTec-AD-layout dataset.")
    ap.add_argument("--fields", nargs="+", default=[],
                    help="Cached fields to score individually.")
    ap.add_argument("--combo", nargs="+", action="append", default=[],
                    help="Fields to average into one map. Repeatable.")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Restrict to these cached categories (default: all). "
                         "Pixel metrics pool a whole category into one sort, so "
                         "a very large category can exhaust memory; this scores "
                         "categories separately to avoid that.")
    ap.add_argument("--pixel-size", type=int, default=518)
    ap.add_argument("--pixel-sigma", type=float, default=4.0)
    ap.add_argument("--gate", nargs="+", default=["none"],
                    choices=["none", "sigmoid", "tanh", "relu"],
                    help="SAM-head objectness gate squashing; repeatable to sweep.")
    ap.add_argument("--gate-sign", nargs="+", default=["pos"], choices=["pos", "neg"],
                    help="Whether high SAM feature norm means foreground (pos) "
                         "or background (neg).")
    ap.add_argument("--gate-strength", nargs="+", type=float, default=[1.0])
    ap.add_argument("--suppress-bg", action="store_true",
                    help="Do not score background patches: they take the image's "
                         "minimum margin. Needs a run with --fg-select otsu.")
    ap.add_argument("--fg-source", default="colour",
                    choices=["sam", "colour", "union", "text"],
                    help="Mask source: border-colour prior (default, "
                         "model-free), SAM-head objectness, their union, or "
                         "'text' (language mask, needs --text-cache).")
    ap.add_argument("--text-cache", default=None,
                    help="Directory of scripts/zs_text_mask.py output for this "
                         "dataset (e.g. results_zs_textmask/visa). Required by "
                         "--fg-source text.")
    ap.add_argument("--path-rebase", nargs=2, default=None,
                    metavar=("OLD_PREFIX", "NEW_PREFIX"),
                    help="Rewrite the OLD_PREFIX->NEW_PREFIX of the absolute image "
                         "paths cached in 'paths' before reading raw pixels for "
                         "--fg-source colour/union. Needed "
                         "when a cache's images were encoded under a path that "
                         "is no longer readable.")
    ap.add_argument("--text-vocab", default="keyword",
                    choices=sorted(__import__("src.zeroshot", fromlist=["x"])
                                   .FGBG_VOCABULARIES),
                    help="fg/bg word ensemble (src.zeroshot.FGBG_VOCABULARIES). "
                         "'keyword' is the literal pair and is name-free, so "
                         "the prompt is identical for every category; 'class' "
                         "is the control for whether naming the object helps.")
    ap.add_argument("--text-field", default="fgctx", choices=["fg", "fgctx"],
                    help="Support the mask is built on: 'fgctx' is the default "
                         "k=5 context window (the same one the map lives on), "
                         "'fg' the raw patch grid.")
    ap.add_argument("--text-split", default="zero", choices=["zero", "otsu"],
                    help="Where a patch stops being foreground: the margin's own "
                         "sign change, or per-image Otsu as "
                         "the control.")
    ap.add_argument("--text-gate", type=float, default=None,
                    help="Drop the mask on images whose MAP-pooled margin is "
                         "below this (images that read as surface/texture "
                         "rather than object; default: off). The paper uses "
                         "0.016, chosen by leave-one-dataset-out validation.")
    ap.add_argument("--suppress-mode", default="floor", choices=["floor", "additive"],
                    help="'floor' vetoes background; 'additive' applies a bounded "
                         "penalty so a mislabelled defect stays recoverable.")
    ap.add_argument("--guard", default="binary", choices=["binary", "continuous"],
                    help="How the border/interior effect size gates suppression: "
                         "a hard |d| < 0.2 cutoff, or strength = tanh(|d|).")
    ap.add_argument("--guard-source", default="sam", choices=["sam", "colour"],
                    help="Which field the guard's effect size is measured on. "
                         "Should match the field --fg-source draws the mask "
                         "from, or the guard cannot see whether that mask "
                         "found a real background.")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="Penalty in units of the per-image margin range.")
    ap.add_argument("--coherence-gate", type=float, default=0.0,
                    help="Exempt an image from suppression when its own mask "
                         "is incoherent (edge_frac above this; 0 = off).")
    ap.add_argument("--fg-fill", default="min",
                    choices=["min", "catmin", "zero", "floor"],
                    help="Score given to background patches: this image's "
                         "minimum (default, and the only choice computable "
                         "from the image being scored), the category-wide "
                         "minimum, the constant 0, or the margin field's "
                         "theoretical floor of -2.")
    ap.add_argument("--fg-vote", default="none",
                    help="Regularize the mask by a cached region partition: "
                         "each k-means region takes the majority label of its "
                         "patches. Pass a PURE region field ('class_reg8', "
                         "'class_reg32'); 'none' (default) leaves the mask "
                         "per-patch.")
    ap.add_argument("--fg-probe", type=int, default=0,
                    help="Saturation probe: dilate the mask's pooling block by "
                         "N and, if it covers most of the grid, drop masking "
                         "for that image entirely. 0 (default) disables it.")
    ap.add_argument("--fg-probe-frac", type=float, default=0.5,
                    help="Patch-grid coverage at which --fg-probe calls an "
                         "image background-free.")
    ap.add_argument("--fg-dilate", type=int, default=1,
                    help="Widen the colour mask's pooling block from 8px to "
                         "8*N, aligned to the patch grid (1 = default).")
    ap.add_argument("--fg-border-scale", type=float, default=None,
                    help="Make _colour_fg's reference-colour border band "
                         "grid-relative (fraction of the grid*8 resize side) "
                         "instead of the fixed 12px. None (default) "
                         "reproduces the paper's numbers.")
    ap.add_argument("--fg-apply", default="patch", choices=["patch", "pixel"],
                    help="Where background suppression is applied. 'patch' "
                         "(default) fills at the patch grid, so the fill "
                         "value matters after upsampling blends it into mask "
                         "boundaries. 'pixel' fills after the upsample+blur, "
                         "which removes that band entirely.")
    args = ap.parse_args()

    jobs = [[f] for f in args.fields] + args.combo
    if not jobs:
        raise SystemExit("Nothing to do: pass --fields and/or --combo.")
    if (args.fg_source == "text" or args.text_gate is not None) and not args.text_cache:
        raise SystemExit("--fg-source text / --text-gate need --text-cache DIR "
                         "(scripts/zs_text_mask.py output for this dataset).")
    if args.text_gate is not None and not args.suppress_bg:
        raise SystemExit("--text-gate decides which images SKIP background "
                         "suppression, so it does nothing without --suppress-bg.")

    gates = [(fn, sg, st) for fn in args.gate
             for sg in (args.gate_sign if fn != "none" else ["pos"])
             for st in (args.gate_strength if fn != "none" else [1.0])]

    # p-F1 (max-F1 over the pixel score threshold) costs nothing: it's derived
    # from the same single sort as AUROC and AP (src/pixel_metrics.py).
    print(f"{'map':<44} {'gate':<18} {'p-AUROC':>8} {'AUPRO':>7} {'p-AP':>7} {'p-F1':>7}")
    for fields in jobs:
        for fn, sg, st in gates:
            m, _ = score(args.score_dir, args.data_root, fields,
                         args.pixel_size, args.pixel_sigma, fn, sg, st,
                         args.suppress_bg, args.fg_fill, args.fg_source,
                         args.suppress_mode, args.lam, args.guard,
                         args.categories, args.guard_source, args.fg_apply,
                         args.fg_vote, args.fg_dilate, args.fg_probe,
                         args.fg_probe_frac, args.coherence_gate,
                         args.text_cache, args.text_vocab, args.text_field,
                         args.text_split, args.text_gate,
                         args.path_rebase, args.fg_border_scale)
            tag = "--" if fn == "none" else f"{fn}/{sg}/{st:g}"
            print(f"{' + '.join(fields):<44} {tag:<18} {m['pixel_auroc']:>8.4f} "
                  f"{m['aupro']:>7.4f} {m['pixel_ap']:>7.4f} {m['pixel_f1']:>7.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
