"""Zero-shot patch-text anomaly scoring.

The margin ``cos(v, t_a) - cos(v, t_n)`` is computed against a WinCLIP-style
prompt ensemble. This module also holds context and region pooling,
affinity refinement, the foreground masks and the fg/bg text anchors.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Prompts: the WinCLIP compositional ensemble (templates x states).
# ---------------------------------------------------------------------------
# WinCLIP's photographic prompt templates (a subset of CLIP's ImageNet ones).
TEMPLATES = [
    "a cropped photo of the {}.", "a cropped photo of a {}.",
    "a close-up photo of a {}.", "a close-up photo of the {}.",
    "a bright photo of a {}.", "a bright photo of the {}.",
    "a dark photo of a {}.", "a dark photo of the {}.",
    "a jpeg corrupted photo of a {}.", "a jpeg corrupted photo of the {}.",
    "a blurry photo of a {}.", "a blurry photo of the {}.",
    "a photo of a {}.", "a photo of the {}.",
    "a photo of a small {}.", "a photo of the small {}.",
    "a photo of a large {}.", "a photo of the large {}.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for anomaly detection.",
    "a photo of the {} for anomaly detection.",
]

# State words composed into the templates; only the class name is
# category-specific, which is what keeps this zero-shot.
NORMAL_STATES = [
    "{}", "flawless {}", "perfect {}", "unblemished {}",
    "{} without flaw", "{} without defect", "{} without damage",
]
ANOMALY_STATES = [
    "damaged {}", "{} with flaw", "{} with defect", "{} with damage",
]

# Clinical state words for medical benchmarks (--states clinical).
CLINICAL_NORMAL_STATES = [
    "{}", "healthy {}", "normal {}", "{} without lesion",
    "{} without abnormality", "{} without pathology",
]
CLINICAL_ANOMALY_STATES = [
    "{} with a lesion", "{} with a tumor", "{} with an abnormality",
    "{} with pathology", "abnormal {}", "diseased {}",
]

# Named vocabularies for --states. 'industrial' is the default everywhere.
STATE_VOCABULARIES = {
    "industrial": (NORMAL_STATES, ANOMALY_STATES),
    "clinical": (CLINICAL_NORMAL_STATES, CLINICAL_ANOMALY_STATES),
}

# Class names as they should read inside a prompt (underscores are not English).
PROMPT_NAMES = {
    "metal_nut": "metal nut",
    "toothbrush": "toothbrush",
    "pcb": "printed circuit board",
    # VisA
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "chewinggum": "chewing gum",
    "01": "object",
    "02": "object",
    "03": "object",
    "bracket_black": "black bracket",
    "bracket_brown": "brown bracket",
    "bracket_white": "white bracket",
    "br35h": "brain MRI",
    "brain_mri": "brain MRI",
    "headct": "head CT",
    "clinicdb": "colonoscopy image",
    "colondb": "colonoscopy image",
    "isic": "dermoscopic image",
    "tn3k": "thyroid ultrasound",
    "Class1": "textured surface",
    "Class2": "textured surface",
    "Class3": "textured surface",
    "Class4": "textured surface",
    "Class5": "textured surface",
    "Class6": "textured surface",
    "Class7": "textured surface",
    "Class8": "textured surface",
    "Class9": "textured surface",
    "Class10": "textured surface",
    "ksdd": "electrical commutator",
    "ksdd2": "production item",
    # DTD-Synthetic: strip the dataset index, keep the texture word.
    "Blotchy_099": "blotchy texture",
    "Fibrous_183": "fibrous texture",
    "Marbled_078": "marbled texture",
    "Matted_069": "matted texture",
    "Mesh_114": "mesh texture",
    "Perforated_037": "perforated texture",
    "Stratified_154": "stratified texture",
    "Woven_001": "woven texture",
    "Woven_068": "woven texture",
    "Woven_104": "woven texture",
    "Woven_125": "woven texture",
    "Woven_127": "woven texture",
    # BMAD medical: the benchmark's own class description, in English.
    "brain": "brain MRI",
    "liver": "liver CT scan",
    "retina": "retinal OCT scan",
}


def prompt_name(category: str) -> str:
    return PROMPT_NAMES.get(category, category.replace("_", " "))


def build_prompts(name: str, states: Sequence[str]) -> List[str]:
    """Every ``template x state`` combination for one class name."""
    return [tmpl.format(state.format(name))
            for state in states for tmpl in TEMPLATES]


@torch.no_grad()
def text_anchors(backbone, category: str, name: Optional[str] = None,
                 normal_states: Optional[Sequence[str]] = None,
                 anomaly_states: Optional[Sequence[str]] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean normal and anomaly text embeddings for a category, L2-normalized.

    ``name`` overrides the class name in the prompts; ``normal_states`` and
    ``anomaly_states`` override the state words (default: industrial).
    """
    name = name or prompt_name(category)

    def _encode(states: Sequence[str]) -> torch.Tensor:
        emb = backbone.encode_text(build_prompts(name, states)).float()
        return F.normalize(F.normalize(emb, dim=-1).mean(0), dim=-1)

    return (_encode(normal_states or NORMAL_STATES),
            _encode(anomaly_states or ANOMALY_STATES))


def patch_margin(patch_feats: torch.Tensor, normal: torch.Tensor,
                 anomaly: torch.Tensor) -> torch.Tensor:
    """``cos(patch, anomaly) - cos(patch, normal)`` for ``(..., D)`` features.

    The object-identity component common to both prompts cancels, leaving
    the axis that separates defect from intact.
    """
    p = F.normalize(patch_feats.float(), dim=-1)
    return p @ anomaly.to(p.device) - p @ normal.to(p.device)


# Feature spaces for affinity refinement and region clustering: RADIO's
# adaptor heads ('dino', 'sam'), its trunk, or the text-aligned patches.
REFINE_MODES = ("none", "dino", "sam", "native", "trunk", "self")
REGION_SPACES = ("dino", "sam", "native", "trunk", "text")


def _kmeans(feats: torch.Tensor, k: int, iters: int = 15,
            seed: int = 0) -> torch.Tensor:
    """Spherical (cosine) k-means over one image's patch features.

    Returns a region id per patch. Cosine distance matches how every feature
    space here is otherwise scored.
    """
    f = F.normalize(feats.float(), dim=-1)
    n = f.shape[0]
    k = max(1, min(k, n))
    g = torch.Generator().manual_seed(seed)
    cent = f[torch.randperm(n, generator=g)[:k].to(f.device)].clone()
    assign = torch.zeros(n, dtype=torch.long, device=f.device)
    for _ in range(iters):
        assign = (f @ cent.t()).argmax(dim=1)
        acc = torch.zeros_like(cent).index_add_(0, assign, f)
        cnt = torch.bincount(assign, minlength=k).unsqueeze(1)
        # Empty clusters keep their previous centroid; normalizing a zero vector
        # would retire them permanently and shrink K.
        cent = torch.where(cnt > 0, F.normalize(acc / cnt.clamp(min=1), dim=-1), cent)
    return assign


def region_margin(text_patch: torch.Tensor, substrate: torch.Tensor,
                  normal: torch.Tensor, anomaly: torch.Tensor,
                  n_regions: int, seed: int = 0,
                  fg: Optional[torch.Tensor] = None,
                  kmeans_iters: int = 15) -> torch.Tensor:
    """Segment the patch grid into regions, score each against text, broadcast back.

    Clustering first and pooling patch embeddings within each region (rather
    than a plain k x k window) gives context that respects object boundaries.
    ``substrate`` is the space the regions are cut in (e.g. DINOv3 or SAM3
    dense features); ``text_patch`` is the text-aligned space being scored.
    """
    pooled, lab = _region_pool(text_patch, substrate, n_regions, seed=seed, fg=fg,
                               kmeans_iters=kmeans_iters)
    return (pooled @ anomaly.to(pooled.device)
            - pooled @ normal.to(pooled.device))[lab]


def _region_pool(text_patch: torch.Tensor, substrate: torch.Tensor,
                 n_regions: int, seed: int = 0,
                 fg: Optional[torch.Tensor] = None,
                 kmeans_iters: int = 15
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The text-free half of :func:`region_margin`: ``(pooled, region id)``.

    Nothing here depends on the text anchors, so this also backs
    :func:`region_basis`.
    """
    if fg is None:
        lab = _kmeans(substrate, n_regions, iters=kmeans_iters, seed=seed)
    else:
        # Cluster the foreground only; background forms one region of its own.
        lab = torch.full((substrate.shape[0],), n_regions,
                         dtype=torch.long, device=substrate.device)
        idx = torch.nonzero(fg, as_tuple=True)[0]
        if idx.numel() >= 2:
            lab[idx] = _kmeans(substrate[idx], n_regions, iters=kmeans_iters, seed=seed)
    p = F.normalize(text_patch.float(), dim=-1)
    k = int(lab.max().item()) + 1
    acc = torch.zeros(k, p.shape[-1], device=p.device, dtype=p.dtype)
    acc.index_add_(0, lab, p)
    cnt = torch.bincount(lab, minlength=k).clamp(min=1).unsqueeze(1)
    return F.normalize(acc / cnt, dim=-1), lab


def region_basis(text_patch: torch.Tensor, substrate: torch.Tensor,
                 n_regions: int, seed: int = 0,
                 fg: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``region_margin`` with the text factored out: the ``(N, D)`` rows it scores.

    ``region_margin(...) == region_basis(...) @ (anomaly - normal)``, which is
    what lets :mod:`src.basis` express the default map as a linear operator
    on the text anchors.
    """
    pooled, lab = _region_pool(text_patch, substrate, n_regions, seed=seed, fg=fg)
    return pooled[lab]


def sam_objectness(sam_patch: torch.Tensor) -> torch.Tensor:
    """Per-image standardized objectness from the SAM head's patch-feature norm.

    The channel L2-norm of the SAM-adaptor dense features is already an
    objectness signal, needing no training images or second model. Returns
    the signed z-score, not a gate; sign and squashing are CPU-side choices
    left to the caller (see ``scripts/zs_combine.py``).
    """
    sal = sam_patch.float().norm(dim=-1)                        # (..., N)
    mu = sal.mean(dim=-1, keepdim=True)
    sd = sal.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (sal - mu) / sd


def _otsu(z: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Otsu's threshold on one image's values: the split maximizing between-class
    variance."""
    lo, hi = z.min(), z.max()
    if not torch.isfinite(lo) or hi <= lo:
        return lo
    hist = torch.histc(z, bins=bins, min=float(lo), max=float(hi))
    p = hist / hist.sum().clamp_min(1e-12)
    centres = torch.linspace(float(lo), float(hi), bins, device=z.device)
    w0 = torch.cumsum(p, 0)                       # P(below split)
    w1 = 1.0 - w0
    m0 = torch.cumsum(p * centres, 0) / w0.clamp_min(1e-12)
    mt = (p * centres).sum()
    m1 = (mt - torch.cumsum(p * centres, 0)) / w1.clamp_min(1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return centres[int(torch.argmax(torch.nan_to_num(between, nan=0.0)))]


def foreground_mask(sam_z: torch.Tensor, grid, border_frac: float = 0.15):
    """Otsu foreground mask on SAM objectness, plus evidence of a background.

    Otsu always splits, even on a background-free texture, so Cohen's d
    between border and interior objectness is returned too. Returns
    ``(mask, d)`` with mask True on foreground patches.
    """
    gh, gw = grid
    b = max(1, int(round(border_frac * min(gh, gw))))
    is_border = torch.zeros(gh, gw, dtype=torch.bool, device=sam_z.device)
    is_border[:b] = is_border[-b:] = True
    is_border[:, :b] = is_border[:, -b:] = True
    is_border = is_border.reshape(-1)

    zb, zi = sam_z[is_border], sam_z[~is_border]
    pooled = torch.sqrt(0.5 * (zb.var(unbiased=False) + zi.var(unbiased=False)))
    d = (zb.mean() - zi.mean()) / pooled.clamp_min(1e-6)
    return sam_z > _otsu(sam_z), d


# ---------------------------------------------------------------------------
# Language foreground mask and gate: patch_margin with fg/bg anchors.
# 'objsurf' (industrial) and 'anatex' (medical) are the gate vocabularies used
# in the paper; the others are figure/ground word pairs.
# ---------------------------------------------------------------------------
FOREGROUND_STATES = ["foreground", "foreground object", "object in focus"]
BACKGROUND_STATES = ["background", "plain background", "empty background"]

FGBG_VOCABULARIES = {
    "keyword": (["foreground"], ["background"]),
    "phrase": (FOREGROUND_STATES, BACKGROUND_STATES),
    "class": (["{}"], ["background"]),
    "objtex": (["object"], ["texture"]),
    "objsurf": (["object", "thing"], ["surface", "texture"]),
    "organ": (["organ"], ["tissue"]),
    "anatex": (["organ", "anatomy"], ["tissue", "texture"]),
}

# Class name for the name-free vocabularies (only 'class' actually uses it).
FGBG_AGNOSTIC_NAME = "object"


@torch.no_grad()
def fgbg_anchors(backbone, category: str, vocab: str = "keyword",
                 name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(foreground, background)`` text embeddings for one category.

    :func:`text_anchors` with a foreground/background state pair instead of
    normal/anomaly.
    """
    if vocab not in FGBG_VOCABULARIES:
        raise ValueError(f"fg/bg vocabulary must be one of "
                         f"{sorted(FGBG_VOCABULARIES)}, got '{vocab}'.")
    fg_states, bg_states = FGBG_VOCABULARIES[vocab]
    name = name or (prompt_name(category) if vocab == "class"
                    else FGBG_AGNOSTIC_NAME)
    # background takes text_anchors' 'normal' slot so patch_margin(p, bg, fg)
    # is cos(p, fg) - cos(p, bg); swapped back on return.
    bg, fg = text_anchors(backbone, category, name=name,
                          normal_states=bg_states, anomaly_states=fg_states)
    return fg, bg


def text_foreground_mask(fg_margin, mode: str = "zero") -> torch.Tensor:
    """Boolean patch mask from a foreground-minus-background text margin.

    ``mode='zero'`` splits at the margin's zero-crossing.
    ``mode='otsu'`` is the per-image Otsu split (:func:`_otsu`), as a control.
    Accepts one image's ``(N,)`` margins or a ``(B, N)`` batch; Otsu is always
    computed per image.
    """
    m = torch.as_tensor(fg_margin).float()
    if mode == "zero":
        return m > 0
    if mode == "otsu":
        if m.dim() == 1:
            return m > _otsu(m)
        return torch.stack([row > _otsu(row) for row in m])
    raise ValueError(f"text foreground mode must be 'zero' or 'otsu', got '{mode}'.")


def masked_neighborhood_aggregate(tokens: torch.Tensor, grid, size: int,
                                  mask: torch.Tensor) -> torch.Tensor:
    """``neighborhood_aggregate`` that averages over *foreground* neighbours only.

    Avoids diluting the object signal with background at object borders, which
    matters most when the object is small relative to the window.
    """
    if size <= 1:
        return tokens
    b, n, dim = tokens.shape
    gh, gw = grid
    m = mask.reshape(b, 1, gh, gw).float()
    x = (tokens.reshape(b, gh, gw, dim).permute(0, 3, 1, 2) * m)

    # Asymmetric padding keeps the grid size for even windows too.
    lo, hi = (size - 1) // 2, size - 1 - (size - 1) // 2
    pool = lambda t: F.avg_pool2d(F.pad(t, (lo, hi, lo, hi)), size, stride=1)
    num, den = pool(x), pool(m)
    out = torch.where(den > 0, num / den.clamp_min(1e-6),
                      tokens.reshape(b, gh, gw, dim).permute(0, 3, 1, 2))
    return out.permute(0, 2, 3, 1).reshape(b, n, dim)


def affinity_refine(margin: torch.Tensor, feats: torch.Tensor, iters: int = 1,
                    temp: float = 0.1, keep: float = 0.5) -> torch.Tensor:
    """Diffuse a margin field over a patch-affinity graph built from ``feats``.

    ``m <- keep * m + (1 - keep) * softmax(f_i . f_j / temp) @ m``: each patch
    keeps ``keep`` of its own score and takes the rest from similar patches.
    """
    f = F.normalize(feats.float(), dim=-1)
    sim = f @ f.transpose(-2, -1)                               # (N, N)
    # Mask self-similarity so the softmax doesn't collapse onto the identity.
    sim.fill_diagonal_(float("-inf"))
    A = torch.softmax(sim / temp, dim=-1)
    m = margin.float()
    for _ in range(iters):
        m = keep * m + (1.0 - keep) * (A @ m)
    return m

