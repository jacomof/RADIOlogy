"""The paper's anomaly map as a linear function of the text anchors.

Context pooling, region pooling and affinity refinement do not depend on the
text, so the map is ``B @ t`` with ``t = t_a - t_n`` and a text-free
``(N, D)`` basis ``B`` per image. main.py uses this to build the map in one
pass.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.utils import neighborhood_aggregate
from src.zeroshot import affinity_refine, region_basis


@dataclass
class BasisConfig:
    """Stage-1 geometry of the default configuration.

    Defaults are the reported run: ``--patch-context 5 --regions 8 32
    --refine trunk --region-space sam``, whose four fields (``reg8mix``,
    ``reg32mix``, and both ``_refined``) are averaged into the reported map.
    """
    patch_context: int = 5
    regions: Tuple[int, ...] = (8, 32)
    refine: str = "trunk"
    region_space: str = "sam"
    refine_iters: int = 1
    refine_temp: float = 0.1
    refine_keep: float = 0.5
    seed: int = 0

    @property
    def fields(self) -> List[str]:
        """The stage-2 ``--combo`` this basis is the mean of."""
        return [f"class_reg{k}mix{sfx}" for k in self.regions
                for sfx in ("", "_refined")]


@torch.no_grad()
def deployed_basis(feats, cfg: BasisConfig = BasisConfig()) -> torch.Tensor:
    """``(B, N, D)`` rows such that ``basis @ t`` is the paper's anomaly map.

    For each K, the region basis is averaged with the 5x5 context basis; that
    field and its affinity-refined version (a fixed linear operator) are kept,
    and all fields are averaged.
    """
    tp = feats.clip_text_patch
    grid = feats.grid
    b = tp.shape[0]
    head = {"dino": feats.dino_patch, "sam": feats.sam_patch, "text": tp,
            "self": tp, "trunk": feats.patch_tokens}
    sub, aff = head[cfg.region_space], (head[cfg.refine] if cfg.refine != "none"
                                        else None)

    r_ctx = F.normalize(
        neighborhood_aggregate(tp, grid, cfg.patch_context).float(), dim=-1)
    fields = []
    for k in cfg.regions:
        r_reg = torch.stack([region_basis(tp[i], sub[i], k, seed=cfg.seed)
                             for i in range(b)], 0)
        r_mix = 0.5 * (r_reg + r_ctx)
        fields.append(r_mix)
        if aff is not None:
            fields.append(torch.stack(
                [affinity_refine(r_mix[i], aff[i], iters=cfg.refine_iters,
                                 temp=cfg.refine_temp, keep=cfg.refine_keep)
                 for i in range(b)], 0))
    return torch.stack(fields, 0).mean(0)


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian truncated at 4 sigma, matching scipy.ndimage.gaussian_filter's default."""
    radius = int(4.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on ``(B, H, W)``, the default sigma=4 smoothing.

    Reflect padding rather than scipy's whole-sample reflection: the two
    differ only in the outermost pixel of the border.
    """
    if sigma <= 0:
        return x
    k = _gaussian_kernel(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    y = x.unsqueeze(1)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return y.squeeze(1)
