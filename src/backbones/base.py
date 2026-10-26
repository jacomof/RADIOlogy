"""Common backbone interface: every backbone returns the same :class:`Features`.

``patch_tokens`` are the structural features. Text-capable backbones also
give ``clip_patch`` (dense teacher features) and ``clip_text_patch`` (the
same patches projected into the joint text space, used for scoring). Inputs
are always ``[0, 1]``; each backbone applies its own normalization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class Features:
    summary: torch.Tensor                     # (B, Ds) structural global token
    patch_tokens: torch.Tensor                # (B, N, Ds) structural patch tokens
    grid: Tuple[int, int]                     # (gh, gw)
    clip_patch: Optional[torch.Tensor] = None       # (B,N,Dc) native CLIP dense features (semantic; for comp)
    clip_text_patch: Optional[torch.Tensor] = None  # (B,N,Dt) patches projected into the text space (for patch-text)
    clip_summary: Optional[torch.Tensor] = None      # (B,Dt) text-aligned image embedding
    sam_patch: Optional[torch.Tensor] = None         # (B,N,Ds) SAM-aligned dense features (localization; for spatial comp)
    patch_ml: Optional[torch.Tensor] = None          # (B,N,Ds*L) last-L-layer concat structural tokens (PatchCore multi-scale; patch branch only)
    dino_patch: Optional[torch.Tensor] = None        # (B,N,Dd) DINOv3-adaptor dense features (structural; second patch-kNN view, for patch_dino)

    @property
    def feat_dim(self) -> int:
        return self.patch_tokens.shape[-1]

    @property
    def has_clip(self) -> bool:
        return self.clip_patch is not None

    def patch_space(self, which: str = "structural") -> torch.Tensor:
        """Return per-patch features in the requested space.

        ``which='clip'`` selects the CLIP-aligned tokens (raises if the backbone
        has none); anything else selects the structural tokens.
        """
        if which == "clip":
            if self.clip_patch is None:
                raise ValueError("Backbone has no CLIP-aligned patch features.")
            return self.clip_patch
        return self.patch_tokens

    def summary_space(self, which: str = "structural") -> torch.Tensor:
        if which == "clip":
            if self.clip_summary is None:
                raise ValueError("Backbone has no CLIP summary.")
            return self.clip_summary
        return self.summary


class Backbone(ABC):
    """Common interface. Subclasses set the attributes in ``__init__``."""

    name: str = "backbone"
    patch_size: int = 16
    min_res_step: int = 16
    feat_dim: int = 0                 # structural patch dim
    clip_dim: Optional[int] = None    # CLIP-space patch dim, if any
    has_text: bool = False
    device: torch.device = torch.device("cpu")

    def set_img_size(self, img_size: int) -> None:
        if img_size % self.min_res_step != 0:
            nearest = round(img_size / self.min_res_step) * self.min_res_step
            raise ValueError(
                f"img_size={img_size} must be a multiple of {self.min_res_step} "
                f"for {self.name}; nearest valid is {nearest}."
            )
        self.img_size = img_size
        g = img_size // self.patch_size
        self.grid = (g, g)

    @abstractmethod
    def encode(self, images: torch.Tensor) -> Features:
        """Encode a ``(B, 3, H, W)`` ``[0, 1]`` batch."""

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Return L2-normalizable ``(T, clip_dim)`` text embeddings.

        Only text-capable backbones implement this; others raise.
        """
        raise NotImplementedError(f"{self.name} has no text encoder.")
