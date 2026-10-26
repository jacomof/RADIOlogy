"""Raw CLIP backbone (HuggingFace), used only by bench_clip_raw_throughput.py.

Pinned to CLIP's native resolution. Patch tokens are projected into the
joint space with CLIP's post-layernorm and visual projection.
"""

from typing import List, Optional

import torch
from transformers import AutoTokenizer, CLIPModel

from .base import Backbone, Features

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPBackbone(Backbone):
    name = "clip"
    has_text = True

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16",
                 device: Optional[str] = None, img_size: int = 224):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.patch_size = self.model.config.vision_config.patch_size
        self.min_res_step = self.patch_size
        self.feat_dim = self.model.config.vision_config.hidden_size
        self.clip_dim = self.model.config.projection_dim
        self._mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

        native = self.model.config.vision_config.image_size
        if img_size != native:
            print(f"[clip] pinning img_size to native {native} (requested {img_size}).")
        self.set_img_size(native)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> Features:
        images = images.to(self.device)
        images = (images - self._mean) / self._std
        gh, gw = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        vm = self.model.vision_model
        out = vm(pixel_values=images)
        hidden = out.last_hidden_state                 # (B, 1+N, D) pre-projection
        structural = hidden[:, 1:]                      # patch tokens (structural)
        # Project every token into the joint (text-aligned) space.
        proj = self.model.visual_projection(vm.post_layernorm(hidden))
        clip_summary = proj[:, 0]                       # pooled CLS in joint space
        clip_patch = proj[:, 1:]                        # per-patch joint embeddings

        # For stock CLIP the joint projected space is both the semantic dense
        # space and the text-aligned one, so clip_patch == clip_text_patch.
        cp = clip_patch.float()
        return Features(
            summary=hidden[:, 0].float(), patch_tokens=structural.float(),
            grid=(gh, gw),
            clip_patch=cp, clip_text_patch=cp, clip_summary=clip_summary.float(),
        )

    @torch.inference_mode()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        # Explicit text tower + projection (get_text_features return type varies
        # across transformers versions); pooler_output is the EOS token embedding.
        text_out = self.model.text_model(**tok)
        return self.model.text_projection(text_out.pooler_output).float()
