"""Backbone factory. This package ships RADIO (C-RADIOv4-h / radio_v2.5-*),
the backbone the default pipeline runs on, plus a raw CLIP backbone kept
only as the comparison point for bench_clip_raw_throughput.py."""

from typing import Optional

from .base import Backbone, Features

__all__ = ["Backbone", "Features", "build_backbone"]


def build_backbone(name: str, device: Optional[str] = None, img_size: int = 448,
                   want_text: bool = True, want_sam: bool = False,
                   model_name: Optional[str] = None, patch_layers: int = 1,
                   want_dino: bool = False, dino_as_struct: bool = False) -> Backbone:
    """Construct a backbone by short name.

    ``name`` is 'c-radio_v4-h' (default), a 'radio_v2.5-*' version ('radio' is
    an alias for 'radio_v2.5-b'), or 'clip' for the throughput baseline.
    ``model_name`` overrides the HF checkpoint for 'clip' only.
    """
    key = name.lower()
    if (key == "radio" or key.startswith("radio_")
            or key.startswith("c-radio") or key.startswith("e-radio")):
        from .radio import RadioBackbone
        version = "radio_v2.5-b" if key == "radio" else name
        return RadioBackbone(version=version, device=device, img_size=img_size,
                             want_text=want_text, want_sam=want_sam,
                             patch_layers=patch_layers, want_dino=want_dino,
                             dino_as_struct=dino_as_struct)
    if key == "clip":
        from .clip import CLIPBackbone
        return CLIPBackbone(model_name=model_name or "openai/clip-vit-base-patch16",
                            device=device, img_size=img_size)
    raise ValueError(f"Unknown backbone '{name}'. This package ships the RADIO "
                     f"family ('c-radio_v4-h' (default) or a 'radio_v2.5-*' "
                     f"version) and 'clip' (the raw-CLIP throughput baseline).")
