"""RADIO backbone: structural and text-aligned features from one forward pass.

The text adaptor's patch features are in the teacher's vision space, so they
are projected into the text space with the teacher's own head: CLIP's
``ln_post + proj`` (v2.5) or the SigLIP2 MAP head per patch (C-RADIOv4).
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from .base import Backbone, Features

# radio version -> (teacher HF repo, checkpoint filename, vision_width, joint_dim).
# radio_v2.5-b/-l distil the DFN5B CLIP ViT-H/14-378 teacher.
_CLIP_TEACHER = {
    "radio_v2.5-b": ("apple/DFN5B-CLIP-ViT-H-14-378", "open_clip_pytorch_model.bin", 1280, 1024),
    "radio_v2.5-l": ("apple/DFN5B-CLIP-ViT-H-14-378", "open_clip_pytorch_model.bin", 1280, 1024),
}

# C-RADIOv4 version -> (SigLIP2 teacher HF repo, vision_width, joint_dim).
_SIGLIP_TEACHER = {
    "c-radio_v4-h":      ("google/siglip2-giant-opt-patch16-384", 1536, 1536),
    "c-radio_v4-so400m": ("google/siglip2-giant-opt-patch16-384", 1536, 1536),
}


class RadioBackbone(Backbone):
    name = "radio"

    def __init__(self, version: str = "radio_v2.5-b", device: Optional[str] = None,
                 img_size: int = 448, want_text: bool = True, want_sam: bool = False,
                 half: bool = False, patch_layers: int = 1,
                 want_dino: bool = False, dino_as_struct: bool = False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.version = version
        self.patch_layers = patch_layers
        # dino_as_struct: expose the DINO adaptor's features as the
        # *structural* space (patch_tokens/summary) instead of RADIO's own.
        self.dino_as_struct = dino_as_struct
        want_dino = want_dino or dino_as_struct
        # Adaptor names differ across RADIO generations: v2.5/v3 distil {clip, dino_v2, sam};
        # C-RADIOv4 distils the newer {siglip2-g, dino_v3, sam3}. Pick per version.
        is_v4 = "v4" in version
        self._dino_name = "dino_v3_7b" if is_v4 else "dino_v2"  # v4 head distils DINOv3-7B
        self._sam_name = "sam3" if is_v4 else "sam"
        self._text_name = "siglip2-g" if is_v4 else "clip"
        adaptor_names = (([self._text_name] if want_text else [])   # 'clip' (v2.5/v3) or 'siglip2-g' (v4)
                         + ([self._sam_name] if want_sam else [])
                         + ([self._dino_name] if want_dino else []))
        # An explicit ref skips torch.hub's default-branch lookup.
        model = torch.hub.load("NVlabs/RADIO:main", "radio_model", version=version,
                               adaptor_names=adaptor_names or None,
                               progress=True, trust_repo=True, skip_validation=True)
        model.eval().to(self.device)
        self.half = half and self.device.type == "cuda"
        if self.half:
            model.half()
        self.model = model

        # model.adaptors is an nn.ModuleDict (no .get); index by membership.
        adaptors = getattr(model, "adaptors", None)
        self.adaptor = (adaptors[self._text_name] if adaptors is not None
                        and self._text_name in adaptors else None)
        self.has_text = self.adaptor is not None
        self.has_sam = (adaptors is not None and self._sam_name in adaptors)
        self.has_dino = (adaptors is not None and self._dino_name in adaptors)
        if self.dino_as_struct and not self.has_dino:
            raise RuntimeError(
                f"dino_as_struct requested but {version} exposes no '{self._dino_name}' adaptor "
                f"(available: {list(adaptors.keys()) if adaptors else None}).")

        self.patch_size = int(model.patch_size)
        self.min_res_step = int(getattr(model, "min_resolution_step", self.patch_size))
        self.feat_dim = int(model.model.embed_dim)

        # Patch-to-text projection from the teacher checkpoint (CLIP linear for
        # v2.5, SigLIP2 MAP head for v4); native features if no teacher is known.
        self._ln_w = self._ln_b = self._proj = None      # CLIP linear lift
        self._sig: Optional[dict] = None                 # SigLIP MAP-head lift
        self._sig_apply_post_ln = True                   # feed teacher post_layernorm before the head
        if self.has_text and version in _CLIP_TEACHER:
            self._load_clip_projection()
            self.clip_dim = _CLIP_TEACHER[version][3]
        elif self.has_text and version in _SIGLIP_TEACHER:
            self._load_siglip_projection()
            self.clip_dim = _SIGLIP_TEACHER[version][2]
        elif self.has_text and self._text_name.startswith("siglip"):
            print(f"[radio] {version} has a '{self._text_name}' head but no registered "
                  f"SigLIP teacher; patch-text features stay un-projected (native).")

        # Multi-layer patch descriptor (PatchCore multi-scale): capture the last
        # ``patch_layers`` inner-ViT block outputs via forward hooks.
        self._ml_buffer: List[torch.Tensor] = []
        self._ml_hooks = []
        if self.patch_layers > 1:
            self._register_ml_hooks()

        self.set_img_size(img_size)

    def _register_ml_hooks(self) -> None:
        inner = getattr(self.model, "model", None)
        blocks = getattr(inner, "blocks", None)
        if blocks is None:
            raise RuntimeError(
                f"RADIO {self.version} inner ViT exposes no `.blocks`; "
                f"multi-layer patch (patch_layers>1) unsupported for this version.")

        def _hook(module, inp, out):
            self._ml_buffer.append(out[0] if isinstance(out, tuple) else out)

        for blk in list(blocks)[-self.patch_layers:]:
            self._ml_hooks.append(blk.register_forward_hook(_hook))

    def _load_clip_projection(self) -> None:
        if self.version not in _CLIP_TEACHER:
            raise ValueError(
                f"No CLIP-teacher projection registered for {self.version}. "
                f"Add it to _CLIP_TEACHER (teacher repo/ckpt/dims) or use want_text=False."
            )
        repo, fname, _, _ = _CLIP_TEACHER[self.version]
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, fname)      # cached; respects HF_HOME
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self._ln_w = sd["visual.ln_post.weight"].float().to(self.device)
        self._ln_b = sd["visual.ln_post.bias"].float().to(self.device)
        self._proj = sd["visual.proj"].float().to(self.device)  # (1280, 1024)

    def _project_clip(self, feat_1280: torch.Tensor) -> torch.Tensor:
        """Lift teacher-width patch features into the joint text space."""
        x = F.layer_norm(feat_1280.float(), (feat_1280.shape[-1],), self._ln_w, self._ln_b)
        return x @ self._proj

    def _load_siglip_projection(self) -> None:
        """Pull the SigLIP2 teacher's MAP-head + post_layernorm weights.

        Only the tiny head/norm tensors are read (via safetensors' lazy
        ``safe_open``), not the full teacher. No second ViT is instantiated --
        the weights are applied by hand in :meth:`_project_siglip`.
        """
        import json

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        from safetensors import safe_open

        repo, vwidth, _ = _SIGLIP_TEACHER[self.version]
        want_prefix = ("vision_model.head.", "vision_model.post_layernorm.")
        try:                                             # sharded checkpoint (giant is 2 shards)
            index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
            wm = index["weight_map"]
            keys = [k for k in wm if k.startswith(want_prefix)]
            shards = sorted({wm[k] for k in keys})
        except EntryNotFoundError:                       # single-file fallback
            shards = ["model.safetensors"]

        sd: dict = {}
        for shard in shards:
            with safe_open(hf_hub_download(repo, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith(want_prefix):
                        sd[k] = f.get_tensor(k)

        dev = self.device
        g = lambda name: sd[name].float().to(dev)
        h = "vision_model.head."
        D = vwidth
        iw, ib = g(h + "attention.in_proj_weight"), g(h + "attention.in_proj_bias")  # [q;k;v]
        self._sig = {
            "post_w": g("vision_model.post_layernorm.weight"),
            "post_b": g("vision_model.post_layernorm.bias"),
            "Wv": iw[2 * D:3 * D, :].contiguous(),       # value slice of the fused QKV proj
            "bv": ib[2 * D:3 * D].contiguous(),
            "Wo": g(h + "attention.out_proj.weight"),
            "bo": g(h + "attention.out_proj.bias"),
            "ln_w": g(h + "layernorm.weight"),
            "ln_b": g(h + "layernorm.bias"),
            "fc1_w": g(h + "mlp.fc1.weight"), "fc1_b": g(h + "mlp.fc1.bias"),
            "fc2_w": g(h + "mlp.fc2.weight"), "fc2_b": g(h + "mlp.fc2.bias"),
        }

    def _project_siglip(self, feat: torch.Tensor) -> torch.Tensor:
        """Project SigLIP2-teacher patch features into the joint text space.

        The MAP head pools patches with an attention probe. For a single patch that
        attention is the identity, so each patch goes through the head's value and
        output projections and then its residual LayerNorm + MLP.
        """
        s = self._sig
        x = feat.float()
        D = x.shape[-1]
        if self._sig_apply_post_ln:
            x = F.layer_norm(x, (D,), s["post_w"], s["post_b"])
        v = F.linear(x, s["Wv"], s["bv"])                # value projection
        a = F.linear(v, s["Wo"], s["bo"])                # attention output projection
        h = F.layer_norm(a, (D,), s["ln_w"], s["ln_b"])
        h = F.linear(h, s["fc1_w"], s["fc1_b"])
        h = F.gelu(h, approximate="tanh")                # SigLIP uses gelu_pytorch_tanh
        h = F.linear(h, s["fc2_w"], s["fc2_b"])
        return a + h                                     # residual + MLP

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> Features:
        images = images.to(self.device)
        if self.half:
            images = images.half()
        gh, gw = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        if self.patch_layers > 1:
            self._ml_buffer = []
        out = self.model(images)
        # With adaptors RADIO returns {'backbone': RadioOutput, 'clip'/'sam': ...};
        # without, a single RadioOutput (summary, features).
        if isinstance(out, dict):
            bb = out["backbone"]
            clip, sam, dino = (out.get(self._text_name), out.get(self._sam_name),
                               out.get(self._dino_name))
        else:
            bb, clip, sam, dino = out, None, None, None

        patch_ml = None
        if self.patch_layers > 1:
            npatch = gh * gw
            caps = self._ml_buffer
            if len(caps) != self.patch_layers:
                raise RuntimeError(
                    f"expected {self.patch_layers} captured blocks, got {len(caps)}.")
            layers = []
            for t in caps:                       # earliest -> latest
                t = t.float()
                prefix = t.shape[1] - npatch     # strip CLS/registers
                layers.append(t[:, prefix:])
            patch_ml = torch.cat(layers, dim=-1)  # (B, N, D*L)

        clip_patch = clip_text_patch = clip_summary = None
        if clip is not None:
            clip_patch = clip.features.float()               # (B, N, D) native dense (semantic)
            # Lift patches into the joint text space with whichever teacher head is
            # loaded (v2.5/v3 CLIP linear, or v4 SigLIP2 MAP head); else expose native.
            if self._proj is not None:
                clip_text_patch = self._project_clip(clip.features)
            elif self._sig is not None:
                clip_text_patch = self._project_siglip(clip.features)
            else:
                clip_text_patch = clip_patch
            clip_summary = clip.summary.float()

        # DINO-adaptor dense features, exposed as ``dino_patch`` when loaded.
        dino_patch = None
        if dino is not None:
            dino_patch = dino.features.float()
            if dino_patch.ndim == 4:                          # (B,H,W,C) -> (B,N,C)
                dino_patch = dino_patch.reshape(dino_patch.shape[0], -1, dino_patch.shape[-1])

        # Default structural space = RADIO backbone; optionally swap to the
        # DINO adaptor instead.
        struct_summary = bb.summary.float()
        struct_patch = bb.features.float()
        if self.dino_as_struct and dino is not None:
            struct_patch = dino_patch
            struct_summary = dino.summary.float()

        return Features(
            summary=struct_summary,
            patch_tokens=struct_patch,
            grid=(gh, gw),
            clip_patch=clip_patch,
            clip_text_patch=clip_text_patch,
            clip_summary=clip_summary,
            sam_patch=sam.features.float() if sam is not None else None,
            patch_ml=patch_ml,
            dino_patch=dino_patch,
        )

    @torch.inference_mode()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        if not self.has_text:
            raise NotImplementedError("RADIO loaded without the CLIP adaptor (want_text=False).")
        tokens = self.adaptor.tokenizer(prompts).to(self.device)
        return self.adaptor.encode_text(tokens, normalize=False).float()
