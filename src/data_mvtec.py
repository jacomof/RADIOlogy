"""Dataset loading for MVTec AD and benchmarks converted to its layout:
``<root>/<category>/{train/good, test/<type>, ground_truth/<defect>}``.

Images are only resized and scaled to ``[0, 1]``; RADIO normalizes itself.
"""

import os
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MVTEC_AD_CLASS_NAMES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]

# VisA, converted to the MVTec layout.
VISA_CLASS_NAMES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

# BTAD: three unnamed products ('01'/'02'/'03').
BTAD_CLASS_NAMES = ["01", "02", "03"]

# MPDD (metal parts), already in exact MVTec layout; no conversion needed.
MPDD_CLASS_NAMES = [
    "bracket_black", "bracket_brown", "bracket_white", "connector",
    "metal_plate", "tubes",
]

# BMAD medical benchmarks (Brain MRI, Liver CT, Retinal OCT), MVTec layout.
MEDAD_CLASS_NAMES = ["brain", "liver", "retina"]

# DTD-Synthetic (Aota et al., WACV'23), the texture AD benchmark. Ships in
# MVTec layout already, so no conversion step exists for it.
DTD_SYN_CLASS_NAMES = [
    "Blotchy_099", "Fibrous_183", "Marbled_078", "Matted_069", "Mesh_114",
    "Perforated_037", "Stratified_154", "Woven_001", "Woven_068",
    "Woven_104", "Woven_125", "Woven_127",
]

# DAGM 2007, the 10-class release. Masks are weak ellipse approximations, not
# pixel-accurate boundaries.
DAGM_CLASS_NAMES = [f"Class{i}" for i in range(1, 11)]

# KolektorSDD / SDD2. One physical product each, so one category each.
KSDD_CLASS_NAMES = ["ksdd"]
KSDD2_CLASS_NAMES = ["ksdd2"]

# Medical suite: MEDIMG has no masks (image metrics only), MEDPIX has no
# normal images (pixel metrics only).
ADACLIP_MEDIMG_CLASS_NAMES = ["br35h", "brain_mri", "headct"]
ADACLIP_MEDPIX_CLASS_NAMES = ["clinicdb", "colondb", "isic", "tn3k"]

# The five MVTec texture categories, reported separately from objects.
MVTEC_AD_TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def default_transform(img_size: int, normalize: bool = False) -> Callable:
    """Resize to a square ``img_size`` and map to a ``[0, 1]`` tensor.

    ``normalize=False`` (the default) is what RADIO wants: it applies
    ImageNet mean/std internally via its input conditioner, so feeding it a
    plain ``[0, 1]`` tensor is correct.
    """
    steps = [
        T.Resize((img_size, img_size),
                 interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ]
    if normalize:
        steps.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(steps)


def _sorted_pngs(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )


def collate_samples(batch: List[dict]) -> dict:
    """Collate dict-samples into a batched dict.

    Stacks the image tensors; keeps everything else as plain lists so
    variable-length / string fields (paths, defect types, original sizes)
    survive batching untouched.
    """
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "defect_type": [b["defect_type"] for b in batch],
        "path": [b["path"] for b in batch],
        "orig_size": [b["orig_size"] for b in batch],
        "mask_dir": [b["mask_dir"] for b in batch],
    }


class MVTecAD(Dataset):
    """One MVTec-layout category, train (good only) or test (good + all defects).

    Items are dicts::

        {"image", "label", "defect_type", "path", "orig_size", "mask_path"}
    """

    def __init__(
        self,
        root: str,
        category: str,
        train: bool = False,
        img_size: int = 448,
        transform: Optional[Callable] = None,
        normalize: bool = False,
        require_masks: bool = True,
    ):
        if category not in (MVTEC_AD_CLASS_NAMES + VISA_CLASS_NAMES
                            + BTAD_CLASS_NAMES + MPDD_CLASS_NAMES
                            + MEDAD_CLASS_NAMES + DTD_SYN_CLASS_NAMES
                            + DAGM_CLASS_NAMES + KSDD_CLASS_NAMES
                            + KSDD2_CLASS_NAMES + ADACLIP_MEDIMG_CLASS_NAMES
                            + ADACLIP_MEDPIX_CLASS_NAMES):
            raise ValueError(
                f"Unknown category '{category}'. Expected one of "
                f"{MVTEC_AD_CLASS_NAMES} (MVTec AD) or "
                f"{VISA_CLASS_NAMES} (VisA), {BTAD_CLASS_NAMES} (BTAD), "
                f"{MPDD_CLASS_NAMES} (MPDD) or "
                f"{MEDAD_CLASS_NAMES} (BMAD medical), "
                f"{DTD_SYN_CLASS_NAMES} (DTD-Synthetic), "
                f"{DAGM_CLASS_NAMES} (DAGM) or "
                f"{KSDD_CLASS_NAMES + KSDD2_CLASS_NAMES} (KolektorSDD)."
            )
        # False only for benchmarks without pixel ground truth.
        self.require_masks = require_masks
        self.root = root
        self.category = category
        self.split = "train" if train else "test"
        self.train = train
        self.img_size = img_size
        self.transform = transform or default_transform(img_size, normalize)

        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(
                f"No images found for category '{category}' "
                f"({self.split}) under {root}."
            )

    def _index_samples(self) -> List[dict]:
        cat_dir = os.path.join(self.root, self.category)
        split_dir = os.path.join(cat_dir, self.split)
        gt_root = os.path.join(cat_dir, "ground_truth")
        samples: List[dict] = []

        if self.train:
            for path in _sorted_pngs(os.path.join(split_dir, "good")):
                samples.append(dict(path=path, label=0, defect_type="good",
                                    mask_path=None))
            return samples

        # Test: 'good' first, then every defect subfolder in sorted order.
        defect_types = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )
        defect_types = ["good"] + [d for d in defect_types if d != "good"]

        for defect in defect_types:
            for path in _sorted_pngs(os.path.join(split_dir, defect)):
                mask_path = None
                if defect != "good":
                    stem = os.path.splitext(os.path.basename(path))[0]
                    # MVTec uses '<stem>_mask.png', the VisA conversion '<stem>.png'.
                    for cand in (f"{stem}_mask.png", f"{stem}.png", f"{stem}.bmp"):
                        cand = os.path.join(gt_root, defect, cand)
                        if os.path.isfile(cand):
                            mask_path = cand
                            break
                    else:
                        if self.require_masks:
                            raise FileNotFoundError(
                                f"No ground-truth mask for defective image {path} "
                                f"(looked for '{stem}_mask.png', '{stem}.png', "
                                f"'{stem}.bmp' under {os.path.join(gt_root, defect)}). "
                                f"Pass require_masks=False only if this benchmark ships "
                                f"no pixel ground truth at all."
                            )
                samples.append(dict(path=path, label=int(defect != "good"),
                                    defect_type=defect, mask_path=mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": s["label"],
            "defect_type": s["defect_type"],
            "path": s["path"],
            "orig_size": image.size,               # (W, H)
            "mask_dir": s["mask_path"] or "",      # name kept for collate compat
        }

    def load_mask(self, idx: int, size: Optional[int] = None) -> np.ndarray:
        """Binary GT mask for item ``idx`` (1 = defective).

        ``size`` resizes to ``size x size`` with nearest-neighbour
        interpolation. Good images give an all-zero mask.
        """
        s = self.samples[idx]
        if s["label"] and not s["mask_path"]:
            # Only reachable with require_masks=False.
            raise RuntimeError(
                f"{self.category}: pixel ground truth was requested for defective image "
                f"{s['path']}, but this dataset ships none (require_masks=False). "
                f"Report image-level metrics only for this benchmark.")
        if s["mask_path"]:
            m = Image.open(s["mask_path"]).convert("L")
        else:
            w, h = Image.open(s["path"]).size
            m = Image.new("L", (w, h), 0)
        if size is not None:
            m = m.resize((size, size), Image.NEAREST)
        return (np.array(m) > 0).astype(np.uint8)


def build_test_loader(
    root: str,
    category: str,
    img_size: int = 448,
    batch_size: int = 8,
    num_workers: int = 8,
    normalize: bool = False,
    require_masks: bool = True,
) -> "torch.utils.data.DataLoader":
    """Test-split loader for one category. Zero-shot needs nothing else."""
    from torch.utils.data import DataLoader

    ds = MVTecAD(root, category, train=False, img_size=img_size,
                 normalize=normalize, require_masks=require_masks)
    loader = DataLoader(ds, shuffle=False, batch_size=batch_size,
                        num_workers=num_workers, collate_fn=collate_samples,
                        pin_memory=True)
    return loader
