"""Pixel-level AUROC, AP, max-F1 and AUPRO (to 30% FPR) for anomaly maps.

All four come from one shared sort on the GPU. Cumulative sums are float64,
ties keep the last point of each equal-score run, and AP follows sklearn's
``precision_recall_curve``. Connected components use scipy on the CPU.
Maps are compared at 518px by default.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import label as cc_label

_PRO_FPR_LIMIT = 0.3       # the standard AUPRO integration limit
_STRUCTURE = np.ones((3, 3), dtype=int)


def upsample_map(patch_scores: np.ndarray, grid: Sequence[int],
                 size: int, sigma: float = 0.0) -> np.ndarray:
    """Patch-grid scores -> a ``size x size`` pixel map (bilinear, optional blur).

    ``sigma > 0`` applies a Gaussian blur in pixel space. Image-level scores
    are always computed on the unsmoothed patch field.
    """
    import torch.nn.functional as F

    gh, gw = grid
    x = torch.from_numpy(np.asarray(patch_scores, dtype=np.float32)).reshape(1, 1, gh, gw)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    out = x[0, 0].numpy()
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=sigma)
    return out


def _region_change_maps(mask: np.ndarray):
    """Per-pixel FP and PRO increments for one ground-truth mask.

    A pixel outside every region contributes 1 to the false-positive count; a
    pixel inside a region of size ``s`` contributes ``1/s``, so each region
    contributes a total of 1 to the PRO numerator regardless of its area.
    """
    labeled, n_components = cc_label(mask, _STRUCTURE)
    ok = labeled == 0
    fp_change = ok.astype(np.float64)
    pro_change = np.zeros(labeled.shape, dtype=np.float64)
    if n_components:
        counts = np.bincount(labeled.ravel(), minlength=n_components + 1)
        inv = np.zeros_like(counts, dtype=np.float64)
        inv[1:] = 1.0 / counts[1:]
        pro_change = inv[labeled]
    return fp_change, pro_change, n_components, int(ok.sum())


def _trapezoid(x: torch.Tensor, y: torch.Tensor,
               x_max: Optional[float] = None) -> torch.Tensor:
    """Trapezoidal area under ``(x, y)``, matching the MVTec AD evaluation code.

    When ``x_max`` falls between samples, the y value there is linearly
    interpolated and the final partial trapezoid added as a correction,
    rather than holding the last y flat.
    """
    correction = x.new_zeros(())
    if x_max is not None:
        ins = int(torch.searchsorted(
            x, torch.as_tensor(x_max, device=x.device, dtype=x.dtype)))
        if ins <= 0 or ins >= len(x):
            raise ValueError(f"x_max={x_max} lies outside the sampled range")
        if x[ins] != x_max:
            y_i = y[ins - 1] + (y[ins] - y[ins - 1]) * \
                (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            correction = 0.5 * (y_i + y[ins - 1]) * (x_max - x[ins - 1])
        keep = x <= x_max
        x, y = x[keep], y[keep]
    return (0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])).sum() + correction


class PixelAccumulator:
    """Collects score maps + GT masks, then computes all pixel metrics at once.

    Per image it stores the flattened score map plus the two per-pixel increment
    arrays AUPRO needs, so ``compute`` is a single pass with no second look at
    the images.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._scores: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._fp: List[np.ndarray] = []
        self._pro: List[np.ndarray] = []
        self._n_ok = 0
        self._n_regions = 0

    def add(self, score_map: np.ndarray, mask: np.ndarray) -> None:
        s = np.asarray(score_map, dtype=np.float32)
        m = np.asarray(mask, dtype=np.uint8)
        if s.shape != m.shape:
            raise ValueError(f"score map {s.shape} != mask {m.shape}.")
        fp, pro, n_comp, n_ok = _region_change_maps(m)
        self._n_regions += n_comp
        self._n_ok += n_ok
        self._scores.append(s.ravel())
        self._labels.append((m > 0).astype(np.float64).ravel())
        self._fp.append(fp.ravel())
        self._pro.append(pro.ravel())

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        if not self._scores:
            return {}
        dev = self.device
        scores = torch.from_numpy(np.concatenate(self._scores)).to(dev)
        labels = torch.from_numpy(np.concatenate(self._labels)).to(dev)
        fp_change = torch.from_numpy(np.concatenate(self._fp)).to(dev)
        pro_change = torch.from_numpy(np.concatenate(self._pro)).to(dev)

        n_pos = float(labels.sum())
        if n_pos == 0 or n_pos == labels.numel():
            return {}

        # The one sort every metric below is derived from.
        scores_sorted, order = torch.sort(scores, descending=True)
        labels, fp_change, pro_change = labels[order], fp_change[order], pro_change[order]
        del scores, order
        keep = torch.ones_like(scores_sorted, dtype=torch.bool)
        keep[:-1] = scores_sorted[1:] != scores_sorted[:-1]
        del scores_sorted

        tps = torch.cumsum(labels, dim=0)[keep]
        fps = torch.cumsum(1.0 - labels, dim=0)[keep]
        n_neg = float(labels.numel()) - n_pos
        del labels

        zero, one = tps.new_zeros(1), tps.new_ones(1)
        auroc = _trapezoid(torch.cat([zero, fps / n_neg, one]),
                           torch.cat([zero, tps / n_pos, one]))

        precision = tps / (tps + fps)
        recall = tps / n_pos
        last = min(int(torch.searchsorted(tps, tps.new_tensor(n_pos))), len(tps) - 1)
        precision = torch.cat([torch.flip(precision[:last + 1], (0,)), one])
        recall = torch.cat([torch.flip(recall[:last + 1], (0,)), zero])
        ap = -(torch.diff(recall) * precision[:-1]).sum()
        f1 = torch.nan_to_num(2 * precision * recall / (precision + recall), nan=0.0).max()

        fprs = torch.clamp(torch.cumsum(fp_change, dim=0)[keep] / self._n_ok, max=1.0)
        pros = torch.clamp(torch.cumsum(pro_change, dim=0)[keep] / self._n_regions, max=1.0)
        aupro = _trapezoid(torch.cat([zero, fprs, one]), torch.cat([zero, pros, one]),
                           x_max=_PRO_FPR_LIMIT) / _PRO_FPR_LIMIT

        return {"pixel_auroc": float(auroc), "pixel_ap": float(ap),
                "pixel_f1": float(f1), "aupro": float(aupro)}


def format_pixel_metrics(m: Dict[str, float]) -> str:
    if not m:
        return "(no pixel metrics)"
    order = ["pixel_auroc", "aupro", "pixel_ap", "pixel_f1"]
    keys = [k for k in order if k in m] + [k for k in m if k not in order]
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys)


def average_pixel_metrics(
    per_category: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    keys = {k for v in per_category.values() for k in v}
    return {
        k: float(np.mean([v[k] for v in per_category.values() if k in v]))
        for k in sorted(keys)
    }
