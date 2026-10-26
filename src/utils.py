"""Small shared helpers: seeding, device, spatial pooling, and results I/O."""

import json
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_as_native)
    print(f"[io] wrote {path}")


def _as_native(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def neighborhood_aggregate(tokens: torch.Tensor, grid: Tuple[int, int],
                           size: int = 3) -> torch.Tensor:
    """Average each patch token with its ``size x size`` spatial neighbourhood
    on the grid: WinCLIP's window context, done post-hoc in feature space
    instead of by re-running the encoder over a window pyramid.

    ``size <= 1`` is a no-op. Border patches average only valid neighbours
    (``count_include_pad=False``) so they aren't biased toward zero.
    """
    if size <= 1:
        return tokens
    b, n, d = tokens.shape
    gh, gw = grid
    x = tokens.reshape(b, gh, gw, d).permute(0, 3, 1, 2)          # (B, D, gh, gw)
    # Asymmetric padding so even ``size`` also yields a 'same'-shaped grid;
    # avg_pool2d's symmetric ``padding=size//2`` returns gh+1 for even size.
    lo, hi = (size - 1) // 2, size - 1 - (size - 1) // 2
    x = F.avg_pool2d(F.pad(x, (lo, hi, lo, hi)), kernel_size=size, stride=1,
                     count_include_pad=False)
    return x.permute(0, 2, 3, 1).reshape(b, n, d)
