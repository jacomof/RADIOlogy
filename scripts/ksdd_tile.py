"""Split KolektorSDD test frames into horizontal bands, written as an MVTec-layout dataset.

Frames whose height/width ratio exceeds 2:1 are cut into ceil(h/w) bands (3 for
every KSDD frame), so each band can be encoded without an anisotropic resize.
Score the bands with main_zeroshot.py, then recombine them with ksdd_stitch.py.

Usage: python scripts/ksdd_tile.py --src /path/to/ksdd --dst /path/to/ksdd_tiled
"""

import argparse
import math
import os

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="KSDD in MVTec layout (contains ksdd/test).")
    ap.add_argument("--dst", required=True, help="Output root for the banded copy.")
    ap.add_argument("--category", default="ksdd")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="Aspect ratio above which a frame is split.")
    args = ap.parse_args()

    n_images = n_bands_total = 0
    test_dir = os.path.join(args.src, args.category, "test")
    for split in sorted(d for d in os.listdir(test_dir)
                        if os.path.isdir(os.path.join(test_dir, d))):
        src_dir = os.path.join(test_dir, split)
        dst_dir = os.path.join(args.dst, args.category, "test", split)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)):
            stem, ext = os.path.splitext(fname)
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            w, h = img.size
            n = math.ceil(h / w) if h / w > args.threshold else 1
            edges = [round(k * h / n) for k in range(n + 1)]
            for k in range(n):
                band = img.crop((0, edges[k], w, edges[k + 1]))
                band.save(os.path.join(dst_dir, f"{stem}_b{k}{ext}"), quality=95)
            n_images += 1
            n_bands_total += n

    print(f"{n_images} frames -> {n_bands_total} bands under {args.dst}")


if __name__ == "__main__":
    main()
