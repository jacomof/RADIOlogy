#!/bin/bash
# Reproduce the paper's MVTec AD numbers in three stages:
#   1. main_zeroshot.py         encode test images, cache margin fields
#   2. scripts/zs_text_mask.py  cache the language-gate inputs
#   3. scripts/zs_combine.py    combine fields, apply mask + gate, score
#
# Usage: ./reproduce_mvtec.sh /path/to/mvtecad [output_dir]

set -euo pipefail

DATA_ROOT=${1:?Usage: ./reproduce_mvtec.sh /path/to/mvtecad [output_dir]}
OUT=${2:-results_mvtec}
TEXT_CACHE_NAME=mvtec

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[1/3] encoding test images and caching margin fields -> $OUT"
python main_zeroshot.py \
    --backbone c-radio_v4-h --img-size 448 \
    --data-root "$DATA_ROOT" \
    --refine trunk --region-space sam --patch-context 5 --regions 8 32 \
    --prompt-variants class \
    --pixel-size 518 --output-dir "$OUT"

echo "[2/3] caching the language foreground-gate ingredients"
python scripts/zs_text_mask.py --name "$TEXT_CACHE_NAME" --data-root "$DATA_ROOT"

echo "[3/3] combining fields, applying the mask + language gate, scoring"
python scripts/zs_combine.py "$OUT" --data-root "$DATA_ROOT" --pixel-size 518 \
    --combo class_reg8mix class_reg32mix class_reg8mix_refined class_reg32mix_refined \
    --suppress-bg --fg-source colour --fg-fill min --guard binary \
    --text-cache "results_zs_textmask/$TEXT_CACHE_NAME" --text-vocab objsurf --text-gate 0.016

echo "Pixel metrics are printed above; image-level metrics are in"
echo "$OUT/summary_c-radio_v4-h.json (field 'overall')."
