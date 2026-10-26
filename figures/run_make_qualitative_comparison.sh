#!/bin/bash
#SBATCH --job-name=qual_fig
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# CPU-only figure assembly (numpy/PIL/matplotlib); see
# make_qualitative_comparison.py for data sources.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
set +u
eval "$($CONDA_EXE shell.bash hook)"
conda activate YOUR_ENV_NAME
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python figures/make_qualitative_comparison.py
echo "DONE"
