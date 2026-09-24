#!/usr/bin/env bash
# Stage 2: fold the 6 restrained tetramer systems with Boltz-2.
# Paper protocol: 10 diffusion samples per system, then geometry-filter + pick top.
# GPU access MUST go through the cluster reservation gate (see ~/.claude/CLAUDE.md).
#
# Usage:  run_folds.sh <gpu_idx>      # idx from: gpu_reserve list
# It reserves that GPU for the whole batch and runs all 6 systems sequentially
# (one task per GPU), releasing on exit.
set -euo pipefail
IDX="${1:?usage: run_folds.sh <gpu_idx>   (see: gpu_reserve list)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLTZ=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/boltz
GATE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
mkdir -p "$HERE/results"

"$GATE" run "$IDX" -- bash -c '
  set -e
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  for y in "'"$HERE"'"/inputs/*.yaml; do
    name=$(basename "$y" .yaml)
    out="'"$HERE"'/results/$name"
    if [ -d "$out" ] && ls "$out"/boltz_results_*/predictions/*/*_model_0.pdb >/dev/null 2>&1; then
      echo "[skip] $name already folded"; continue
    fi
    echo "[fold] $name  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
    "'"$BOLTZ"'" predict "$y" --out_dir "$out" \
        --diffusion_samples 10 --output_format pdb \
        --accelerator gpu --devices 1 --override
  done
'
echo "done -> $HERE/results"
