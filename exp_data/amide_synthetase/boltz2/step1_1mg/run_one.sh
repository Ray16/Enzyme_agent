#!/usr/bin/env bash
# Fold ONE enzyme's 1-Mg step-1 input with Boltz-2. Args: GPU_IDX ENZYME
# GPU is passed by gpu_reserve via CUDA_VISIBLE_DEVICES; we ignore $1 for device sel
# (gpu_reserve already exported it) and just use $2 = enzyme.
set -eo pipefail   # NOT -u: conda's binutils activate script uses unbound vars
ENZ=$2
ROOT=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent
BASE=$ROOT/exp_data/amide_synthetase/boltz2/step1_1mg
YAML=$BASE/inputs/step1_$ENZ.yaml
OUT=$BASE/results/$ENZ

source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
echo "[$(date +%T)] start $ENZ on CUDA=$CUDA_VISIBLE_DEVICES"
boltz predict "$YAML" \
    --out_dir "$OUT" \
    --diffusion_samples 50 \
    --output_format pdb \
    --accelerator gpu --devices 1 \
    --num_workers 1 \
    --override
echo "[$(date +%T)] done $ENZ"
