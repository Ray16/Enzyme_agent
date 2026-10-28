#!/bin/bash
set -eo pipefail
YAML="$1"; OUT="$2"; SEED="$3"; CORES="$4"
OMP="${OMP:-4}"; NW="${NW:-2}"
R="/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_design_agent/campaigns/diad_AS9_20261026/amidation_q2_20261027/runner"
set +u
source /homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
set -u
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP OMP_THREAD_LIMIT=$OMP
mkdir -p "$OUT"
echo "WORKER host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} cores=$CORES omp=$OMP nw=$NW yaml=$YAML out=$OUT seed=$SEED start=$(date -Is)"
exec taskset -c "$CORES" python "$R/fp32_predict.py" predict "$YAML" \
    --out_dir "$OUT" --cache /nfs/lambda_stor_01/homes/rzhu/.boltz \
    --devices 1 --accelerator gpu --recycling_steps 3 --sampling_steps 200 \
    --diffusion_samples 50 --max_parallel_samples 1 --output_format pdb \
    --num_workers "$NW" --seed "$SEED" --override
