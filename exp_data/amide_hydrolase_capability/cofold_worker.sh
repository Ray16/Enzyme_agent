#!/bin/bash
# One boltz worker: runs on a single GPU (CUDA_VISIBLE_DEVICES set by gpu_reserve run).
# Args: yaml basenames (without .yaml) to process, in order.
set -u
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_hydrolase_capability
ENV=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2
export BOLTZ_CACHE=/nfs/lambda_stor_01/homes/rzhu/.boltz
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
SAMPLES=50
INP=$BASE/inputs
OUTROOT=$BASE/cofold

for id in "$@"; do
  y=$INP/$id.yaml
  o=$OUTROOT/$id
  done_dir=$o/boltz_results_$id/predictions/$id
  n=$(ls "$done_dir"/${id}_model_*.pdb 2>/dev/null | wc -l)
  if [ "$n" -ge "$SAMPLES" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $id (done, $n models)"; continue
  fi
  mkdir -p "$o"
  echo "[$(date +%H:%M:%S)] FOLD $id on CVD=${CUDA_VISIBLE_DEVICES:-?}"
  "$ENV/bin/boltz" predict "$y" --out_dir "$o" \
    --diffusion_samples $SAMPLES --recycling_steps 3 --sampling_steps 200 \
    --output_format pdb --use_msa_server --cache "$BOLTZ_CACHE" \
    --accelerator gpu --devices 1 --num_workers 2 --override \
    > "$o/boltz.log" 2>&1
  rc=$?
  n=$(ls "$done_dir"/${id}_model_*.pdb 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] DONE $id rc=$rc models=$n"
done
