#!/usr/bin/env bash
# Fold the 6 VERIFIED regioselectivity systems (isolated from the collided parent files).
# 40 diffusion samples (paper used 10, but Nyl50 needed more to seat the scissile bond);
# pocket + 2 atom-contact restraints. GPU via gpu_reserve (shared lock => no collision
# with the peer session's jobs). Run ON the GPU host (lambda13).  Usage: run_folds.sh <idx>...
set -uo pipefail
[ "$#" -ge 1 ] || { echo "usage: run_folds.sh <gpu_idx> [<gpu_idx> ...]"; exit 1; }
IDXS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
mkdir -p "$HERE/results" "$HERE/logs"
yamls=("$HERE"/inputs/*.yaml)
echo "systems: ${#yamls[@]}  gpus: ${IDXS[*]}  host: $(hostname)"
pids=(); i=0
for y in "${yamls[@]}"; do
  name=$(basename "$y" .yaml); out="$HERE/results/$name"
  ls "$out"/boltz_results_*/predictions/*/*_model_0.pdb >/dev/null 2>&1 && { echo "[skip] $name"; continue; }
  idx=${IDXS[$(( i % ${#IDXS[@]} ))]}
  ( "$GATE" run "$idx" -- bash -c '
      source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
      conda activate boltz-2
      export LD_LIBRARY_PATH=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mpi-shim/lib:${LD_LIBRARY_PATH:-}
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
      echo "[$(date +%H:%M:%S)] '"$name"' start CUDA=$CUDA_VISIBLE_DEVICES"
      boltz predict "'"$y"'" --out_dir "'"$out"'" --diffusion_samples 40 --output_format pdb \
          --accelerator gpu --devices 1 --num_workers 1 --override
      rc=$?; echo "[$(date +%H:%M:%S)] '"$name"' exit $rc"; exit $rc
    ' ) > "$HERE/logs/$name.log" 2>&1 &
  pids+=($!); i=$((i+1)); sleep 5
done
echo "launched ${#pids[@]}; waiting..."; rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL DONE (rc=$rc) -> $HERE/results"; exit $rc
