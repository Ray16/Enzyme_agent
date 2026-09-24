#!/usr/bin/env bash
# Stage 2: fold the 6 restrained tetramer systems with Boltz-2 (paper protocol:
# 10 diffusion samples). Fans out one system per GPU across the given indices.
# GPU access goes through the cluster reservation gate (~/.claude/CLAUDE.md).
# MUST be run ON the GPU host (e.g. lambda13).
#
# Usage:  run_folds.sh <idx> [<idx> ...]     # free GPU indices on THIS host
#         (see: gpu_reserve list)
set -uo pipefail
[ "$#" -ge 1 ] || { echo "usage: run_folds.sh <gpu_idx> [<gpu_idx> ...]"; exit 1; }
IDXS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLTZ=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/boltz
GATE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
mkdir -p "$HERE/results" "$HERE/logs"

yamls=("$HERE"/inputs/*.yaml)
echo "systems: ${#yamls[@]}  gpus: ${IDXS[*]}  host: $(hostname)"

pids=()
i=0
for y in "${yamls[@]}"; do
  name=$(basename "$y" .yaml)
  out="$HERE/results/$name"
  if ls "$out"/boltz_results_*/predictions/*/*_model_0.pdb >/dev/null 2>&1; then
    echo "[skip] $name already folded"; continue
  fi
  idx=${IDXS[$(( i % ${#IDXS[@]} ))]}
  log="$HERE/logs/$name.log"
  echo "[launch] $name -> gpu $idx  (log: $log)"
  (
    "$GATE" run "$idx" -- bash -c '
      source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
      conda activate boltz-2
      export LD_LIBRARY_PATH=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mpi-shim/lib:${LD_LIBRARY_PATH:-}
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
      echo "[$(date +%H:%M:%S)] '"$name"' start on CUDA=$CUDA_VISIBLE_DEVICES"
      boltz predict "'"$y"'" --out_dir "'"$out"'" \
          --diffusion_samples 10 --output_format pdb \
          --accelerator gpu --devices 1 --num_workers 1 --override
      rc=$?; echo "[$(date +%H:%M:%S)] '"$name"' exit $rc"; exit $rc
    '
  ) > "$log" 2>&1 &
  pids+=($!)
  i=$((i+1))
  sleep 5           # stagger model loads
done

echo "launched ${#pids[@]} jobs; waiting..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL DONE (rc=$rc) -> $HERE/results"
exit $rc
