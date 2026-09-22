#!/usr/bin/env bash
# fold_loop.sh — one-GPU claim-based, resumable, self-healing Boltz-2 worker for the
# 1-Mg ANL panel re-fold. Launched via `gpu_reserve run <idx> -- bash fold_loop.sh`
# (gpu_reserve exports CUDA_VISIBLE_DEVICES). Multi-node safe: claims are atomic
# `mkdir` on NFS, released on failure so another GPU/node retries.
set -eo pipefail   # NOT -u: conda activate uses unbound vars
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1_1mg
FLEET=$BASE/fleet; CLAIMS=$FLEET/claims; LOGS=$FLEET/logs
mkdir -p "$CLAIMS" "$LOGS"
H=$(hostname); G=${CUDA_VISIBLE_DEVICES:-NA}

source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
# libmpi shim: some nodes (lambda4/11/13) lack a system OpenMPI that mpi4py (pulled
# in by boltz) needs. This NFS env provides libmpi.so.40 cluster-wide (matches the
# system OpenMPI 4.1 on lambda1/5), so boltz-2 runs on ALL nodes.
export LD_LIBRARY_PATH=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mpi-shim/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# (1) PREFLIGHT — a node that can't run the REAL import chain must NOT claim anything.
if ! python -c "import sys, torch, mpi4py.MPI, boltz; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "$H gpu$G PREFLIGHT FAIL -> no-op"; exit 3
fi

done_test(){ local e=$1 d="$BASE/results/$1/boltz_results_step1_$1/predictions/step1_$1"
  [ "$(ls "$d"/step1_${e}_model_*.pdb 2>/dev/null | wc -l)" -ge 50 ]; }

fails=0; ran=0
while read -r e; do
  [ -z "$e" ] && continue
  done_test "$e" && continue                       # already folded
  mkdir "$CLAIMS/$e" 2>/dev/null || continue        # atomic claim (skip if taken)
  done_test "$e" && continue
  echo "[$(date +%T)] $H gpu$G start $e"
  if boltz predict "$BASE/inputs/step1_$e.yaml" \
        --out_dir "$BASE/results/$e" \
        --diffusion_samples 50 --output_format pdb \
        --accelerator gpu --devices 1 --num_workers 1 --override \
        > "$LOGS/$e.log" 2>&1 && done_test "$e"; then
    echo "[$(date +%T)] $H gpu$G DONE $e"; ran=$((ran+1)); fails=0
  else
    echo "[$(date +%T)] $H gpu$G FAIL $e"; rmdir "$CLAIMS/$e" 2>/dev/null || true
    fails=$((fails+1)); [ "$fails" -ge 3 ] && { echo "$H gpu$G CIRCUIT-BREAKER (ran $ran)"; exit 4; }
  fi
done < "$FLEET/items.txt"
echo "$H gpu$G finished ran=$ran"
