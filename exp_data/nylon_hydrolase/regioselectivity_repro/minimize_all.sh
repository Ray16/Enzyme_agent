#!/usr/bin/env bash
# Parallel classical-FF minimization (FF19SB + GAFF v1 + AM1-BCC) of the 6 selected
# poses. Robust to GPU count: each GPU gets a queue and runs its systems SEQUENTIALLY,
# so at most one job per GPU at a time (never oversubscribes). Must run ON the GPU host.
#   usage: minimize_all.sh <gpu_idx> [<gpu_idx> ...]
set -uo pipefail
[ "$#" -ge 1 ] || { echo "usage: minimize_all.sh <gpu_idx>..."; exit 1; }
IDXS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
mkdir -p "$HERE/minimized"
systems=(Nyl10__MAXA_precursor Nyl10__XAMA_precursor Nyl12__MAXA_precursor \
         Nyl12__XAMA_precursor Nyl50__MAXA_precursor Nyl50__XAMA_precursor)

run_gpu() {                      # $1=gpu idx, rest=systems to run sequentially on it
  local g=$1; shift
  for s in "$@"; do
    "$GATE" run "$g" -- bash -c '
        source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
        conda activate fep
        export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
        python '"$HERE"'/minimize.py '"$s"'
      ' > "$HERE/minimized/$s.log" 2>&1
    echo "[gpu $g] $s done"
  done
}

# distribute systems round-robin into per-GPU lists
declare -A lists
i=0
for s in "${systems[@]}"; do
  g=${IDXS[$(( i % ${#IDXS[@]} ))]}
  lists[$g]="${lists[$g]:-} $s"; i=$((i+1))
done

pids=()
for g in "${!lists[@]}"; do
  run_gpu "$g" ${lists[$g]} &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL MIN DONE rc=$rc"
grep -h "kcal/mol" "$HERE"/minimized/*.log 2>/dev/null
