#!/usr/bin/env bash
# Parallel classical-FF minimization of the 6 selected poses, one system per GPU
# through the reservation gate. Must run ON the GPU host. fep env has
# antechamber+tleap+OpenMM(CUDA)+parmed.
#   usage: minimize_all.sh <gpu_idx> [<gpu_idx> ...]
set -uo pipefail
[ "$#" -ge 1 ] || { echo "usage: minimize_all.sh <gpu_idx>..."; exit 1; }
IDXS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
mkdir -p "$HERE/minimized"
systems=(Nyl10__MAXA_precursor Nyl10__XAMA_precursor Nyl12__MAXA_precursor \
         Nyl12__XAMA_precursor Nyl50__MAXA_precursor Nyl50__XAMA_precursor)
i=0; pids=()
for s in "${systems[@]}"; do
  idx=${IDXS[$(( i % ${#IDXS[@]} ))]}
  echo "[launch] $s -> gpu $idx"
  ( "$GATE" run "$idx" -- bash -c '
      source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
      conda activate fep
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
      python '"$HERE"'/minimize.py '"$s"'
    ' ) > "$HERE/minimized/$s.log" 2>&1 &
  pids+=($!); i=$((i+1)); sleep 3
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL MIN DONE rc=$rc"
grep -h "kcal/mol" "$HERE"/minimized/*.log 2>/dev/null
exit $rc
