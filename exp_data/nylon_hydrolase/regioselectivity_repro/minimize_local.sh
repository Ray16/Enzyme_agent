#!/usr/bin/env bash
# Local CPU minimization of the 6 selected poses (FF19SB + GAFF v1 + AM1-BCC).
# sqm-bound (~5 min/system); OpenMM min on CPU. Runs all 6 in parallel locally,
# no GPU reservation needed. Meant to run as ONE tracked background task.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/minimized"
source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate fep
export OPENMM_PLATFORM=CPU OPENMM_CPU_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
systems=(Nyl10__MAXA_precursor Nyl10__XAMA_precursor Nyl12__MAXA_precursor \
         Nyl12__XAMA_precursor Nyl50__MAXA_precursor Nyl50__XAMA_precursor)
pids=()
for s in "${systems[@]}"; do
  python "$HERE/minimize.py" "$s" > "$HERE/minimized/$s.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "ALL MIN DONE rc=$rc"
grep -h "kcal/mol" "$HERE"/minimized/*.log 2>/dev/null
