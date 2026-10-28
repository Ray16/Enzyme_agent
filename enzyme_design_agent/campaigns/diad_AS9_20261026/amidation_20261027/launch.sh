#!/bin/bash
# Launch the six step-2 (amidation) cofolds through the cluster reservation gate.
# Every GPU is taken via `gpu_reserve run`, which re-checks freeness at submission and
# releases on exit; no CUDA_VISIBLE_DEVICES is ever set by hand. One job per GPU.
# Hosts restricted to lambda1/lambda2 (boltz 2.2.1 / torch 2.10.0+cu128) to match the
# step-1 environment exactly -- lambda5 runs torch 2.9.1 and is deliberately excluded.
set -uo pipefail

BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_design_agent/campaigns/diad_AS9_20261026/amidation_20261027
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
W=$BASE/runner/worker.sh

# job:host:gpu:cores:seed
JOBS=(
  "WT_AS9:lambda1:0:0-7:20261027"
  "WT_AS9_rep2:lambda1:1:8-15:20261028"
  "WT_AS1:lambda1:2:16-23:20261027"
  "AS9_diad_pose0_fixed_cap_r2:lambda2:0:0-7:20261027"
  "AS9_diad_pose0_internal_cap_r2:lambda2:1:8-15:20261027"
  "AS9_diad_pose3_internal_cap_r2:lambda2:2:16-23:20261027"
)

mkdir -p "$BASE/logs" "$BASE/cofold_out"

for spec in "${JOBS[@]}"; do
  IFS=: read -r job host gpu cores seed <<<"$spec"
  yaml=$BASE/cofold_inputs/$job.yaml
  out=$BASE/cofold_out/$job
  log=$BASE/logs/$job.log
  if [[ ! -f $yaml ]]; then echo "SKIP $job: missing $yaml"; continue; fi
  echo "launch $job -> $host gpu$gpu cores$cores seed=$seed"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" \
    "setsid nohup env OMP=4 NW=2 $GR run $gpu -- $W '$yaml' '$out' '$seed' '$cores' \
       > '$log' 2>&1 < /dev/null & echo started pid=\$! on \$(hostname)"
done

echo
echo "launched; poll with: bash $BASE/check.sh"
