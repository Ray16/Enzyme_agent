#!/usr/bin/env bash
# launch_all.sh [MAX] — fan the 1-Mg re-fold across every free GPU the system-wide
# gate reports, one detached `gpu_reserve run` worker per GPU (reserves->runs->releases).
# Courtesy: freeness + no-contention come straight from `gpu_reserve list` (the gate).
set -eo pipefail
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1_1mg
FLEET=$BASE/fleet
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
EXCLUDE="lambda3 lambda9"          # unreachable / no home mount
MAX=${1:-45}
n=0
# Launch all workers in PARALLEL (background each ssh; ssh -n so it never eats the
# piped GPU list). One detached `gpu_reserve run` worker per free GPU.
while read -r host idx; do
  [ -z "$host" ] && continue
  [ "$n" -ge "$MAX" ] && break
  if echo "$EXCLUDE" | grep -qw "$host"; then continue; fi
  ( ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "setsid nohup $GR run $idx -- bash $FLEET/fold_loop.sh \
         > $FLEET/logs/worker_${host}_gpu${idx}.log 2>&1 < /dev/null & \
       echo launched $host gpu$idx" \
    || echo "SKIP $host gpu$idx (ssh/reserve fail)" ) &
  n=$((n+1))
done < <("$GR" list)
wait
echo "=== dispatched $n worker launches ==="
