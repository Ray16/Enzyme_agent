#!/usr/bin/env bash
# launch_all.sh [MAX] — fan the literature_cofolding STEP-1 set across free GPUs the
# system-wide gate reports, one detached `gpu_reserve run` worker per GPU
# (reserve -> run fold_loop -> release). Courtesy: freeness comes from `gpu_reserve list`.
# 30 jobs; MAX caps workers so we don't hog a shared node.
set -eo pipefail
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/campaigns/regio_desd_figures_20261021
FLEET=$BASE/fleet
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
EXCLUDE="lambda3 lambda9"
MAX=${1:-16}
n=0
while read -r host idx; do
  [ -z "$host" ] && continue
  [ "$n" -ge "$MAX" ] && break
  if echo "$EXCLUDE" | grep -qw "$host"; then continue; fi
  ( ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "setsid nohup $GR run $idx -- env ITEMS=$FLEET/${ITEMS_NAME:-items_step1.txt} bash $FLEET/fold_loop.sh \
         > $FLEET/logs/worker_${host}_gpu${idx}.log 2>&1 < /dev/null & \
       echo launched $host gpu$idx" \
    || echo "SKIP $host gpu$idx (ssh/reserve fail)" ) &
  n=$((n+1))
done < <("$GR" list)
wait
echo "=== dispatched $n worker launches (MAX=$MAX) ==="
