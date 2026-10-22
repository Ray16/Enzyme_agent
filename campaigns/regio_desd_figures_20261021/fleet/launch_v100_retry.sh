#!/usr/bin/env bash
# launch_v100_retry.sh -- run the OOM-casualty list ONLY on 32 GB cards.
# lambda5 is 20x Tesla T4 (15 GB); these large-ligand complexes OOM there, burn ~50 min each and
# leave 0 poses. fold_loop.sh now refuses any card under MINMEM, so a T4 worker that picks up one
# of these no-ops instead of failing -- but we still steer the retries at V100 hosts directly.
set -eo pipefail
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/campaigns/regio_desd_figures_20261021
FLEET=$BASE/fleet
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
V100_HOSTS="lambda1 lambda2"
MAX=${1:-6}
n=0
while read -r host idx; do
  [ -z "$host" ] && continue
  echo "$V100_HOSTS" | grep -qw "$host" || continue
  [ "$n" -ge "$MAX" ] && break
  ( ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "setsid nohup $GR run $idx -- env ITEMS=$FLEET/items_retry.txt bash $FLEET/fold_loop.sh \
         > $FLEET/logs/retry_${host}_gpu${idx}.log 2>&1 < /dev/null & echo launched $host gpu$idx" \
    || echo "SKIP $host gpu$idx" ) &
  n=$((n+1))
done < <("$GR" list)
wait
echo "=== dispatched $n V100 retry workers ==="
