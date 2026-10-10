#!/usr/bin/env bash
# launch_all.sh [MAX] — fan the canonical step1 cofold across free GPUs the system-wide
# gate reports, one detached `gpu_reserve run` worker per GPU (reserves->runs->releases).
# Claim-based + resumable: safe to re-run; already-done items are skipped, and a worker
# joining late just claims whatever is left. Freeness/no-contention come from the gate.
set -eo pipefail
FLEET=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling/cofold/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl/fleet
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
EXCLUDE="lambda3 lambda9"          # unreachable / no home mount
MAX=${1:-10}                        # courtesy cap: leave headroom on the shared fleet
mkdir -p "$FLEET/logs"
n=0
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
echo "=== dispatched $n worker launches (MAX=$MAX) ==="
