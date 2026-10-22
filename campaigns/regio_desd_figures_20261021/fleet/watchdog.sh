#!/usr/bin/env bash
# watchdog.sh -- drive BOTH fold sets to completion without supervision.
#
# Three jobs, in order:
#   1. RELEASE STALE CLAIMS. A worker that dies hard never reaches its claim-release line, so the
#      job keeps a held claim with <50 poses and looks "in progress" forever. That is exactly how
#      the HSC control sat at 0/50 for 80 minutes. A claim is judged stale when the job has <50
#      poses AND its boltz log has not been touched for STALE_MIN minutes (boltz writes progress
#      continuously, so a quiet log means a dead process). Claims are MOVED to archive/, never
#      deleted.
#   2. LAUNCH on whatever is free. Large-ligand jobs go to >=20 GB cards only (fold_loop.sh
#      enforces MINMEM and no-ops elsewhere), so a T4 picking one up wastes nothing.
#   3. REPORT counts so progress is visible.
#
# Idempotent and safe to run repeatedly: atomic mkdir claims prevent double-running.
set -o pipefail   # NOT -e: `[ cond ] && cmd` would abort silently
cd "$(dirname "$0")/.."
BASE=$PWD
FLEET=$BASE/fleet
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
STALE_MIN=${STALE_MIN:-20}
V100_HOSTS="lambda1 lambda2"
mkdir -p archive/stale_claims

count_done () {  # $1 = step1|step2
  ls -d cofolds/$1/results/*/boltz_results_*/predictions/*/ 2>/dev/null | while read -r d; do
    [ "$(ls "$d"*.pdb 2>/dev/null | wc -l)" -ge 50 ] && echo x
  done | wc -l
}

# ---- 1. release stale claims -------------------------------------------------------------
released=0
for c in fleet/claims/*; do
  [ -d "$c" ] || continue
  stem=$(basename "$c" | sed 's|^cofolds_step[12]_inputs_||;s|\.yaml$||')
  step=$(basename "$c" | sed -n 's|^cofolds_\(step[12]\)_inputs_.*|\1|p')
  [ -z "$step" ] && continue
  pred="cofolds/$step/results/$stem/boltz_results_$stem/predictions/$stem"
  n=$(ls "$pred"/*.pdb 2>/dev/null | wc -l)
  if [ "$n" -ge 50 ]; then continue; fi            # finished; claim is correct
  log="fleet/logs/$stem.log"
  if [ -f "$log" ] && [ -z "$(find "$log" -mmin -$STALE_MIN 2>/dev/null)" ]; then
    mv "$c" "archive/stale_claims/${stem}_$(date +%H%M%S)" 2>/dev/null && released=$((released+1)) \
      && echo "  released stale claim: $stem (${n}/50 poses, log quiet >${STALE_MIN}m)"
  fi
done
[ "$released" -gt 0 ] && echo "  -> $released stale claim(s) released to archive/" || true

# ---- 2. rebuild the outstanding list and launch on free GPUs -------------------------------
for step in step1 step2; do
  items=$FLEET/items_${step}.txt
  if [ ! -f "$items" ]; then continue; fi
  out=$FLEET/outstanding_${step}.txt
  : > "$out"
  while read -r rel; do
    [ -z "$rel" ] && continue
    stem=$(basename "$rel" .yaml)
    pred="cofolds/$step/results/$stem/boltz_results_$stem/predictions/$stem"
    [ "$(ls "$pred"/*.pdb 2>/dev/null | wc -l)" -ge 50 ] || echo "$rel" >> "$out"
  done < "$items"
  n_out=$(wc -l < "$out")
  if [ "$n_out" -eq 0 ]; then echo "  $step: complete"; continue; fi
  launched=0
  while read -r host idx; do
    [ -z "$host" ] && continue
    [ "$launched" -ge "${MAXLAUNCH:-10}" ] && break
    # step-1 leftovers are the large-ligand casualties -> V100 hosts only
    if [ "$step" = step1 ] && ! echo "$V100_HOSTS" | grep -qw "$host"; then continue; fi
    ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "setsid nohup $GR run $idx -- env ITEMS=$out bash $FLEET/fold_loop.sh \
         > $FLEET/logs/wd_${step}_${host}_gpu${idx}.log 2>&1 < /dev/null &" \
      >/dev/null 2>&1 && launched=$((launched+1))
  done < <("$GR" list)
  echo "  $step: $n_out outstanding, $launched worker(s) launched"
done

echo "  TOTALS: step1 $(count_done step1)/58 · step2 $(count_done step2)/90"
