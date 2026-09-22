#!/usr/bin/env bash
# Self-healing keeper for the 1-Mg panel. Each poll:
#   1. mark done (>=50 pdb);
#   2. reap STALE claims — claimed+unfinished whose per-enzyme log hasn't been
#      touched in >STALE_MIN min (worker died) -> rmdir the claim (never rm);
#   3. if any unfinished item is unclaimed AND GPUs are free -> relaunch workers
#      (idempotent claim dispatch picks them up). No waiting for 0-running.
# Exits 0 at 49/49. Does NOT contend: launches only through gpu_reserve (free GPUs).
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1_1mg
FLEET=$BASE/fleet; CLAIMS=$FLEET/claims; LOGS=$FLEET/logs
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
ENZ=(SfaB $(seq -f 'AS%g' 1 48)); STALE_MIN=6
done_test(){ local e=$1 d="$BASE/results/$1/boltz_results_step1_$1/predictions/step1_$1"
  [ "$(ls "$d"/step1_${e}_model_*.pdb 2>/dev/null | wc -l)" -ge 50 ]; }

for i in $(seq 1 120); do
  d=0; unclaimed=0
  for e in "${ENZ[@]}"; do
    if done_test "$e"; then d=$((d+1)); continue; fi
    if [ -d "$CLAIMS/$e" ]; then
      # stale-claim reap: no log, or log idle > STALE_MIN min
      lg="$LOGS/$e.log"
      if [ ! -f "$lg" ] || [ -n "$(find "$lg" -mmin +$STALE_MIN 2>/dev/null)" ]; then
        rmdir "$CLAIMS/$e" 2>/dev/null && echo "  reaped stale claim $e"
      fi
    fi
    [ -d "$CLAIMS/$e" ] || unclaimed=$((unclaimed+1))
  done
  echo "[$(date +%T)] done=$d/49 unclaimed_unfinished=$unclaimed"
  [ "$d" -ge 49 ] && { echo "ALL_DONE 49/49"; exit 0; }
  if [ "$unclaimed" -gt 0 ] && [ "$($GR list 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  orphans present + GPUs free -> relaunch"
    bash "$FLEET/launch_all.sh" "$unclaimed" >> "$LOGS/keeper_relaunch.log" 2>&1
  fi
  sleep 90
done
echo "TIMEOUT done=$d/49"; exit 1
