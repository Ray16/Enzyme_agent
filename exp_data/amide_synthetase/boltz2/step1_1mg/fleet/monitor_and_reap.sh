#!/usr/bin/env bash
# Poll the 1-Mg fleet until all 49 enzymes have 50 pdb models. Reap orphans:
# if no workers are running cluster-wide but the panel is incomplete, clear stale
# claims for unfinished enzymes and relaunch across free GPUs. Exits 0 when complete.
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1_1mg
FLEET=$BASE/fleet; CLAIMS=$FLEET/claims
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
ENZ=(SfaB $(seq -f 'AS%g' 1 48))
NODES="lambda0 lambda1 lambda4 lambda5 lambda6 lambda11 lambda12 lambda13"

done_test(){ local e=$1 d="$BASE/results/$1/boltz_results_step1_$1/predictions/step1_$1"
  [ "$(ls "$d"/step1_${e}_model_*.pdb 2>/dev/null | wc -l)" -ge 50 ]; }
count_done(){ local d=0; for e in "${ENZ[@]}"; do done_test "$e" && d=$((d+1)); done; echo $d; }
count_running(){ local t=0; for h in $NODES; do
    n=$(ssh -n -o BatchMode=yes -o ConnectTimeout=6 "$h" 'pgrep -u rzhu -f "boltz predict" | wc -l' 2>/dev/null || echo 0)
    t=$((t+n)); done; echo $t; }

stall=0
for i in $(seq 1 120); do
  d=$(count_done)
  echo "[$(date +%T)] done=$d/49  claims=$(ls "$CLAIMS" 2>/dev/null | wc -l)"
  if [ "$d" -ge 49 ]; then echo "ALL_DONE 49/49"; exit 0; fi
  r=$(count_running)
  if [ "$r" -eq 0 ]; then
    stall=$((stall+1))
    echo "  no workers running (stall=$stall); done=$d"
    if [ "$stall" -ge 2 ]; then
      echo "  REAPING orphans -> clearing stale claims for unfinished + relaunch"
      for e in "${ENZ[@]}"; do done_test "$e" || rmdir "$CLAIMS/$e" 2>/dev/null || true; done
      bash "$FLEET/launch_all.sh" 45 >> "$FLEET/logs/relaunch.log" 2>&1
      stall=0
    fi
  else
    stall=0
  fi
  sleep 90
done
echo "TIMEOUT after 120 polls; done=$(count_done)/49"; exit 1
