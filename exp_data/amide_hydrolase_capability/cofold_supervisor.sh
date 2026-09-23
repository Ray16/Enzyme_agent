#!/bin/bash
# Supervisor: run ON lambda5. Each round: shard the REMAINING enzymes across free GPUs
# (disjoint), launch one gpu_reserve-gated boltz worker per GPU, WAIT for all to finish,
# then re-evaluate. Waiting between rounds guarantees no enzyme is assigned to two GPUs.
# Resilient/idempotent: skips already-done (>=50 pdb models); safe to restart.
set -u
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_hydrolase_capability
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
INP=$BASE/inputs
OUTROOT=$BASE/cofold
SAMPLES=50
MAXGPU=6
MAXROUNDS=15
HOST=$(hostname -s)
mkdir -p "$OUTROOT" "$BASE/superv_logs"

remaining() {
  local rem=()
  for y in "$INP"/*.yaml; do
    id=$(basename "$y" .yaml)
    dd=$OUTROOT/$id/boltz_results_$id/predictions/$id
    n=$(ls "$dd"/${id}_model_*.pdb 2>/dev/null | wc -l)
    [ "$n" -ge "$SAMPLES" ] || rem+=("$id")
  done
  echo "${rem[@]}"
}

round=0
while [ $round -lt $MAXROUNDS ]; do
  round=$((round+1))
  rem=($(remaining))
  ntot=$(ls "$INP"/*.yaml | wc -l)
  echo "[$(date +%F_%H:%M:%S)] === round $round: $((ntot-${#rem[@]}))/$ntot done, ${#rem[@]} remaining ==="
  if [ ${#rem[@]} -eq 0 ]; then echo "ALL DONE"; break; fi

  free=($($GR list --hosts $HOST 2>/dev/null | awk -v h=$HOST '$1==h{print $2}'))
  k=${#free[@]}; [ $k -gt $MAXGPU ] && k=$MAXGPU
  # don't spin more workers than remaining enzymes
  [ $k -gt ${#rem[@]} ] && k=${#rem[@]}
  if [ $k -le 0 ]; then echo "  no free GPUs on $HOST; wait 180s"; sleep 180; continue; fi

  declare -A shard; for i in $(seq 0 $((k-1))); do shard[$i]=""; done
  idx=0
  for id in "${rem[@]}"; do shard[$((idx % k))]+=" $id"; idx=$((idx+1)); done

  pids=()
  for i in $(seq 0 $((k-1))); do
    gpu=${free[$i]}; ids=${shard[$i]}; [ -z "$ids" ] && continue
    log=$BASE/superv_logs/worker_gpu${gpu}_r${round}.log
    echo "  GPU $gpu <-$ids"
    $GR run $gpu --label amidehyd -- bash $BASE/cofold_worker.sh $ids > "$log" 2>&1 &
    pids+=($!)
    sleep 4
  done
  echo "  waiting on ${#pids[@]} workers..."
  wait "${pids[@]}"
  echo "  round $round workers finished."
done
echo "[$(date +%F_%H:%M:%S)] supervisor exit; final remaining: $(remaining)"
