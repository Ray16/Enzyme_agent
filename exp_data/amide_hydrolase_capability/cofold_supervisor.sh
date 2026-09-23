#!/bin/bash
# Supervisor: run ON lambda5. Keeps launching gpu_reserve-gated boltz workers on free
# GPUs until all 40 enzymes are cofolded (>=50 pdb models) or MAXROUNDS reached.
# Resilient: idempotent, skips done, re-shards remaining each round.
set -u
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_hydrolase_capability
GR=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
INP=$BASE/inputs
OUTROOT=$BASE/cofold
SAMPLES=50
MAXGPU=6
MAXROUNDS=60
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
  echo "[$(date +%F_%H:%M:%S)] round $round: $((ntot-${#rem[@]}))/$ntot done, ${#rem[@]} remaining"
  if [ ${#rem[@]} -eq 0 ]; then echo "ALL DONE"; break; fi

  # free GPU indices on THIS host
  free=($($GR list --hosts $HOST 2>/dev/null | awk -v h=$HOST '$1==h{print $2}'))
  # how many workers currently running (their run holds a lock)
  running=$(pgrep -fc "cofold_worker.sh" 2>/dev/null || echo 0)
  slots=$((MAXGPU - running))
  [ $slots -lt 0 ] && slots=0
  k=${#free[@]}; [ $k -gt $slots ] && k=$slots
  if [ $k -le 0 ]; then echo "  no free slots (running=$running free=${#free[@]}); wait"; sleep 120; continue; fi

  # shard remaining across k workers, round-robin
  declare -A shard
  for i in $(seq 0 $((k-1))); do shard[$i]=""; done
  idx=0
  for id in "${rem[@]}"; do shard[$((idx % k))]+=" $id"; idx=$((idx+1)); done

  for i in $(seq 0 $((k-1))); do
    gpu=${free[$i]}
    ids=${shard[$i]}
    [ -z "$ids" ] && continue
    log=$BASE/superv_logs/worker_gpu${gpu}_r${round}.log
    echo "  launch worker on GPU $gpu :$ids"
    nohup $GR run $gpu --label amidehyd -- bash $BASE/cofold_worker.sh $ids > "$log" 2>&1 &
    sleep 3
  done
  # let workers make progress before re-evaluating
  sleep 300
done
echo "[$(date +%F_%H:%M:%S)] supervisor exit; final:"
rem=($(remaining)); echo "remaining: ${rem[*]:-none}"
