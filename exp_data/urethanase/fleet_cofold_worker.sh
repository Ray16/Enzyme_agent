#!/usr/bin/env bash
# One worker = one GPU. Claims items from a shared NFS queue, folds, releases.
#
# Size-partitioned on purpose: a 16 GB T4 cannot be assumed to hold the largest
# chains, so MAX_LEN bounds what this worker will take. Longer jobs are left in
# the queue for a worker on a larger card rather than being sent to a GPU that
# will OOM after twenty minutes.
set -uo pipefail

REPO=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent
GPU="${1:?usage: fleet_cofold_worker.sh <gpu_index> [max_len]}"
MAX_LEN="${2:-100000}"
NSAMPLES=50

IN=$REPO/exp_data/urethanase/panel/cofold_inputs
OUT=$REPO/exp_data/urethanase/panel/cofold_out
CLAIMS=$REPO/exp_data/urethanase/panel/.claims
LOGD=$REPO/logs/carbamate_campaigns/fleet
BOLTZ=/homes/rzhu/miniforge3/envs/boltz-2/bin/boltz
RESERVE=/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve
mkdir -p "$OUT" "$CLAIMS" "$LOGD"
LOG=$LOGD/worker_$(hostname)_gpu${GPU}.log

# PREFLIGHT: a node missing the env must never touch the queue, or it will
# claim and fail every item in seconds and starve the healthy workers.
if [ ! -x "$BOLTZ" ]; then echo "PREFLIGHT FAIL: no boltz on $(hostname)" | tee -a "$LOG"; exit 1; fi
if [ ! -x "$RESERVE" ]; then echo "PREFLIGHT FAIL: no gpu_reserve" | tee -a "$LOG"; exit 1; fi
echo "[$(date +%H:%M:%S)] worker up on $(hostname) gpu$GPU max_len=$MAX_LEN" >> "$LOG"

fails=0
for yaml in $(ls "$IN"/*.yaml | sort); do
  name=$(basename "$yaml" .yaml)
  # size gate
  len=$(grep -m1 'sequence:' "$yaml" | awk '{print length($2)}')
  [ "$len" -gt "$MAX_LEN" ] && continue
  # already done?
  # "done" means a COMPLETE ensemble. A partial or stale run (e.g. a 5-pose
  # output left by an earlier setting) must be redone, not skipped, or the panel
  # silently mixes ensemble sizes and the per-enzyme descriptors stop being
  # comparable.
  have=$(ls "$OUT/boltz_results_${name}/predictions/${name}/"*.cif 2>/dev/null | wc -l)
  [ "$have" -ge "$NSAMPLES" ] && continue
  # atomic claim (mkdir is NFS-safe)
  mkdir "$CLAIMS/$name" 2>/dev/null || continue

  echo "[$(date +%H:%M:%S)] START $name (${len} aa)" >> "$LOG"
  if "$RESERVE" run "$GPU" -- "$BOLTZ" predict "$yaml" \
        --out_dir "$OUT" --diffusion_samples $NSAMPLES --recycling_steps 3 \
        --output_format mmcif --override >> "$LOG" 2>&1
  then
    echo "[$(date +%H:%M:%S)] OK    $name" >> "$LOG"; fails=0
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name" >> "$LOG"
    rmdir "$CLAIMS/$name" 2>/dev/null   # release so a healthy worker retries
    fails=$((fails+1))
    if [ "$fails" -ge 3 ]; then
      echo "[$(date +%H:%M:%S)] CIRCUIT BREAK after $fails consecutive failures" >> "$LOG"
      exit 1
    fi
  fi
done
echo "[$(date +%H:%M:%S)] worker done" >> "$LOG"
