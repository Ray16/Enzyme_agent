#!/usr/bin/env bash
# One worker = one GPU. Claims items from a shared NFS queue, folds, releases.
#
# Size-partitioned on purpose: a 16 GB T4 cannot be assumed to hold the largest
# chains, so MAX_LEN bounds what this worker will take. Longer jobs are left in
# the queue for a worker on a larger card rather than being sent to a GPU that
# will OOM after twenty minutes.
#
# MEMORY, measured: on a 15 GB T4 the panel split exactly by length -- every
# chain <=392 aa produced a full 50-pose ensemble, every chain >=436 aa died.
# The cause is that --diffusion_samples N holds all N samples on the card at
# once, so peak memory scales with N x length. The fix is NOT to cut the sample
# count (50 poses is the statistical floor here), it is to cap how many are held
# in parallel: --max_parallel_samples chunks the diffusion into sequential
# batches and still writes the full ensemble. Memory then tracks the chunk, not
# the ensemble, so the longest chains fit on the same cards.
set -uo pipefail

REPO=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent
GPU="${1:?usage: fleet_cofold_worker.sh <gpu_index> [max_len] [max_parallel]}"
MAX_LEN="${2:-100000}"
MAX_PARALLEL="${3:-10}"
NSAMPLES=50

# N workers share a node's cores; without this each one spawns threads per core
# and the node thrashes.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

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
# $HOME is on NFS so every node sees the same boltz, but the runtime libs are
# node-local: on the V100 nodes rdkit picks up the system /lib libstdc++, which
# predates GLIBCXX_3.4.31, and the import dies. Pointing the loader at the
# env's own lib dir fixes it and is what makes those nodes usable at all --
# without it they resolve the binary, fail every item in seconds, and starve the
# healthy workers. Do not rely on the login profile: an interactive ssh sources
# it and a detached worker does not, so the two disagree about whether the node
# works.
export LD_LIBRARY_PATH="$(dirname "$BOLTZ")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Preflight through the REAL entry point. A proxy check (importing rdkit with a
# separate python call) passed on nodes where boltz itself then failed, which is
# how three items were burned on a node already known to be broken.
if ! "$BOLTZ" --help >/dev/null 2>&1; then
  echo "PREFLIGHT FAIL: boltz will not start on $(hostname)" | tee -a "$LOG"; exit 1
fi
if [ ! -x "$RESERVE" ]; then echo "PREFLIGHT FAIL: no gpu_reserve" | tee -a "$LOG"; exit 1; fi
echo "[$(date +%H:%M:%S)] worker up on $(hostname) gpu$GPU max_len=$MAX_LEN" >> "$LOG"

# Release our claim if the worker is killed. Without this a kill leaves the
# claim behind, the job is never folded and never retried, and the panel
# silently stalls short -- which is invisible because a missing enzyme looks
# the same as one that has not started yet.
#
# Release ONLY a claim we still own. A claim is stamped with host:pid at the
# moment it is taken; a worker that is killed after its item was re-claimed by
# someone else must not delete the new owner's claim, or two workers fold the
# same enzyme while another is never folded at all. Observed: a TERMed worker
# deleted a claim a second worker had taken 90 s earlier.
CURRENT=""
ME="$(hostname):$$"
release_if_mine() {
  [ -n "$1" ] || return 0
  [ "$(cat "$CLAIMS/$1/owner" 2>/dev/null)" = "$ME" ] || return 0
  rm -f "$CLAIMS/$1/owner" 2>/dev/null; rmdir "$CLAIMS/$1" 2>/dev/null
}
cleanup() { release_if_mine "$CURRENT"; exit 130; }
trap cleanup INT TERM HUP

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
  # count across every layout this panel has used: direct, per-attempt, and
  # the original shared one, so a finished ensemble is never re-folded
  have=$( { ls "$OUT/$name/boltz_results_${name}/predictions/${name}/"*.cif \
                "$OUT/$name/"*/boltz_results_${name}/predictions/${name}/*.cif \
                "$OUT/boltz_results_${name}/predictions/${name}/"*.cif ; } 2>/dev/null | wc -l)
  [ "$have" -ge "$NSAMPLES" ] && continue
  # atomic claim (mkdir is NFS-safe)
  mkdir "$CLAIMS/$name" 2>/dev/null || continue
  echo "$ME" > "$CLAIMS/$name/owner"

  CURRENT="$name"
  echo "[$(date +%H:%M:%S)] START $name (${len} aa)" >> "$LOG"
  JOBOUT="$OUT/$name/attempt_$(date +%s)_$$"
  mkdir -p "$JOBOUT"
  if "$RESERVE" run "$GPU" -- "$BOLTZ" predict "$yaml" \
        --out_dir "$JOBOUT" --diffusion_samples $NSAMPLES \
        --max_parallel_samples "$MAX_PARALLEL" --recycling_steps 3 \
        --output_format mmcif --override >> "$LOG" 2>&1
  then
    # Boltz can exit 0 having written nothing (observed: a missing processed
    # .npz leaves zero structures behind). Exit status is therefore not the
    # success test -- the ensemble on disk is. Trusting the exit code logs OK,
    # keeps the claim, and the job is never retried, so the panel silently ends
    # up short with no error anywhere.
    produced=$(ls "$JOBOUT/boltz_results_${name}/predictions/${name}/"*.cif 2>/dev/null | wc -l)
    if [ "$produced" -ge "$NSAMPLES" ]; then
      echo "[$(date +%H:%M:%S)] OK    $name ($produced poses)" >> "$LOG"; fails=0; CURRENT=""
    else
      # Name the cause. An OOM here is a capacity problem (lower MAX_PARALLEL or
      # use a bigger card); anything else is not, and the two need opposite
      # responses. Without this line both look identical in the log, which is
      # how a length-driven OOM was mistaken for a caching bug once already.
      why="unknown"
      tail -n 200 "$LOG" | grep -q "ran out of memory" && why="CUDA OOM"
      echo "[$(date +%H:%M:%S)] SILENT-FAIL $name (exit 0 but $produced/$NSAMPLES poses; $why)" >> "$LOG"
      release_if_mine "$name"; CURRENT=""
      fails=$((fails+1))
      [ "$fails" -ge 3 ] && { echo "[$(date +%H:%M:%S)] CIRCUIT BREAK" >> "$LOG"; exit 1; }
    fi
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name" >> "$LOG"
    release_if_mine "$name"   # release so a healthy worker retries
    fails=$((fails+1))
    if [ "$fails" -ge 3 ]; then
      echo "[$(date +%H:%M:%S)] CIRCUIT BREAK after $fails consecutive failures" >> "$LOG"
      exit 1
    fi
  fi
done
echo "[$(date +%H:%M:%S)] worker done" >> "$LOG"
