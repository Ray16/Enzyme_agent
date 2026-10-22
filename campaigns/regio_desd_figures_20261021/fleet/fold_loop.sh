#!/usr/bin/env bash
# fold_loop.sh -- one-GPU claim-based, resumable, self-healing Boltz-2 worker for the regio-paper
# campaign. Launched via `gpu_reserve run <idx> -- bash fold_loop.sh` (gpu_reserve exports
# CUDA_VISIBLE_DEVICES). Multi-node safe: atomic mkdir claims on NFS.
# ITEMS holds one YAML relpath per job; results go to cofolds/<step>/results/<stem>/.
set -eo pipefail   # NOT -u: conda activate uses unbound vars
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/campaigns/regio_desd_figures_20261021
FLEET=$BASE/fleet; CLAIMS=$FLEET/claims; LOGS=$FLEET/logs
mkdir -p "$CLAIMS" "$LOGS"
H=$(hostname); G=${CUDA_VISIBLE_DEVICES:-NA}

source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
# libmpi shim so boltz-2 (mpi4py) runs on ALL nodes (lambda4/11/13 lack system OpenMPI)
export LD_LIBRARY_PATH=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mpi-shim/lib:${LD_LIBRARY_PATH:-}
# courtesy: cap threads so N GPU workers don't oversubscribe cores on a shared node
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# PREFLIGHT: a node that can't run the real import chain must NOT claim anything.
# NOTE: the inherited version of this check also required `mpi4py.MPI`. The current boltz-2 env
# (torch 2.10.0+cu128) has no mpi4py at all and `import boltz` succeeds without it, so that clause
# rejected EVERY node in the fleet. Check only what boltz actually needs to run.
if ! python -c "import sys, torch, boltz; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "$H gpu$G PREFLIGHT FAIL -> no-op"; exit 3
fi

# MEMORY PREFLIGHT -- added after 8 jobs died on 15 GB cards.
# lambda5 is 20x Tesla T4 (15360 MiB) while lambda1/2 are V100-SXM2-32GB. These 592-647 aa
# complexes OOM on a T4, and Boltz "skips the batch" and then dies in the affinity stage looking
# for a .npz the failed structure step never wrote -- leaving 0 poses and a HELD claim, so the job
# looks pending forever. Worse, the casualties were not random: they were the LARGEST ligands and
# therefore concentrated in specific panels (acid Z -> Fig 2B, acid H -> Fig 2C/2D), which would
# have biased the surviving dataset rather than merely thinning it.
MINMEM=${MINMEM:-20000}
GPU_MEM=$(python - <<'PY'
import torch
print(int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024) if torch.cuda.is_available() else 0)
PY
)
if [ "${GPU_MEM:-0}" -lt "$MINMEM" ]; then
  echo "$H gpu$G has ${GPU_MEM} MiB < ${MINMEM} MiB required -> no-op (undersized card)"; exit 5
fi

done_test(){ # $1 = stem, $2 = outdir -- Boltz exits 0 on OOM, so COUNT the structures
  [ "$(ls "$2/boltz_results_$1/predictions/$1/${1}_model_"*.pdb 2>/dev/null | wc -l)" -ge 50 ]; }

fails=0; ran=0
while read -r rel; do
  [ -z "$rel" ] && continue
  yaml="$BASE/$rel"
  stem=$(basename "$rel" .yaml)
  outdir="$BASE/$(dirname "$rel" | sed 's|/inputs$|/results|')/$stem"
  claim="$CLAIMS/$(echo "$rel" | tr '/' '_')"
  done_test "$stem" "$outdir" && continue             # already folded
  mkdir "$claim" 2>/dev/null || continue              # atomic claim
  done_test "$stem" "$outdir" && continue
  mkdir -p "$outdir"
  echo "[$(date +%T)] $H gpu$G start $stem"
  # --max_parallel_samples is MANDATORY here: --diffusion_samples 50 alone holds all 50 poses
  # on-card and OOMs these 592-647 aa chains at exit 0 with zero structures.
  if boltz predict "$yaml" \
        --out_dir "$outdir" \
        --diffusion_samples 50 --max_parallel_samples 5 --output_format pdb \
        --accelerator gpu --devices 1 --num_workers 1 --override \
        > "$LOGS/$stem.log" 2>&1 && done_test "$stem" "$outdir"; then
    echo "[$(date +%T)] $H gpu$G DONE $stem"; ran=$((ran+1)); fails=0
  else
    n=$(ls "$outdir/boltz_results_$stem/predictions/$stem/${stem}_model_"*.pdb 2>/dev/null | wc -l)
    oom=$(grep -ci "out of memory" "$LOGS/$stem.log" 2>/dev/null || true)
    echo "[$(date +%T)] $H gpu$G FAIL $stem (pdbs=$n oom_lines=$oom)"
    rmdir "$claim" 2>/dev/null || true
    fails=$((fails+1)); [ "$fails" -ge 3 ] && { echo "$H gpu$G CIRCUIT-BREAKER ran=$ran"; exit 4; }
  fi
done < "${ITEMS:-$FLEET/items_step1.txt}"
echo "$H gpu$G finished ran=$ran"
