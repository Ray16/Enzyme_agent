#!/bin/bash
# Fold ONE dimer YAML to 50 diffusion samples on ONE already-reserved GPU, chunked.
# V100-32GB cannot hold 50 samples in one boltz call (trunk ~20GB + ~1.2GB/sample);
# so run NCHUNKS calls of K samples each with DISTINCT seeds (boltz seed defaults to None
# -> identical copies without this) and collect all model PDBs under predictions/<name>/.
# Usage: fold_chunked.sh <dimer.yaml> <out_root> [K] [NCHUNKS]
set -uo pipefail
YAML=$1; OUT_ROOT=$2; K=${3:-5}; NCHUNKS=${4:-10}
ENV=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
name=$(basename "$YAML" .yaml)                 # e.g. dimer_P78061
final="$OUT_ROOT/$name/predictions/$name"
mkdir -p "$final"
echo "[fold_chunked] $name : $NCHUNKS chunks x $K samples on CUDA=${CUDA_VISIBLE_DEVICES:-?}"
for s in $(seq 1 $NCHUNKS); do
  cdir="$OUT_ROOT/$name/chunks/seed_$s"
  # skip chunk if already has K models (resumable)
  have=$(find "$cdir" -name "*_model_*.pdb" 2>/dev/null | wc -l)
  if [ "$have" -ge "$K" ]; then echo "  chunk $s: already $have, skip"; else
    mkdir -p "$cdir"
    $ENV/bin/boltz predict "$YAML" --out_dir "$cdir" --diffusion_samples $K \
        --seed $s --output_format pdb --override >"$cdir/boltz.log" 2>&1
  fi
  # collect models with globally-unique names
  idx=0
  for p in $(find "$cdir" -name "*_model_*.pdb" 2>/dev/null | sort); do
    cp "$p" "$final/${name}_s${s}_m${idx}.pdb"; idx=$((idx+1))
  done
done
tot=$(find "$final" -name "*.pdb" 2>/dev/null | wc -l)
echo "[fold_chunked] $name DONE : $tot model PDBs in $final"
