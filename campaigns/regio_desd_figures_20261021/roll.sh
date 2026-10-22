#!/usr/bin/env bash
# roll.sh -- rolling analysis: structural gate -> score with A -> replot any complete panel.
# Safe to run repeatedly while the fleet works; everything is idempotent.
set -eo pipefail
cd "$(dirname "$0")"
source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # courtesy on a shared node
# the structural gate must pass before anything is scored
python3 validate_outputs.py --quiet > logs/validate_outputs.txt 2>&1 || {
  echo "STRUCTURAL GATE FAILED -- not scoring; see logs/validate_outputs.txt"; tail -8 logs/validate_outputs.txt; exit 1; }
python3 analyze_step1.py > logs/analyze_step1.txt 2>&1
python3 plot_panels.py
