#!/usr/bin/env bash
# roll.sh -- rolling analysis. Structural gate -> score -> replot panels -> REASSEMBLE the
# composite. Safe to run repeatedly while the fleet works; everything is idempotent.
#
# NOTE: assemble_figure.py must stay in this list. It was omitted at first, so the composite
# silently drifted 18 min behind the per-panel figures and the scores CSV -- a stale headline
# figure is worse than a missing one, because nothing about it looks wrong.
set -eo pipefail
cd "$(dirname "$0")"
source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # courtesy on a shared node
mkdir -p logs

# the structural gate must pass before anything is scored
python3 validate_outputs.py --quiet > logs/validate_outputs.txt 2>&1 || {
  echo "STRUCTURAL GATE FAILED -- not scoring; see logs/validate_outputs.txt"
  tail -8 logs/validate_outputs.txt; exit 1; }

python3 analyze_step1.py  > logs/analyze_step1.txt 2>&1
python3 analyze_step2.py  > logs/analyze_step2.txt 2>&1 || true   # step 2 lags step 1
python3 analyze_control.py > logs/analyze_control.txt 2>&1 || true # control may not be folded yet
python3 plot_panels.py
python3 plot_diagnostics.py > logs/plot_diagnostics.txt 2>&1
python3 plot_two_step.py > logs/plot_two_step.txt 2>&1 || true
python3 run_tests.py > logs/run_tests.txt 2>&1 || true   # regenerates analysis/PREREGISTERED_RESULTS.md
python3 assemble_figure.py | tail -2
