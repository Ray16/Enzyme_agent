#!/usr/bin/env bash
# Pull step-1 discovery cofold predictions from Polaris -> local results/output_configs.
# Grabs only the predictions payload (pdb + confidence/affinity json); skips the bulky
# processed/ feature npz and pae/pde/plddt npz. Preserves the layout step1_nac.py expects:
#   results/output_configs/step1_<acc>/boltz_results_step1_<acc>/predictions/...
set -euo pipefail

REMOTE="polaris:/grand/FRAME-IDP/rzhu/boltz_runs/discovery_step1/output_configs/"
LOCAL="/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling/results/output_configs/"
mkdir -p "$LOCAL"

rsync -avz --prune-empty-dirs \
  --include='*/' \
  --include='predictions/**/*.pdb' \
  --include='predictions/**/confidence_*.json' \
  --include='predictions/**/affinity_*.json' \
  --exclude='*' \
  "$REMOTE" "$LOCAL"

echo "--- pulled configs with model_0.pdb ---"
find "$LOCAL" -path '*predictions*' -name '*_model_0.pdb' | wc -l
