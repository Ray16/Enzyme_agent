#!/usr/bin/env bash
set -eo pipefail
source /nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh
conda activate boltz-2
export LD_LIBRARY_PATH=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mpi-shim/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
boltz predict "/homes/rzhu/Enzyme_agent/structural_modeling/cofold/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl/step1/step1_P90518.yaml" --out_dir "/homes/rzhu/Enzyme_agent/structural_modeling/cofold/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl/results/P90518"   --diffusion_samples 50 --max_parallel_samples 1 --output_format pdb   --accelerator gpu --devices 1 --num_workers 1 --override
n=$(ls "/homes/rzhu/Enzyme_agent/structural_modeling/cofold/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl/results/P90518/boltz_results_step1_P90518/predictions/step1_P90518/step1_P90518_model_"*.pdb 2>/dev/null | wc -l)
echo "P90518 lowmem done models=$n"
