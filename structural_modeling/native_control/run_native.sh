#!/bin/bash -l
#PBS -N nat_ctrl
#PBS -A FRAME-IDP
#PBS -q debug
#PBS -l select=2:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -j oe
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
source /home/ray16/anaconda3/etc/profile.d/conda.sh
conda activate boltz-cp
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp
RUN=/lus/grand/projects/FRAME-IDP/rzhu/boltz_runs/native_control
export BOLTZ_INPUTDIR="$RUN/configs" BOLTZ_OUTPUTDIR="$RUN/output_configs"
export BOLTZ_CACHE=/lus/grand/projects/FRAME-IDP/rzhu/boltz_runs/.boltz_cache
export BOLTZ_TOTAL_SAMPLES=50
mkdir -p logs "$BOLTZ_OUTPUTDIR"
NR=$(( $(wc -l < "$PBS_NODEFILE") * 4 ))
export BOLTZ_NRANKS=$NR
mpiexec -n $NR --ppn 4 --cpu-bind depth -d 8 python "$RUN/run_boltz.py" > logs/run_configs.log 2>&1
