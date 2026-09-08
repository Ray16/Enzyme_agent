#!/bin/bash -l
#PBS -N smoke_disc_step1
#PBS -A FRAME-IDP
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:grand
#PBS -j oe
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
source /home/ray16/anaconda3/etc/profile.d/conda.sh
conda activate boltz-cp
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp
RUN=/lus/grand/projects/FRAME-IDP/rzhu/boltz_runs/discovery_step1
export BOLTZ_INPUTDIR="$RUN/configs_smoke" BOLTZ_OUTPUTDIR="$RUN/output_configs_smoke"
export BOLTZ_CACHE=/lus/grand/projects/FRAME-IDP/rzhu/boltz_runs/.boltz_cache
export BOLTZ_TOTAL_SAMPLES=2
mkdir -p logs "$BOLTZ_OUTPUTDIR"
NR=4
export BOLTZ_NRANKS=$NR
mpiexec -n $NR --ppn 4 --cpu-bind depth -d 8 python "$RUN/run_boltz.py" > logs/run_smoke.log 2>&1
