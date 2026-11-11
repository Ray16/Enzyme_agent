#!/bin/bash
# Legacy exploratory cluster scan for ONE adenylation pose; no validated free-energy barrier.
# args: <cofold_pdb> <enzyme_id> <out_dir> [wt_barrier]
set -e
COF="$1"; ID="$2"; OUT="$3"; WT="${4:-}"
EB=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent
MM=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mlmm-toolkit
MD=/nfs/lambda_stor_01/homes/rzhu/Tool/envs/processivity-md-20261106/bin/python
UMA=/nfs/lambda_stor_01/homes/rzhu/Tool/envs/amide-uma-20261105/bin/python
SPEC=$EB/enzyme_barrier/specs/adenylation_adipate_cofold.json
export PATH=$MM/bin:$PATH AMBERHOME=$MM
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-${OMP_NUM_THREADS:-1}}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_NUM_THREADS UMA_NUM_THREADS=$OMP_NUM_THREADS
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo 'Launch through gpu_reserve run and bind the assigned CPUs with taskset.' >&2
  exit 2
fi
mkdir -p "$OUT"
echo "[$ID] stage1 prep (protonate pH8 + hydrate)"
$MM/bin/python -c "import sys;sys.path.insert(0,'$EB');from enzyme_barrier.prep_cofold import split_cofold,param_substrate,build_tleap;o,atp,m,s=split_cofold('$COF','$OUT');param_substrate(s,atp,'$OUT','SUB',-2,ph=8.0);ok,p,c=build_tleap('$OUT','SUB',pol_overrides={'c5':1.061,'nv':1.090});assert ok,'tleap failed'"
echo "[$ID] stage2 minimize"
PYTHONPATH=$EB $MD -c "from enzyme_barrier.prep import minimize;minimize('$OUT/complex_1264.prmtop','$OUT/complex.inpcrd','$OUT',spec_path='$SPEC')"
echo "[$ID] stage3 carve R6 + scan DeltaE"
PYTHONPATH=$EB $UMA -m enzyme_barrier carve --spec $SPEC --prmtop $OUT/complex_1264.prmtop --coords $OUT/minimized.pdb --id $ID --out $OUT
PYTHONPATH=$EB $UMA -m enzyme_barrier scan --meta $OUT/cluster_$ID.meta.json ${WT:+--wt $WT}
PYTHONPATH=$EB $UMA -m enzyme_barrier.plot $OUT/barrier_$ID.json $OUT/energy_diagram_$ID.png || true
echo "[$ID] DONE"
