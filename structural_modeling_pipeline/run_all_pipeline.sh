#!/bin/bash
set -e
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling_pipeline
BOLTZ_BIN=/homes/rzhu/miniforge3/envs/boltz-2/bin/boltz
export LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/boltz-2/lib:$LD_LIBRARY_PATH

echo "=== STEP 1: Wait for apo folding to finish ==="
while ps aux | grep "boltz predict" | grep -v grep > /dev/null 2>&1; do
    count=$(ps aux | grep "boltz predict" | grep -v grep | wc -l)
    active=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100' | wc -l)
    echo "  $(date '+%H:%M:%S') $count boltz processes, $active GPUs active"
    sleep 120
done
echo "Apo folding done at $(date)"
sleep 15
echo "Syn PDBs: $(find $BASE/amide_synthetase/boltz_output/ -path '*/predictions/*' -name '*_model_0.pdb' 2>/dev/null | wc -l)"
echo "Hyd PDBs: $(find $BASE/amide_hydrolase/boltz_output/ -path '*/predictions/*' -name '*_model_0.pdb' 2>/dev/null | wc -l)"

echo ""
echo "=== STEP 2: Run synthetase cofolding with MSA ==="
python3 $BASE/run_cofold.py $BASE/amide_synthetase --skip-prepare --use-msa-server 2>&1

echo ""
echo "=== STEP 3: Run hydrolase cofolding with MSA ==="
python3 $BASE/run_cofold.py $BASE/amide_hydrolase --skip-prepare --use-msa-server 2>&1

echo ""
echo "=== STEP 4: Run pocket analysis on Boltz-2 apo structures ==="
python3 $BASE/boltz_apo_pocket_analysis.py 2>&1

echo ""
echo "=== STEP 5: Run apo pocket analysis ==="
python3 $BASE/apo_pocket_analysis.py 2>&1

echo ""
echo "=== STEP 6: Analyze cofolding results ==="
python3 $BASE/analyze_cofolding.py $BASE/amide_synthetase 2>&1
python3 $BASE/analyze_cofolding.py $BASE/amide_hydrolase 2>&1

echo ""
echo "=== STEP 7: Run holo pocket analysis ==="
python3 $BASE/holo_pocket_analysis.py 2>&1

echo ""
echo "=== STEP 8: Generate comprehensive table ==="
python3 $BASE/generate_comprehensive_table.py 2>&1

echo ""
echo "=== STEP 9: Generate plots ==="
python3 $BASE/plot_pocket_chemistry.py 2>&1
python3 $BASE/plot_boltz_vs_alphafold.py 2>&1

echo ""
echo "=== ALL DONE at $(date) ==="
