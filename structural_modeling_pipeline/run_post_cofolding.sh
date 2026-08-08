#!/bin/bash
set -e
cd /nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling_pipeline
LOG="post_cofolding_pipeline.log"

echo "$(date): Waiting for cofolding jobs to complete..." | tee $LOG

# Wait for all boltz predict processes to finish
while ps aux | grep "boltz predict" | grep -v grep > /dev/null 2>&1; do
    n_running=$(ps aux | grep "boltz predict" | grep -v grep | wc -l)
    echo "$(date): $n_running Boltz processes still running..." | tee -a $LOG
    sleep 60
done
echo "$(date): All Boltz jobs finished." | tee -a $LOG

# Wait for run_cofold.py to finish (it does select_best_model)
while ps aux | grep "run_cofold.py" | grep -v grep > /dev/null 2>&1; do
    echo "$(date): run_cofold.py still running (best model selection)..." | tee -a $LOG
    sleep 30
done
echo "$(date): run_cofold.py finished." | tee -a $LOG

# Check completion counts
SYN_DONE=$(ls amide_synthetase/cofold_output_tight_constraints/*/boltz_results_*/predictions/*/*.pdb 2>/dev/null | sed 's|.*/cofold_output_tight_constraints/||' | cut -d/ -f1 | sort -u | wc -l)
HYD_DONE=$(ls amide_hydrolase/cofold_output_tight_constraints/*/boltz_results_*/predictions/*/*.pdb 2>/dev/null | sed 's|.*/cofold_output_tight_constraints/||' | cut -d/ -f1 | sort -u | wc -l)
echo "$(date): Synthetases complete: $SYN_DONE, Hydrolases complete: $HYD_DONE" | tee -a $LOG

# ── Step 0: Rename cofold_output_tight_constraints → cofold_output ──
echo "" | tee -a $LOG
echo "$(date): Step 0 - Renaming output directories..." | tee -a $LOG

for proj in amide_synthetase amide_hydrolase; do
    if [ -d "$proj/cofold_output_tight_constraints" ]; then
        # Remove old cofold_output if it exists
        if [ -d "$proj/cofold_output" ]; then
            rm -rf "$proj/cofold_output"
            echo "  Removed old $proj/cofold_output" | tee -a $LOG
        fi
        mv "$proj/cofold_output_tight_constraints" "$proj/cofold_output"
        echo "  Renamed $proj/cofold_output_tight_constraints → cofold_output" | tee -a $LOG
    fi
done
echo "$(date): Step 0 done." | tee -a $LOG

# ── Step 1: Analyze cofolding metrics ──
echo "" | tee -a $LOG
echo "$(date): Step 1 - Analyzing cofolding metrics..." | tee -a $LOG
python analyze_cofolding.py 2>&1 | tee -a $LOG
echo "$(date): Step 1 done." | tee -a $LOG

# ── Step 2: Check clashes ──
echo "" | tee -a $LOG
echo "$(date): Step 2 - Checking steric clashes..." | tee -a $LOG
python check_clashes.py 2>&1 | tee -a $LOG
echo "$(date): Step 2 done." | tee -a $LOG

# ── Step 3: Pocket volume analysis ──
echo "" | tee -a $LOG
echo "$(date): Step 3 - Pocket volume analysis..." | tee -a $LOG
python pocket_volume_analysis.py 2>&1 | tee -a $LOG
echo "$(date): Step 3 done." | tee -a $LOG

# ── Step 4: Select best model ──
echo "" | tee -a $LOG
echo "$(date): Step 4 - Selecting best models..." | tee -a $LOG
python select_best_model.py amide_synthetase 2>&1 | tee -a $LOG
python select_best_model.py amide_hydrolase 2>&1 | tee -a $LOG
echo "$(date): Step 4 done." | tee -a $LOG

# ── Step 5: Apo pocket chemistry ──
echo "" | tee -a $LOG
echo "$(date): Step 5 - Apo pocket chemistry..." | tee -a $LOG
python apo_pocket_analysis.py 2>&1 | tee -a $LOG
echo "$(date): Step 5 done." | tee -a $LOG

# ── Step 6: Holo pocket analysis ──
echo "" | tee -a $LOG
echo "$(date): Step 6 - Holo pocket analysis..." | tee -a $LOG
python holo_pocket_analysis.py 2>&1 | tee -a $LOG
echo "$(date): Step 6 done." | tee -a $LOG

# ── Step 7: Generate comprehensive table ──
echo "" | tee -a $LOG
echo "$(date): Step 7 - Generating comprehensive table..." | tee -a $LOG
python generate_comprehensive_table.py 2>&1 | tee -a $LOG
echo "$(date): Step 7 done." | tee -a $LOG

# ── Step 8: Generate HTML reports ──
echo "" | tee -a $LOG
echo "$(date): Step 8 - Generating HTML pocket visualizations..." | tee -a $LOG
python generate_pocket_html.py 2>&1 | tee -a $LOG
echo "$(date): Step 8 done." | tee -a $LOG

# ── Step 9: Clean up redundant folders ──
echo "" | tee -a $LOG
echo "$(date): Step 8 - Cleaning up redundant folders..." | tee -a $LOG

# Synthetase cleanups
# KEEP: cofold_input (current), cofold_output (just renamed from tight_constraints)
for dir in amide_synthetase/cofold_input_backup_before_coherence \
           amide_synthetase/cofold_input_backup_before_pocket_fix \
           amide_synthetase/cofold_rerun_input \
           amide_synthetase/cofold_output_before_catalytic_merge; do
    if [ -d "$dir" ]; then
        echo "  Removing: $dir" | tee -a $LOG
        rm -rf "$dir"
    fi
done

# Hydrolase cleanups
# KEEP: cofold_input (current), cofold_output (just renamed from tight_constraints)
for dir in amide_hydrolase/cofold_input_backup_before_pocket_fix \
           amide_hydrolase/cofold_output_before_catalytic_merge; do
    if [ -d "$dir" ]; then
        echo "  Removing: $dir" | tee -a $LOG
        rm -rf "$dir"
    fi
done

# Remove test/temp files
rm -f amide_hydrolase/pocket_analysis_test.json
rm -f amide_hydrolase/pocket_analysis_backup_before_spatial.json
rm -f test_spatial_transfer.py

echo "$(date): Step 7 done." | tee -a $LOG
echo "" | tee -a $LOG
echo "$(date): Post-cofolding pipeline complete!" | tee -a $LOG
