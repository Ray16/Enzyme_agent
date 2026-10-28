#!/bin/bash
BASE=/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_design_agent/campaigns/diad_AS9_20261026/amidation_q2_20261027
printf "%-34s %8s %-9s %s\n" JOB POSES/50 STATE TAIL
for j in WT_AS9 WT_AS9_rep2 WT_AS1 AS9_diad_pose0_fixed_cap_r2 AS9_diad_pose0_internal_cap_r2 AS9_diad_pose3_internal_cap_r2; do
  d="$BASE/cofold_out/$j/boltz_results_$j/predictions/$j"
  n=$(ls "$d"/${j}_model_*.pdb 2>/dev/null | wc -l)
  log="$BASE/logs/$j.log"
  st=pending
  [[ -f $log ]] && st=running
  grep -qi "ran out of memory" "$log" 2>/dev/null && st=OOM
  grep -qi "Traceback" "$log" 2>/dev/null && st=ERROR
  [[ $n -eq 50 ]] && st=DONE
  printf "%-34s %8s %-9s %s\n" "$j" "$n" "$st" "$(tail -c 110 "$log" 2>/dev/null | tr '\n' ' ' | tail -c 70)"
done
