# Inspect AS9 step-2 active site: is acyl-AMP(B)+Mg(D) in the pocket, where is HMDA(C)?
load /nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/sfab_step2_results/output_configs/step2_AS9/boltz_results_step2_AS9/predictions/step2_AS9/step2_AS9_model_0.pdb, AS9_step2
hide everything; show cartoon, chain A; color grey80, chain A
show sticks, chain B; color yellow, chain B      # adipyl-AMP
show sticks, chain C; color cyan, chain C        # HMDA (nucleophile amine = first N)
show spheres, chain D; color green, chain D       # Mg
# distance: HMDA first amine N -> acyl-AMP carbonyl C region
zoom (chain B or chain C or chain D), 5
bg_color white
