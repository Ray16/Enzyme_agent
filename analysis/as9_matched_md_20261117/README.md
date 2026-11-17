# Matched AS9 step-1 MD screen

Three WT AS9 poses (models 16, 23, 29), selected from the existing 50-pose ATP/glutarate/one-Mg ensemble by shortest nucleophile–Pα distance among angles >=120°, extending the existing `barrier.select_pose` rule. This selects starting hypotheses; it does not change the frozen candidate-ranking metric.

Each pose receives standard preparation at pH 8 with the campaign ATP assets and 12-6-4 metal parameters, 500 ps staged equilibration and 500 ps unrestrained production, seed 17, 4 fs HMR, geometry every 10 ps, DCD every 50 ps. No new cofolding or reactive geometry restraint is added. Starting distances are 4.74, 4.87, and 4.99 Å; these are the closest available poses, not near-attack assertions.

Runtime output: `structural_modeling_agent/runs/as9_step1_multipose_md_20261117/`. The manifest records source hashes, host/GPU/CPU assignments. Each worker uses an atomic claim, one bounded attempt, physical-validation gate, shared GPU reservation, and contention monitoring that stops only its own process group. Existing partial outputs are preserved; they require explicit resume review. Runtime logs and preflight reports remain in the run directory.

The standard preparation protonation audit is retained. A review flag is not evidence that the protonation model is validated; it must accompany downstream interpretation, as in the comparison campaign. Successful MD completion also requires scientific review.
