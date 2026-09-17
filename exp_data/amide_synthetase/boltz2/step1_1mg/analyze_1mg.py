#!/usr/bin/env python3
"""Compare 1-Mg vs 2-Mg step-1 cofolds for SfaB + the 3 measured actives.

Readout per enzyme = adipate carboxyl-O near-attack on ATP Pa over the 50 diffusion poses:
  min_d   : closest approach of any adipate O to Pa (near-attack needs <4 A)
  n_gated : #poses with an O within 4 A of Pa
  angle   : mean in-line attack angle among gated poses (the validated proxy score)
2-Mg baseline is the original boltz2/step1 run. If SfaB now reaches Pa with 1 Mg while the
actives still dock, the extra Mg mattered; if SfaB still fails, the miss is substrate-affinity,
not the metal count.
"""
import glob, os, sys
import numpy as np

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
sys.path.insert(0, f"{ROOT}/structural_modeling/scripts")
import step1_nac as s

ENZ = ["SfaB", "AS9", "AS17", "AS23"]
RUNS = {
    "2Mg": f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs/step1_{{e}}/boltz_results_step1_{{e}}/predictions/step1_{{e}}",
    "1Mg": f"{ROOT}/exp_data/amide_synthetase/boltz2/step1_1mg/results/{{e}}/boltz_results_step1_{{e}}/predictions/step1_{{e}}",
}
D_GATE = 4.0


def stats(pred_dir):
    pdbs = sorted(glob.glob(f"{pred_dir}/*model_*.pdb"))
    rows = [g for p in pdbs if (g := s.geom_model(p))]
    if not rows:
        return None
    d = np.array([r["d_pa"] for r in rows])
    a = np.array([r["ang_pa"] for r in rows])
    gm = d < D_GATE
    return dict(n=len(d), min_d=float(d.min()), n_gated=int(gm.sum()),
                mean_angle=float(a[gm].mean()) if gm.any() else float("nan"))


if __name__ == "__main__":
    print(f"{'enz':6} {'run':4} {'n':>3} {'min_d':>6} {'n_gated':>7} {'mean_angle':>10} {'verdict':>14}")
    for e in ENZ:
        for tag, tmpl in RUNS.items():
            st = stats(tmpl.format(e=e))
            if not st:
                print(f"{e:6} {tag:4}  (no results yet)")
                continue
            v = "near-attack" if st["n_gated"] >= 5 else "INCONCLUSIVE"
            print(f"{e:6} {tag:4} {st['n']:>3} {st['min_d']:>6.1f} {st['n_gated']:>7} "
                  f"{st['mean_angle']:>10.1f} {v:>14}")
