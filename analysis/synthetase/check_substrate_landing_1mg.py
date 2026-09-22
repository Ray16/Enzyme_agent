#!/usr/bin/env python3
"""Does adipate land in the active site after the 1-Mg correction?

For the corrected adenylate discovery candidates (13 NIS/DdaG + SfaB), compare the
near-attack geometry of adipate's carboxylate O to ATP Pα between the original 2-Mg
cofold and the corrected 1-Mg re-fold:
  min_d   = closest approach of any adipate O to Pα over the 50 poses (<4 A = docked)
  n_gated = # poses with an adipate O within 4 A of Pα
  angle   = mean in-line attack angle among gated poses
Improvement (min_d down, n_gated up) = the extra Mg was pushing the substrate out.
Uses Pα directly (adenylate mechanism), independent of step1_nac's family classifier.
"""
import glob, sys
import numpy as np
ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
sys.path.insert(0, f"{ROOT}/structural_modeling/scripts")
import step1_nac as S
RUN = f"{ROOT}/structural_modeling/cofold/20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl"
DIRS = {
    "2Mg": f"{RUN}/results/output_configs/step1_{{e}}/boltz_results_step1_{{e}}/predictions/step1_{{e}}",
    "1Mg": f"{RUN}/rerun_1mg/results/{{e}}/boltz_results_step1_{{e}}/predictions/step1_{{e}}",
}
ACCS = ["E2JA32","A0A0F7RJ52","P0DXD6","Q2FW70","Q2FW72","Q2G1M8","Q2G1M9",
        "Q2G1N1","Q47316","Q47318","Q76BS5","Q76BS7","Q81RQ8","A0A2H4T912"]
LABEL = {"E2JA32":"DdaG","A0A2H4T912":"SfaB"}
D_GATE = 4.0

def stats(pdir):
    rows = [g for p in sorted(glob.glob(f"{pdir}/*model_*.pdb")) if (g := S.geom_model(p))]
    if not rows:
        return None
    d = np.array([r["d_pa"] for r in rows]); a = np.array([r["ang_pa"] for r in rows])
    g = d < D_GATE
    return len(d), float(d.min()), int(g.sum()), (float(a[g].mean()) if g.any() else float("nan"))

print(f"{'acc':12}{'name':7}{'run':5}{'n':>4}{'min_d':>7}{'n<4A':>6}{'angle':>7}  verdict")
for e in ACCS:
    line = {}
    for tag, tmpl in DIRS.items():
        st = stats(tmpl.format(e=e))
        line[tag] = st
        if st:
            n, md, ng, ang = st
            v = "docked" if ng >= 5 else ("rare" if ng >= 1 else "OUT")
            print(f"{e:12}{LABEL.get(e,''):7}{tag:5}{n:>4}{md:>7.2f}{ng:>6}{ang:>7.1f}  {v}")
        else:
            print(f"{e:12}{LABEL.get(e,''):7}{tag:5}   (no results)")
    if line.get("1Mg") and line.get("2Mg"):
        print(f"           -> min_d {line['2Mg'][1]:.2f}->{line['1Mg'][1]:.2f} A, "
              f"n<4A {line['2Mg'][2]}->{line['1Mg'][2]}")
    print()
