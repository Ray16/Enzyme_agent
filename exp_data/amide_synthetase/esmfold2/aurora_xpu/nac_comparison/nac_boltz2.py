"""Same near-attack geometry over the matched 49x50 Boltz-2 1-Mg panel (PDB poses)."""
import csv, pathlib, sys
sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/code")
from synthetase_geometry import geometry

ROOT = pathlib.Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/"
                    "boltz2/step1_1mg/results")
rows = []
for tgt in sorted(ROOT.iterdir()):
    if not tgt.is_dir():
        continue
    pred = tgt / f"boltz_results_step1_{tgt.name}" / "predictions" / f"step1_{tgt.name}"
    models = sorted(pred.glob("*model_*.pdb"))
    assert len(models) == 50, (tgt, len(models))
    for m in models:
        d, a = geometry(m, "PA")
        rows.append({"target": tgt.name, "sample": m.stem,
                     "d_A": round(d, 4), "angle_deg": round(a, 3)})
out = pathlib.Path(sys.argv[1])
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "poses ->", out)
