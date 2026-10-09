"""Run ligand-axis descriptor over QC-passing cofold poses. Fast + single-threaded (no
oversubscription). Writes per-pose and per-enzyme aggregates."""
import sys, csv, os
from pathlib import Path
import numpy as np

ROOT = Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent")
HERE = ROOT / "exp_data/nylon_hydrolase/analysis/selectivity_proxy_20261009"
QC = ROOT / "exp_data/nylon_hydrolase/analysis/proxy_verification_20261008/qc/model_qc.csv"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, str(HERE))
import ligand_axis as LA  # noqa

STEP2 = 2 * LA.STEP  # axis_fail threshold

qc = [r for r in csv.DictReader(open(QC)) if r["qc_pass"] == "True"]
by_enz = {}
for r in qc:
    by_enz.setdefault(r["id"], []).append(r["path"])

per_pose, agg = [], {}
for enz, paths in by_enz.items():
    for p in sorted(paths):
        try:
            rows = LA.score(ROOT / p)
        except Exception as e:  # noqa
            per_pose.append(dict(id=enz, path=p, site="ERR", error=str(e)))
            continue
        for row in rows:
            fail = row["total_extent"] <= STEP2
            per_pose.append(dict(id=enz, path=p, axis_fail=fail, **row))
            if not fail:
                agg.setdefault(enz, {"asym": [], "distal": []})
                agg[enz]["asym"].append(row["asymmetry"])
                agg[enz]["distal"].append(row["distal_extent"])

with open(HERE / "ligand_axis_per_pose.csv", "w", newline="") as fh:
    keys = ["id", "path", "site", "axis_fail", "e_plus", "e_minus",
            "distal_extent", "total_extent", "asymmetry", "error"]
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader()
    for r in per_pose:
        w.writerow(r)
with open(HERE / "ligand_axis_per_enzyme.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "n_obs", "asymmetry", "distal_capacity"])
    for enz, d in sorted(agg.items()):
        w.writerow([enz, len(d["asym"]), np.mean(d["asym"]), np.mean(d["distal"])])
print(f"enzymes with usable data: {len(agg)} / {len(by_enz)}; "
      f"poses scored: {len(per_pose)}")
