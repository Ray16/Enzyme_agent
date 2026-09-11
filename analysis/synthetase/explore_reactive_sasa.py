#!/usr/bin/env python3
"""EXPLORATORY (read-only, NOT pipeline): TRUE SASA water-exclusion of the reactive
center vs measured yield (SfaB panel step-1 cofolds), via Biopython ShrakeRupley.

Hypothesis: reactive acyl center shielded from solvent -> hydrolysis suppressed ->
amine wins -> higher yield. Metric = solvent-accessible surface area (A^2) of the
adipate reactive carboxyl C + its two carbonyl O's, computed IN CONTEXT (protein +
ATP + Mg present). LOWER SASA = more buried. Gate to docked poses (reactive C within
GATE A of an ATP phosphate); mean SASA over docked poses.

Correlate: hypothesis predicts NEGATIVE Spearman(SASA, activity). Reported as
"burial signal" = Spearman(-SASA, activity) so positive = buried->active.
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/exploratory_reactive_sasa.csv"
GATE = 6.0
N_MIN = 5
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()


def pose_sasa(pdb):
    """Return (dP, reactive_C_SASA, carbonylO_SASA, center_SASA) or None."""
    m = _P.get_structure("x", pdb)[0]
    E = [a for a in m["E"].get_atoms()] if "E" in m else []
    B = [a for a in m["B"].get_atoms()] if "B" in m else []
    Ps = [a.coord for a in B if a.element == "P"]
    if not E or not Ps:
        return None
    # adipate carboxyl carbons: C with >=2 O neighbours <1.7A within chain E
    carbs = []
    for a in E:
        if a.element != "C":
            continue
        onb = [b for b in E if b.element == "O" and np.linalg.norm(a.coord - b.coord) < 1.7]
        if len(onb) >= 2:
            carbs.append((a, onb))
    if not carbs:
        return None
    react = min(carbs, key=lambda co: min(np.linalg.norm(co[0].coord - p) for p in Ps))
    rc, onb = react
    dP = float(min(np.linalg.norm(rc.coord - p) for p in Ps))
    _SR.compute(m, level="A")     # per-atom SASA in full-complex context
    c_sasa = float(rc.sasa)
    o_sasa = float(sum(o.sasa for o in onb))
    return dP, c_sasa, o_sasa, c_sasa + o_sasa


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step1_"):] if base.startswith("step1_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        try:
            r = pose_sasa(pdb)
        except Exception:
            r = None
        if r:
            per.append(r)
    if not per:
        return None
    dPs = np.array([p[0] for p in per])
    docked = dPs < GATE
    row = {"id": name, "n": len(per), "n_docked": int(docked.sum()),
           "median_dP": round(float(np.median(dPs)), 2)}
    if docked.sum() >= N_MIN:
        idx = np.where(docked)[0]
        row["reactiveC_sasa"] = round(float(np.mean([per[i][1] for i in idx])), 2)
        row["carbonylO_sasa"] = round(float(np.mean([per[i][2] for i in idx])), 2)
        row["center_sasa"] = round(float(np.mean([per[i][3] for i in idx])), 2)
        row["status"] = "scored"
    else:
        for k in ("reactiveC_sasa", "carbonylO_sasa", "center_sasa"):
            row[k] = None
        row["status"] = "INCONCLUSIVE"
    return row


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def auroc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    pos, neg = s[l == 1], s[l == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def perm_p(scores, labels, n=20000, seed=0):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    obs = auroc(s, l); rng = np.random.default_rng(seed)
    ge = sum(auroc(s, rng.permutation(l)) >= obs for _ in range(n))
    return (ge + 1) / (n + 1)


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 24)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["id", "n", "n_docked", "median_dP", "reactiveC_sasa", "carbonylO_sasa",
              "center_sasa", "status", "activity_N66L1"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    y = [r["activity_N66L1"] for r in m]
    labels = [1 if v >= 0.5 else 0 for v in y]
    print(f"=== TRUE SASA (ShrakeRupley) reactive-center water-exclusion vs N66 yield ===")
    print(f"{len(rows)} enzymes | {sum(r['status']=='scored' for r in rows)} scored | activity+scored n={len(m)} | actives={sum(labels)}\n")
    print(f"{'proxy (lower SASA=buried)':26s} {'Spear(-SASA)':>13} {'AUROC':>8} {'perm_p':>8}")
    for key in ("reactiveC_sasa", "carbonylO_sasa", "center_sasa"):
        x = [r[key] for r in m]
        rho = spearman([-v for v in x], y)          # burial signal
        au = auroc([-v for v in x], labels)          # buried scored high
        pp = perm_p([-v for v in x], labels)
        print(f"{key:26s} {rho:>+13.3f} {au:>8.3f} {pp:>8.4f}")
    print("\nactive enzymes' center_sasa (lower=buried):")
    ranked = sorted(m, key=lambda r: r["center_sasa"])   # most buried first
    for r in sorted([r for r in m if r["activity_N66L1"] >= 0.5], key=lambda r: -r["activity_N66L1"]):
        rk = ranked.index(r) + 1
        print(f"  {r['id']:6s} act={r['activity_N66L1']:>5.2f}  center_sasa={r['center_sasa']:>6}  rank {rk}/{len(ranked)} (1=most buried)")
    allc = sorted(r["center_sasa"] for r in m)
    print(f"\ncenter_sasa dist: min {allc[0]:.1f}  median {allc[len(allc)//2]:.1f}  max {allc[-1]:.1f} A^2")
    print(f"actives center_sasa: {sorted(r['center_sasa'] for r in m if r['activity_N66L1']>=0.5)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
