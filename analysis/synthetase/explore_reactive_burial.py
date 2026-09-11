#!/usr/bin/env python3
"""EXPLORATORY (read-only, NOT part of the pipeline): does burial / water-exclusion
of the reactive center predict measured yield on the SfaB panel?

Hypothesis: yield is set by partitioning of the activated acyl intermediate between
aminolysis (productive) and hydrolysis (futile). A reactive center more shielded from
solvent -> hydrolysis suppressed -> amine wins -> higher yield. Geometric handle
(no SASA lib needed): count of protein heavy atoms packed around the reactive carbon
= burial; more neighbours = less water access.

Structures: step-1 cofolds (A=protein, B=ATP, C/D=Mg, E=adipate), 50 models/enzyme.
Reactive center = adipate carboxyl C nearest an ATP phosphate (the one poised to be
activated). Gate to DOCKED poses (reactive C within GATE A of a phosphate) so burial
is measured only where the diacid is actually in the pocket, then mean over poses.

Reads only; writes one CSV under a scratch results path. Does not import or alter
any pipeline module.
"""
import csv, glob, json, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/exploratory_reactive_burial.csv"
GATE = 6.0            # reactive C within this of a phosphate = docked/poised
N_MIN = 5            # min docked poses to score
SHELLS = (6.0, 8.0, 10.0)


def parse(path):
    A, B, E = [], [], []          # A=protein heavy-atom xyz; B=(elem,xyz); E=(elem,xyz)
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip()
        elem = (ln[76:78].strip() or an[0]).upper()
        if elem == "H":
            continue
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ch == "A": A.append(xyz)
        elif ch == "B": B.append((elem, xyz))
        elif ch == "E": E.append((elem, xyz))
    return (np.array(A) if A else np.zeros((0, 3))), B, E


def reactive_carbon(B, E):
    """adipate carboxyl C nearest an ATP phosphate; return (C_xyz, [O_xyz...], min_d_to_P)."""
    Ps = [x for (e, x) in B if e == "P"]
    if not Ps or not E:
        return None
    Ecoords = [x for (e, x) in E]
    Es = [(e, x) for (e, x) in E]
    # carboxyl carbons: C with >=2 O neighbours within 1.7A (within adipate)
    carbs = []
    for e, x in Es:
        if e != "C":
            continue
        o_nb = [ox for (oe, ox) in Es if oe == "O" and np.linalg.norm(x - ox) < 1.7]
        if len(o_nb) >= 2:
            carbs.append((x, o_nb))
    if not carbs:
        return None
    # the carboxyl closest to any phosphate = the reactive one
    best = min(carbs, key=lambda co: min(np.linalg.norm(co[0] - p) for p in Ps))
    cx, o_nb = best
    dP = min(np.linalg.norm(cx - p) for p in Ps)
    return cx, o_nb, dP


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step1_"):] if base.startswith("step1_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        A, B, E = parse(pdb)
        rc = reactive_carbon(B, E)
        if rc is None or len(A) == 0:
            continue
        cx, o_nb, dP = rc
        dA = np.linalg.norm(A - cx, axis=1)
        # burial of reactive C: protein heavy atoms within each shell
        bur = {r: int((dA < r).sum()) for r in SHELLS}
        # burial of the reactive carbonyl oxygens (mean over the O's)
        oburs = []
        for ox in o_nb:
            do = np.linalg.norm(A - ox, axis=1)
            oburs.append(int((do < 8.0).sum()))
        per.append((dP, bur, float(np.mean(oburs))))
    if not per:
        return None
    dPs = np.array([p[0] for p in per])
    docked = dPs < GATE
    row = {"id": name, "n": len(per), "n_docked": int(docked.sum()),
           "median_dP": round(float(np.median(dPs)), 2)}
    if docked.sum() >= N_MIN:
        idx = np.where(docked)[0]
        for r in SHELLS:
            row[f"burial_{int(r)}A"] = round(float(np.mean([per[i][1][r] for i in idx])), 1)
        row["carbonylO_burial_8A"] = round(float(np.mean([per[i][2] for i in idx])), 1)
        row["status"] = "scored"
    else:
        for r in SHELLS:
            row[f"burial_{int(r)}A"] = None
        row["carbonylO_burial_8A"] = None
        row["status"] = "INCONCLUSIVE"
    return row


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12)), len(x)


def auroc(scores, labels):
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s); s, l = s[m], l[m]
    pos = s[l == 1]; neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["id", "n", "n_docked", "median_dP"] + [f"burial_{int(r)}A" for r in SHELLS] + \
             ["carbonylO_burial_8A", "status", "activity_N66L1"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    print(f"=== reactive-center BURIAL / water-exclusion vs N66 yield (step-1 cofolds) ===")
    print(f"{len(rows)} enzymes | {sum(r['status']=='scored' for r in rows)} scored (docked>= {N_MIN}) | n with activity+scored = {len(m)}")
    y = [r["activity_N66L1"] for r in m]
    labels = [1 if v >= 0.5 else 0 for v in y]
    print(f"actives (>=0.5): {sum(labels)}\n")
    print(f"{'proxy':22s} {'Spearman':>10} {'AUROC':>8}")
    for key in [f"burial_{int(r)}A" for r in SHELLS] + ["carbonylO_burial_8A"]:
        x = [r[key] for r in m]
        rho, n = spearman(x, y)
        au = auroc(x, labels)
        print(f"{key:22s} {rho:>+10.3f} {au:>8.3f}   (higher burial -> higher yield?)")
    print("\nactive enzymes' burial_8A standing:")
    ranked = sorted([r for r in m], key=lambda r: -(r["burial_8A"] or 0))
    for r in sorted([r for r in m if r["activity_N66L1"] >= 0.5], key=lambda r: -r["activity_N66L1"]):
        rk = [x["id"] for x in ranked].index(r["id"]) + 1
        print(f"  {r['id']:6s} act={r['activity_N66L1']:.2f}  burial_8A={r['burial_8A']}  "
              f"dockfrac={r['n_docked']}/{r['n']}  rank {rk}/{len(ranked)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
