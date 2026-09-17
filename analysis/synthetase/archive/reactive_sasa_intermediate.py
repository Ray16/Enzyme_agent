#!/usr/bin/env python3
"""Reactive-center burial on the ACTIVATED INTERMEDIATE (step-2 adipyl-AMP), the
mechanistically-correct version of the step-1 SASA proxy.

Hypothesis (hydrolysis-vs-aminolysis partitioning): the electrophile that partitions
between the amine and water is the acyl-adenylate CARBONYL CARBON, not the resting
adipate. A buried / water-excluded activated carbonyl suppresses hydrolysis -> amine
wins -> higher yield. So burial should be measured on the acyl-AMP carbonyl in the
step-2 structures, not on pre-activation adipate (which is what the step-1 SASA
exploration used; that gave AUROC 0.97 but on the wrong species).

Chains (step2 unrestrained): A=protein  B=adipyl-AMP  C=free HMDA  D=Mg.
Reactive center = the anhydride ACYL carbon (carbonyl C bonded through the bridging O
to phosphorus; = the carbonyl C nearest P) + its carbonyl O.

SASA context = protein + adipyl-AMP(self) + Mg, with the free HMDA REMOVED, so we
measure ENZYME-POCKET water-exclusion, not occlusion by the mobile amine reactant.
(A with-amine variant is also reported for contrast.)

Gate = intermediate BOUND: acyl carbon within GATE_BOUND A of a protein heavy atom;
>= N_MIN such poses or the enzyme is INCONCLUSIVE. Score = mean SASA over bound poses,
LOWER = more buried. Burial signal reported as Spearman(-SASA, activity) and AUROC
(actives > ACT_THR ug/ml). Distal free-carboxyl SASA is a negative control (should
NOT track activity if the story is specific to the reactive acyl carbon).
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step2/unrestrained/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/reactive_sasa_intermediate.csv"
PREFIX = "step2u_"
GATE_BOUND = 5.0    # acyl carbon within this of protein -> intermediate bound in pocket
N_MIN = 5
ACT_THR = 0.2
_P = PDBParser(QUIET=True)
_SR = ShrakeRupley()


def _d(a, b):
    return float(np.linalg.norm(a.coord - b.coord))


def find_reactive(B):
    """(acyl_C, carbonyl_O, distal_carboxyl_C) or None, from adipyl-AMP chain B."""
    Ps = [a for a in B if a.element == "P"]
    if not Ps:
        return None
    P = Ps[0]
    o_on_p = [o for o in B if o.element == "O" and _d(o, P) < 1.75]
    acyl_c = acyl_o = None
    for o in o_on_p:                                   # bridging ester O -> acyl C
        for c in [c for c in B if c.element == "C" and _d(c, o) < 1.6]:
            dbl = [oo for oo in B if oo.element == "O" and oo is not o and _d(c, oo) < 1.36]
            if dbl:
                acyl_c, acyl_o = c, dbl[0]
    if acyl_c is None:
        return None
    # distal free carboxyl C = the OTHER carbonyl-type C (2 O neighbours), farthest from P
    carbox = [c for c in B if c.element == "C" and c is not acyl_c
              and len([o for o in B if o.element == "O" and _d(c, o) < 1.36]) >= 2]
    distal = max(carbox, key=lambda c: _d(c, P)) if carbox else None
    return acyl_c, acyl_o, distal


def pose(pdb):
    m = _P.get_structure("x", pdb)[0]
    if "A" not in m or "B" not in m:
        return None
    B = list(m["B"].get_atoms())
    r = find_reactive(B)
    if not r:
        return None
    acyl_c, acyl_o, distal = r
    prot = list(m["A"].get_atoms())
    dprot = min(_d(acyl_c, a) for a in prot)
    out = {"dprot": dprot}
    # (i) enzyme-pocket context: remove free amine (chain C), keep protein+adipyl-AMP+Mg
    if "C" in m:
        m.detach_child("C")
    _SR.compute(m, level="A")
    out["acylC"] = float(acyl_c.sasa)
    out["carbonylO"] = float(acyl_o.sasa)
    out["center"] = float(acyl_c.sasa + acyl_o.sasa)
    out["distalC"] = float(distal.sasa) if distal is not None else np.nan
    return out


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len(PREFIX):] if base.startswith(PREFIX) else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        try:
            r = pose(pdb)
        except Exception:
            r = None
        if r:
            per.append(r)
    if not per:
        return None
    dp = np.array([p["dprot"] for p in per])
    bound = dp < GATE_BOUND
    row = {"id": name, "n": len(per), "n_bound": int(bound.sum()),
           "median_dprot": round(float(np.median(dp)), 2)}
    keys = ("acylC", "carbonylO", "center", "distalC")
    if bound.sum() >= N_MIN:
        idx = np.where(bound)[0]
        for k in keys:
            row[k + "_sasa"] = round(float(np.nanmean([per[i][k] for i in idx])), 2)
        row["status"] = "scored"
    else:
        for k in keys:
            row[k + "_sasa"] = None
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
    cfgdirs = sorted(glob.glob(f"{RES}/{PREFIX}*"))
    with Pool(min(len(cfgdirs), 24)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["id", "n", "n_bound", "median_dprot", "acylC_sasa", "carbonylO_sasa",
              "center_sasa", "distalC_sasa", "status", "activity_N66L1"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    y = [r["activity_N66L1"] for r in m]
    labels = [1 if v > ACT_THR else 0 for v in y]
    print("=== Reactive-center burial on the ACTIVATED INTERMEDIATE (acyl-AMP, step-2) ===")
    print(f"{len(rows)} enzymes | {sum(r['status']=='scored' for r in rows)} scored | "
          f"activity+scored n={len(m)} | actives={sum(labels)} "
          f"({[r['id'] for r,l in zip(m,labels) if l]})\n")
    print(f"{'metric (lower SASA=buried)':22s} {'Spear(-SASA)':>13} {'AUROC':>8} {'perm_p':>8}")
    for key in ("acylC_sasa", "carbonylO_sasa", "center_sasa", "distalC_sasa"):
        x = [r[key] for r in m]
        note = "  <- negative control" if key == "distalC_sasa" else ""
        print(f"{key:22s} {spearman([-v for v in x], y):>+13.3f} "
              f"{auroc([-v for v in x], labels):>8.3f} {perm_p([-v for v in x], labels):>8.4f}{note}")
    ranked = sorted(m, key=lambda r: r["center_sasa"])   # most buried first
    print("\nactive enzymes (rank by center_sasa, 1=most buried):")
    for r in sorted([r for r in m if r["activity_N66L1"] > ACT_THR], key=lambda r: -r["activity_N66L1"]):
        print(f"  {r['id']:6s} act={r['activity_N66L1']:>5.2f}  center_sasa={r['center_sasa']:>6}  "
              f"rank {ranked.index(r)+1}/{len(ranked)}")
    allc = sorted(r["center_sasa"] for r in m)
    print(f"\ncenter_sasa dist: min {allc[0]:.1f}  median {allc[len(allc)//2]:.1f}  max {allc[-1]:.1f} A^2")
    inc = [r["id"] for r in rows if r["status"] != "scored"]
    print(f"INCONCLUSIVE (intermediate not bound, <{N_MIN} poses within {GATE_BOUND}A): {inc}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
