#!/usr/bin/env python3
"""Aminolysis attack geometry (step-2 NAC) for the SfaB panel.

Reaction: HMDA free amine attacks the acyl-AMP carbonyl carbon (-> amide + AMP).
  nucleophile  = HMDA neutral amine N (chain C)
  electrophile = acyl-AMP carbonyl C (chain B): the C bearing a carbonyl =O whose
                 OTHER oxygen neighbour is bonded to P (the mixed-anhydride linkage)
  d     = |N - C|                 (attack distance)
  angle = N - C = O               (Burgi-Dunitz, ideal ~107 deg)
  NAC   = d < 3.5 A AND 90 <= angle <= 130

Nucleophile choice: HMDA SMILES is 'NCCCCCC[NH3+]' -> the FIRST amine (neutral, N1
in Boltz atom order) is the nucleophile; the [NH3+] end is protonated. We take the
N that is NOT the protonated terminus. If atom names are ambiguous we fall back to
the amine nearest the reactive carbonyl. (Confirm atom naming from a real PDB.)

Aggregates over 50 models/enzyme, merges N66 L1 activity, tests discrimination.
"""
import csv, glob, os, sys
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/sfab_step2_results/output_configs"
ACT = f"{ROOT}/exp_data/SfaB_diversity_panel.csv"
OUT = f"{ROOT}/exp_data/sfab_step2_results/nac_step2.csv"
D_CUT, A_LO, A_HI = 3.5, 90.0, 130.0
NUC_ATOM = os.environ.get("NUC_ATOM", "N1")   # override after confirming PDB naming


def parse(path):
    """Return (B_atoms, C_atoms) as lists of (name, elem, xyz)."""
    B, C = [], []
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip(); elem = (ln[76:78].strip() or an[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ch == "B":
            B.append((an, elem, xyz))
        elif ch == "C":
            C.append((an, elem, xyz))
    return B, C


def acyl_carbonyl(B):
    """Carbonyl C of the acyl-AMP linkage: C with a =O and an O neighbour bonded to P."""
    Cs = [(n, x) for (n, e, x) in B if e == "C"]
    Os = [x for (n, e, x) in B if e == "O"]
    Ps = [x for (n, e, x) in B if e == "P"]
    for cn, cx in Cs:
        o_nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.6]
        if len(o_nb) < 2:
            continue
        # need one O double-bonded (carbonyl, no P near) and one O bonded to P (ester)
        ester = [ox for ox in o_nb if any(np.linalg.norm(ox - px) < 1.8 for px in Ps)]
        carbonyl = [ox for ox in o_nb if ox is not None and not any(np.linalg.norm(ox - px) < 1.8 for px in Ps)]
        if ester and carbonyl:
            return cx, carbonyl[0]      # (carbonyl C, carbonyl O)
    return None


def nucleophile_N(C, carbonyl_c):
    # HMDA SMILES 'NCCCCCC[NH3+]': the FIRST N (in atom order) is the neutral
    # -NH2 nucleophile; the last N is the protonated [NH3+]. Boltz preserves
    # SMILES atom order, so the first N encountered in chain C is the nucleophile.
    for (n, e, x) in C:
        if e == "N":
            return x
    return None


def process(cfgdir):
    name = os.path.basename(cfgdir)[len("step2_"):] if os.path.basename(cfgdir).startswith("step2_") else os.path.basename(cfgdir)
    base = os.path.basename(cfgdir)
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    ds = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        B, C = parse(pdb)
        ac = acyl_carbonyl(B)
        if ac is None or not C:
            continue
        cc, co = ac
        nn = nucleophile_N(C, cc)
        if nn is None:
            continue
        ds.append(float(np.linalg.norm(nn - cc)))     # ATTACK DISTANCE only
    if not ds:
        return None
    ds = np.array(ds)
    return {"id": name, "n": len(ds),
            "min_d": round(float(ds.min()), 2),
            "median_d": round(float(np.median(ds)), 2),
            "mean_d": round(float(ds.mean()), 2),
            "frac_lt4": round(float((ds < 4.0).mean()), 3),
            "frac_lt5": round(float((ds < 5.0).mean()), 3)}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx@ry)/(np.sqrt((rx@rx)*(ry@ry))+1e-12))


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
    rows.sort(key=lambda r: r["min_d"])     # best (shortest) attack distance first
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["activity_N66L1"] is not None]
    y = np.array([r["activity_N66L1"] for r in m])
    print(f"Analyzed {len(rows)} enzymes ({sum(r['n'] for r in rows)} models)")
    print("STEP-2 ATTACK DISTANCE only (HMDA amine -> acyl-AMP carbonyl C), n={}\n".format(len(m)))
    for key, lab in [("min_d","min attack dist"), ("median_d","median dist"),
                     ("mean_d","mean dist"), ("frac_lt4","frac models <4A"), ("frac_lt5","frac models <5A")]:
        # for distances, shorter = better -> correlate with -dist; for fracs, higher=better
        x = np.array([-r[key] if key.endswith("_d") else r[key] for r in m])
        print(f"  Spearman ({lab:16s}) vs activity: {spearman(x, y):+.3f}")
    active = [r for r in m if r["activity_N66L1"] >= 0.5]
    inactive = [r for r in m if r["activity_N66L1"] < 0.5]
    print(f"\nactive (>=0.5) n={len(active)}: min_d mean {np.mean([r['min_d'] for r in active]):.2f}")
    print(f"inactive     n={len(inactive)}: min_d mean {np.mean([r['min_d'] for r in inactive]):.2f}")
    print(f"\nactive enzymes' standing (ranked by min attack distance, 1=closest):")
    for r in sorted(active, key=lambda r: -r["activity_N66L1"]):
        rank = [x["id"] for x in rows].index(r["id"]) + 1
        print(f"  {r['id']:6s} act={r['activity_N66L1']:.2f}  min_d={r['min_d']}  median_d={r['median_d']}  rank {rank}/{len(rows)}")
    print(f"\nClosest 8 by min attack distance:")
    for r in rows[:8]:
        a = "" if r["activity_N66L1"] is None else f"{r['activity_N66L1']:.3f}"
        print(f"  {r['id']:6s} min_d={r['min_d']}  median_d={r['median_d']}  frac<4A={r['frac_lt4']}  act={a}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
