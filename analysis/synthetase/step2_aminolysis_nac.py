#!/usr/bin/env python3
"""STEP-2 (aminolysis) near-attack: does HMDA's neutral amine attack the acyl-adenylate
carbonyl, and does that surface the active enzymes (esp. AS9) that step-1 geometry misses?

Reaction step 2: adipyl-AMP + H2N-R  ->  amide + AMP. The neutral amine N adds to the
ELECTROPHILIC acyl carbonyl C (the carbonyl esterified to the AMP phosphate). Near-attack
= N...C distance < 3.5 A AND Burgi-Dunitz angle N-C=O in ~[90,120] deg (ideal ~107).

Per structure (chain-agnostic):
  acyl carbonyl C = the ligand C with a =O and an ester O that is itself bonded to P
                    (distinguishes it from adipate's distal free carboxylate).
  attacking N     = the amine N (a separate small ligand chain) nearest that carbonyl C.
Score per enzyme over 50 poses: frac_nac (d<3.5 & BD in [90,120]), n_gated (d<3.5),
mean_bd among gated, min_d. Evaluated vs activity (top3 + Spearman).
"""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np

def parse(pdb):
    het = []  # (chain, name, elem, xyz)
    for ln in open(pdb):
        if not ln.startswith("HETATM"):
            continue
        elem = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        het.append((ln[21], ln[12:16].strip(), elem, xyz))
    return het

def features(pdb):
    het = parse(pdb)
    if not het:
        return None
    P = [a[3] for a in het if a[2] == "P"]
    Cs = [a for a in het if a[2] == "C"]
    Os = [a for a in het if a[2] == "O"]
    Ns = [a for a in het if a[2] == "N"]
    if not P or not Cs or not Ns:
        return None
    # acyl carbonyl C: has a carbonyl O (~1.2A) and an ester O (~1.35A) that is bonded to a P
    acyl = None
    for c in Cs:
        oxy = [o for o in Os if np.linalg.norm(c[3] - o[3]) < 1.6]
        if len(oxy) < 2:
            continue
        ester = [o for o in oxy if any(np.linalg.norm(o[3] - p) < 1.75 for p in P)]
        carbonyl = [o for o in oxy if o not in ester]
        if ester and carbonyl:
            acyl = (c, carbonyl[0]); break
    if acyl is None:
        return None
    cC, cO = acyl
    # attacking amine N = amine N nearest the acyl carbonyl (exclude adenine ring N: those
    # are in the same chain as P; the HMDA N's are in a different, P-free, small chain)
    pchains = {a[0] for a in het if a[2] == "P"}
    amineN = [n for n in Ns if n[0] not in pchains]
    if not amineN:
        amineN = Ns
    n = min(amineN, key=lambda x: np.linalg.norm(x[3] - cC[3]))
    d = float(np.linalg.norm(n[3] - cC[3]))
    v1 = n[3] - cC[3]; v2 = cO[3] - cC[3]
    bd = float(np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))))
    return d, bd

def enzyme(args):
    e, root = args
    pdbs = sorted(glob.glob(f"{root}/output_configs/step2_{e}/boltz_results_step2_{e}/predictions/step2_{e}/*model_*.pdb"))
    per = [f for p in pdbs if (f := features(p))]
    if not per:
        return None
    d = np.array([x[0] for x in per]); bd = np.array([x[1] for x in per])
    gate = d < 3.5
    return dict(id=e, n=len(per), n_gated=int(gate.sum()), min_d=round(float(d.min()), 2),
                frac_nac=round(float(np.mean((d < 3.5) & (bd > 90) & (bd < 120))), 3),
                mean_bd_gated=round(float(bd[gate].mean()), 1) if gate.any() else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); a = ap.parse_args()
    import csv
    ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
    act = {r["id"]: float(r["activity_N66L1_ugml_avg"]) for r in
           csv.DictReader(open(f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"))
           if r["activity_N66L1_ugml_avg"]}
    enz = [os.path.basename(d)[len("step2_"):] for d in glob.glob(f"{a.root}/output_configs/step2_*")]
    with Pool(12) as pool:
        rows = [r for r in pool.map(enzyme, [(e, a.root) for e in enz]) if r]
    out = f"{ROOT}/exp_data/amide_synthetase/analysis/step2_aminolysis_nac.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} enzymes)")
    # evaluate: does frac_nac / n_gated surface AS9 & the actives?
    from scipy.stats import spearmanr
    cands = [r for r in rows if r["id"] != "SfaB" and r["id"] in act]
    sfab = act["SfaB"]; top3 = set(sorted((r["id"] for r in cands), key=lambda e: -act[e])[:3])
    print(f"actives (>SfaB): {sorted((r['id'] for r in cands if act[r['id']]>sfab), key=lambda e:-act[e])}")
    for col in ("frac_nac", "n_gated"):
        order = sorted(cands, key=lambda r: -r[col])
        rho, p = spearmanr([r[col] for r in cands], [act[r["id"]] for r in cands])
        top = [f"{r['id']}{'*' if r['id'] in top3 else ''}" for r in order[:8]]
        print(f"  {col:10} rho={rho:+.3f} p={p:.3f}  top8: {' '.join(top)}")
    a9 = next((r for r in cands if r["id"] == "AS9"), None)
    if a9:
        print(f"  AS9: n_gated={a9['n_gated']} frac_nac={a9['frac_nac']} min_d={a9['min_d']} "
              f"rank_by_frac_nac={sorted(cands,key=lambda r:-r['frac_nac']).index(a9)+1}/{len(cands)}")

if __name__ == "__main__":
    main()
