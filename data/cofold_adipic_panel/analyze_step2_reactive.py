#!/usr/bin/env python3
"""Accommodation analysis for the REACTIVE step-2 run (amine biased to carbonyl).

The attack distance is now biased by the contact constraint, so it is NOT the
readout. We measure how well each active site ACCOMMODATES the imposed near-attack:
  - accommodation fraction = frac of 50 models where the constraint is actually
    satisfied (N18-C35 < 3.5 A)   [enzymes that resist stay long]
  - Burgi-Dunitz angle achieved (N18 - C35 = O), median over models
  - pose confidence: complex_plddt, ligand_iptm, ptm(HMDA=chain2)
Then score each vs N66 activity with the standard evaluation block.

Atom names are deterministic per SMILES: nucleophile = C/N18, acyl carbonyl = B/C35.
"""
import csv, glob, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from proxy_eval import evaluate, fmt

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/sfab_step2_reactive_results/output_configs"
ACT = f"{ROOT}/exp_data/SfaB_diversity_panel.csv"
OUT = f"{ROOT}/exp_data/sfab_step2_reactive_results/step2_reactive_accommodation.csv"
NUC, ACYLC = "N18", "C35"


def geom(pdb):
    N = C = None; Bpos = []
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip(); el = (ln[76:78].strip() or an[0]).upper()
        x = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ch == "C" and an == NUC:
            N = x
        elif ch == "B":
            Bpos.append((an, el, x))
            if an == ACYLC:
                C = x
    if N is None or C is None:
        return None
    Ps = [x for a, e, x in Bpos if e == "P"]
    # carbonyl O = O neighbour of C35 NOT bonded to P
    carO = None
    for a, e, x in Bpos:
        if e == "O" and np.linalg.norm(x - C) < 1.6 and not any(np.linalg.norm(x - p) < 1.8 for p in Ps):
            carO = x; break
    d = float(np.linalg.norm(N - C))
    ang = None
    if carO is not None:
        v1, v2 = N - C, carO - C
        ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))))
    return d, ang


def process(cd):
    base = os.path.basename(cd); name = base[len("step2r_"):] if base.startswith("step2r_") else base
    pred = f"{cd}/boltz_results_{base}/predictions/{base}"
    ds, angs, pl, li, pc = [], [], [], [], []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        g = geom(pdb)
        if g:
            ds.append(g[0])
            if g[1] is not None:
                angs.append(g[1])
    for j in sorted(glob.glob(f"{pred}/confidence_{base}_model_*.json")):
        d = json.load(open(j)); pl.append(d.get("complex_plddt", 0)); li.append(d.get("ligand_iptm", 0))
        pc.append(float(d.get("chains_ptm", {}).get("2", 0)))
    if not ds:
        return None
    ds = np.array(ds); angs = np.array(angs) if angs else np.array([np.nan])
    return {"id": name, "n": len(ds),
            "accom_frac": round(float((ds < 3.5).mean()), 3),   # constraint satisfied
            "min_d": round(float(ds.min()), 2), "median_d": round(float(np.median(ds)), 2),
            "median_bd": round(float(np.nanmedian(angs)), 1),
            "plddt": round(float(np.mean(pl)), 3), "ligand_iptm": round(float(np.mean(li)), 3),
            "ptm_hmda": round(float(np.mean(pc)), 3)}


def main():
    act = {r["id"]: float(r["activity_N66L1_ugml_avg"]) if r["activity_N66L1_ugml_avg"] not in ("", None) else None
           for r in csv.DictReader(open(ACT))}
    with Pool(32) as p:
        rows = [r for r in p.map(process, sorted(glob.glob(f"{RES}/*"))) if r]
    for r in rows:
        r["activity"] = act.get(r["id"])
    rows.sort(key=lambda r: -r["accom_frac"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["activity"] is not None]
    lab = [1 if r["activity"] >= 0.5 else 0 for r in m]
    print(f"REACTIVE step-2 accommodation ({len(rows)} enzymes, {sum(r['n'] for r in rows)} models)\n")
    for key, hb in [("accom_frac", True), ("median_bd", True), ("min_d", False),
                    ("plddt", True), ("ligand_iptm", True), ("ptm_hmda", True)]:
        print("  " + fmt(evaluate([r[key] for r in m], lab, f"reactive {key}", hb)))
    print("\nactive enzymes:")
    for r in sorted([x for x in m if x["activity"] >= 0.5], key=lambda r: -r["activity"]):
        rk = [x["id"] for x in rows].index(r["id"]) + 1
        print(f"  {r['id']:6s} act={r['activity']:.2f} accom_frac={r['accom_frac']} median_bd={r['median_bd']} plddt={r['plddt']} rank {rk}/{len(rows)}")
    print(f"\nhighest accom_frac (was high-NAC inactives poorly accommodating?):")
    for r in rows[:8]:
        a = "" if r["activity"] is None else f"{r['activity']:.3f}"
        print(f"  {r['id']:6s} accom_frac={r['accom_frac']} median_bd={r['median_bd']} act={a}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
