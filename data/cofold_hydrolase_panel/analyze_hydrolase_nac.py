#!/usr/bin/env python3
"""Hydrolysis attack-geometry (NAC) for the nylon-hydrolase panel.

Ntn-hydrolase mechanism: the beta-chain N-terminal catalytic Thr Ogamma attacks a
scissile amide carbonyl of the nylon-6,6 tetrad substrate.

  nucleophile  = chain B residue 1 (catalytic Thr) atom OG1
  electrophile = nearest SCISSILE amide carbonyl C of the tetrad (chain C):
                 a C bonded to a carbonyl O (=O, <1.6 A) AND an amide N (<1.6 A)
                 -> distinguishes amide C(=O)N from the terminal carboxylate C(OO)
  d     = |OG1 - C|            (attack distance)
  angle = OG1 - C = O          (Burgi-Dunitz, ideal ~107 deg for carbonyl attack)
  NAC   = d < 3.5 A AND 90 <= angle <= 130

Aggregates over the 50 models per enzyme, merges with L1/L2 product release, and
tests whether the geometry predicts activity.
"""
import csv, glob, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/nylon_hydrolase_results/output_configs"
ACT = f"{ROOT}/exp_data/amide_hydrolase_biorep_activity.csv"
OUT = f"{ROOT}/exp_data/nylon_hydrolase_results/hydrolase_nac.csv"
D_CUT, A_LO, A_HI = 3.5, 90.0, 130.0


def parse(path):
    og1 = None; C = []  # C = tetrad atoms (name,elem,xyz)
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip(); elem = (ln[76:78].strip() or an[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ln.startswith("ATOM") and ch == "B" and int(ln[22:26]) == 1 and an == "OG1":
            og1 = xyz
        elif ch == "C":
            C.append((elem, xyz))
    return og1, C


def scissile(og1, C):
    """Return (min attack distance, BD angle at that carbonyl) over amide carbonyls."""
    Os = [x for (e, x) in C if e == "O"]; Ns = [x for (e, x) in C if e == "N"]
    Cs = [x for (e, x) in C if e == "C"]
    best = None
    for cx in Cs:
        o_nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.6]
        n_nb = [nx for nx in Ns if np.linalg.norm(cx - nx) < 1.6]
        if len(o_nb) == 1 and len(n_nb) >= 1:          # amide carbonyl (not carboxylate)
            d = float(np.linalg.norm(og1 - cx))
            v1 = og1 - cx; v2 = o_nb[0] - cx
            ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
            if best is None or d < best[0]:
                best = (d, ang)
    return best


def process(cfgdir):
    name = os.path.basename(cfgdir)
    pred = f"{cfgdir}/boltz_results_{name}/predictions/{name}"
    ds, angs = [], []
    for pdb in sorted(glob.glob(f"{pred}/{name}_model_*.pdb")):
        og1, C = parse(pdb)
        if og1 is None or not C:
            continue
        r = scissile(og1, C)
        if r:
            ds.append(r[0]); angs.append(r[1])
    if not ds:
        return None
    ds = np.array(ds); angs = np.array(angs)
    npass = int(np.sum((ds < D_CUT) & (angs >= A_LO) & (angs <= A_HI)))
    return {"id": name, "n": len(ds), "nac_pass": npass, "nac_frac": round(npass/len(ds), 3),
            "min_d": round(float(ds.min()), 2), "median_d": round(float(np.median(ds)), 2),
            "bd_at_min_d": round(float(angs[int(ds.argmin())]), 1)}


def actkey(name):
    return "Nyl" + str(int(name[3:])) if name.startswith("Nyl") and name[3:].isdigit() else name


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx@ry)/(np.sqrt((rx@rx)*(ry@ry))+1e-12))


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        if r["type"] != "Powder" or r["enzyme"] in act:
            continue
        def f(v):
            try: return float(v)
            except: return None
        act[r["enzyme"]] = {"L1": f(r["L1_ugml"]), "L2": f(r["L2_ugml"]),
                            "L1sig": r["L1_significant"], "L2sig": r["L2_significant"]}

    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        a = act.get(actkey(r["id"]), {})
        r["L1"] = a.get("L1"); r["L2"] = a.get("L2")
        r["total"] = (r["L1"] or 0) + (r["L2"] or 0) if a else None
        r["L1sig"] = a.get("L1sig"); r["L2sig"] = a.get("L2sig")

    rows.sort(key=lambda r: (-r["nac_frac"], r["min_d"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    m = [r for r in rows if r["total"] is not None]
    print(f"Analyzed {len(rows)} hydrolases ({sum(r['n'] for r in rows)} models); {len(m)} with activity.")
    print(f"NAC: d<{D_CUT}A & BD in [{A_LO},{A_HI}]  (Thr-Ogamma -> scissile amide carbonyl)\n")
    for lab, key in [("L1 (diad)", "L1"), ("L2 (dimer)", "L2"), ("L1+L2", "total")]:
        y = np.array([r[key] or 0 for r in m])
        print(f"  Spearman vs {lab:12s}:  NAC_frac {spearman(np.array([r['nac_frac'] for r in m]), y):+.3f}"
              f"   -min_d {spearman(np.array([-r['min_d'] for r in m]), y):+.3f}")
    # active (either L1 or L2 significant) vs inactive
    def sig(r): return r["L1sig"] == "1" or r["L2sig"] == "1"
    A = [r["nac_frac"] for r in m if sig(r)]; I = [r["nac_frac"] for r in m if not sig(r)]
    print(f"\nactive (L1 or L2 significant) n={len(A)}: NAC_frac mean {np.mean(A):.3f}")
    print(f"inactive n={len(I)}: NAC_frac mean {np.mean(I):.3f}")
    print(f"\nTop 10 by NAC_frac:")
    print(f"  {'id':7s} {'nac_frac':>8s} {'min_d':>6s} {'bd':>6s} {'L1':>7s} {'L2':>7s} {'sig':>4s}")
    for r in rows[:10]:
        s = "Y" if (r["L1sig"]=="1" or r["L2sig"]=="1") else "n"
        print(f"  {r['id']:7s} {r['nac_frac']:>8.2f} {r['min_d']:>6.2f} {r['bd_at_min_d']:>6.1f} "
              f"{(r['L1'] or 0):>7.3f} {(r['L2'] or 0):>7.3f} {s:>4s}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
