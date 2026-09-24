#!/usr/bin/env python3
"""Stage A: active-site geometry filter + top-model selection (paper protocol:
'filtered based on active site geometry and sorted by model confidence').

For each of the 10 Boltz models per system, at the catalytic centre of heterodimer-1:
  nucleophile  = chain B (beta1) residue 1  Thr OG1
  scissile     = ligand chain I  carbonyl C / O   (names per framing, substrates.json)
  oxyanion     = chain A backbone N of Phe/Tyr (d2) + chain A Asn ND2 (d3)
  d1  = |OG1 - C|                          (paper filter < 3.2 A)
  aBD = OG1-C=O  Burgi-Dunitz               (80-120 deg; >90 favoured)
  aFL = Flippin-Lodge lateral (0=symmetric) (paper 70-110 -> |lat| small favoured)
  d2,d3 oxyanion H-bonds                     (< 3.8 A)
Pick the top model = passes geometry gate, then max Boltz confidence_score; if none
pass, fall back to smallest d1. Writes a per-model table + the selected model paths.
"""
import csv, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results"
SUBS = json.load(open(f"{HERE}/substrates/substrates.json"))
# per-enzyme oxyanion residues in chain A (full-length == alpha numbering)
OXY = {"Nyl10": {"fy": 106, "asn": 180}, "Nyl12": {"fy": 113, "asn": 186},
       "Nyl50": {"fy": 105, "asn": 180}}
D1, ABD = 3.2, (80, 120)
DOXY = 3.8


def atoms(path):
    """dict: (chain,resnum,atomname)->xyz ; and ligand atoms by name."""
    d = {}
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; rn = int(ln[22:26]); an = ln[12:16].strip()
        d[(ch, rn, an)] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return d


def ang(a, b, c):
    v1 = a - b; v2 = c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) /
                 (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))


def flippin_lodge(og1, C, O, N, Cacyl):
    z = O - C; z /= np.linalg.norm(z) + 1e-9
    n = np.cross(N - C, Cacyl - C); n /= np.linalg.norm(n) + 1e-9
    y = np.cross(n, z); y /= np.linalg.norm(y) + 1e-9
    t = og1 - C; t /= np.linalg.norm(t) + 1e-9
    return float(np.degrees(np.arcsin(np.clip(np.dot(t, y), -1, 1))))


def lig_neighbors(path, cname, oname):
    """Find, in ligand chain I, the scissile C's amide N and acyl C neighbours."""
    lig = []
    for ln in open(path):
        if ln.startswith("HETATM") and ln[21] == "I":
            an = ln[12:16].strip(); el = (ln[76:78].strip() or an[0]).upper()
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            lig.append((an, el, xyz))
    C = next(x for a, e, x in lig if a == cname)
    Ns = [x for a, e, x in lig if e == "N" and np.linalg.norm(x - C) < 1.6]
    Cs = [x for a, e, x in lig if e == "C" and 0.1 < np.linalg.norm(x - C) < 1.7]
    return (Ns[0] if Ns else None), (Cs[0] if Cs else None)


def process(system):
    enz, frame = system.split("__")
    sub = SUBS[frame]; cn, on = sub["scissile_C_name"], sub["scissile_O_name"]
    oxy = OXY[enz]
    pred = glob.glob(f"{RES}/{system}/boltz_results_*/predictions/*")[0]
    rows = []
    for m in range(10):
        pdb = f"{pred}/{system}_model_{m}.pdb"
        if not os.path.exists(pdb):
            continue
        a = atoms(pdb)
        og1 = a.get(("B", 1, "OG1"))
        C = a.get(("I", 1, cn)); O = a.get(("I", 1, on))
        if og1 is None or C is None or O is None:
            continue
        d1 = float(np.linalg.norm(og1 - C))
        aBD = ang(og1, C, O)
        Nlig, Cacyl = lig_neighbors(pdb, cn, on)
        aFL = flippin_lodge(og1, C, O, Nlig, Cacyl) if (Nlig is not None and Cacyl is not None) else float("nan")
        fyN = a.get(("A", oxy["fy"], "N")); asnD = a.get(("A", oxy["asn"], "ND2"))
        d2 = float(np.linalg.norm(O - fyN)) if fyN is not None else float("nan")
        d3 = float(np.linalg.norm(O - asnD)) if asnD is not None else float("nan")
        conf = json.load(open(f"{pred}/confidence_{system}_model_{m}.json"))
        gate = (d1 < D1 and ABD[0] <= aBD <= ABD[1]
                and (np.isnan(d2) or d2 < DOXY) and (np.isnan(d3) or d3 < DOXY))
        rows.append({"model": m, "d1": round(d1, 2), "aBD": round(aBD, 1),
                     "aFL": round(aFL, 1), "d2": round(d2, 2), "d3": round(d3, 2),
                     "conf": round(conf["confidence_score"], 3),
                     "iptm": round(conf["iptm"], 3),
                     "lig_iptm": round(conf["ligand_iptm"], 3),
                     "plddt": round(conf["complex_plddt"], 3),
                     "pass": int(gate), "pdb": pdb})
    if not rows:
        return None
    passed = [r for r in rows if r["pass"]]
    top = (max(passed, key=lambda r: r["conf"]) if passed
           else min(rows, key=lambda r: r["d1"]))
    return enz, frame, rows, top


def main():
    systems = sorted(os.path.basename(d) for d in glob.glob(f"{RES}/*")
                     if os.path.isdir(d) and "__" in os.path.basename(d))
    sel = {}
    allrows = []
    for s in systems:
        r = process(s)
        if not r:
            print(f"{s}: no scorable models"); continue
        enz, frame, rows, top = r
        npass = sum(x["pass"] for x in rows)
        print(f"\n=== {enz}  {frame}  ({npass}/10 pass geometry gate) ===")
        print(f"{'m':>2} {'d1':>5} {'aBD':>6} {'aFL':>6} {'d2':>5} {'d3':>5} "
              f"{'conf':>6} {'ligIptm':>8} {'pass':>4}")
        for x in sorted(rows, key=lambda r: (-r["pass"], r["d1"])):
            mark = "  <== TOP" if x["model"] == top["model"] else ""
            print(f"{x['model']:>2} {x['d1']:>5} {x['aBD']:>6} {x['aFL']:>6} "
                  f"{x['d2']:>5} {x['d3']:>5} {x['conf']:>6} {x['lig_iptm']:>8} "
                  f"{x['pass']:>4}{mark}")
            xr = dict(x); xr["system"] = s; allrows.append(xr)
        sel[s] = top
    json.dump({k: v["pdb"] for k, v in sel.items()},
              open(f"{HERE}/selected_models.json", "w"), indent=2)
    with open(f"{HERE}/geometry_all_models.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "model", "d1", "aBD", "aFL", "d2", "d3", "conf",
                    "iptm", "lig_iptm", "plddt", "pass"])
        for x in allrows:
            w.writerow([x["system"], x["model"], x["d1"], x["aBD"], x["aFL"],
                        x["d2"], x["d3"], x["conf"], x["iptm"], x["lig_iptm"],
                        x["plddt"], x["pass"]])
    print(f"\nselected top model per system -> selected_models.json")


if __name__ == "__main__":
    main()
