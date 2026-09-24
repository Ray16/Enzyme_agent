#!/usr/bin/env python3
"""Clean, self-contained pose selection reading ONLY my Polaris fold (results/, 10
models) and my own substrates (substrates/substrates.json). Avoids the externally
modified score_geometry.py / selected_models.json / verified/ tree.

Parse contract (verified explicitly): catalytic Thr = chain B residue 1 atom OG1;
scissile carbonyl C/O = ligand chain I atoms with the Boltz names from substrates.json;
oxyanion = chain A Asn ND2 + chain A Phe/Tyr backbone N. Picks, per system, the best
geometry-competent pose (d1<3.2, aBD 80-120, oxyanion<3.8) by Boltz confidence; falls
back to smallest d1. Writes myrun/selected.json.
"""
import csv, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results"                                   # MY Polaris 10-model fold
SUBS = json.load(open(f"{HERE}/substrates/substrates.json"))
OUT = f"{HERE}/myrun"; os.makedirs(OUT, exist_ok=True)
OXY = {"Nyl10": {"fy": 106, "asn": 180}, "Nyl12": {"fy": 113, "asn": 186},
       "Nyl50": {"fy": 105, "asn": 180}}


def parse(path):
    d = {}
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            key = (ln[21], int(ln[22:26]), ln[12:16].strip())
            d[key] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return d


def ang(a, b, c):
    v1, v2 = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) /
                 (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1))))


def main():
    sel = {}; report = []
    for enz in ["Nyl10", "Nyl12", "Nyl50"]:
        for frame in ["MAXA_precursor", "XAMA_precursor"]:
            system = f"{enz}__{frame}"
            cn, on = SUBS[frame]["scissile_C_name"], SUBS[frame]["scissile_O_name"]
            oxy = OXY[enz]
            preds = glob.glob(f"{RES}/{system}/boltz_results_*/predictions/*")
            if not preds:
                print(f"{system}: NO results"); continue
            pred = preds[0]
            rows = []
            for m in range(10):
                pdb = f"{pred}/{system}_model_{m}.pdb"
                if not os.path.exists(pdb):
                    continue
                a = parse(pdb)
                og1 = a.get(("B", 1, "OG1")); C = a.get(("I", 1, cn)); O = a.get(("I", 1, on))
                if og1 is None or C is None or O is None:
                    continue
                d1 = float(np.linalg.norm(og1 - C)); aBD = ang(og1, C, O)
                fyN = a.get(("A", oxy["fy"], "N")); asnD = a.get(("A", oxy["asn"], "ND2"))
                d2 = float(np.linalg.norm(O - fyN)) if fyN is not None else 99
                d3 = float(np.linalg.norm(O - asnD)) if asnD is not None else 99
                conf_json = f"{pred}/confidence_{system}_model_{m}.json"
                conf = json.load(open(conf_json))["confidence_score"] if os.path.exists(conf_json) else 0
                gate = (d1 < 3.2 and 80 <= aBD <= 120 and d2 < 3.8 and d3 < 3.8)
                rows.append(dict(model=m, d1=d1, aBD=aBD, d2=d2, d3=d3, conf=conf,
                                 gate=gate, pdb=pdb))
            passed = [r for r in rows if r["gate"]]
            top = (max(passed, key=lambda r: r["conf"]) if passed
                   else min(rows, key=lambda r: r["d1"]))
            sel[system] = top["pdb"]
            report.append((system, len(passed), top["model"], round(top["d1"], 2),
                           round(top["aBD"], 1), top["gate"]))
    json.dump(sel, open(f"{OUT}/selected.json", "w"), indent=2)
    print(f"{'system':26}{'nPass/10':>9}{'topModel':>9}{'d1':>7}{'aBD':>7}{'competent':>10}")
    for r in report:
        print(f"{r[0]:26}{r[1]:>9}{r[2]:>9}{r[3]:>7}{r[4]:>7}{str(r[5]):>10}")
    print(f"\nwrote {OUT}/selected.json (poses from MY results/ only)")


if __name__ == "__main__":
    main()
