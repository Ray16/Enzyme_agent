#!/usr/bin/env python
"""Fold-agnostic amide-hydrolysis CAPABILITY proxy from Boltz-2 cofolds.

For each enzyme cofolded with N-methylacetamide (CC(=O)NC), over its 50 poses:
  * locate the substrate amide carbonyl C (ligand chain B: a C with a ~1.2 A =O
    neighbor AND an N neighbor) and its carbonyl O.
  * candidate nucleophiles = protein Ser OG, Thr OG1, Cys SG.
  * per pose: min nucleophile->carbonyl-C distance and the Burgi-Dunitz attack
    angle (Nu...C=O, ideal ~105 deg). Near-attack gate: dist < 3.5 A AND
    BD angle in [90,120].
  * oxyanion-hole proxy: # H-bond donors (backbone amide N, Ser/Thr/Tyr OH)
    within 3.5 A of the carbonyl O.
  * flag whether any catalytic metal ion (ZN/MG/MN/FE/NI/CO/CA/CD) sits within
    4 A of the carbonyl C (metallo-amidases use a metal-activated water that the
    residue-nucleophile proxy would otherwise miss).
Aggregate per enzyme: near_attack_fraction, min_nuc_dist, best_BD_angle,
mean_oxyanion_donors, metal_near_frac, and mean complex_plddt.

Usage: python capability_proxy.py            # scores all enzymes, writes CSV
Outputs: exp_data/amide_hydrolase_capability/capability_scores.csv
"""
import os, glob, json, math, csv
from multiprocessing import Pool

BASE = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_hydrolase_capability"
COFOLD = os.path.join(BASE, "cofold")
BENCH = os.path.join(BASE, "benchmark_set.csv")
OUT_CSV = os.path.join(BASE, "capability_scores.csv")

DIST_GATE = 3.5      # A, nucleophile -> carbonyl C
BD_LO, BD_HI = 90.0, 120.0
OXY_CUT = 3.5        # A, donor -> carbonyl O
METAL_CUT = 4.0      # A, metal -> carbonyl C
METALS = {"ZN", "MG", "MN", "FE", "NI", "CO", "CA", "CD", "CU"}
NUC = {("SER", "OG"), ("THR", "OG1"), ("CYS", "SG")}
OH_DONOR = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}


def parse_pdb(path):
    """Return list of atom dicts: chain, resname, resseq, name, elem, xyz."""
    atoms = []
    with open(path) as f:
        for line in f:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            elem = line[76:78].strip().upper()
            if not elem:  # fall back to atom name
                elem = ''.join(c for c in name if c.isalpha())[:2].upper()
            atoms.append({"chain": chain, "resname": resname, "resseq": resseq,
                          "name": name, "elem": elem, "xyz": (x, y, z)})
    return atoms


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def angle(a, b, c):
    """angle at b (a-b-c) in degrees."""
    ba = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    bc = (c[0]-b[0], c[1]-b[1], c[2]-b[2])
    nba = math.sqrt(sum(v*v for v in ba)); nbc = math.sqrt(sum(v*v for v in bc))
    if nba == 0 or nbc == 0:
        return float("nan")
    cosv = sum(ba[i]*bc[i] for i in range(3)) / (nba*nbc)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def find_carbonyl(lig):
    """In ligand atoms, find (C_carbonyl, O_carbonyl). C has an O neighbor at
    ~1.2 A and an N neighbor at ~1.3-1.5 A."""
    cs = [a for a in lig if a["elem"] == "C"]
    os_ = [a for a in lig if a["elem"] == "O"]
    ns = [a for a in lig if a["elem"] == "N"]
    best = None
    for c in cs:
        # nearest O
        o_near = min(os_, key=lambda o: dist(c["xyz"], o["xyz"]), default=None)
        if o_near is None:
            continue
        d_co = dist(c["xyz"], o_near["xyz"])
        n_near = min(ns, key=lambda n: dist(c["xyz"], n["xyz"]), default=None)
        if n_near is None:
            continue
        d_cn = dist(c["xyz"], n_near["xyz"])
        if d_co <= 1.35 and d_cn <= 1.6:
            score = abs(d_co - 1.23) + abs(d_cn - 1.35)
            if best is None or score < best[0]:
                best = (score, c, o_near)
    if best is None:
        return None, None
    return best[1], best[2]


def score_pose(path):
    atoms = parse_pdb(path)
    lig = [a for a in atoms if a["chain"] == "B"]
    prot = [a for a in atoms if a["chain"] == "A"]
    if not lig or not prot:
        return None
    cC, cO = find_carbonyl(lig)
    if cC is None:
        return None
    cCxyz, cOxyz = cC["xyz"], cO["xyz"]
    # nucleophiles
    min_d = float("inf"); bd_at_min = float("nan")
    for a in prot:
        if (a["resname"], a["name"]) in NUC:
            d = dist(a["xyz"], cCxyz)
            if d < min_d:
                min_d = d
                bd_at_min = angle(a["xyz"], cCxyz, cOxyz)
    near_attack = (min_d < DIST_GATE) and (not math.isnan(bd_at_min)) and (BD_LO <= bd_at_min <= BD_HI)
    # oxyanion donors near carbonyl O
    oxy = 0
    for a in prot:
        is_bbN = (a["name"] == "N" and a["elem"] == "N")
        is_oh = (a["resname"], a["name"]) in OH_DONOR
        if is_bbN or is_oh:
            if dist(a["xyz"], cOxyz) <= OXY_CUT:
                oxy += 1
    # metal near carbonyl C (from any chain)
    metal_near = False
    for a in atoms:
        if a["elem"] in METALS and dist(a["xyz"], cCxyz) <= METAL_CUT:
            metal_near = True
            break
    return {"min_d": min_d, "bd": bd_at_min, "near_attack": near_attack,
            "oxy": oxy, "metal": metal_near}


def pred_dir(_id):
    return os.path.join(COFOLD, _id, f"boltz_results_{_id}", "predictions", _id)


def score_enzyme(args):
    _id, label = args
    pdbs = sorted(glob.glob(os.path.join(pred_dir(_id), f"{_id}_model_*.pdb")))
    if not pdbs:
        return {"id": _id, "label": label, "n_poses": 0}
    results = [score_pose(p) for p in pdbs]
    results = [r for r in results if r]
    n = len(results)
    if n == 0:
        return {"id": _id, "label": label, "n_poses": 0}
    naf = sum(1 for r in results if r["near_attack"]) / n
    min_nuc = min(r["min_d"] for r in results)
    # best BD = angle closest to 105 among poses whose nucleophile is within 6 A
    close = [r for r in results if r["min_d"] < 6.0 and not math.isnan(r["bd"])]
    if close:
        best_bd = min(close, key=lambda r: abs(r["bd"] - 105.0))["bd"]
    else:
        best_bd = float("nan")
    mean_oxy = sum(r["oxy"] for r in results) / n
    metal_frac = sum(1 for r in results if r["metal"]) / n
    # complex_plddt
    plddts = []
    for p in pdbs:
        stem = os.path.basename(p).replace(".pdb", "")
        cj = os.path.join(pred_dir(_id), f"confidence_{stem}.json")
        if os.path.exists(cj):
            try:
                plddts.append(json.load(open(cj)).get("complex_plddt", float("nan")))
            except Exception:
                pass
    plddts = [x for x in plddts if isinstance(x, (int, float)) and not math.isnan(x)]
    mean_plddt = sum(plddts) / len(plddts) if plddts else float("nan")
    return {"id": _id, "label": label, "n_poses": n,
            "near_attack_fraction": round(naf, 4),
            "min_nuc_dist": round(min_nuc, 3),
            "best_BD_angle": round(best_bd, 2) if not math.isnan(best_bd) else "",
            "mean_oxyanion_donors": round(mean_oxy, 3),
            "metal_near_frac": round(metal_frac, 4),
            "complex_plddt": round(mean_plddt, 4) if not math.isnan(mean_plddt) else ""}


def main():
    rows = []
    with open(BENCH) as f:
        for r in csv.DictReader(f):
            rows.append((r["id"], r["label"]))
    with Pool(min(20, len(rows))) as pool:
        scored = pool.map(score_enzyme, rows)
    cols = ["id", "label", "n_poses", "near_attack_fraction", "min_nuc_dist",
            "best_BD_angle", "mean_oxyanion_donors", "metal_near_frac", "complex_plddt"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in scored:
            w.writerow({c: s.get(c, "") for c in cols})
    done = sum(1 for s in scored if s.get("n_poses", 0) > 0)
    print(f"scored {done}/{len(scored)} enzymes with poses -> {OUT_CSV}")
    for s in sorted(scored, key=lambda x: x["id"]):
        print(s["id"], s["label"], "poses=", s.get("n_poses"),
              "naf=", s.get("near_attack_fraction"), "mind=", s.get("min_nuc_dist"),
              "plddt=", s.get("complex_plddt"))


if __name__ == "__main__":
    main()
