#!/usr/bin/env python3
"""Adenylation-step NAC analysis for the SfaB diversity panel (step-1 cofolds).

Replicates the DesD NAC definition (select_diad_nac_50models.py):
  nucleophile  = adipate (LIG chain E) carboxylate O nearest ATP PA
  electrophile = ATP alpha-phosphate PA (LIG chain B)
  leaving axis = O3A
  d = |O_nuc - PA| ;  angle = O_nuc - PA - O3A  (in-line, ideal 180)
  NAC pass = d < 4.5 A AND angle > 140 deg
Carboxylate O found geometrically (C with >=2 O within 1.6 A), so naming-independent.

Aggregates over the 50 diffusion samples per enzyme, writes a per-enzyme table,
and correlates NAC propensity with measured N66 L1 activity.
"""
import csv, glob, json, os
import numpy as np

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/sfab_step1_results/output_configs"
ACT = f"{ROOT}/exp_data/SfaB_diversity_panel.csv"
OUT = f"{ROOT}/exp_data/sfab_step1_results/nac_step1"
# In-line nucleophilic attack: ideal angle 180 deg. Primary cutoff 150 deg
# (within 30 deg of ideal); 140 kept as a looser sensitivity comparison.
D_CUT = 4.5
A_CUT = 150.0          # primary (was 140 in the DesD script)
A_CUT_LOOSE = 140.0


def parse_pdb(path):
    atoms = {}
    with open(path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip(); resname = line[17:20].strip()
            chain = line[21].strip(); elem = (line[76:78].strip() or name[0]).upper()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            atoms.setdefault((chain, resname), []).append((name, elem, xyz))
    return atoms


def angle_deg(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def carboxylate_oxygens(lst):
    C = [(n, x) for (n, e, x) in lst if e == "C"]
    O = [(n, x) for (n, e, x) in lst if e == "O"]
    out = []
    for cn, cx in C:
        nb = [(on, ox) for (on, ox) in O if np.linalg.norm(cx - ox) < 1.6]
        if len(nb) >= 2:
            out.extend(nb)
    return out


def nac_model(pdb):
    atoms = parse_pdb(pdb)
    pa = o3a = None
    for (ch, rn), lst in atoms.items():
        nm = {n: x for (n, e, x) in lst}
        if "PA" in nm and "O3A" in nm:
            pa, o3a = nm["PA"], nm["O3A"]; break
    sub = None
    for (ch, rn), lst in atoms.items():
        if any(n == "PA" for (n, e, x) in lst):
            continue
        if carboxylate_oxygens(lst):
            sub = lst; break
    if pa is None or sub is None:
        return None
    cb = carboxylate_oxygens(sub)
    on, ox = min(cb, key=lambda t: np.linalg.norm(t[1] - pa))
    d = float(np.linalg.norm(ox - pa)); a = angle_deg(ox, pa, o3a)
    return d, a


def main():
    # load activity
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None

    rows = []
    for cfgdir in sorted(glob.glob(f"{RES}/step1_*")):
        name = os.path.basename(cfgdir)[len("step1_"):]
        pred = f"{cfgdir}/boltz_results_step1_{name}/predictions/step1_{name}"
        ds, ang = [], []
        for pdb in sorted(glob.glob(f"{pred}/step1_{name}_model_*.pdb")):
            r = nac_model(pdb)
            if r is None:
                continue
            d, a = r; ds.append(d); ang.append(a)
        if not ds:
            continue
        ds = np.array(ds); ang = np.array(ang)
        p150 = int(np.sum((ds < D_CUT) & (ang > A_CUT)))
        p140 = int(np.sum((ds < D_CUT) & (ang > A_CUT_LOOSE)))
        rows.append({
            "id": name, "n_models": len(ds),
            "nac_pass_150": p150, "nac_frac_150": round(p150 / len(ds), 3),
            "nac_frac_140": round(p140 / len(ds), 3),
            "min_d": round(float(ds.min()), 2),
            "median_d": round(float(np.median(ds)), 2),
            "best_angle_at_min_d": round(float(ang[int(ds.argmin())]), 1),
            "max_angle": round(float(ang.max()), 1),
            "activity_N66L1": act.get(name),
        })

    rows.sort(key=lambda r: (-r["nac_frac_150"], r["min_d"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(f"{OUT}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        return float((rx @ ry) / (np.sqrt((rx@rx) * (ry@ry)) + 1e-12))
    m = [r for r in rows if r["activity_N66L1"] is not None]
    y = np.array([r["activity_N66L1"] for r in m])
    rho_150 = spearman(np.array([r["nac_frac_150"] for r in m]), y)
    rho_140 = spearman(np.array([r["nac_frac_140"] for r in m]), y)
    rho_d = spearman(np.array([-r["min_d"] for r in m]), y)

    print(f"Analyzed {len(rows)} enzymes ({sum(r['n_models'] for r in rows)} models).")
    print(f"NAC pass: d<{D_CUT}A AND angle>{A_CUT}deg (primary); 140deg shown for comparison\n")
    print(f"{'id':6s} {'frac150':>7s} {'frac140':>7s} {'min_d':>6s} {'ang@min':>7s} {'maxang':>6s} {'act_N66L1':>10s}")
    for r in rows[:15]:
        a = "" if r["activity_N66L1"] is None else f"{r['activity_N66L1']:.3f}"
        print(f"{r['id']:6s} {r['nac_frac_150']:>7.2f} {r['nac_frac_140']:>7.2f} {r['min_d']:>6.2f} "
              f"{r['best_angle_at_min_d']:>7.1f} {r['max_angle']:>6.1f} {a:>10s}")
    print(f"\nSpearman rho vs N66 L1 activity (n={len(m)}):")
    print(f"  NAC frac @150 : {rho_150:.3f}")
    print(f"  NAC frac @140 : {rho_140:.3f}")
    print(f"  -min_d        : {rho_d:.3f}")
    print(f"\nWrote {OUT}.csv")


if __name__ == "__main__":
    main()
