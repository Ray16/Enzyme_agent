#!/usr/bin/env python3
"""UNRESTRAINED step-2 aminolysis geometry, scored with the VALIDATED step-1 recipe
applied to step 2: no restraints -> gate to poses where the amine actually reached
the carbonyl (docked) -> mean Burgi-Dunitz attack angle among docked poses.

Reaction: HMDA free amine attacks the acyl-AMP carbonyl C (-> amide + AMP).
  nucleophile  = HMDA neutral amine N (chain C, first N in SMILES 'NCCCCCC[NH3+]')
  electrophile = acyl-AMP carbonyl C (chain B): C with a =O and an O neighbour bonded to P
  d      = |N - C|                        attack distance
  angle  = N - C = O   (Burgi-Dunitz)     ideal nucleophilic addition ~107 deg

Go/no-go logic (mirrors step-1):
  - DOCKING GATE: pose counts as docked if d < D_GATE (amine reached the carbonyl).
    If < N_MIN docked poses -> INCONCLUSIVE (the free amine never docked -> cofold
    cannot reach the amine half; this is the expected failure mode we're testing).
  - SCORE = mean B-D angle among docked poses; also frac near-ideal (95-125 deg).
  - Report docking-success fraction itself (n_docked/n) as a separate axis.

Then correlate every axis with N66-L1 activity (Spearman + active/inactive split).
Set RESULTS_DIR env to point at the unrestrained output_configs when it lands.
"""
import csv, glob, json, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = os.environ.get("RESULTS_DIR",
    f"{ROOT}/exp_data/amide_synthetase/boltz2/step2/unrestrained/results/output_configs")
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/boltz2/step2/unrestrained/results/step2u_bd_angle.csv"
D_GATE = float(os.environ.get("D_GATE", "4.0"))   # amine reached carbonyl
N_MIN = int(os.environ.get("N_MIN", "5"))         # min docked poses to score (else INCONCLUSIVE)
BD_LO, BD_HI = 95.0, 125.0                          # near-ideal Burgi-Dunitz window


def parse(path):
    B, C = [], []
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; an = ln[12:16].strip(); elem = (ln[76:78].strip() or an[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if ch == "B": B.append((an, elem, xyz))
        elif ch == "C": C.append((an, elem, xyz))
    return B, C


def acyl_carbonyl(B):
    """(carbonyl C, carbonyl O) of the acyl-AMP mixed anhydride."""
    Cs = [(n, x) for (n, e, x) in B if e == "C"]
    Os = [x for (n, e, x) in B if e == "O"]
    Ps = [x for (n, e, x) in B if e == "P"]
    for cn, cx in Cs:
        o_nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.6]
        if len(o_nb) < 2:
            continue
        ester = [ox for ox in o_nb if any(np.linalg.norm(ox - px) < 1.8 for px in Ps)]
        carbonyl = [ox for ox in o_nb if not any(np.linalg.norm(ox - px) < 1.8 for px in Ps)]
        if ester and carbonyl:
            return cx, carbonyl[0]
    return None


def nucleophile_N(C):
    for (n, e, x) in C:
        if e == "N":
            return x                    # first N = neutral -NH2 nucleophile
    return None


def bd_angle(n, c, o):
    """angle N - C = O in degrees."""
    v1 = n - c; v2 = o - c
    cos = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step2u_"):] if base.startswith("step2u_") else \
           base[len("step2_"):] if base.startswith("step2_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    ds, angs = [], []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        B, C = parse(pdb)
        ac = acyl_carbonyl(B)
        if ac is None or not C:
            continue
        cc, co = ac
        nn = nucleophile_N(C)
        if nn is None:
            continue
        d = float(np.linalg.norm(nn - cc))
        ds.append(d)
        angs.append(bd_angle(nn, cc, co))
    if not ds:
        return None
    ds, angs = np.array(ds), np.array(angs)
    gated = ds < D_GATE
    n_dock = int(gated.sum())
    row = {"id": name, "n": len(ds), "n_docked": n_dock,
           "dock_frac": round(n_dock / len(ds), 3),
           "min_d": round(float(ds.min()), 2),
           "median_d": round(float(np.median(ds)), 2)}
    if n_dock >= N_MIN:
        g = angs[gated]
        row["mean_bd_angle"] = round(float(g.mean()), 1)
        row["frac_near_ideal"] = round(float(((g >= BD_LO) & (g <= BD_HI)).mean()), 3)
        row["status"] = "scored"
    else:
        row["mean_bd_angle"] = None
        row["frac_near_ideal"] = None
        row["status"] = "INCONCLUSIVE"
    return row


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))


def main():
    if not glob.glob(f"{RES}/*"):
        print(f"No results yet at {RES}"); return
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])
    rows.sort(key=lambda r: (r["status"] != "scored", -(r["mean_bd_angle"] or -999)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["id", "n", "n_docked", "dock_frac", "min_d", "median_d",
              "mean_bd_angle", "frac_near_ideal", "status", "activity_N66L1"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    n_scored = sum(r["status"] == "scored" for r in rows)
    print(f"=== UNRESTRAINED step-2 (D_GATE={D_GATE}A, N_MIN={N_MIN}) ===")
    print(f"{len(rows)} enzymes | {n_scored} scored | {len(rows)-n_scored} INCONCLUSIVE (amine never docked)")
    print(f"docking success (mean dock_frac): {np.mean([r['dock_frac'] for r in rows]):.3f}")

    # go/no-go #1: does the amine dock at all? (dock_frac vs activity, all enzymes)
    m0 = [r for r in rows if r["activity_N66L1"] is not None]
    y0 = np.array([r["activity_N66L1"] for r in m0])
    print(f"\nSpearman dock_frac vs activity (n={len(m0)}): "
          f"{spearman(np.array([r['dock_frac'] for r in m0]), y0):+.3f}")

    # go/no-go #2: among scored, does gated B-D angle track activity?
    m = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    if len(m) >= 5:
        y = np.array([r["activity_N66L1"] for r in m])
        for key, lab in [("mean_bd_angle", "mean B-D angle"), ("frac_near_ideal", "frac near-ideal")]:
            x = np.array([r[key] for r in m])
            print(f"Spearman {lab:16s} vs activity (scored n={len(m)}): {spearman(x, y):+.3f}")
        active = [r for r in m if r["activity_N66L1"] >= 0.5]
        print(f"\nactive (>=0.5) among scored: n={len(active)}")
        for r in sorted(active, key=lambda r: -r["activity_N66L1"]):
            print(f"  {r['id']:6s} act={r['activity_N66L1']:.2f}  bd_angle={r['mean_bd_angle']}  "
                  f"dock_frac={r['dock_frac']}  n_docked={r['n_docked']}")
    else:
        print(f"\nOnly {len(m)} enzymes scored -> too few to correlate. "
              f"If most are INCONCLUSIVE, that IS the result: the free amine does not dock.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
