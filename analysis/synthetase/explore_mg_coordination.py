#!/usr/bin/env python3
"""EXPLORATORY (read-only, NOT pipeline): catalytic Mg2+ coordination geometry of the
reactive carboxylate vs measured yield (SfaB panel step-1 cofolds).

Mechanistic idea: in ANL acyl-adenylate chemistry the catalytic Mg2+ ions coordinate
the ATP phosphates and help align/activate the attacking carboxylate. A pose in which
a Mg2+ properly coordinates (inner-sphere ~2.0-2.2A) the adipate reactive carboxylate
O -- especially BRIDGING an ATP phosphate O and the carboxylate O -- is a productively
aligned Michaelis complex. Hypothesis: better Mg coordination/bridging -> higher yield.

Chain layout A=protein, B=ATP (PA/PB/PG + phosphate O's), C=MG, D=MG, E=adipate.
Reactive carboxyl = adipate carboxyl C nearest an ATP P (Pα=PA is the reactive
phosphate for acyl-adenylate). Docked-pose gate = reactive C within GATE A of an
ATP P; require >=N_MIN docked poses to score. Aggregate over gated poses.

Reuses parsing idiom + spearman/auroc/perm_p from explore_reactive_sasa.py.
Writes exp_data/amide_synthetase/analysis/exploratory_mg_coordination.csv. Only reads
structures; creates one new script + one CSV; no pipeline files touched.
"""
import csv, glob, os, warnings
import numpy as np
from multiprocessing import Pool
warnings.filterwarnings("ignore")

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
IPTM = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/protein_substrate_iptm.csv"
MANIFEST = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_manifest.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/exploratory_mg_coordination.csv"
GATE = 6.0
N_MIN = 5
COORD_CUT = 2.5          # inner-sphere Mg-O coordination cutoff (swept for robustness)
CUT_SWEEP = [2.2, 2.5, 3.0, 3.5]


def parse(path):
    """Return dict of chain -> list of (atomname, elem, xyz)."""
    ch = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        c = ln[21]
        if c not in ch:
            continue
        an = ln[12:16].strip()
        elem = (ln[76:78].strip() or an[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        ch[c].append((an, elem, xyz))
    return ch


def pose_geometry(path):
    """Per-pose Mg-coordination features; returns dict or None."""
    ch = parse(path)
    B, E = ch["B"], ch["E"]
    mgs = [xyz for (an, e, xyz) in (ch["C"] + ch["D"]) if e == "MG"]
    Ps = [xyz for (an, e, xyz) in B if e == "P"]
    Pa = [xyz for (an, e, xyz) in B if an == "PA"]        # reactive phosphate (acyl-adenylate)
    phos_O = [xyz for (an, e, xyz) in B if e == "O"]       # ATP oxygens (mostly phosphate)
    if not E or not Ps or not mgs:
        return None
    # reactive carboxyl C = C in E with >=2 O neighbours, nearest an ATP P
    carbs = []
    for (an, e, x) in E:
        if e != "C":
            continue
        onb = [ox for (a2, e2, ox) in E if e2 == "O" and np.linalg.norm(x - ox) < 1.7]
        if len(onb) >= 2:
            carbs.append((x, onb))
    if not carbs:
        return None
    cx, o_nb = min(carbs, key=lambda co: min(np.linalg.norm(co[0] - p) for p in Ps))
    dP = float(min(np.linalg.norm(cx - p) for p in Ps))

    # Mg -> reactive carboxylate O : nearest over (Mg, carboxylate O)
    mgO = min(np.linalg.norm(mg - o) for mg in mgs for o in o_nb)
    # Mg -> nearest ATP phosphate P and phosphate O (placement sanity)
    mgP = min(np.linalg.norm(mg - p) for mg in mgs for p in Ps)
    mgPa = min((np.linalg.norm(mg - p) for mg in mgs for p in Pa), default=np.nan)
    # bridging at a given cutoff: some single Mg within cut of BOTH a phosphate O and a carboxylate O
    def bridged(cut):
        for mg in mgs:
            near_phos = any(np.linalg.norm(mg - po) < cut for po in phos_O) if phos_O else False
            near_carb = any(np.linalg.norm(mg - o) < cut for o in o_nb)
            if near_phos and near_carb:
                return True
        return False
    return {"dP": dP, "mgO": float(mgO), "mgP": float(mgP), "mgPa": float(mgPa),
            "bridge": {c: bridged(c) for c in CUT_SWEEP}}


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step1_"):] if base.startswith("step1_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        try:
            g = pose_geometry(pdb)
        except Exception:
            g = None
        if g:
            per.append(g)
    if not per:
        return None
    dPs = np.array([p["dP"] for p in per])
    docked = dPs < GATE
    # placement sanity uses ALL poses (does Boltz put Mg near ATP at all?)
    row = {"id": name, "n": len(per), "n_docked": int(docked.sum()),
           "median_dP": round(float(np.median(dPs)), 2),
           "mgP_all_median": round(float(np.median([p["mgP"] for p in per])), 2)}
    if docked.sum() >= N_MIN:
        idx = np.where(docked)[0]
        mgO = [per[i]["mgO"] for i in idx]
        row["min_mgO"] = round(float(np.min(mgO)), 2)
        row["mean_mgO"] = round(float(np.mean(mgO)), 2)
        row["frac_coord"] = round(float(np.mean([per[i]["mgO"] < COORD_CUT for i in idx])), 3)
        for c in CUT_SWEEP:
            row[f"frac_bridge_{c}"] = round(float(np.mean([per[i]["bridge"][c] for i in idx])), 3)
        row["mean_mgPa"] = round(float(np.nanmean([per[i]["mgPa"] for i in idx])), 2)
        row["status"] = "scored"
    else:
        for k in ["min_mgO", "mean_mgO", "frac_coord", "mean_mgPa"] + [f"frac_bridge_{c}" for c in CUT_SWEEP]:
            row[k] = None
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
    if not len(s):
        return float("nan")
    obs = auroc(s, l); rng = np.random.default_rng(seed)
    ge = sum(auroc(s, rng.permutation(l)) >= obs for _ in range(n))
    return (ge + 1) / (n + 1)


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    iptm = {}
    for r in csv.DictReader(open(IPTM)):
        v = r.get("ps_iptm_mean", "")
        if v not in ("", None):
            iptm[r["id"]] = float(v)
    length = {}
    for r in csv.DictReader(open(MANIFEST)):
        try:
            length[r["name"]] = int(r["length_aa"])
        except (KeyError, ValueError, TypeError):
            pass

    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 24)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["id", "n", "n_docked", "median_dP", "mgP_all_median", "min_mgO", "mean_mgO",
              "frac_coord"] + [f"frac_bridge_{c}" for c in CUT_SWEEP] + \
             ["mean_mgPa", "status", "activity_N66L1"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # -------- Mg placement sanity (ALL enzymes) --------
    print("=== Mg2+ COORDINATION geometry vs N66 yield (SfaB panel step-1) ===\n")
    mgP_all = np.array([r["mgP_all_median"] for r in rows], float)
    print(f"[Mg placement sanity] median(per-enzyme median nearest Mg-to-ATP-P distance) = "
          f"{np.median(mgP_all):.2f} A")
    print(f"  enzymes with Mg within 4A of an ATP P (median pose): {(mgP_all < 4).sum()}/{len(rows)}; "
          f"within 6A: {(mgP_all < 6).sum()}/{len(rows)}")
    print("  (Mg-P ~2.8-3.5A = inner-sphere phosphate coordination; if most are far, Boltz mis-placed Mg)\n")

    m = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    y = [r["activity_N66L1"] for r in m]
    labels = [1 if v >= 0.5 else 0 for v in y]
    print(f"{len(rows)} enzymes | {sum(r['status']=='scored' for r in rows)} scored | "
          f"activity+scored n={len(m)} | actives={sum(labels)}\n")

    # distance metrics: SHORTER = better -> correlate with -dist ; fracs: HIGHER = better
    print(f"{'metric':22s} {'signed_dir':>10} {'Spearman':>9} {'AUROC':>7} {'perm_p':>8}")
    def report(key, higher_better):
        x = [r[key] for r in m]
        sc = x if higher_better else [-v if v is not None else None for v in x]
        rho = spearman(sc, y); au = auroc(sc, labels); pp = perm_p(sc, labels)
        arrow = "high=act" if higher_better else "near=act"
        print(f"{key:22s} {arrow:>10} {rho:>+9.3f} {au:>7.3f} {pp:>8.4f}")
    report("min_mgO", False)
    report("mean_mgO", False)
    report("frac_coord", True)
    for c in CUT_SWEEP:
        report(f"frac_bridge_{c}", True)
    report("mean_mgPa", False)

    # -------- robustness of the coordination/bridge cutoff --------
    print("\n[Robustness] bridge fraction across coordination cutoffs (guard vs free-param overfit):")
    print(f"{'cutoff(A)':>9} {'Spearman':>9} {'AUROC':>7} {'perm_p':>8} {'mean_frac':>10}")
    for c in CUT_SWEEP:
        x = [r[f"frac_bridge_{c}"] for r in m]
        rho = spearman(x, y); au = auroc(x, labels); pp = perm_p(x, labels)
        mf = np.mean([v for v in x if v is not None])
        print(f"{c:>9} {rho:>+9.3f} {au:>7.3f} {pp:>8.4f} {mf:>10.3f}")

    # -------- confound checks --------
    print("\n[Confounds] (Spearman of each Mg metric vs ps_iptm and protein length):")
    ipt = [iptm.get(r["id"], np.nan) for r in m]
    ln = [length.get(r["id"], np.nan) for r in m]
    for key in ["min_mgO", "mean_mgO", "frac_coord", f"frac_bridge_{COORD_CUT}"]:
        x = [r[key] for r in m]
        print(f"  {key:20s} vs ps_iptm {spearman(x, ipt):+.3f} | vs length {spearman(x, ln):+.3f}")
    print(f"  (reference: ps_iptm vs activity = {spearman(ipt, y):+.3f})")

    # -------- active-enzyme standings --------
    print("\nactive enzymes' Mg coordination standing:")
    ranked_o = sorted(m, key=lambda r: r["mean_mgO"])          # closest Mg-O first
    for r in sorted([r for r in m if r["activity_N66L1"] >= 0.5], key=lambda r: -r["activity_N66L1"]):
        rk = ranked_o.index(r) + 1
        print(f"  {r['id']:6s} act={r['activity_N66L1']:>5.2f}  mean_mgO={r['mean_mgO']:>5}A  "
              f"frac_coord={r['frac_coord']}  frac_bridge_2.5={r['frac_bridge_2.5']}  "
              f"mean_mgO_rank {rk}/{len(m)} (1=closest)")
    allo = sorted(r["mean_mgO"] for r in m)
    print(f"\nmean_mgO dist: min {allo[0]:.2f}  median {allo[len(allo)//2]:.2f}  max {allo[-1]:.2f} A")
    fb = [r["frac_bridge_2.5"] for r in m]
    print(f"frac_bridge_2.5 dist: {sum(v>0 for v in fb)}/{len(fb)} enzymes have ANY bridging pose; "
          f"mean {np.mean(fb):.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
