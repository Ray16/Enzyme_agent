#!/usr/bin/env python3
"""EXPLORATORY (read-only, NOT pipeline): active-site PREORGANIZATION / pose
consistency vs measured yield (SfaB panel step-1 cofolds).

Reframe: instead of "can it reach near-attack" (single best pose), ask "how
RELIABLY does it reach near-attack across the 50 diffusion samples". A preorganized
active site holds the substrate poised in every pose; a floppy one only occasionally.

Per enzyme, over the docked poses (reactive carboxyl C within GATE A of an ATP
phosphate), we measure the DISPERSION of the in-line attack geometry:
  - angle_std, angle_iqr  (dispersion of the in-line attack angle)
  - d_std,     d_iqr      (dispersion of the attack distance)
and over ALL 50 poses a CONSISTENCY fraction:
  - consist_frac = frac of poses with d < DCUT and angle > ACUT (near-attack achieved)

In-line geometry follows the pipeline convention (structural_modeling/scripts/
step1_nac.py): all-adenylate panel -> reactive phosphate PA, leaving-group bridge
O3A. Nucleophile = adipate carboxylate O nearest PA. angle = angle(O_nuc, PA, O3A).

Hypothesis: LOW dispersion / HIGH consist_frac = preorganized = higher yield.
Chain layout A=protein B=ATP C/D=MG E=adipate. Read-only: writes one CSV.
"""
import csv, glob, os
import numpy as np
from multiprocessing import Pool

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
ACT = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv"
PSI = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/protein_substrate_iptm.csv"
MAN = f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_manifest.csv"
OUT = f"{ROOT}/exp_data/amide_synthetase/analysis/exploratory_pose_consistency.csv"
GATE = 6.0        # reactive carboxyl C within this of an ATP P = docked
N_MIN = 5         # min docked poses to compute dispersion
DCUT, ACUT = 4.0, 150.0   # near-attack criterion for consist_frac (swept in robustness)


def angle_deg(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def parse(path):
    """Return (ATP name->xyz, adipate list[(elem,xyz)])."""
    chains = {}
    for ln in open(path):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        ch = ln[21]; nm = ln[12:16].strip()
        el = (ln[76:78].strip() or nm[0]).upper()
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        chains.setdefault(ch, []).append((nm, el, xyz))
    atp = {nm: x for (nm, e, x) in chains.get("B", [])}
    adip = [(e, x) for (nm, e, x) in chains.get("E", [])]
    return atp, adip


def carboxyls(adip):
    """List of (C_xyz, [O_xyz,...]) for adipate carboxylate carbons."""
    Cs = [x for (e, x) in adip if e == "C"]
    Os = [x for (e, x) in adip if e == "O"]
    out = []
    for cx in Cs:
        nb = [ox for ox in Os if np.linalg.norm(cx - ox) < 1.7]
        if len(nb) >= 2:
            out.append((cx, nb))
    return out


def pose_geom(path):
    """Return (dP, d_attack, angle) for one model, or None."""
    atp, adip = parse(path)
    if "PA" not in atp or "O3A" not in atp or not adip:
        return None
    PA, O3A = atp["PA"], atp["O3A"]
    Ps = [x for (nm, x) in atp.items() if nm in ("PA", "PB", "PG")]
    carbs = carboxyls(adip)
    if not carbs:
        return None
    # nucleophile = carboxylate O nearest PA; reactive C = its parent carbon
    react = min(carbs, key=lambda co: min(np.linalg.norm(o - PA) for o in co[1]))
    rc, onb = react
    o_nuc = min(onb, key=lambda o: np.linalg.norm(o - PA))
    d_attack = float(np.linalg.norm(o_nuc - PA))
    ang = angle_deg(o_nuc, PA, O3A)
    dP = float(min(np.linalg.norm(rc - p) for p in Ps))
    return dP, d_attack, ang


def process(cfgdir):
    base = os.path.basename(cfgdir)
    name = base[len("step1_"):] if base.startswith("step1_") else base
    pred = f"{cfgdir}/boltz_results_{base}/predictions/{base}"
    per = []
    for pdb in sorted(glob.glob(f"{pred}/{base}_model_*.pdb")):
        g = pose_geom(pdb)
        if g:
            per.append(g)
    if not per:
        return None
    dP = np.array([p[0] for p in per]); dA = np.array([p[1] for p in per]); ang = np.array([p[2] for p in per])
    docked = dP < GATE
    row = {"id": name, "n": len(per), "n_docked": int(docked.sum())}
    # consistency over ALL poses (default cutoffs); dispersion over docked poses
    row["consist_frac"] = round(float(np.mean((dA < DCUT) & (ang > ACUT))), 3)
    if docked.sum() >= N_MIN:
        a = ang[docked]; d = dA[docked]
        row["mean_angle"] = round(float(a.mean()), 1)
        row["angle_std"] = round(float(a.std()), 2)
        row["angle_iqr"] = round(float(np.percentile(a, 75) - np.percentile(a, 25)), 2)
        row["d_std"] = round(float(d.std()), 3)
        row["d_iqr"] = round(float(np.percentile(d, 75) - np.percentile(d, 25)), 3)
        row["status"] = "scored"
    else:
        for k in ("mean_angle", "angle_std", "angle_iqr", "d_std", "d_iqr"):
            row[k] = None
        row["status"] = "INCONCLUSIVE"
    # keep raw arrays for the robustness sweep
    row["_dA"] = dA; row["_ang"] = ang
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


def load_col(path, key, val, cast=float):
    d = {}
    for r in csv.DictReader(open(path)):
        v = r.get(val, "")
        d[r[key]] = cast(v) if v not in ("", None) else None
    return d


def main():
    act = load_col(ACT, "id", "activity_N66L1_ugml_avg")
    psi = load_col(PSI, "id", "ps_iptm_mean")
    length = load_col(MAN, "name", "length_aa", int)
    cfgdirs = sorted(glob.glob(f"{RES}/*"))
    with Pool(min(len(cfgdirs), 24)) as p:
        rows = [r for r in p.map(process, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])

    # write CSV (drop raw arrays)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["id", "n", "n_docked", "status", "consist_frac", "mean_angle",
              "angle_std", "angle_iqr", "d_std", "d_iqr", "activity_N66L1"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    scored = [r for r in rows if r["status"] == "scored" and r["activity_N66L1"] is not None]
    y = [r["activity_N66L1"] for r in scored]
    lab = [1 if v >= 0.5 else 0 for v in y]
    print("=== POSE CONSISTENCY / PREORGANIZATION vs N66 yield (step-1 cofolds) ===")
    print(f"{len(rows)} enzymes | {len(scored)} scored (docked>={N_MIN}) | actives={sum(lab)}\n")

    # directionality: low dispersion -> active => use -metric as the 'preorg signal';
    # consist_frac: high -> active => use as-is
    print(f"{'metric':14s} {'dir':4s} {'Spearman':>9} {'AUROC':>7} {'perm_p':>8}  (signal orientation)")
    specs = [("consist_frac", +1, "high=active"),
             ("angle_std", -1, "low=active"), ("angle_iqr", -1, "low=active"),
             ("d_std", -1, "low=active"), ("d_iqr", -1, "low=active")]
    for key, sign, note in specs:
        xr = [(r.get(key) if r.get(key) is not None else np.nan) for r in scored]
        sig = [sign * v for v in xr]
        rho = spearman(sig, y); au = auroc(sig, lab); pp = perm_p(sig, lab)
        d = "+" if sign > 0 else "-"
        print(f"{key:14s} {d:4s} {rho:>+9.3f} {au:>7.3f} {pp:>8.4f}  ({note})")

    # --- confound checks ---
    print("\n--- confounds (Spearman of raw metric vs ...) ---")
    conf = {"ps_iptm": [psi.get(r["id"], np.nan) for r in scored],
            "length": [length.get(r["id"], np.nan) for r in scored],
            "mean_angle": [r.get("mean_angle", np.nan) for r in scored],
            "n_docked": [r["n_docked"] for r in scored]}
    for key, _, _ in specs:
        xr = [(r.get(key) if r.get(key) is not None else np.nan) for r in scored]
        line = f"{key:14s}"
        for cname, cvals in conf.items():
            line += f"  {cname}={spearman(xr, cvals):+.2f}"
        print(line)

    # --- robustness sweep for consist_frac (its cutoffs are the only free params) ---
    print("\n--- consist_frac robustness sweep (d_cut x angle_cut): Spearman / AUROC ---")
    print(f"{'':6s}" + "".join(f"{'a>'+str(a):>14s}" for a in (140, 150, 160)))
    for dc in (3.5, 4.0, 4.5, 5.0):
        cells = []
        for ac in (140.0, 150.0, 160.0):
            xr = [float(np.mean((r["_dA"] < dc) & (r["_ang"] > ac))) for r in scored]
            cells.append(f"{spearman(xr, y):+.2f}/{auroc(xr, lab):.2f}")
        print(f"d<{dc:<4}" + "".join(f"{c:>14s}" for c in cells))

    # actives' standing on the headline preorg metric (angle_std, low=preorganized)
    print("\nactives' standing (angle_std, rank 1 = most consistent/lowest std):")
    ranked = sorted(scored, key=lambda r: r["angle_std"])
    for r in sorted([r for r in scored if r["activity_N66L1"] >= 0.5], key=lambda r: -r["activity_N66L1"]):
        rk = ranked.index(r) + 1
        print(f"  {r['id']:6s} act={r['activity_N66L1']:>5.2f}  angle_std={r['angle_std']:>6}  "
              f"consist_frac={r['consist_frac']:>5}  mean_angle={r['mean_angle']}  rank {rk}/{len(ranked)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
