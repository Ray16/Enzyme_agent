#!/usr/bin/env python3
"""Cross-model step-1 near-attack driver (amide-synthetase panel).

Reusable driver for the rolling cross-model comparison. For each cofold model
present under exp_data/amide_synthetase/<model>/step1_1mg/, computes the
validated acyl-adenylate near-attack proxy per target (adipate proximal
carboxyl O -> ATP Palpha; gate d<4A THEN in-line angle>150deg), aggregates over
all diffusion samples, and scores predictivity vs measured activity.

Primary metric (user's validated proxy) = mean_gated_angle: among poses that pass
the d<4A gate, the mean in-line angle. na_frac (fraction passing both gates) is a
coarser secondary. Activity + label_gt_sfab are read from the panel activity table
(model-independent). Only models whose structures are present are scored, so it can
be re-run as each panel lands (rolling). Boltz outputs are PDB; the others mmCIF.

Usage: python crossmodel_driver.py [--base DIR] [--tag STR]
"""
import os, sys, glob, csv, argparse, math

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from crossmodel_nearattack import near_attack_pose, D_GATE, ANG_MIN  # noqa: E402

# Per-model: how to find one target's structure files (recursive), given target id T.
MODELS = {
    "boltz2":   ("boltz2/step1_1mg/results/{T}",      "*.pdb"),
    "protenix": ("protenix/step1_1mg/{T}",            "*sample_*.cif"),
    "af3":      ("af3/step1_1mg/step1_{T}",           "*_model.cif"),
    "esmfold2": ("esmfold2/step1_1mg/{T}",            "*.cif"),
}
MODEL_ORDER = ["boltz2", "protenix", "af3", "esmfold2"]


def load_activity(base):
    """{id: (activity_float, label_gt_sfab_int)}. Prefer the archived crossmodel csv
    (activity+label already derived, model-independent); else panel_activity.csv."""
    arch = sorted(glob.glob(os.path.join(base, "analysis/archive_*/crossmodel_nearattack.csv")))
    if arch:
        act = {}
        with open(arch[-1]) as f:
            for row in csv.DictReader(f):
                act[row["id"]] = (float(row["activity"]), int(row["label_gt_sfab"]))
        if act:
            return act
    # fallback: panel_activity.csv (+ derive label vs SfaB)
    pa = os.path.join(base, "assay_data/panel_activity.csv")
    rows = list(csv.DictReader(open(pa)))
    def actcol(r):
        for k in ("activity", "mean_activity", "yield", "product"):
            if k in r and r[k] not in ("", None):
                return float(r[k])
        raise KeyError("no activity column in panel_activity.csv")
    idcol = "id" if "id" in rows[0] else list(rows[0])[0]
    vals = {r[idcol]: actcol(r) for r in rows}
    sfab = vals.get("SfaB")
    return {k: (v, int(v > sfab)) for k, v in vals.items() if k != "SfaB" and sfab is not None}


def aggregate_target(base, model, T):
    root, pat = MODELS[model]
    d = os.path.join(base, root.format(T=T))
    if not os.path.isdir(d):
        return None
    files = glob.glob(os.path.join(d, "**", pat), recursive=True)
    poses = []
    for p in files:
        try:
            r = near_attack_pose(p)
        except Exception:
            r = None
        if r is not None:
            poses.append(r)
    if not poses:
        return None
    ds = [d_ for d_, a in poses]
    gated = [a for d_, a in poses if d_ < D_GATE]
    na = sum(1 for d_, a in poses if d_ < D_GATE and a > ANG_MIN)
    return {
        "n_poses": len(poses),
        "na_frac": na / len(poses),
        "min_d": min(ds),
        "mean_gated_angle": (sum(gated) / len(gated)) if gated else float("nan"),
        "n_gated": len(gated),
    }


def predictivity(ids, score, activity, labels):
    """Spearman(score vs activity), AUROC(score vs label), and rank of AS9 (1=best).
    Higher score = more near-attack = predicted more active. NaN scores drop from
    Spearman/AUROC; for ranking they sort last."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    pairs = [(s, activity[i], labels[i], i) for i, s in zip(ids, score)]
    valid = [(s, a, l) for s, a, l, i in pairs if not (isinstance(s, float) and math.isnan(s))]
    rho = auroc = float("nan")
    if len(valid) >= 3:
        ss, aa, ll = zip(*valid)
        rho = spearmanr(ss, aa).correlation
        if len(set(ll)) == 2:
            auroc = roc_auc_score(ll, ss)
    # AS9 rank by score descending (NaN last)
    order = sorted(pairs, key=lambda x: (-(x[0]) if not (isinstance(x[0], float) and math.isnan(x[0])) else float("inf")))
    as9_rank = next((k + 1 for k, (s, a, l, i) in enumerate(order) if i == "AS9"), None)
    n_scored = len(valid)
    return rho, auroc, as9_rank, n_scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--tag", default="50sample")
    args = ap.parse_args()
    base = os.path.abspath(args.base)
    activity = load_activity(base)
    ids = [i for i in activity if i.startswith("AS")]  # AS1..AS48 (SfaB is the ref)
    present = [m for m in MODEL_ORDER
               if os.path.isdir(os.path.join(base, m, "step1_1mg"))
               and any(aggregate_target(base, m, T) for T in ids[:3])]
    print(f"[driver] base={base} tag={args.tag}")
    print(f"[driver] models present: {present}")

    # per-target metrics
    per = {m: {T: aggregate_target(base, m, T) for T in ids} for m in present}
    outdir = os.path.join(base, "analysis"); os.makedirs(outdir, exist_ok=True)
    pt_csv = os.path.join(outdir, "crossmodel_nearattack.csv")
    cols = ["id", "activity", "label_gt_sfab"]
    for m in present:
        cols += [f"{m}_na_frac", f"{m}_mean_gated_angle", f"{m}_min_d", f"{m}_n_poses"]
    with open(pt_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for T in ids:
            row = [T, activity[T][0], activity[T][1]]
            for m in present:
                r = per[m].get(T)
                if r: row += [round(r["na_frac"], 3), (round(r["mean_gated_angle"], 1) if not math.isnan(r["mean_gated_angle"]) else ""), round(r["min_d"], 2), r["n_poses"]]
                else: row += ["", "", "", 0]
            w.writerow(row)

    # predictivity summary
    from sklearn.metrics import roc_auc_score as _auc
    act = {T: activity[T][0] for T in ids}
    lab = {T: activity[T][1] for T in ids}
    top3 = set(sorted(ids, key=lambda i: -act[i])[:3])   # 3 most active catalysts
    sum_csv = os.path.join(outdir, "crossmodel_predictivity_summary.csv")
    summary = []
    for m in present:
        for metric in ("mean_gated_angle", "na_frac"):
            score = [ (per[m][T][metric] if per[m].get(T) else float("nan")) for T in ids ]
            rho, auroc, as9_rank, n = predictivity(ids, score, act, lab)
            npos = [per[m][T]["n_poses"] for T in ids if per[m].get(T)]
            as9 = per[m].get("AS9") or {}
            # top-3 surfacing: AUROC(label=in top-3) and the ranks of the 3 top actives
            sm = list(zip(ids, score))
            valid = [(i, s) for i, s in sm if not (isinstance(s, float) and math.isnan(s))]
            order = sorted(sm, key=lambda x: (-(x[1]) if not (isinstance(x[1], float) and math.isnan(x[1])) else float("inf")))
            rk = {i: k + 1 for k, (i, s) in enumerate(order)}
            t3_ranks = sorted(rk[i] for i in top3)
            yy = [1 if i in top3 else 0 for i, s in valid]; xx = [s for i, s in valid]
            t3_auroc = round(_auc(yy, xx), 3) if len(set(yy)) == 2 else ""
            summary.append({"model": m, "metric": metric, "n_scored": n,
                            "top3_auroc": t3_auroc, "top3_ranks": "|".join(map(str, t3_ranks)),
                            "auroc": round(auroc, 3) if not math.isnan(auroc) else "",
                            "spearman": round(rho, 3) if not math.isnan(rho) else "",
                            "as9_rank": as9_rank, "n_targets_scored_of": len(ids),
                            "as9_min_d": round(as9.get("min_d", float("nan")), 2) if as9 else "",
                            "as9_na_frac": round(as9.get("na_frac", float("nan")), 2) if as9 else "",
                            "median_n_poses": (sorted(npos)[len(npos)//2] if npos else 0)})
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader()
        for r in summary: w.writerow(r)

    print(f"[driver] wrote {pt_csv}")
    print(f"[driver] wrote {sum_csv}")
    print(f"\n{'model':10s} {'metric':17s} {'n':>3s} {'AUROC':>6s} {'rho':>6s} {'AS9rk':>6s} {'top3AUROC':>9s} {'top3ranks':>11s}")
    for r in summary:
        print(f"{r['model']:10s} {r['metric']:17s} {r['n_scored']:>3d} {str(r['auroc']):>6s} {str(r['spearman']):>6s} {str(r['as9_rank']):>6s} {str(r['top3_auroc']):>9s} {str(r['top3_ranks']):>11s}")
    return summary, per, ids, activity, present


if __name__ == "__main__":
    main()
