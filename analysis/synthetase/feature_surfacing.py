#!/usr/bin/env python3
"""Which structural feature best surfaces the top synthetase catalysts?

Merges every per-target structural feature we've computed on the Boltz-2 1-Mg
step-1 panel (near-attack geometry, SASA/burial, Mg coordination, amine-subsite
electrostatics, acceptor-tunnel geometry, reactive preorganization, ps-iPTM), and
ranks each feature by its ability to surface the most-active enzymes:
  - top3_auroc : AUROC with label = "in the 3 most active" (AS9, AS17, AS23)
  - as9_rank   : rank of AS9 (the 5x-best) by the feature, best-direction
  - top3_ranks : ranks of the 3 top actives
  - spearman   : rho(feature, activity)  (sign = direction)
Features are ranked by |top3_auroc-0.5| (discrimination, direction-agnostic).

Usage: python feature_surfacing.py [--base exp_data/amide_synthetase]
"""
import os, csv, argparse, math
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# feature tables to merge (Boltz-2, 1-Mg panel); columns that are metadata, not features
SRC = {
    "battery_1mg.csv": "bat",
    "hypothesis_battery.csv": "hyp",
    "reactive_preorganization.csv": "preorg",
    "reactive_sasa_intermediate.csv": "sasa",
    "amine_subsite_electrostatics.csv": "elec",
    "acceptor_tunnel_geometry.csv": "tunnel",
    "exploratory_mg_coordination.csv": "mg",
    "psiptm_enzyme_diacid_1mg.csv": "psiptm",
    "crossmodel_nearattack.csv": "boltz",   # only boltz2_* cols kept below
}
META = {"id", "activity", "act", "activity_n66l1", "label_gt_sfab", "status", "is_active",
        "source_override", "prediction_source", "phosphate", "n", "n_bound", "n_docked"}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=".")
    base = os.path.abspath(ap.parse_args().base); an = os.path.join(base, "analysis")

    # activity + label from crossmodel csv (48 AS targets, validated)
    act, lab = {}, {}
    for r in load_csv(os.path.join(an, "crossmodel_nearattack.csv")):
        act[r["id"]] = float(r["activity"]); lab[r["id"]] = int(r["label_gt_sfab"])
    ids = [i for i in act]                         # AS1..AS48
    top3 = set(sorted(ids, key=lambda i: -act[i])[:3])   # AS9, AS17, AS23

    # merge features
    feats = {}   # colname -> {id: value}
    for fn, tag in SRC.items():
        p = os.path.join(an, fn)
        if not os.path.isfile(p):
            continue
        for r in load_csv(p):
            i = r.get("id")
            if i not in act:
                continue
            for k, v in r.items():
                if k.lower() in META or k == "id":
                    continue
                if fn == "crossmodel_nearattack.csv" and not k.startswith("boltz2_"):
                    continue
                col = f"{tag}:{k}"
                try:
                    feats.setdefault(col, {})[i] = float(v)
                except (ValueError, TypeError):
                    pass

    rows = []
    for col, vals in feats.items():
        xs = [(i, vals[i]) for i in ids if i in vals and not math.isnan(vals[i])]
        if len(xs) < 20:                      # need coverage
            continue
        obs_ids = [i for i, _ in xs]; x = [v for _, v in xs]
        a = [act[i] for i in obs_ids]
        y3 = [1 if i in top3 else 0 for i in obs_ids]
        yb = [lab[i] for i in obs_ids]
        rho = spearmanr(x, a).correlation
        au3 = roc_auc_score(y3, x) if len(set(y3)) == 2 else float("nan")
        aub = roc_auc_score(yb, x) if len(set(yb)) == 2 else float("nan")
        # best-direction ranking: if higher-feature associates with active (au3>=.5) rank desc else asc
        desc = (au3 >= 0.5) if not math.isnan(au3) else (rho >= 0)
        order = sorted(obs_ids, key=lambda i: -vals[i] if desc else vals[i])
        rk = {i: k + 1 for k, i in enumerate(order)}
        t3r = sorted(rk[i] for i in top3 if i in rk)
        rows.append({
            "feature": col, "n": len(xs),
            "top3_auroc": round(au3, 3) if not math.isnan(au3) else "",
            "top3_disc": round(abs(au3 - 0.5) + 0.5, 3) if not math.isnan(au3) else "",
            "dir": "+" if desc else "-",
            "as9_rank": rk.get("AS9"), "top3_ranks": "|".join(map(str, t3r)),
            "binary_auroc": round(aub, 3) if not math.isnan(aub) else "",
            "spearman": round(rho, 3) if rho is not None and not math.isnan(rho) else "",
        })
    rows.sort(key=lambda r: (r["top3_disc"] if r["top3_disc"] != "" else 0), reverse=True)

    out = os.path.join(an, "feature_surfacing_ranking.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"top-3 actives: {sorted(top3, key=lambda i:-act[i])}  (AS9,AS17,AS23)")
    print(f"ranked {len(rows)} features -> {out}\n")
    print(f"{'feature':30s} {'n':>3s} {'top3AUROC':>9s} {'dir':>3s} {'AS9rk':>5s} {'top3ranks':>10s} {'binAUROC':>8s} {'rho':>6s}")
    for r in rows[:25]:
        print(f"{r['feature']:30s} {r['n']:>3d} {str(r['top3_auroc']):>9s} {r['dir']:>3s} {str(r['as9_rank']):>5s} {r['top3_ranks']:>10s} {str(r['binary_auroc']):>8s} {str(r['spearman']):>6s}")
    return rows


if __name__ == "__main__":
    main()
