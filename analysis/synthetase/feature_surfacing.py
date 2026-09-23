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
    "esp_reactive.csv": "esp",              # electrostatics: ESP + projected LEF at reactive atom
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

    topk = {k: set(sorted(ids, key=lambda i: -act[i])[:k]) for k in (3, 5, 10)}

    def au(y, x):
        return roc_auc_score(y, x) if len(set(y)) == 2 else float("nan")

    rows = []
    for col, vals in feats.items():
        xs = [(i, vals[i]) for i in ids if i in vals and not math.isnan(vals[i])]
        if len(xs) < 20:                      # need coverage
            continue
        obs_ids = [i for i, _ in xs]; x = [v for _, v in xs]
        a = [act[i] for i in obs_ids]
        rho = spearmanr(x, a).correlation
        rho = 0.0 if (rho is None or math.isnan(rho)) else rho
        # ONE direction per feature (higher feature -> more active), from continuous rho
        desc = rho >= 0
        xo = x if desc else [-v for v in x]           # oriented so higher=more active
        aur = {k: au([1 if i in topk[k] else 0 for i in obs_ids], xo) for k in (3, 5, 10)}
        aub = au([lab[i] for i in obs_ids], xo)       # activity > SfaB (natural cutoff, ~7 actives)
        order = sorted(obs_ids, key=lambda i: -vals[i] if desc else vals[i])
        rk = {i: k + 1 for k, i in enumerate(order)}
        t3r = sorted(rk[i] for i in topk[3] if i in rk)
        # robustness: mean AUROC across the 4 cutoffs (all oriented same way)
        vals4 = [v for v in (aur[3], aur[5], aur[10], aub) if not math.isnan(v)]
        robust = sum(vals4) / len(vals4) if vals4 else float("nan")
        def R(v): return round(v, 3) if not math.isnan(v) else ""
        rows.append({
            "feature": col, "n": len(xs), "dir": "+" if desc else "-",
            "robust_meanAUROC": R(robust),
            "top3_auroc": R(aur[3]), "top5_auroc": R(aur[5]), "top10_auroc": R(aur[10]),
            "binary_auroc_gtSfaB": R(aub), "spearman_all": round(rho, 3),
            "as9_rank": rk.get("AS9"), "top3_ranks": "|".join(map(str, t3r)),
        })
    rows.sort(key=lambda r: (r["robust_meanAUROC"] if r["robust_meanAUROC"] != "" else 0), reverse=True)

    out = os.path.join(an, "feature_surfacing_ranking.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"top-3 actives: {sorted(top3, key=lambda i:-act[i])}  (AS9,AS17,AS23)")
    print(f"ranked {len(rows)} features -> {out}\n")
    print(f"{'feature':30s} {'robust':>6s} {'top3':>5s} {'top5':>5s} {'top10':>5s} {'>SfaB':>6s} {'rho':>6s} {'dir':>3s} {'AS9':>4s}")
    for r in rows[:25]:
        print(f"{r['feature']:30s} {str(r['robust_meanAUROC']):>6s} {str(r['top3_auroc']):>5s} {str(r['top5_auroc']):>5s} {str(r['top10_auroc']):>5s} {str(r['binary_auroc_gtSfaB']):>6s} {str(r['spearman_all']):>6s} {r['dir']:>3s} {str(r['as9_rank']):>4s}")
    return rows


if __name__ == "__main__":
    main()
