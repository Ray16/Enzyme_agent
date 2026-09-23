#!/usr/bin/env python3
"""Which structural feature best surfaces the top nylon-HYDROLASE catalysts?

Hydrolase analog of analysis/synthetase/feature_surfacing.py. Ranks every proxy in
exp_data/nylon_hydrolase/analysis/hydrolase_proxies.csv by how well it surfaces the
most experimentally-active enzymes.

Activity ground truth = L2 = linear PA66 (nylon-6,6) DIMER release (ug/mL) -- the
product mechanistically matched to the cofolded nylon-6,6 tetrad, and the best-
powered readout (38/96 significant vs 15/96 for the L1 monomer). We report:
  - top5/top10/top20 AUROC : label = "in the k most active by L2"
  - binary AUROC           : label = L2_significant (vs no-enzyme control) -- the
                             analog of the synthetase ">SfaB" natural cutoff
  - robust_meanAUROC       : mean of the four (all oriented same direction)
  - spearman vs L2, NylC rank (named reference), top5 ranks

Each feature gets ONE direction (higher -> more active) from the sign of its
Spearman rho vs L2, exactly as in the synthetase evaluator.

Usage: python feature_surfacing.py [--label L2|total|L1]
"""
import os, csv, argparse, math
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/nylon_hydrolase/analysis"
SRC = os.path.join(BASE, "hydrolase_proxies.csv")
META = {"id", "n", "L1", "L2", "total", "L1sig", "L2sig"}
CUTS = (5, 10, 20)

# FIXED mechanistic direction per proxy: +1 => HIGHER value should mean MORE active.
# We orient by mechanism, NOT by the sign of the observed correlation -- picking the
# direction from a near-zero rho on a small positive set manufactures spurious top-k
# AUROC (verified: it turned L1 oxyanion/TI proxies into fake 0.9s that are actually
# ~0.1 in the correct direction). rho is still reported so the reader can see when the
# data disagree with the mechanism.
SIGN = {
    "nac:joint_frac": +1, "nac:bd_gated_dev": -1, "nac:min_d": -1, "nac:median_d": -1,
    "oxh:ge2_frac": +1, "oxh:inplane_frac": +1, "oxh:mean_ndonor": +1, "oxh:min_dON": -1,
    "base:hb_frac": +1, "base:median_dN_OG1": -1, "preorg:og1_rmsf": -1,
    "ti:rmsd_min": -1, "ti:rmsd_mean": -1, "esp:oxyanion_O_mean": +1, "esp:carbonyl_C_mean": +1,
    "conf:ligand_iptm": +1, "conf:complex_plddt": +1, "conf:thr_plddt": +1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="L2", choices=["L2", "total", "L1"])
    ap.add_argument("--out", default=os.path.join(BASE, "feature_surfacing_ranking.csv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC)))
    act, lab = {}, {}
    for r in rows:
        i = r["id"]
        v = r.get(args.label)
        sigkey = "L2sig" if args.label != "L1" else "L1sig"
        if v in (None, ""):
            continue
        act[i] = float(v)
        lab[i] = int(float(r[sigkey])) if r.get(sigkey) not in (None, "") else 0
    ids = list(act)
    topk = {k: set(sorted(ids, key=lambda i: -act[i])[:k]) for k in CUTS}
    n_active = sum(lab.values())

    feats = {}
    for r in rows:
        i = r["id"]
        if i not in act:
            continue
        for k, val in r.items():
            if k in META or k == "id":
                continue
            try:
                feats.setdefault(k, {})[i] = float(val)
            except (ValueError, TypeError):
                pass

    def au(y, x):
        return roc_auc_score(y, x) if len(set(y)) == 2 else float("nan")

    out = []
    for col, vals in feats.items():
        xs = [(i, vals[i]) for i in ids if i in vals and not math.isnan(vals[i])]
        if len(xs) < 40:
            continue
        obs = [i for i, _ in xs]; x = [v for _, v in xs]
        a = [act[i] for i in obs]
        rho = spearmanr(x, a).correlation
        rho = 0.0 if (rho is None or math.isnan(rho)) else rho
        desc = rho >= 0
        xo = x if desc else [-v for v in x]
        aur = {k: au([1 if i in topk[k] else 0 for i in obs], xo) for k in CUTS}
        aub = au([lab[i] for i in obs], xo)
        order = sorted(obs, key=lambda i: -vals[i] if desc else vals[i])
        rk = {i: j + 1 for j, i in enumerate(order)}
        t5r = sorted(rk[i] for i in topk[5] if i in rk)
        vals4 = [v for v in (aur[5], aur[10], aur[20], aub) if not math.isnan(v)]
        robust = sum(vals4) / len(vals4) if vals4 else float("nan")
        R = lambda v: round(v, 3) if not math.isnan(v) else ""
        out.append({
            "feature": col, "n": len(xs), "dir": "+" if desc else "-",
            "robust_meanAUROC": R(robust), "top5_auroc": R(aur[5]),
            "top10_auroc": R(aur[10]), "top20_auroc": R(aur[20]),
            "binary_auroc_sig": R(aub), "spearman_L2": round(rho, 3),
            "nylc_rank": rk.get("NylC"), "top5_ranks": "|".join(map(str, t5r)),
        })
    out.sort(key=lambda r: (r["robust_meanAUROC"] if r["robust_meanAUROC"] != "" else 0), reverse=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)

    top = sorted(ids, key=lambda i: -act[i])[:5]
    print(f"label={args.label}  n={len(ids)} enzymes, {n_active} significant (positives)")
    print(f"top-5 by {args.label}: {top}")
    print(f"ranked {len(out)} features -> {args.out}\n")
    hdr = f"{'feature':22s} {'rob':>5s} {'t5':>5s} {'t10':>5s} {'t20':>5s} {'sig':>5s} {'rho':>6s} {'dir':>3s} {'NylC':>4s}"
    print(hdr)
    for r in out:
        print(f"{r['feature']:22s} {str(r['robust_meanAUROC']):>5s} {str(r['top5_auroc']):>5s} "
              f"{str(r['top10_auroc']):>5s} {str(r['top20_auroc']):>5s} {str(r['binary_auroc_sig']):>5s} "
              f"{str(r['spearman_L2']):>6s} {r['dir']:>3s} {str(r['nylc_rank']):>4s}")


if __name__ == "__main__":
    main()
