#!/usr/bin/env python3
"""Protein-substrate pair ipTM (ps_iptm) for the SfaB step-1 cofolds.

ps_iptm = pair_chains_iptm['0']['4']  (protein chain A vs adipate chain E),
following the DesD convention (analyze_binding_vs_catalysis_v2.py). Excludes
ATP (chain 1) and Mg (2,3). Aggregates mean/max over the 50 models per enzyme,
then tests whether ps_iptm separates active from inactive synthetases.
"""
import csv, glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from math import erf, sqrt

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
RES = f"{ROOT}/exp_data/sfab_step1_results/output_configs"
ACT = f"{ROOT}/exp_data/SfaB_diversity_panel.csv"
OUT = f"{ROOT}/exp_data/sfab_step1_results/ps_iptm_step1.csv"
ACT_CUT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
SUB_CHAIN = "4"   # adipate (E); protein is "0"


def process_config(cfgdir):
    name = os.path.basename(cfgdir)[len("step1_"):]
    pred = f"{cfgdir}/boltz_results_step1_{name}/predictions/step1_{name}"
    vals = []
    for j in sorted(glob.glob(f"{pred}/confidence_step1_{name}_model_*.json")):
        pc = json.load(open(j)).get("pair_chains_iptm", {})
        if "0" in pc and SUB_CHAIN in pc["0"]:
            vals.append(float(pc["0"][SUB_CHAIN]))
    if not vals:
        return None
    v = np.array(vals)
    return {"id": name, "n": len(v),
            "ps_iptm_mean": round(float(v.mean()), 3),
            "ps_iptm_max": round(float(v.max()), 3),
            "ps_iptm_median": round(float(np.median(v)), 3)}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt((rx@rx)*(ry@ry)) + 1e-12))


def mwu_gt(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); na, nb = len(a), len(b)
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort"); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, ranks); ranks = (s/cnt)[inv]
    Ua = ranks[:na].sum() - na*(na+1)/2; mu = na*nb/2; n = na+nb
    tie = np.sum(cnt.astype(float)**3 - cnt)
    sd = np.sqrt(na*nb/12*((n+1) - tie/(n*(n-1)))) if n > 1 else 0
    if sd == 0: return Ua, float("nan")
    z = (Ua-mu)/sd; return Ua, 1 - 0.5*(1+erf(z/sqrt(2)))


def main():
    act = {}
    for r in csv.DictReader(open(ACT)):
        v = r.get("activity_N66L1_ugml_avg", "")
        act[r["id"]] = float(v) if v not in ("", None) else None
    cfgdirs = sorted(glob.glob(f"{RES}/step1_*"))
    with Pool(min(len(cfgdirs), 32)) as p:
        rows = [r for r in p.map(process_config, cfgdirs) if r]
    for r in rows:
        r["activity_N66L1"] = act.get(r["id"])

    # rank by mean ps_iptm
    rows.sort(key=lambda r: -r["ps_iptm_mean"])
    for i, r in enumerate(rows):
        r["ps_iptm_rank"] = i + 1
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    meas = [r for r in rows if r["activity_N66L1"] is not None]
    y = np.array([r["activity_N66L1"] for r in meas])
    for key in ("ps_iptm_mean", "ps_iptm_max"):
        rho = spearman(np.array([r[key] for r in meas]), y)
        print(f"Spearman rho ({key} vs activity, n={len(meas)}): {rho:.3f}")

    active = [r for r in meas if r["activity_N66L1"] >= ACT_CUT]
    inactive = [r for r in meas if r["activity_N66L1"] < ACT_CUT]
    print(f"\nACTIVE (>= {ACT_CUT}): {len(active)}   INACTIVE: {len(inactive)}")
    print(f"  {'id':6s} {'activity':>9s} {'ps_mean':>8s} {'ps_max':>7s} {'rank(mean)':>10s}")
    for r in sorted(active, key=lambda r: -r["activity_N66L1"]):
        print(f"  {r['id']:6s} {r['activity_N66L1']:>9.3f} {r['ps_iptm_mean']:>8.3f} {r['ps_iptm_max']:>7.3f} {r['ps_iptm_rank']:>5d}/{len(meas)}")
    am = np.array([r["ps_iptm_mean"] for r in active]); im = np.array([r["ps_iptm_mean"] for r in inactive])
    print(f"\nps_iptm_mean  active: mean {am.mean():.3f} median {np.median(am):.3f}  |  inactive: mean {im.mean():.3f} median {np.median(im):.3f}")
    U, pv = mwu_gt(am, im); print(f"Mann-Whitney (active > inactive): U={U:.1f}, one-sided p={pv:.4f}")
    ids_a = {r["id"] for r in active}
    for K in (3, 5, 10):
        hit = [r["id"] for r in rows[:K] if r["id"] in ids_a]
        print(f"  top {K:>2d} by ps_iptm_mean: {len(hit)}/{len(active)} active  [{', '.join(hit) or '-'}]")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
