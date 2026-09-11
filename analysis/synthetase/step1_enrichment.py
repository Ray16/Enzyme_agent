#!/usr/bin/env python3
"""Does adenylation-NAC discriminate the ACTIVE synthetases from the inactive panel?

Reads the per-enzyme NAC table (near_attack_geometry.csv) + N66 L1 activity and asks:
  - where do the active enzymes rank by NAC fraction (and by min_d)?
  - is NAC fraction higher in the active group than the inactive group?
    (Mann-Whitney U, one-sided: active > inactive)
  - enrichment: of the top-K enzymes by NAC, how many are the active ones?

"Active" = N66 L1 activity above ACT_CUT (default 0.5 ug/mL; the panel is
near-background <0.15 except AS9/AS17/AS23). ACT_CUT overridable via argv[1].
"""
import csv, sys
import numpy as np

NAC = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step1/results/near_attack_geometry.csv"
ACT_CUT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

rows = list(csv.DictReader(open(NAC)))
for r in rows:
    r["nac_frac"] = float(r["nac_frac"]); r["min_d"] = float(r["min_d"])
    r["activity"] = float(r["activity_N66L1"]) if r["activity_N66L1"] not in ("", "None", None) else None

meas = [r for r in rows if r["activity"] is not None]
# ranks: 1 = best (highest nac_frac; ties broken by smaller min_d)
by_nac = sorted(meas, key=lambda r: (-r["nac_frac"], r["min_d"]))
for i, r in enumerate(by_nac):
    r["nac_rank"] = i + 1
N = len(meas)

active = [r for r in meas if r["activity"] >= ACT_CUT]
inactive = [r for r in meas if r["activity"] < ACT_CUT]

print(f"n={N} measured enzymes; ACTIVE = activity >= {ACT_CUT} ug/mL -> {len(active)} active, {len(inactive)} inactive\n")

print("ACTIVE enzymes and their NAC standing:")
print(f"  {'id':6s} {'activity':>9s} {'nac_frac':>8s} {'nac_rank':>9s} {'min_d':>6s}")
for r in sorted(active, key=lambda r: -r["activity"]):
    print(f"  {r['id']:6s} {r['activity']:>9.3f} {r['nac_frac']:>8.2f} {r['nac_rank']:>4d}/{N:<4d} {r['min_d']:>6.2f}")

af = np.array([r["nac_frac"] for r in active]); inf_ = np.array([r["nac_frac"] for r in inactive])
print(f"\nNAC fraction — active:   mean {af.mean():.3f}  median {np.median(af):.3f}  (min {af.min():.2f}, max {af.max():.2f})")
print(f"NAC fraction — inactive: mean {inf_.mean():.3f}  median {np.median(inf_):.3f}  (min {inf_.min():.2f}, max {inf_.max():.2f})")

# Mann-Whitney U (one-sided: active > inactive), normal approx with tie correction
def mannwhitney_u_gt(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); na, nb = len(a), len(b)
    allv = np.concatenate([a, b]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks); avg = sums / cnt
    ranks = avg[inv]
    Ra = ranks[:na].sum(); Ua = Ra - na * (na + 1) / 2
    mu = na * nb / 2.0
    n = na + nb
    tie = np.sum(cnt.astype(float) ** 3 - cnt)
    sd = np.sqrt(na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    if sd == 0: return Ua, float("nan")
    z = (Ua - mu) / sd
    # one-sided p (active > inactive)
    from math import erf, sqrt
    p = 1 - 0.5 * (1 + erf(z / sqrt(2)))
    return Ua, p

U, p = mannwhitney_u_gt(af, inf_)
print(f"Mann-Whitney U (active > inactive): U={U:.1f}, one-sided p={p:.4f}")

# Enrichment: of the top-K by NAC, how many are active?
print("\nTop-K by NAC fraction — how many active captured:")
ids_active = {r["id"] for r in active}
for K in (3, 5, 10):
    topk = by_nac[:K]; hit = sum(1 for r in topk if r["id"] in ids_active)
    print(f"  top {K:>2d}: {hit}/{len(active)} active enzymes  [{', '.join(r['id'] for r in topk if r['id'] in ids_active) or '-'}]")
