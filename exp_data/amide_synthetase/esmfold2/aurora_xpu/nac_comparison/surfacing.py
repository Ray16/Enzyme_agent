"""Retrieval view: does A surface the 3 measured actives, ESMFold2 vs Boltz-2?"""
import json, sys, itertools
import numpy as np
from scipy import stats
D = json.load(open(f"{sys.argv[1]}/A_both_models.json"))
tg = sorted(D["esmfold2"])
POS = ["AS9", "AS17", "AS23"]
lab = np.array([t in POS for t in tg])
METRICS = ["A", "Dh", "ng", "nac"]

def auroc(x, y, higher=True):
    s = x if higher else -x
    r = stats.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1*(n1+1)/2) / (n1*n0)

def ranks(x, higher=True):
    order = np.argsort(-x if higher else x, kind="stable")
    pos = {tg[i]: k+1 for k, i in enumerate(order)}
    return pos

print(f"panel n={len(tg)}, positives={POS}")
print(f"measured activity: " + ", ".join(f"{p}={D['activity'][p][0]:.3f}" for p in POS))
print("  (next highest non-positive: " +
      ", ".join(f"{t}={D['activity'][t][0]:.3f}" for t in
                sorted(tg, key=lambda t: -D['activity'][t][0])[3:6]) + ")")

res = {}
for model in ("esmfold2", "boltz2"):
    print(f"\n=== {model} ===")
    for met in METRICS:
        hi = met != "Dh"
        x = np.array([D[model][t][met] for t in tg], float)
        rk = ranks(x, hi)
        a = auroc(x, lab, hi)
        rs = [rk[p] for p in POS]
        top5 = sum(r <= 5 for r in rs); top10 = sum(r <= 10 for r in rs)
        res[(model, met)] = (a, rs)
        print(f"  {met:4} AUROC={a:.3f}  ranks " +
              " ".join(f"{p}={rk[p]}" for p in POS) +
              f"  | top5 {top5}/3  top10 {top10}/3  median rank {np.median(rs):.0f}/{len(tg)}")

# enrichment of a 3-positive panel: exact hypergeometric for top-k recall
from scipy.stats import hypergeom
print("\nchance baseline: top-10 of 49 catches a given positive 20.4% of the time;")
print("P(>=2 of 3 positives in top 10) =",
      f"{1-hypergeom.cdf(1, 49, 3, 10):.3f}; P(all 3) = {hypergeom.pmf(3,49,3,10):.4f}")

# selection-corrected null over the 8-test battery (4 metrics x 2 models)
rng = np.random.default_rng(2)
obs = {k: v[0] for k, v in res.items()}
best = max(obs, key=obs.get)
X = {k: (np.array([D[k[0]][t][k[1]] for t in tg], float), k[1] != "Dh") for k in res}
N = 20000
mx = np.empty(N)
for i in range(N):
    y = np.zeros(len(tg), bool); y[rng.choice(len(tg), 3, replace=False)] = True
    mx[i] = max(auroc(x, y, hi) for x, hi in X.values())
print(f"\nobserved max AUROC over the 8-test battery: {obs[best]:.3f} at {best}")
print(f"null 95th pct = {np.percentile(mx,95):.3f}; selection-corrected p = {(mx>=obs[best]).mean():.3f}")
for m in ("esmfold2", "boltz2"):
    a = obs[(m, "A")]
    print(f"  pre-registered A, {m:9}: AUROC={a:.3f}  one-test p={(mx>=a).mean():.3f} (corrected)")

# paired: does A's AUROC differ between models? bootstrap over the 46 negatives
ae, _ = X[("esmfold2","A")]; ab, _ = X[("boltz2","A")]
neg = np.where(~lab)[0]; pos_i = np.where(lab)[0]
d = []
for _ in range(20000):
    i = np.concatenate([pos_i, rng.choice(neg, len(neg), replace=True)])
    y = np.zeros(len(i), bool); y[:3] = True
    d.append(auroc(ae[i], y) - auroc(ab[i], y))
d = np.array(d); lo, hi_ = np.percentile(d, [2.5, 97.5])
print(f"\nAUROC(A, ESMFold2) - AUROC(A, Boltz-2) = {obs[('esmfold2','A')]-obs[('boltz2','A')]:+.3f}, "
      f"bootstrap 95% CI [{lo:+.3f},{hi_:+.3f}], p={2*min((d<=0).mean(),(d>=0).mean()):.3f}")
