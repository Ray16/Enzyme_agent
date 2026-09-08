#!/usr/bin/env python3
"""Standard evaluation block for a geometric proxy vs binary activity.

For a skewed activity panel, judge a proxy as a *ranking classifier*:
  - AUROC  (Mann-Whitney: P(active ranks above inactive))
  - bootstrap 95% CI on AUROC (resample enzymes)  -> fragility with few positives
  - permutation p (shuffle labels)                -> is the separation better than chance
  - precision@k (top-k by proxy)                  -> screening usefulness (AUROC hides this)

Higher proxy score should mean "more active"; pass higher_better=False for a
distance-like proxy (shorter = better) and it is negated internally.
"""
import numpy as np
from scipy.stats import rankdata


def auroc(s, y):
    """Mann-Whitney AUROC via tie-averaged ranks (vectorised, O(n log n))."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)                        # tie-averaged ranks
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def evaluate(scores, labels, name="", higher_better=True,
             ks=(3, 5, 10), nboot=5000, nperm=20000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, float) * (1.0 if higher_better else -1.0)
    y = np.asarray(labels, int)
    obs = auroc(s, y)
    # bootstrap CI (resample enzymes)
    boot = []
    for _ in range(nboot):
        idx = rng.integers(0, len(y), len(y))
        a = auroc(s[idx], y[idx])
        if not np.isnan(a):
            boot.append(a)
    lo, hi = (np.percentile(boot, 2.5), np.percentile(boot, 97.5)) if boot else (np.nan, np.nan)
    # permutation p (shuffle labels)
    perm = np.array([auroc(s, rng.permutation(y)) for _ in range(nperm)])
    pval = float((perm >= obs).mean())
    # precision@k (top-k by proxy score)
    order = np.argsort(-s, kind="mergesort")
    prec = {k: (int(y[order[:k]].sum()), k) for k in ks}
    return {"name": name, "n": len(y), "n_active": int(y.sum()),
            "auroc": round(float(obs), 3), "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
            "perm_p": round(pval, 4), "precision_at_k": prec}


def fmt(res):
    pk = "  ".join(f"P@{k}={h}/{k}" for k, (h, k) in res["precision_at_k"].items())
    return (f"{res['name']:34s} AUROC={res['auroc']:.3f} "
            f"[{res['ci_lo']:.2f},{res['ci_hi']:.2f}]  perm_p={res['perm_p']:.3f}  "
            f"n_act={res['n_active']}/{res['n']}  {pk}")
