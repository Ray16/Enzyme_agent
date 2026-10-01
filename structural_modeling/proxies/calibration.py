"""Deciding whether a proxy's score means anything.

The library is small on purpose, but even a small library is a multiple-comparison
problem when a panel has few actives. Reporting the best proxy out of several
without a matching null is uninterpretable: the maximum over a set of uninformative
features is biased upward, and the bias grows with the size of the set and shrinks
with the number of actives. On a panel with a handful of actives that upward bias
is large enough to produce respectable-looking AUROCs from pure noise.

``selection_null`` therefore reproduces the ACTUAL selection procedure -- permute
the labels, score every proxy, keep the maximum -- so a candidate is compared
against the right reference rather than against 0.5. ``screen`` refuses to promote
a proxy that does not clear it.

Three gates, from ``knowledge/structural_proxy_prior.yaml`` cross_cutting:
  direction_audit  the sign must match the pre-declared ``expected_direction``;
                   a proxy that only works sign-flipped after seeing labels is
                   fitting noise
  confound_check   the proxy must survive controlling for a nuisance covariate,
                   typically model confidence: prediction confidence alone can be
                   a serviceable activity predictor without reflecting catalysis
  selection_null   the score must beat the best-of-library noise distribution
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                            # SciPy is optional here
    from scipy.stats import rankdata, spearmanr
except ImportError:                             # pragma: no cover
    rankdata = spearmanr = None


def _rank(x):
    if rankdata is not None:
        return rankdata(x)
    order = np.argsort(np.argsort(x))
    return order.astype(float) + 1.0


def _spearman(x, y) -> float:
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUROC with ties at 0.5. No sklearn dependency."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = _rank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def robust_auroc(score, activity, oriented: bool = True, ks=(3, 5, 10)) -> float:
    """Mean AUROC over nested top-k labels. ``oriented=False`` keeps the proxy's
    declared direction; ``True`` lets the data choose it, which is what the
    selection null must also be allowed to do to stay comparable."""
    score, activity = np.asarray(score, float), np.asarray(activity, float)
    ok = np.isfinite(score) & np.isfinite(activity)
    score, activity = score[ok], activity[ok]
    if len(score) < 10:
        return float("nan")
    sign = 1.0
    if oriented and _spearman(score, activity) < 0:
        sign = -1.0
    out = []
    for k in ks:
        if k >= len(activity):
            continue
        y = (activity >= np.sort(activity)[-k]).astype(int)
        out.append(auroc(sign * score, y))
    return float(np.nanmean(out)) if out else float("nan")


@dataclass(frozen=True)
class ProxyVerdict:
    proxy_id: str
    score: float
    direction_ok: bool
    partial_rho: float          # vs activity, controlling for the confound
    selection_p: float          # P(noise library reaches this score)
    promoted: bool
    note: str


def partial_spearman(x, y, control) -> float:
    """Spearman of x vs y after linearly removing ``control`` from both ranks."""
    x, y, control = (np.asarray(v, float) for v in (x, y, control))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if ok.sum() < 8:
        return float("nan")
    rx, ry, rz = _rank(x[ok]), _rank(y[ok]), _rank(control[ok])
    design = np.c_[np.ones(ok.sum()), rz]

    def resid(v):
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def selection_null(features: np.ndarray, activity: np.ndarray,
                   n_perm: int = 300, seed: int = 0) -> np.ndarray:
    """Null distribution of the BEST score across the library.

    ``features`` is (n_proxies, n_complexes). Permuting activity and taking the max
    over proxies reproduces the real selection procedure, which is the only null
    that makes "we tried N and kept the best" interpretable.
    """
    rng = np.random.default_rng(seed)
    activity = np.asarray(activity, float)
    out = []
    for _ in range(n_perm):
        perm = rng.permutation(activity)
        out.append(max(robust_auroc(f, perm) for f in features))
    return np.array(out)


def screen(scores: dict[str, np.ndarray], activity: np.ndarray,
           library, confound: np.ndarray | None = None,
           n_perm: int = 300, alpha: float = 0.05) -> list[ProxyVerdict]:
    """Run all three gates and return a verdict per proxy, best first.

    ``scores``   proxy id -> per-complex value (NaN where unsupported)
    ``confound`` per-complex nuisance covariate (e.g. mean complex pLDDT).
                 Strongly recommended: model confidence alone is frequently a
                 serviceable activity predictor, so a proxy correlated with it
                 inherits that predictiveness without reflecting catalysis.
    """
    by_id = {p.id: p for p in library}
    usable = {k: np.asarray(v, float) for k, v in scores.items()
              if np.isfinite(np.asarray(v, float)).sum() >= 10}
    if not usable:
        return []
    filled = []
    for v in usable.values():
        w = v.copy()
        if not np.isfinite(w).all():
            w[~np.isfinite(w)] = np.nanmedian(w)
        filled.append(w)
    null = selection_null(np.array(filled), activity, n_perm=n_perm)

    verdicts = []
    for pid, v in usable.items():
        proxy = by_id.get(pid)
        score = robust_auroc(v, activity)
        rho = _spearman(v[np.isfinite(v)], np.asarray(activity, float)[np.isfinite(v)])
        want = getattr(proxy, "expected_direction", "+")
        direction_ok = (rho >= 0) if want == "+" else (rho <= 0)
        pr = partial_spearman(v, activity, confound) if confound is not None else float("nan")
        p_sel = float(np.mean(null >= score)) if np.isfinite(score) else float("nan")

        notes = []
        if not direction_ok:
            notes.append(f"direction audit FAILED (expected {want}, saw rho={rho:+.2f})")
        if p_sel > alpha:
            notes.append(f"does not clear the best-of-library null (p={p_sel:.3f})")
        if confound is not None and np.isfinite(pr) and abs(pr) < 0.1:
            notes.append(f"vanishes under the confound control (partial rho={pr:+.2f})")
        verdicts.append(ProxyVerdict(
            proxy_id=pid, score=score, direction_ok=direction_ok, partial_rho=pr,
            selection_p=p_sel, promoted=direction_ok and p_sel <= alpha,
            note="; ".join(notes) or "clears all gates"))
    return sorted(verdicts, key=lambda v: (-(v.score if np.isfinite(v.score) else -1)))
