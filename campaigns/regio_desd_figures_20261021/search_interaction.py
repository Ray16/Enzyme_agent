#!/usr/bin/env python3
"""Amendment 7 -- does ground-state geometry predict the enzyme x substrate INTERACTION?

Every earlier test stratified by acid donor, which leaves the enzyme main effect intact; amendment
6 showed that effect is confounded with training-set membership. This is the mirror design:
stratify by ENZYME and ask whether the model predicts which substrates a given enzyme accepts.
DesD is 10/10 active, so it contributes zero informative pairs BY CONSTRUCTION -- the memorisation
confound is structurally absent rather than controlled for after the fact.

The null preserves BOTH margins of the activity matrix (checkerboard / curveball swaps), so a
descriptor cannot score by tracking enzyme promiscuity or substrate difficulty. Only genuine
interaction survives.

Design, battery, null and decision rule were written into PREREGISTRATION.md amendment 7 before
this file was run. Nothing here may be edited in response to a score.
"""
import csv, os, pathlib, sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
from search_descriptors import build_cells, ORIENT, spearman  # noqa: E402

NDRAW, BURN, BETWEEN, SEED = 20000, 20000, 2000, 20261023


def concordance_parts(cells, looks):
    """Precompute, per look and per enzyme, the pairwise comparison matrix.

    Ranks do not change under label permutation, so these matrices are computed once. For a label
    vector y within an enzyme, the concordant-pair count is the quadratic form y^T M (1-y).
    Row sums of the activity matrix are invariant under checkerboard swaps, so the DENOMINATOR
    (sum of n_active*n_inactive per enzyme) is constant across draws and needs no recomputation.
    """
    enzymes = sorted({c["enzyme"] for c in cells})
    idx = {e: [i for i, c in enumerate(cells) if c["enzyme"] == e] for e in enzymes}
    mats = {}
    for lk in looks:
        v = np.array([cells[i][lk] * ORIENT_SIGN[lk] for i in range(len(cells))], float)
        per = {}
        for e in enzymes:
            ii = idx[e]
            a = v[ii][:, None]; b = v[ii][None, :]
            per[e] = np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))
        mats[lk] = per
    return enzymes, idx, mats


def C_of(y, enzymes, idx, mats, lk, denom):
    num = 0.0
    for e in enzymes:
        ye = y[idx[e]]
        if ye.sum() == 0 or ye.sum() == len(ye):
            continue
        num += float(ye @ mats[lk][e] @ (1.0 - ye))
    return num / denom


def main():
    cells = build_cells()

    # attach A and S from the frozen head-to-head export (same pdb_geometry anchors)
    extra = {}
    f = CAMP / "analysis/compare_A_vs_S.csv"
    for r in csv.DictReader(f.open()):
        extra[(r["enzyme"], r["acid_code"])] = (float(r["A_deg"]), float(r["S"]))
    keep = []
    for c in cells:
        k = (c["enzyme"], c["donor"])
        if k not in extra:
            continue
        c["s1_A_frozen"], c["s1_S_frozen"] = extra[k]
        keep.append(c)
    cells = keep

    # battery: the evaluable singles from amendment 6 (complete data only), plus A and S
    global ORIENT_SIGN
    ORIENT_SIGN = dict(ORIENT)
    ORIENT_SIGN["s1_A_frozen"] = +1          # larger angle = more in-line
    ORIENT_SIGN["s1_S_frozen"] = +1          # larger S = more productive geometry
    looks = [k for k in ORIENT if all(c.get(k) is not None for c in cells)]
    looks += ["s1_A_frozen", "s1_S_frozen"]
    dropped = [k for k in ORIENT if k not in looks]

    y = np.array([c["active"] for c in cells], float)
    print(f"n = {len(cells)} labelled cells; {int(y.sum())} detected")
    print(f"battery: {len(looks)} looks" + (f"; dropped (incomplete): {dropped}" if dropped else ""))

    enzymes, idx, mats = concordance_parts(cells, looks)
    denom = sum(int(y[idx[e]].sum()) * int(len(idx[e]) - y[idx[e]].sum()) for e in enzymes)
    print("within-enzyme informative pairs: "
          + " | ".join(f"{e} {int(y[idx[e]].sum())}x{int(len(idx[e]) - y[idx[e]].sum())}"
                       for e in enzymes) + f"  -> {denom}")

    obs = {lk: C_of(y, enzymes, idx, mats, lk, denom) for lk in looks}
    rank = sorted(looks, key=lambda k: -obs[k])
    print("\n--- observed within-enzyme concordance (UNADJUSTED) ---")
    for i, lk in enumerate(rank[:12], 1):
        print(f"  {i:>2} {lk:<26} {obs[lk]:.4f}")
    best = rank[0]

    # ---- margin-preserving null -----------------------------------------------------------
    panels = sorted({c["panel"] for c in cells})
    pos = {(c["enzyme"], c["panel"]): i for i, c in enumerate(cells)}
    R, Cn = len(enzymes), len(panels)
    M = np.full((R, Cn), -1, int)                     # -1 = cell not observed
    for (e, p), i in pos.items():
        M[enzymes.index(e), panels.index(p)] = int(y[i])
    print(f"\nactivity matrix {R}x{Cn}; observed cells {int((M >= 0).sum())}, "
          f"missing {int((M < 0).sum())}")
    print("  row sums (enzyme actives): "
          + ", ".join(f"{e}:{int(M[i][M[i] >= 0].sum())}" for i, e in enumerate(enzymes)))

    rng = np.random.default_rng(SEED)
    W = M.copy()

    def swaps(n):
        """Checkerboard swaps; both margins invariant. Only fully observed 2x2 blocks move."""
        done = 0
        for _ in range(n * 6):                        # oversample attempts; many 2x2 are invalid
            r1, r2 = rng.integers(0, R, 2)
            c1, c2 = rng.integers(0, Cn, 2)
            if r1 == r2 or c1 == c2:
                continue
            a, b, c, d = W[r1, c1], W[r1, c2], W[r2, c1], W[r2, c2]
            if min(a, b, c, d) < 0:
                continue
            if a == 1 and b == 0 and c == 0 and d == 1:
                W[r1, c1] = W[r2, c2] = 0; W[r1, c2] = W[r2, c1] = 1; done += 1
            elif a == 0 and b == 1 and c == 1 and d == 0:
                W[r1, c1] = W[r2, c2] = 1; W[r1, c2] = W[r2, c1] = 0; done += 1
            if done >= n:
                break
        return done

    swaps(BURN)
    yv = np.empty(len(cells))
    null_max = np.empty(NDRAW)
    for t in range(NDRAW):
        swaps(BETWEEN)
        for (e, p), i in pos.items():
            yv[i] = W[enzymes.index(e), panels.index(p)]
        null_max[t] = max(C_of(yv, enzymes, idx, mats, lk, denom) for lk in looks)

    p_sel = float((null_max >= obs[best]).mean())
    print(f"\n--- SELECTION-CORRECTED, BOTH-MARGIN-PRESERVING NULL ({NDRAW} draws) ---")
    print(f"  best observed  C = {obs[best]:.4f}  ({best})")
    print(f"  null max: median {np.median(null_max):.4f}, 95th {np.quantile(null_max, .95):.4f}, "
          f"max {null_max.max():.4f}")
    print(f"  p_sel = {p_sel:.4f}   DECISION: {'FINDING' if p_sel < 0.05 else 'NON-FINDING'}")

    # ---- declared controls ------------------------------------------------------------------
    print("\n--- declared controls ---")
    bv = [c[best] * ORIENT_SIGN[best] for c in cells]
    rho, pr = spearman(bv, [c["s1_occupancy"] for c in cells])
    print(f"  rho({best}, s1_occupancy) = {rho:+.3f} (p = {pr:.4f})")
    for e in enzymes:
        sub = [c for c in cells if c["enzyme"] != e]
        ys = np.array([c["active"] for c in sub], float)
        ez, ix, mt = concordance_parts(sub, [best])
        dn = sum(int(ys[ix[q]].sum()) * int(len(ix[q]) - ys[ix[q]].sum()) for q in ez)
        if dn == 0:
            print(f"  without {e:<5} no informative pairs remain")
            continue
        print(f"  without {e:<5} C = {C_of(ys, ez, ix, mt, best, dn):.4f}  ({dn} pairs)")
    seen, uniq = set(), []
    for c in cells:
        if c["s1_stem"] in seen:
            continue
        seen.add(c["s1_stem"]); uniq.append(c)
    yu = np.array([c["active"] for c in uniq], float)
    ez, ix, mt = concordance_parts(uniq, [best])
    dn = sum(int(yu[ix[q]].sum()) * int(len(ix[q]) - yu[ix[q]].sum()) for q in ez)
    print(f"  unique step-1 ensembles (n={len(uniq)}): "
          + (f"C = {C_of(yu, ez, ix, mt, best, dn):.4f} ({dn} pairs)" if dn else "no pairs"))


if __name__ == "__main__":
    main()
