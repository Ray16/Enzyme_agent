#!/usr/bin/env python3
"""Stage 2 of amendment 6: the declared descriptor search, with its selection-corrected null.

Deliberately a SEPARATE file from mine_descriptors.py, so that the descriptor definitions cannot be
edited in response to a score without that edit showing up as a change to the other file.

What it does, exactly as amendment 6 declares:
  * builds one row per (enzyme, acid donor, amine acceptor) labelled cell
  * orients every descriptor so "higher = mechanistically more favourable" (orientation fixed in
    ORIENT below, not inferred from the data -- an unoriented AUROC lets a sign flip count as a hit
    and silently doubles the battery)
  * scores each single descriptor by AUROC, and each cross-step PAIR by the fit-free rule
    min(within-stratum percentile rank of step-1 descriptor, same for step-2): a cell scores well
    only if it is good at BOTH steps. No fitted weights, no tuned thresholds.
  * computes p_sel by permuting labels WITHIN ACID STRATUM and retaining, for each draw, the
    MAXIMUM AUROC over the whole battery. A per-descriptor p-value is meaningless with ~170 looks.

Reports the mandatory confound controls for the winner whatever p_sel says: correlation with
docking occupancy at both steps, AUROC recomputed within each fold family, and a unique-step-1
sensitivity check (cells sharing an acid donor reuse one step-1 ensemble).
"""
import csv, math, pathlib, sys

import numpy as np

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
from plot_panels import SOURCES, load  # noqa: E402
from build_step2_inputs import STEP2_PAIRS  # noqa: E402

NDRAW = 20000
SEED = 20261022          # fixed so the null is reproducible
FOLD = {"DdaG": "ANL", "SfaB": "ANL", "AsbA": "NIS", "AsbB": "NIS", "AcsA": "NIS", "DesD": "NIS"}

# +1 = higher is mechanistically more favourable; -1 = lower is. FIXED BEFORE RUNNING.
ORIENT = {
    "s1_A": +1,                    # closer to in-line 180
    "s1_Dh": -1,                   # shorter attack distance
    "s1_occupancy": +1,
    "s1_min_d": -1,
    "s1_nac_frac": +1,
    "s1_best_angle_gated": +1,
    "s1_mg_carboxylate": -1,       # Mg assisting the nucleophile
    "s1_mg_phosphate": -1,         # correct cofactor placement
    "s1_carboxylate_hbonds": +1,
    "s1_cation_contacts": +1,
    "s1_donor_contacts": +1,
    "s1_centroid_spread": -1,      # converged docking
    "s1_pocket_apolar": -1,        # a dicarboxylate wants a polar pocket
    "s2_BD_dev": -1,
    "s2_Dh": -1,
    "s2_occupancy": +1,
    "s2_min_d": -1,
    "s2_nac_frac": +1,
    "s2_best_bd_gated": -1,
    "s2_oxyanion_hbonds": +1,
    "s2_leaving_anti": +1,         # nucleophile anti to the AMP leaving group
    "s2_n_burial": +1,
    "s2_acceptor_contacts": +1,
    "s2_distal_charge": -1,
    "s2_centroid_spread": -1,
    "s2_acyl_intact": +1,
}


def rankdata(v):
    """Average ranks, 1-based (ties share the mean rank) -- scipy-free."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def pct_within(values, strata):
    """Within-stratum percentile rank in (0,1); higher value -> higher percentile."""
    out = np.zeros(len(values), float)
    for s in set(strata):
        idx = [i for i, t in enumerate(strata) if t == s]
        sub = rankdata([values[i] for i in idx])
        k = len(idx)
        for pos, i in enumerate(idx):
            out[i] = (sub[pos] - 0.5) / k
    return out


def auroc_from_ranks(R, y):
    """AUROC for every row of R (n_desc x n_cells) against binary y."""
    y = np.asarray(y, float)
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.full(R.shape[0], np.nan)
    return (R @ y - npos * (npos + 1) / 2.0) / (npos * nneg)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        return float("nan"), float("nan")
    rho = float((ra * rb).sum() / den)
    n = len(a)
    if n > 3 and abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # two-sided p from a t distribution via its incomplete beta, series-free approximation
        p = 2 * (1 - _studt_cdf(abs(t), n - 2))
    else:
        p = float("nan")
    return rho, p


def _studt_cdf(t, df):
    x = df / (df + t * t)
    return 1 - 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x) / a
    return 1 - _betainc(b, a, 1 - x)      # symmetry; recurse on swapped parameters


def _cf(a, b, x, iters=300):
    f, c, d = 1.0, 1.0, 0.0
    for i in range(iters + 1):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1 - c * d) < 1e-10:
            break
    return f - 1.0


# --------------------------------------------------------------------------------------------
def build_cells():
    d1 = {}
    for r in csv.DictReader((CAMP / "analysis/descriptors_step1.csv").open()):
        if r["n_mg"] == "1":
            d1[(r["enzyme"], r["acid_code"])] = r
    d2 = {(r["enzyme"], r["donor_code"], r["acceptor_code"]): r
          for r in csv.DictReader((CAMP / "analysis/descriptors_step2.csv").open())}

    labels = {}
    for _n, (p, _pp, _o) in SOURCES.items():
        if p.exists():
            labels.update(load(p))

    pair2panel = {}
    for donor, acceptor, lab in STEP2_PAIRS:
        if "other regiochemistry" in lab:      # a product the panel does not measure
            continue
        pair2panel[(donor, acceptor)] = f"Figure {lab.split()[0]}"

    cells = []
    for (donor, acceptor), panel in pair2panel.items():
        lab = labels.get(panel)
        if not lab:
            continue
        for enz, v in lab.items():
            a, b = d1.get((enz, donor)), d2.get((enz, donor, acceptor))
            if not a or not b:
                continue
            row = dict(panel=panel, enzyme=enz, donor=donor, acceptor=acceptor,
                       fold=FOLD[enz], active=1 if not v["below_LOD"] else 0,
                       s1_stem=a["stem"])
            for k in ORIENT:
                src = a if k.startswith("s1_") else b
                row[k] = float(src[k]) if src.get(k, "") != "" else None
            cells.append(row)
    return cells


def main():
    cells = build_cells()
    y = np.array([c["active"] for c in cells], float)
    strata = [c["donor"] for c in cells]
    print(f"n = {len(cells)} labelled (enzyme, donor, acceptor) cells; "
          f"{int(y.sum())} detected, {int(len(y) - y.sum())} below LOD")
    print(f"acid strata: " + ", ".join(f"{s}:{sum(1 for t in strata if t == s)}"
                                       for s in sorted(set(strata))))

    # ---- which declared descriptors are actually evaluable on the common cell set ------------
    s1_names = [k for k in ORIENT if k.startswith("s1_")]
    s2_names = [k for k in ORIENT if k.startswith("s2_")]
    dropped = {}
    usable = []
    for k in s1_names + s2_names:
        vals = [c[k] for c in cells]
        if any(v is None for v in vals):
            dropped[k] = f"missing for {sum(1 for v in vals if v is None)}/{len(vals)} cells"
        elif len(set(vals)) < 2:
            dropped[k] = f"DEGENERATE -- constant at {vals[0]}"
        else:
            usable.append(k)
    if dropped:
        print("\nDeclared descriptors NOT evaluable on the common cell set "
              "(amendment 6 declared 26; this is a recorded deviation):")
        for k, why in dropped.items():
            print(f"  {k:24s} {why}")
        print("  Excluding them SHRINKS the battery, which makes p_sel slightly less conservative,"
              "\n  so each is additionally searched on its own subset at the end.")

    u1 = [k for k in usable if k.startswith("s1_")]
    u2 = [k for k in usable if k.startswith("s2_")]

    # ---- build the score matrix: singles, then fit-free pairs --------------------------------
    names, scores = [], []
    for k in usable:
        names.append(k)
        scores.append(np.array([c[k] for c in cells], float) * ORIENT[k])
    pct = {k: pct_within(np.array([c[k] for c in cells], float) * ORIENT[k], strata)
           for k in usable}
    for a in u1:
        for b in u2:
            names.append(f"{a} + {b}")
            scores.append(np.minimum(pct[a], pct[b]))     # good at BOTH steps
    S = np.vstack(scores)
    R = np.vstack([rankdata(s) for s in S])
    n_looks = len(names)
    print(f"\nbattery actually searched: {len(u1)} step-1 + {len(u2)} step-2 singles "
          f"+ {len(u1) * len(u2)} pairs = {n_looks} looks")

    obs = auroc_from_ranks(R, y)
    order = np.argsort(-obs)

    print("\nTop 15 by observed AUROC (UNADJUSTED -- these are not p-values yet):")
    print(f"  {'#':>3s} {'descriptor / pair':52s} {'AUROC':>7s}")
    for rank, i in enumerate(order[:15], 1):
        print(f"  {rank:3d} {names[i]:52s} {obs[i]:7.3f}")

    # ---- the declared null: max AUROC over the WHOLE battery, labels permuted within stratum --
    rng = np.random.default_rng(SEED)
    idx_by_s = {s: np.array([i for i, t in enumerate(strata) if t == s]) for s in set(strata)}
    L = np.empty((NDRAW, len(cells)), float)
    for d in range(NDRAW):
        yy = np.empty(len(cells), float)
        for s, idx in idx_by_s.items():
            yy[idx] = rng.permutation(y[idx])
        L[d] = yy
    npos = y.sum(); nneg = len(y) - npos
    null = (L @ R.T - npos * (npos + 1) / 2.0) / (npos * nneg)     # (NDRAW x n_looks)
    null_max = null.max(axis=1)

    best_i = int(order[0])
    p_sel = float((null_max >= obs[best_i]).mean())
    print(f"\n--- SELECTION-CORRECTED NULL ({NDRAW} draws, labels permuted within acid stratum) ---")
    print(f"  best observed AUROC            = {obs[best_i]:.3f}   ({names[best_i]})")
    print(f"  null distribution of the MAX   : median {np.median(null_max):.3f}, "
          f"95th pct {np.percentile(null_max, 95):.3f}, max {null_max.max():.3f}")
    print(f"  p_sel = {p_sel:.4f}")
    print(f"  DECISION (amendment 6.3): "
          f"{'FINDING' if p_sel < 0.05 else 'NOT A FINDING -- p_sel >= 0.05'}")
    if p_sel >= 0.05:
        print("  The best AUROC is reported, but with ~{} looks a value at or below the null's\n"
              "  95th percentile ({:.3f}) is what selection alone produces.".format(
                  n_looks, np.percentile(null_max, 95)))

    # per-descriptor unadjusted p, for contrast only
    p_unadj = (null >= obs[None, :]).mean(axis=0)
    n_nominal = int((p_unadj < 0.05).sum())
    print(f"\n  For contrast: {n_nominal}/{n_looks} looks clear an UNADJUSTED p < 0.05 "
          f"(expected by chance alone: ~{0.05 * n_looks:.0f}).")

    # ---- mandatory confound controls on the winner -------------------------------------------
    print("\n--- MANDATORY CONFOUND CONTROLS on the best look (amendment 6.4) ---")
    bs = S[best_i]
    for occ in ("s1_occupancy", "s2_occupancy"):
        rho, p = spearman(bs, [c[occ] for c in cells])
        print(f"  rho(best, {occ:14s}) = {rho:+.3f}  (p = {p:.4f})"
              + ("   <-- tracks docking, not catalysis" if abs(rho) > 0.5 else ""))
    for fam in ("ANL", "NIS"):
        idx = [i for i, c in enumerate(cells) if c["fold"] == fam]
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            a = auroc_from_ranks(rankdata(bs[idx])[None, :], yy)[0]
            print(f"  AUROC within {fam} only ({len(idx)} cells, {int(yy.sum())} active) = {a:.3f}")
        else:
            print(f"  AUROC within {fam}: not evaluable ({len(idx)} cells, "
                  f"{int(yy.sum())} active -- one class only)")
    # unique step-1 ensembles only (cells sharing an acid donor reuse one step-1 fold)
    seen, keep = set(), []
    for i, c in enumerate(cells):
        if c["s1_stem"] not in seen:
            seen.add(c["s1_stem"]); keep.append(i)
    yy = y[keep]
    if 0 < yy.sum() < len(yy):
        a = auroc_from_ranks(rankdata(bs[keep])[None, :], yy)[0]
        print(f"  AUROC on unique step-1 ensembles only ({len(keep)} cells, "
              f"{int(yy.sum())} active) = {a:.3f}")

    # ---- ROBUSTNESS: is the winner carried by one enzyme, or by training-set recall? ---------
    # DesD is 10/10 active AND its native complex (7TGM) is very likely in Boltz-2's training set,
    # so a signal that disappears without DesD is recall, not prediction. Leave-one-enzyme-out is
    # the check that distinguishes them.
    print("\n--- ROBUSTNESS: leave one enzyme out (winner only) ---")
    for drop in sorted(FOLD):
        idx = [i for i, c in enumerate(cells) if c["enzyme"] != drop]
        yy = y[idx]
        if not (0 < yy.sum() < len(yy)):
            print(f"  without {drop:5s}: one class only -- not evaluable"); continue
        a = auroc_from_ranks(rankdata(bs[idx])[None, :], yy)[0]
        sub = [strata[i] for i in idx]
        ibs = {t: np.array([j for j, u in enumerate(sub) if u == t]) for t in set(sub)}
        cnt = 0
        for _d in range(5000):
            z = np.empty(len(idx), float)
            for t, ii in ibs.items():
                z[ii] = rng.permutation(yy[ii])
            cnt += auroc_from_ranks(rankdata(bs[idx])[None, :], z)[0] >= a
        print(f"  without {drop:5s}: {len(idx)} cells, {int(yy.sum())} active, "
              f"AUROC = {a:.3f}, single-look within-stratum p = {cnt/5000:.4f}")

    # p_sel recomputed on unique step-1 ensembles -- the non-independence sensitivity, corrected
    print("\n--- p_sel recomputed over the WHOLE battery on unique step-1 ensembles only ---")
    yk = y[keep]
    Rk = np.vstack([rankdata(S[i][keep]) for i in range(len(names))])
    obs_k = auroc_from_ranks(Rk, yk)
    sub = [strata[i] for i in keep]
    ibs = {t: np.array([j for j, u in enumerate(sub) if u == t]) for t in set(sub)}
    nmax = np.empty(NDRAW)
    npk = yk.sum(); nnk = len(yk) - npk
    Lk = np.empty((NDRAW, len(keep)), float)
    for d in range(NDRAW):
        z = np.empty(len(keep), float)
        for t, ii in ibs.items():
            z[ii] = rng.permutation(yk[ii])
        Lk[d] = z
    nullk = (Lk @ Rk.T - npk * (npk + 1) / 2.0) / (npk * nnk)
    nmax = nullk.max(axis=1)
    bk = int(np.argmax(obs_k))
    print(f"  best on this subset = {names[bk]} at AUROC {obs_k[bk]:.3f}")
    print(f"  the overall winner ({names[best_i]}) scores {obs_k[best_i]:.3f} here")
    print(f"  null max: median {np.median(nmax):.3f}, 95th {np.percentile(nmax,95):.3f}")
    print(f"  p_sel (best on subset) = {(nmax >= obs_k[bk]).mean():.4f}")

    # ---- write the full table ----------------------------------------------------------------
    out = CAMP / "analysis/descriptor_search.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "descriptor_or_pair", "kind", "auroc", "p_unadjusted",
                    "p_sel_of_best", "null_max_p95", "n_cells", "n_active", "n_looks"])
        for rank, i in enumerate(order, 1):
            kind = "pair" if " + " in names[i] else ("step1" if names[i].startswith("s1_")
                                                     else "step2")
            w.writerow([rank, names[i], kind, round(float(obs[i]), 4),
                        round(float(p_unadj[i]), 4), round(p_sel, 4),
                        round(float(np.percentile(null_max, 95)), 4),
                        len(cells), int(y.sum()), n_looks])
    print(f"\nwrote {out}")

    # ---- the excluded descriptors, searched on their own subset -------------------------------
    for k, why in dropped.items():
        if "DEGENERATE" in why:
            print(f"\n{k}: not searchable -- {why}")
            continue
        idx = [i for i, c in enumerate(cells) if c[k] is not None]
        yy = y[idx]
        if not (0 < yy.sum() < len(yy)):
            print(f"\n{k}: subset has one class only -- not evaluable")
            continue
        v = np.array([cells[i][k] for i in idx], float) * ORIENT[k]
        a = auroc_from_ranks(rankdata(v)[None, :], yy)[0]
        sub_str = [strata[i] for i in idx]
        ibs = {s: np.array([j for j, t in enumerate(sub_str) if t == s]) for s in set(sub_str)}
        cnt = 0
        for _d in range(5000):
            z = np.empty(len(idx), float)
            for s, ii in ibs.items():
                z[ii] = rng.permutation(yy[ii])
            if 0 < z.sum() < len(z):
                cnt += auroc_from_ranks(rankdata(v)[None, :], z)[0] >= a
        print(f"\n{k} on its own subset ({len(idx)} cells, {int(yy.sum())} active): "
              f"AUROC = {a:.3f}, within-stratum p = {cnt / 5000:.4f} "
              f"(single look, NOT selection-corrected)")


if __name__ == "__main__":
    main()
