#!/usr/bin/env python3
"""Head-to-head of the two first-step geometry metrics on THIS panel.

Why this script exists: `A` (inverse_distance_attack_angle_v1) replaced the Gaussian score `S`
as the project's standing ranking policy, but the two have never been compared on the same data.
On the ASX panel `S` was SELECTED (81 settings, retrospective), so its edge there is in-sample and
uninformative. This panel is out-of-sample for `S` -- its four constants were frozen before these
folds existed -- and `A` is parameter-free, so neither metric has seen these labels. That makes
this the only clean comparison available.

Both scorers are IMPORTED from their frozen implementations, never reimplemented, and both read
the same per-pose (distance, angle) pairs from the same `pdb_geometry` anchors that produced the
campaign's recorded numbers. A difference between them is therefore a difference in the scoring
function and nothing else.

  A = sum(angle/d) / sum(1/d)                      parameter-free, scale-invariant, in DEGREES
  S = sum(w*q) / (0.1 + sum(w))                    4 frozen constants, bounded [0,1), validity-aware
      w = sigmoid((3.5 - d)/0.25),  q = exp(-0.5*((180 - angle)/30)^2)

Endpoint and strata are the campaign's declared ones: detected vs below-LOD, permuted/compared
WITHIN acid stratum. No new endpoint is introduced here.
"""
import csv, os, pathlib, sys

os.environ.setdefault("OMP_NUM_THREADS", "1")        # courtesy on a shared node
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CAMP))

from structural_modeling_agent.proxies.attack_angle import (  # noqa: E402
    pdb_geometry, weighted_attack_angle)
# the frozen Gaussian score, imported from the ASX panel's own implementation
sys.path.insert(0, str(ROOT / "exp_data/amide_synthetase/scripts"))
from continuous_score import geometry_score  # noqa: E402

RESULTS = CAMP / "cofolds/step1/results"
GATE_A = 4.0


def per_pose(stem):
    """Return (distances, angles) for one step-1 ensemble, from the frozen anchors."""
    pred = RESULTS / stem / f"boltz_results_{stem}" / "predictions" / stem
    ds, angs = [], []
    for p in sorted(pred.glob(f"{stem}_model_*.pdb")):
        try:
            d, a = pdb_geometry(p, substrate_chain="D", nucleotide_chain="B",
                                phosphate="PA", bridge="O3A")
        except (ValueError, KeyError):
            continue
        ds.append(d); angs.append(a)
    return ds, angs


def auroc(scores, y):
    """Mann-Whitney AUROC with midranks; `scores` oriented so larger = predicted active."""
    s = np.asarray(scores, float); y = np.asarray(y, float)
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); i = 0
    sv = s[order]
    while i < len(sv):                                  # midranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return None
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def within_stratum(cells, key):
    """Fraction of concordant active/inactive pairs formed WITHIN an acid stratum."""
    good = bad = tie = 0
    for s in {c["donor"] for c in cells}:
        grp = [c for c in cells if c["donor"] == s]
        for p in [c for c in grp if c["active"] == 1]:
            for n in [c for c in grp if c["active"] == 0]:
                dv = p[key] - n[key]
                good += dv > 0; bad += dv < 0; tie += dv == 0
    tot = good + bad + tie
    return ((good + 0.5 * tie) / tot if tot else None), tot


def main():
    jm = {r["stem"]: r for r in csv.DictReader((CAMP / "cofolds/step1/job_map.csv").open())}
    rows, skipped = {}, []
    for stem, meta in sorted(jm.items()):
        if meta.get("n_mg") != "1":              # the declared panel is the 1-Mg one
            continue
        ds, angs = per_pose(stem)
        if len(ds) != 50:
            # geometry_score contracts on exactly 50 poses; a short ensemble is reported,
            # never silently padded -- padding would invent geometry.
            skipped.append((stem, len(ds)))
            continue
        A = weighted_attack_angle(ds, angs)["weighted_attack_angle_deg"]
        S, total_w, eff = geometry_score(ds, angs)
        rows[(meta["enzyme"], meta["acid_code"])] = dict(
            stem=stem, A=float(A), S=float(S), total_w=float(total_w), eff=float(eff),
            n_gated=sum(1 for d in ds if d < GATE_A), min_d=min(ds))

    print(f"scored {len(rows)} one-Mg step-1 ensembles"
          + (f"; skipped {len(skipped)} short: {skipped}" if skipped else ""))

    out = CAMP / "analysis/compare_A_vs_S.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["enzyme", "acid_code", "stem", "A_deg", "S", "total_weight",
                    "effective_poses", "n_gated", "min_d"])
        for (e, a), v in sorted(rows.items()):
            w.writerow([e, a, v["stem"], round(v["A"], 2), round(v["S"], 4),
                        round(v["total_w"], 3), round(v["eff"], 2), v["n_gated"],
                        round(v["min_d"], 3)])
    print(f"wrote {out.relative_to(CAMP)}")

    # ---- join to the declared labels, using the campaign's own cell builder -------------
    from search_descriptors import build_cells
    cells = []
    for c in build_cells():
        v = rows.get((c["enzyme"], c["donor"]))
        if v is None:
            continue
        cells.append(dict(c, A=v["A"], S=v["S"], n_gated=v["n_gated"]))
    y = [c["active"] for c in cells]
    print(f"\nlabelled cells: {len(cells)}  ({sum(y)} detected, {len(y) - sum(y)} below LOD)")

    print("\n--- POOLED AUROC (all labelled cells; shared step-1 folds counted once per cell) ---")
    for k in ("A", "S"):
        print(f"  {k:<2} {auroc([c[k] for c in cells], y):.4f}")

    print("\n--- WITHIN-ACID-STRATUM concordance (the comparison the declared null tests) ---")
    for k in ("A", "S"):
        v, n = within_stratum(cells, k)
        print(f"  {k:<2} {v:.4f}  ({n} informative active/inactive pairs)")

    print("\n--- deduplicated to UNIQUE step-1 ensembles (shared acid folds counted once) ---")
    seen, uniq = set(), []
    for c in cells:
        if c["s1_stem"] in seen:
            continue
        seen.add(c["s1_stem"]); uniq.append(c)
    yu = [c["active"] for c in uniq]
    print(f"  n = {len(uniq)} ({sum(yu)} detected)")
    for k in ("A", "S"):
        print(f"  {k:<2} {auroc([c[k] for c in uniq], yu):.4f}")

    print("\n--- the phantom-angle cells: what each metric says where nothing docked ---")
    for c in sorted(cells, key=lambda c: c["n_gated"]):
        if c["n_gated"] > 0:
            continue
        print(f"  {c['enzyme']:<6} donor {c['donor']:<4} n_gated=0/50 ->"
              f"  A = {c['A']:6.1f} deg      S = {c['S']:.4f}")


if __name__ == "__main__":
    main()
