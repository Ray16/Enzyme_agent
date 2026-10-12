#!/usr/bin/env python3
"""The pre-registered proxy test, exactly as written in PREREGISTRATION.md.

Run ONCE, in a single pass. Everything below -- the cohort, the five
descriptors, the primary, the selection correction, the covariates -- was fixed
before any structure was folded. This script adds nothing to that list; if a
sixth descriptor or a second cutoff ever appears here, the test has become
exploratory and must be reported as such.

The headline is the SELECTION-CORRECTED p: the probability that the maximum
|rho| over all five descriptors, under label permutation, exceeds the observed
maximum. The winner's nominal p is reported beside it and is not the result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PANEL = HERE / "panel"
N_PERM = 10_000
SEED = 0

DESCRIPTORS = [
    ("D1_near_attack_fraction", +1, "near-attack fraction"),
    ("D2_proximity_weighted_angle", +1, "proximity-weighted angle"),
    ("D3_d_nuc_carbonyl", -1, "d(O_nuc,C_carb)"),
    ("D4_leaving_group_alignment", +1, "leaving-group alignment"),
    ("D5_oxyanion_occupancy", +1, "oxyanion occupancy"),
]
PRIMARY = "D1_near_attack_fraction"


def partial_spearman(x, y, z):
    """Spearman of x,y with z partialled out, on ranks."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    return stats.pearsonr(ex, ey)


def main() -> int:
    labels = json.loads((PANEL / "activity_labels.json").read_text())
    desc = {d["enzyme"]: d for d in json.loads((HERE / "descriptors.json").read_text())}

    rows = []
    for m in labels["measurements"]:
        act = m.get("purp1_specific_activity_umol_min_umol")
        if act is None or isinstance(act, str):
            continue
        d = desc.get(m["enzyme"])
        if d is None or d.get("n_poses", 0) < 50:
            print(f"  MISSING ensemble for labelled enzyme {m['enzyme']} -- cohort incomplete")
            return 1
        rows.append({
            "enzyme": m["enzyme"], "act": float(act), "log10": np.log10(float(act)),
            "tm": m.get("tm_c"), "fam": m.get("family_interpro", ""),
            **{k: d[k] for k, _, _ in DESCRIPTORS},
        })
    rows.sort(key=lambda r: -r["act"])
    n = len(rows)
    print(f"cohort n = {n}  (pre-registered: 11)")
    print(f"activity span: {rows[0]['act']:g} .. {rows[-1]['act']:g} "
          f"({rows[0]['log10'] - rows[-1]['log10']:.1f} orders of magnitude)\n")

    y = np.array([r["log10"] for r in rows])

    # --- degeneracy check, declared failure mode 3 -----------------------
    print("dynamic range (declared failure mode 3: saturation is NOT a weak hit)")
    degenerate = set()
    for key, _, lab in DESCRIPTORS:
        v = np.array([r[key] for r in rows], float)
        uniq = len(np.unique(v))
        flat = uniq <= max(3, n // 4)
        if flat:
            degenerate.add(key)
        print(f"  {key:30s} {uniq:>2}/{n} distinct  range {v.min():.3f}..{v.max():.3f}"
              f"{'   <-- NO DYNAMIC RANGE' if flat else ''}")
    print()

    # --- observed correlations ------------------------------------------
    obs = {}
    print("Spearman vs log10(specific activity)")
    for key, sign, lab in DESCRIPTORS:
        v = np.array([r[key] for r in rows], float)
        rho, p2 = stats.spearmanr(v, y)
        # one-sided in the PRE-REGISTERED direction
        p1 = p2 / 2 if (rho * sign) > 0 else 1 - p2 / 2
        obs[key] = rho
        tag = " [PRIMARY]" if key == PRIMARY else ""
        tag += "  (degenerate)" if key in degenerate else ""
        print(f"  {key:30s} rho={rho:+.3f}  p_1s={p1:.4f}  p_2s={p2:.4f}"
              f"  predicted {'+' if sign > 0 else '-'}{tag}")
    print()

    # --- selection-corrected null: max |rho| over all five --------------
    rng = np.random.default_rng(SEED)
    mats = np.array([[r[k] for k, _, _ in DESCRIPTORS] for r in rows], float)  # n x 5
    obs_max = max(abs(v) for v in obs.values())
    null_max = np.empty(N_PERM)
    for i in range(N_PERM):
        yp = rng.permutation(y)
        null_max[i] = max(abs(stats.spearmanr(mats[:, j], yp)[0]) for j in range(5))
    p_sel = float((null_max >= obs_max).mean())
    best = max(obs, key=lambda k: abs(obs[k]))

    print("SELECTION-CORRECTED TEST  (the headline)")
    print(f"  best descriptor      : {best}  rho={obs[best]:+.3f}")
    print(f"  observed max |rho|   : {obs_max:.3f}")
    print(f"  null max |rho| (95th): {np.percentile(null_max, 95):.3f}   "
          f"median {np.median(null_max):.3f}")
    print(f"  p_selection_corrected: {p_sel:.4f}   <-- REPORT THIS")
    print(f"  (winner's nominal two-sided p = "
          f"{stats.spearmanr(np.array([r[best] for r in rows]), y)[1]:.4f}, NOT the result)")
    print()

    verdict = "HIT" if p_sel < 0.05 else "NULL"
    print(f"VERDICT: {verdict}")

    # --- covariates, only meaningful if something survived ---------------
    if p_sel < 0.05:
        v = np.array([r[best] for r in rows], float)
        tm = np.array([r["tm"] if r["tm"] is not None else np.nan for r in rows], float)
        if not np.isnan(tm).any():
            r_p, p_p = partial_spearman(v, y, tm)
            print(f"  Tm partialled out: rho={r_p:+.3f} p={p_p:.4f}")
        amid = [i for i, r in enumerate(rows) if "Amidase" in r["fam"]]
        if len(amid) >= 5:
            ra, pa = stats.spearmanr(v[amid], y[amid])
            print(f"  within amidases (n={len(amid)}): rho={ra:+.3f} p={pa:.4f} "
                  f"(needs |rho|>=0.71)")
    else:
        print("  Covariate checks not run: nothing survived the selection correction,")
        print("  and conditioning a null on covariates only invites a second look.")

    print("\nper-enzyme table (sorted by measured activity)")
    print(f"  {'enzyme':<12}{'act':>10}  " + "".join(f"{k.split('_')[0]:>8}" for k, _, _ in DESCRIPTORS))
    for r in rows:
        print(f"  {r['enzyme']:<12}{r['act']:>10.5g}  "
              + "".join(f"{r[k]:>8.2f}" for k, _, _ in DESCRIPTORS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
