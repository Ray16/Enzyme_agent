#!/usr/bin/env python3
"""Score the step-1 ensembles with the FROZEN proximity-weighted attack angle `A`.

Metric is imported, never reimplemented:
  structural_modeling_agent.proxies.attack_angle  (METRIC_ID inverse_distance_attack_angle_v1)
    A   = sum(angle_i / d_i) / sum(1 / d_i)
    D_h = n / sum(1 / d_i)
with d = nearest substrate carboxylate O -> ATP PA, angle = O...PA-O3A (PPi leaving axis).

Chain map is fixed by the build (A protein, B ATP, C[/E..] MG, D acid donor). Note Boltz renames
every CCD ligand to `LIG` in the PDB and RENUMBERS the substrate's oxygens (adipate came back as
O11-O14), so neither residue name nor atom name can be trusted -- the frozen extractor identifies
the carboxylate by element and bonding, which is why it survives that.

Reports A, D_h and n_gated together: n_gated (poses with d < 4 A) is a VALIDITY PRECONDITION, not
a ranking input. An ensemble with n_gated == 0 is reported as "did not dock" and flagged
non-interpretable -- but it is NOT dropped, because silently dropping non-dockers is how "did the
ligand dock at all" masquerades as a catalytic signal.
"""
import csv, math, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP.parents[1]))
from structural_modeling_agent.proxies.attack_angle import (  # noqa: E402
    METRIC_ID, pdb_geometry, weighted_attack_angle)

GATE_A = 4.0          # validity precondition only
RESULTS = CAMP / "cofolds/step1/results"


def score_one(stem):
    pred = RESULTS / stem / f"boltz_results_{stem}" / "predictions" / stem
    pdbs = sorted(pred.glob(f"{stem}_model_*.pdb"))
    if len(pdbs) < 50:
        return None, f"incomplete ensemble ({len(pdbs)} poses)"
    ds, angs, bad = [], [], 0
    for p in pdbs:
        try:
            d, a = pdb_geometry(p, substrate_chain="D", nucleotide_chain="B",
                                phosphate="PA", bridge="O3A")
        except (ValueError, KeyError):
            bad += 1
            continue
        ds.append(d); angs.append(a)
    if not ds:
        return None, f"no pose yielded a geometry ({bad} extraction failures)"
    res = weighted_attack_angle(ds, angs)
    n_gated = sum(1 for d in ds if d < GATE_A)
    return dict(
        stem=stem, n_poses=res["n_poses"], n_extract_fail=bad,
        A_deg=round(res["weighted_attack_angle_deg"], 2),
        D_h_A=round(res["harmonic_distance_A"], 3),
        n_gated=n_gated, gated_frac=round(n_gated / len(ds), 3),
        min_d=round(min(ds), 3), median_d=round(sorted(ds)[len(ds) // 2], 3),
        max_angle=round(max(angs), 1),
        docked=n_gated > 0,
    ), None


def main():
    jm = {r["stem"]: r for r in csv.DictReader((CAMP / "cofolds/step1/job_map.csv").open())}
    rows, skipped = [], []
    for stem in sorted(jm):
        res, why = score_one(stem)
        if res is None:
            skipped.append((stem, why)); continue
        res.update(enzyme=jm[stem]["enzyme"], acid_code=jm[stem]["acid_code"],
                   n_mg=jm[stem]["n_mg"], has_label=jm[stem]["has_label"],
                   panels=jm[stem]["panels"], metric_id=METRIC_ID)
        rows.append(res)

    if rows:
        cols = ["stem", "enzyme", "acid_code", "n_mg", "has_label", "A_deg", "D_h_A",
                "n_gated", "gated_frac", "min_d", "median_d", "max_angle", "docked",
                "n_poses", "n_extract_fail", "metric_id", "panels"]
        out = CAMP / "analysis/step1_scores.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows({c: r[c] for c in cols} for r in rows)
        print(f"scored {len(rows)} ensembles -> {out}")
        print(f"{'stem':28s} {'A':>7s} {'D_h':>6s} {'n_gated':>8s} {'min_d':>6s}  docked")
        for r in sorted(rows, key=lambda r: -r["A_deg"]):
            print(f"{r['stem']:28s} {r['A_deg']:7.2f} {r['D_h_A']:6.2f} "
                  f"{r['n_gated']:8d} {r['min_d']:6.2f}  {r['docked']}")
    print(f"\n{len(skipped)} not yet scorable:")
    for s, why in skipped[:8]:
        print(f"  {s}: {why}")
    if len(skipped) > 8:
        print(f"  ... and {len(skipped) - 8} more")


if __name__ == "__main__":
    main()
