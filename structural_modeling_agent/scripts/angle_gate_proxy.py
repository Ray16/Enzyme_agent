#!/usr/bin/env python3
"""Mechanism-aware angle-gate proxy for unrestrained Boltz-2 ensembles.

The conditional mean angle was retrospectively validated for P-alpha attack in
the AMP-binding/ANL panel. P-alpha use in other adenylating families and P-gamma
use in acyl-phosphate families are mechanism-based transfers, not validated
activity models.

Proxy per enzyme = mean mechanism-appropriate attack angle among poses with
nucleophile-to-phosphorus distance <4 A. At least five gated poses are required.
Ligand iPTM is reported as QC but is deliberately not a pose filter: no iPTM
threshold has been validated on the assay panel.

Interpretation: in-line angle is NECESSARY but not SUFFICIENT -> use as a recall-oriented
screening annotation, not a quantitative activity prediction or automatic veto.

Usage: angle_gate_proxy.py <results/output_configs> [--phosphate pa|pg|auto]
                           [--reps reps.csv] [--override ID=PREDICTIONS_DIR]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import step1_nac as s
from proxy_policy import (  # noqa: E402
    COMPATIBLE_ANGLE,
    MIN_GATED_POSES,
    PROXY_VERSION,
    STRONG_ANGLE,
    proxy_mechanism_for_family,
)

A_CUT = COMPATIBLE_ANGLE
D_GATE = 4.0        # near-attack distance gate: keep only poses where the carboxylate O actually
                    # reached attack distance of the phosphate, THEN score the mean in-line angle
                    # among them. Physically principled (true near-attack ~3.5-4A) AND empirically
                    # best on the panel (rho ~0.30 vs 0.27 at 8A, 0.09 for raw NAC). Distance is a
                    # GATE (condition on near-attack), never a score component. <5 poses -> INCONCLUSIVE.
def mechanism_phosphate(family):
    """Backward-compatible helper; unknown families now abstain."""
    spec = proxy_mechanism_for_family(family)
    return spec.phosphate.lower() if spec else None


def _bootstrap_mean_summary(values, seed=0, n_boot=5000):
    """Deterministic bootstrap uncertainty for the gated ensemble mean."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None, None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.05, 0.95])
    return (
        float(low),
        float(high),
        float(np.mean(means >= COMPATIBLE_ANGLE)),
        float(np.mean(means >= STRONG_ANGLE)),
    )


def score_config(pred, tag, confidence_indices=None):
    per = []
    for pdb in sorted(glob.glob(f"{pred}/*model_*.pdb")):
        g = s.geom_model(pdb)
        if not g:
            continue
        mn = pdb.split("model_")[-1].split(".pdb")[0]
        cj = glob.glob(f"{pred}/confidence_*model_{mn}.json")
        if cj and confidence_indices:
            pair_iptm = json.load(open(cj[0]))["pair_chains_iptm"]
            protein_index = str(confidence_indices["protein"])
            substrate_index = str(confidence_indices["substrate"])
            ip = pair_iptm.get(protein_index, {}).get(substrate_index, np.nan)
        else:
            ip = np.nan
        per.append((g[f"ang_{tag}"], g[f"d_{tag}"], ip))
    if not per:
        return None
    a = np.array([p[0] for p in per]); d = np.array([p[1] for p in per]); ip = np.array([p[2] for p in per])
    gm = d < D_GATE                                       # DISTANCE gate: docked in catalytic subsite
    n_gated = int(gm.sum())
    ag = a[gm]
    ci_low, ci_high, p150, p165 = _bootstrap_mean_summary(ag)
    mean_iptm = float(np.nanmean(ip)) if np.isfinite(ip).any() else None
    return {
        "n_models": len(per),
        "n_gated": n_gated,
        "gated_fraction": round(n_gated / len(per), 3),
        "mean_iptm": round(mean_iptm, 3) if mean_iptm is not None else None,
        "docking_ok": n_gated >= MIN_GATED_POSES,
        "mean_inline_angle": round(float(ag.mean()), 1) if n_gated else None,
        "angle_std": round(float(ag.std(ddof=1)), 1) if n_gated > 1 else None,
        "angle_ci90_low": round(ci_low, 1) if ci_low is not None else None,
        "angle_ci90_high": round(ci_high, 1) if ci_high is not None else None,
        "prob_mean_ge_150": round(p150, 3) if p150 is not None else None,
        "prob_mean_ge_165": round(p165, 3) if p165 is not None else None,
        "inline_frac_gated": round(float(np.mean(ag > A_CUT)), 3) if n_gated else None,
        "inline_frac_all": round(float(np.mean(a > A_CUT)), 3),
        "best_angle": round(float(a.max()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--phosphate", default="auto", choices=["pa", "pg", "auto"])
    ap.add_argument("--reps", default=None, help="representatives.csv for auto mechanism (interpro_family)")
    ap.add_argument("--prefix", default="step1_")
    ap.add_argument("--out", default=None)
    ap.add_argument("--confidence-indices", help="JSON mapping candidate IDs to explicit protein/substrate confidence matrix indices; missing maps leave ipTM unreported")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="ID=PREDICTIONS_DIR",
        help="replace one candidate's predictions directory (repeatable)",
    )
    args = ap.parse_args()
    confidence_indices = json.load(open(args.confidence_indices)) if args.confidence_indices else {}

    overrides = {}
    for value in args.override:
        if "=" not in value:
            ap.error(f"invalid --override {value!r}; expected ID=PREDICTIONS_DIR")
        eid, pred = value.split("=", 1)
        if not eid or not os.path.isdir(pred):
            ap.error(f"invalid --override {value!r}; predictions directory not found")
        overrides[eid] = pred

    fam = {}
    if args.reps and os.path.exists(args.reps):
        import csv
        fam = {r["accession"]: r.get("interpro_family", "") for r in csv.DictReader(open(args.reps))}

    rows = []
    for cfg in sorted(glob.glob(f"{args.results}/{args.prefix}*")):
        eid = os.path.basename(cfg)[len(args.prefix):]
        if eid in overrides:
            pred = [overrides[eid]]
        else:
            pred = glob.glob(f"{cfg}/boltz_results_*/predictions/*")
            if not pred:
                continue
        spec = proxy_mechanism_for_family(fam.get(eid, "")) if args.phosphate == "auto" else None
        tag = args.phosphate if args.phosphate != "auto" else (
            spec.phosphate.lower() if spec else None
        )
        if tag is None:
            rows.append({
                "id": eid,
                "phosphate": "",
                "proxy_version": PROXY_VERSION,
                "mechanism_status": "unresolved",
                "evidence_scope": "unresolved",
                "docking_ok": False,
                "n_models": len(glob.glob(f"{pred[0]}/*model_*.pdb")),
                "n_gated": 0,
                "prediction_source": os.path.abspath(pred[0]),
                "source_override": eid in overrides,
            })
            continue
        r = score_config(pred[0], tag, confidence_indices.get(eid))
        if r:
            r.update(
                id=eid,
                phosphate=tag.upper(),
                proxy_version=PROXY_VERSION,
                mechanism_status="assigned" if spec or args.phosphate != "auto" else "unresolved",
                evidence_scope=spec.evidence_scope if spec else "explicit_coordinate",
                prediction_source=os.path.abspath(pred[0]),
                source_override=eid in overrides,
            )
            rows.append(r)

    rows.sort(
        key=lambda r: -(
            r.get("mean_inline_angle")
            if r.get("docking_ok") and r.get("mean_inline_angle") is not None
            else -1
        )
    )
    cols = [
        "id", "proxy_version", "mechanism_status", "evidence_scope", "phosphate",
        "n_models", "n_gated", "gated_fraction", "mean_inline_angle", "angle_std",
        "angle_ci90_low", "angle_ci90_high", "prob_mean_ge_150",
        "prob_mean_ge_165", "inline_frac_gated", "docking_ok", "mean_iptm", "best_angle",
        "source_override", "prediction_source",
    ]
    print(f"{'id':10} {'P':3} {'mean_angle':>10} {'inline_frac':>11} {'dock_ok':>8} {'n_gated':>8} {'iptm':>6}")
    for r in rows:
        v = r.get("mean_inline_angle") if r.get("docking_ok") else "INCONCL"
        print(
            f"{r['id']:10} {r.get('phosphate', ''):3} {str(v):>10} "
            f"{str(r.get('inline_frac_gated')):>11} {str(r.get('docking_ok')):>8} "
            f"{r.get('n_gated', 0):>8} {str(r.get('mean_iptm')):>6}"
        )

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
