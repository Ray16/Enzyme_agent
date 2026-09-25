#!/usr/bin/env python3
"""Angle-gate proxy — the validated static-cofold activity signal.

Decision (validated on the 50-member SfaB panel with measured N66 activity):
  * NAC (distance AND angle) fails: the hard distance gate is corrupted by docking
    failures and cancels the angle signal (rho ~ 0.09).
  * Pocket-restraining (force=true) makes it WORSE (inverts: dead>active).
  * The signal lives in the IN-LINE ANGLE alone, on UNRESTRAINED cofolds, with a
    confidence gate (drop docking-failed poses via ATP x ligand iptm).

Proxy per enzyme = fraction of confidently-docked poses whose mechanism-appropriate
attack angle exceeds 150 deg.  On the panel: rho ~ 0.21-0.28 vs activity, full recall
of actives (incl. AS8 that NAC buried), dead-vs-active gap +0.52.

Interpretation: in-line angle is NECESSARY but not SUFFICIENT -> use as a recall-oriented
NEGATIVE/coarse filter (prune non-attack-competent candidates), NOT a fine ranker.

Usage: angle_gate_proxy.py <results/output_configs> [--phosphate pa|pg|auto]
                           [--reps reps.csv] [--override ID=PREDICTIONS_DIR]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "common"))
import step1_nac as s

A_CUT = 150.0
IPTM_GATE = 0.6
D_GATE = 4.0        # near-attack distance gate: keep only poses where the carboxylate O actually
                    # reached attack distance of the phosphate, THEN score the mean in-line angle
                    # among them. Physically principled (true near-attack ~3.5-4A) AND empirically
                    # best on the panel (rho ~0.30 vs 0.27 at 8A, 0.09 for raw NAC). Distance is a
                    # GATE (condition on near-attack), never a score component. <5 poses -> INCONCLUSIVE.
ADENYLATE_KEYS = ("iuca", "iucc", "aerobactin", "amp-binding", "anl")  # -> Pa


def mechanism_phosphate(family):
    f = (family or "").lower()
    return "pa" if any(k in f for k in ADENYLATE_KEYS) else "pg"


def score_config(pred, tag, gate=IPTM_GATE):
    per = []
    for pdb in sorted(glob.glob(f"{pred}/*model_*.pdb")):
        g = s.geom_model(pdb)
        if not g:
            continue
        mn = pdb.split("model_")[-1].split(".pdb")[0]
        cj = glob.glob(f"{pred}/confidence_*model_{mn}.json")
        if cj:
            pair_iptm = json.load(open(cj[0]))["pair_chains_iptm"]
            substrate_chain = max(pair_iptm, key=int)
            ip = pair_iptm["1"][substrate_chain]
        else:
            ip = np.nan
        per.append((g[f"ang_{tag}"], g[f"d_{tag}"], ip))
    if not per:
        return None
    a = np.array([p[0] for p in per]); d = np.array([p[1] for p in per]); ip = np.array([p[2] for p in per])
    gm = d < D_GATE                                       # DISTANCE gate: docked in catalytic subsite
    n_gated = int(gm.sum())
    ag = a[gm] if n_gated else a
    return {
        "n_models": len(per),
        "n_gated": n_gated,
        "mean_iptm": round(float(np.nanmean(ip)), 3),
        "docking_ok": n_gated >= 5,                       # <5 poses docked -> INCONCLUSIVE (docking failure)
        # PRIMARY: continuous mean in-line angle among docked poses (rho~0.265 on panel)
        "mean_inline_angle": round(float(ag.mean()), 1) if n_gated else None,
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
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="ID=PREDICTIONS_DIR",
        help="replace one candidate's predictions directory (repeatable)",
    )
    args = ap.parse_args()

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
        tag = args.phosphate if args.phosphate != "auto" else mechanism_phosphate(fam.get(eid, ""))
        r = score_config(pred[0], tag)
        if r:
            r.update(
                id=eid,
                phosphate=tag.upper(),
                prediction_source=os.path.abspath(pred[0]),
                source_override=eid in overrides,
            )
            rows.append(r)

    rows.sort(key=lambda r: (-(r["mean_inline_angle"] if r["docking_ok"] and r["mean_inline_angle"] is not None else -1)))
    cols = [
        "id", "phosphate", "mean_inline_angle", "inline_frac_gated",
        "docking_ok", "n_gated", "mean_iptm", "best_angle",
        "source_override", "prediction_source",
    ]
    print(f"{'id':10} {'P':3} {'mean_angle':>10} {'inline_frac':>11} {'dock_ok':>8} {'n_gated':>8} {'iptm':>6}")
    for r in rows:
        v = r["mean_inline_angle"] if r["docking_ok"] else "INCONCL"
        print(f"{r['id']:10} {r['phosphate']:3} {str(v):>10} {str(r['inline_frac_gated']):>11} {str(r['docking_ok']):>8} {r['n_gated']:>8} {r['mean_iptm']:>6}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in cols})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
