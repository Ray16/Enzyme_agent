#!/usr/bin/env python3
"""Cross-model step-1 near-attack figure (amide-synthetase, 50-sample).

Reads exp_data/amide_synthetase/analysis/{crossmodel_nearattack.csv,
crossmodel_predictivity_summary.csv} and renders a 3-panel decision figure:
 (A) AS9 closest approach d(O..Palpha) per model vs the 4A gate;
 (B) AS9 rank (by the validated d<4A->in-line-angle proxy) among scored targets;
 (C) predictivity AUROC per model for the angle proxy vs the coarser na_frac.
Rolling: renders whatever models are present in the summary. Usage:
  python make_crossmodel_figure.py [--base exp_data/amide_synthetase]
"""
import os, csv, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["boltz2", "protenix", "af3", "esmfold2"]
NICE = {"boltz2": "Boltz-2", "protenix": "Protenix-v2", "af3": "AF3", "esmfold2": "ESMFold2"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="exp_data/amide_synthetase")
    args = ap.parse_args(); base = os.path.abspath(args.base)
    an = os.path.join(base, "analysis")
    summ = list(csv.DictReader(open(os.path.join(an, "crossmodel_predictivity_summary.csv"))))
    models = [m for m in ORDER if any(r["model"] == m for r in summ)]

    def get(model, metric, field):
        for r in summ:
            if r["model"] == model and r["metric"] == metric:
                return r[field]
        return None

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    xs = range(len(models)); labels = [NICE[m] for m in models]
    col = "#2E6E9E"

    # A: AS9 closest approach vs 4A gate
    dmin = [float(get(m, "mean_gated_angle", "as9_min_d") or "nan") for m in models]
    ax[0].bar(xs, dmin, color=col)
    ax[0].axhline(4.0, ls="--", c="#c0392b", lw=1.5); ax[0].text(len(models)-0.5, 4.03, "4 Å gate", color="#c0392b", ha="right", fontsize=9)
    ax[0].set_xticks(list(xs)); ax[0].set_xticklabels(labels, rotation=15)
    ax[0].set_ylabel("AS9 min d(O$_{nuc}$···P$_\\alpha$)  (Å)"); ax[0].set_title("A. AS9 near-attack reach (50 samples)")
    for i, v in enumerate(dmin):
        ax[0].text(i, v+0.03, f"{v:.2f}", ha="center", fontsize=9)

    # B: AS9 rank (mean_gated_angle) among scored; lower better
    ranks = [int(get(m, "mean_gated_angle", "as9_rank") or 0) for m in models]
    ns = [int(get(m, "mean_gated_angle", "n_scored") or 0) for m in models]
    ax[1].bar(xs, ranks, color=["#27ae60" if r <= 10 else col for r in ranks])
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(labels, rotation=15)
    ax[1].set_ylabel("AS9 rank (1 = best)"); ax[1].set_title("B. AS9 rank by d<4Å→in-line angle")
    for i, (r, n) in enumerate(zip(ranks, ns)):
        ax[1].text(i, r+0.4, f"{r}/{n}", ha="center", fontsize=9)
    ax[1].invert_yaxis()

    # C: AUROC per model, angle proxy vs na_frac
    w = 0.38
    a_ang = [float(get(m, "mean_gated_angle", "auroc") or "nan") for m in models]
    a_naf = [float(get(m, "na_frac", "auroc") or "nan") for m in models]
    ax[2].bar([x - w/2 for x in xs], a_ang, w, label="d<4Å→angle", color="#2E6E9E")
    ax[2].bar([x + w/2 for x in xs], a_naf, w, label="na_frac", color="#95a5a6")
    ax[2].axhline(0.5, ls=":", c="k", lw=1)
    ax[2].set_xticks(list(xs)); ax[2].set_xticklabels(labels, rotation=15)
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("AUROC (activity > SfaB)"); ax[2].set_title("C. Predictivity AUROC"); ax[2].legend(fontsize=8)

    fig.suptitle("Cross-model step-1 near-attack — amide-synthetase panel (50 samples/complex)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(base, "figures", "synthetase_crossmodel_nearattack_50sample.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out, "| models:", models)


if __name__ == "__main__":
    main()
