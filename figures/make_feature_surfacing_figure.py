#!/usr/bin/env python3
"""Which structural feature surfaces the top synthetase catalysts? (Boltz-2, 50-sample)

Form = heatmap (features x surfacing-metrics), the data's job is to compare many
features across several agreement metrics at once. Monochrome sequential gray
(magnitude = how well a feature ranks the top actives); NO categorical colour.
Each cell = AUROC in that feature's activity-consistent direction (0.5 = random,
darker = better). Columns are nested cutoffs (top-3/5/10 most active, and
activity > SfaB) plus their mean. Rows grouped: always-defined vs conditional on
the d<4A near-attack gate (marked +, survivorship-biased).

Descriptive row names so the mechanism is legible at a glance. The bare gated
in-line angle is deliberately excluded (meaningless without the distance gate);
the always-defined near-attack pose fraction is its rigorous stand-in.

Usage: python make_feature_surfacing_figure.py [--base exp_data/amide_synthetase]
"""
import os, csv, argparse, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 18  # base text size (user spec)

# curated: ranking-CSV key -> (descriptive name, group). One representative per
# mechanism family. group "A" = always-defined (n=48); "C" = conditional on the
# d<4A gate (n<48, >=5 gated poses required) -> survivorship-biased, marked with +.
ROWS = [
    # --- always-defined ---
    ("mg:mgP_all_median",   "Mg²⁺–phosphate distance",          "A"),
    ("bat:mg_pa_dist",      "Mg²⁺–Pα distance",            "A"),
    ("hyp:H7_adenineBurial","ATP adenine burial",                              "A"),
    ("sasa:distalC_sasa",   "Distal-carboxyl solvent exposure",                "A"),
    ("boltz:boltz2_min_d",  "Adipate–Pα closest approach",           "A"),
    ("bat:near_attack_frac","Near-attack pose fraction (d<4Å, in-line)",  "A"),
    ("bat:reactive_sasa",   "Reactive-carbon burial (resting adipate)",        "A"),
    ("mg:frac_coord",       "Mg²⁺ inner-sphere coordination",        "A"),
    ("psiptm:es_mean",      "Enzyme–substrate confidence (ps-iPTM)",      "A"),
    # --- conditional on the d<4A gate (survivorship-biased) ---
    ("preorg:posn_rmsf",    "Reactive-center positional RMSF",                 "C"),
    ("preorg:approach_std", "Attack-angle spread",                             "C"),
    ("tunnel:enc_tight",    "Acceptor-tunnel enclosure",                       "C"),
    ("elec:acidO_8",        "Amine-subsite acidic O (8Å)",                "C"),
    ("elec:nearBase_6",     "Amine-subsite base proximity (6Å)",          "C"),
]
COLS = [("robust_meanAUROC", "mean"), ("top3_auroc", "top-3"),
        ("top5_auroc", "top-5"), ("top10_auroc", "top-10"),
        ("binary_auroc_gtSfaB", ">SfaB")]


def fnum(v):
    try: return float(v)
    except (ValueError, TypeError): return math.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="exp_data/amide_synthetase")
    base = os.path.abspath(ap.parse_args().base)
    rk = {r["feature"]: r for r in csv.DictReader(
        open(os.path.join(base, "analysis/feature_surfacing_ranking.csv")))}

    # keep only curated features present, preserving group order; sort by mean within group
    rows = [(k, lbl, g) for k, lbl, g in ROWS if k in rk]
    rows.sort(key=lambda t: (t[2], -fnum(rk[t[0]]["robust_meanAUROC"])))
    labels = [(lbl + ("†" if g == "C" else ""), g) for _, lbl, g in rows]
    M = np.array([[fnum(rk[k][c]) for c, _ in COLS] for k, _, _ in rows])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(12.6, 0.62 * n + 2.4))
    im = ax.imshow(M, cmap="Blues", vmin=0.42, vmax=0.95, aspect="auto")

    # cell annotations (black on light, white on dark)
    for i in range(n):
        for j in range(len(COLS)):
            v = M[i, j]
            if math.isnan(v): continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS - 2,
                    color="white" if v >= 0.74 else "black")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([c for _, c in COLS], fontsize=FS)
    ax.set_yticks(range(n))
    ax.set_yticklabels([l for l, _ in labels], fontsize=FS)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

    # separator line between always-defined (above) and conditional (below, +)
    first_C = next((i for i, (_, _, g) in enumerate(rows) if g == "C"), None)
    if first_C is not None:
        ax.axhline(first_C - 0.5, color="black", lw=1.6)
    # thin white gridlines between cells for legibility
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("surfacing AUROC  (0.5 = random)", fontsize=FS - 3)
    cb.ax.tick_params(labelsize=FS - 4)

    fig.text(0.01, 0.005,
             "† conditional on the d<4Å near-attack gate (survivorship-biased); "
             "cells = AUROC in each feature's activity-consistent direction.",
             fontsize=FS - 6, color="#555")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(base, "figures", "synthetase_feature_surfacing.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out, "|", n, "features x", len(COLS), "metrics")


if __name__ == "__main__":
    main()
