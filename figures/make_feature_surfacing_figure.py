#!/usr/bin/env python3
"""Which structural feature surfaces the top synthetase catalysts? (Boltz-2, 50-sample)

Curated, MECHANISM-MOTIVATED panel: one representative proxy per catalytic
determinant of the ANL-family adenylating amide synthetase reaction
(adipate + ATP + HMDA -> nylon-6,6 amide), NOT a maximal feature dump. Weak /
physically-unmotivated proxies are excluded (e.g. ps-iPTM = model confidence,
not chemistry; raw near-attack fraction = near-random).

Rows are grouped along the reaction coordinate:
  1. Ground-state binding      - acyl donor (distal carboxyl) + cofactor (ATP)
  2. Adenylation (step 1)      - in-line attack of the carboxylate on ATP-Palpha
  3. Metal-ion organization    - Mg2+ positioning of the phosphates / phosphorane TS
  4. Active-site preorganization - rigidity of the reactive center across the ensemble
  5. Aminolysis (step 2)       - delivery of the amine to the acyl-adenylate carbonyl

Form = heatmap (features x surfacing-metrics). Sequential single-hue blue
(white = random 0.5, deep blue = strong). Each cell = AUROC in that feature's
activity-consistent direction. Columns are nested cutoffs (top-3/5/10 most
active, activity > SfaB) plus their mean. Rows marked + are conditional on the
d<4A near-attack gate (survivorship-biased); all others are always-defined.

Usage: python make_feature_surfacing_figure.py [--base exp_data/amide_synthetase]
"""
import os, csv, argparse, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 18  # base text size (user spec)

# Final physically-motivated panel, one flat list in reaction-coordinate order.
# Each entry: (ranking-CSV key, descriptive name, marker). marker:
#   ""  = always-defined (all 48 enzymes)
#   "†" = conditional on the d<4 Å near-attack gate (survivorship-biased)
#   "‡" = His-present subset (n=44 of 48)
# Metal (Mg) proxies dropped: Boltz-2 Mg placement can't resolve catalytic
# geometry (distance ranks backwards; β,γ-bridging & coordination near-random).
ROWS = [
    ("sasa:distalC_sasa",    "Distal-carboxyl anchoring",         ""),   # i  ground-state
    ("hyp:H7_adenineBurial", "ATP adenine burial",                ""),   # i  ground-state
    ("bat:mean_inline_angle","In-line attack angle (d<4Å→angle)", "†"),  # ii NAC geometry
    ("preorg:posn_rmsf",     "Reactive-center positional RMSF",   "†"),  # vi dynamics/preorg
    ("esp:lef_pa_std",       "Electric-field preorganization (LEF)", ""),# iii electrostatics
    ("pka:s1_his_dev7",      "Catalytic His pKa (poised ~7)",     "‡"),  # iv nucleophile/base
    ("esp:esp_carbonyl",     "Carbonyl-oxyanion ESP (step 2)",    ""),   # iii electrostatics
    ("tunnel:enc_tight",     "Acceptor-tunnel enclosure",         "†"),  # i/ii step-2 shape
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

    # Single continuous list of feature rows, in reaction-coordinate order
    # (no category bands/separators).
    rows = []          # (label, key)
    for key, lbl, mark in ROWS:
        if key not in rk:
            continue
        rows.append((lbl + mark, key))

    n = len(rows)
    M = np.array([[fnum(rk[key][c]) for c, _ in COLS] for _, key in rows])

    cmap = plt.get_cmap("Blues").copy()

    fig, ax = plt.subplots(figsize=(12.8, 0.62 * n + 2.0))
    im = ax.imshow(M, cmap=cmap, vmin=0.42, vmax=0.95, aspect="auto")

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
    ax.set_yticklabels([t for t, _ in rows], fontsize=FS)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

    # thin white gridlines between cells for legibility
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("AUROC  (0.5 = random)", fontsize=FS - 3)
    cb.ax.tick_params(labelsize=FS - 4)

    fig.text(0.01, 0.005,
             "† conditional on the d<4Å near-attack gate (survivorship-biased);  "
             "‡ His-present subset (n=44 of 48);  "
             "cells = AUROC in each feature's activity-consistent direction.",
             fontsize=FS - 6, color="#555")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(base, "figures", "synthetase_feature_surfacing.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out, "|", n, "features x", len(COLS), "metrics")


if __name__ == "__main__":
    main()
