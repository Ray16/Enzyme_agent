#!/usr/bin/env python3
"""Holistic structural-feature comparison for surfacing synthetase catalysts.

Reads exp_data/amide_synthetase/analysis/feature_surfacing_ranking.csv and renders
ONE essential figure with two panels:
  (A) heatmap: top-N features x cutoff AUROCs (top-3/5/10 actives, activity>SfaB),
      oriented so higher = better — shows each feature's behaviour across "how many
      count as active".
  (B) scatter: enrichment (top-5 AUROC) vs global rank-correlation (|Spearman rho|),
      coloured by feature category — separates "enrich the very top" features (high y,
      low x) from "rank-order the whole panel" features (high both).
Usage: python make_feature_surfacing_figure.py [--base exp_data/amide_synthetase] [--top 16]
"""
import os, csv, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CATS = [
    ("near-attack angle", "#2E6E9E", ("inline_angle", "gated_angle", "inline_frac", "near_attack", "preorg:angle", "min_d")),
    ("preorganization",   "#27ae60", ("posn_rmsf", "approach_std", "clamp")),
    ("Mg coordination",   "#8e44ad", ("mg", "bridge")),
    ("electrostatics",    "#e67e22", ("base", "netchg", "acido", "elec:")),
    ("burial / SASA",     "#c0392b", ("sasa", "burial", "distalsasa", "oxyanion", "h1_", "h2_")),
    ("tunnel geometry",   "#16a085", ("tunnel", "channel", "enc_", "exit", "l2.", "l3.", "h3_", "h4_")),
    ("confidence/other",  "#7f8c8d", ()),
]
def catcol(name):
    n = name.lower()
    for lab, col, keys in CATS[:-1]:
        if any(k in n for k in keys): return lab, col
    return CATS[-1][:2]

def fnum(v):
    try: return float(v)
    except: return np.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="exp_data/amide_synthetase"); ap.add_argument("--top", type=int, default=16)
    a = ap.parse_args(); base = os.path.abspath(a.base)
    rows = list(csv.DictReader(open(os.path.join(base, "analysis/feature_surfacing_ranking.csv"))))
    for r in rows: r["_rob"] = fnum(r["robust_meanAUROC"])
    rows = [r for r in rows if not np.isnan(r["_rob"])]
    rows.sort(key=lambda r: r["_rob"], reverse=True)
    topR = rows[:a.top][::-1]

    fig = plt.figure(figsize=(14, max(5, 0.5*len(topR))))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.35)

    # ---- A: heatmap ----
    axA = fig.add_subplot(gs[0, 0])
    mcols = [("top3_auroc","top-3"),("top5_auroc","top-5"),("top10_auroc","top-10"),("binary_auroc_gtSfaB",">SfaB")]
    M = np.array([[fnum(r[c]) for c,_ in mcols] for r in topR])
    im = axA.imshow(M, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    axA.set_xticks(range(len(mcols))); axA.set_xticklabels([l for _,l in mcols])
    axA.set_yticks(range(len(topR))); axA.set_yticklabels([r["feature"] for r in topR], fontsize=8)
    for i in range(len(topR)):
        for j in range(len(mcols)):
            if not np.isnan(M[i,j]): axA.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
    # color feature labels by category
    for tick, r in zip(axA.get_yticklabels(), topR):
        tick.set_color(catcol(r["feature"])[1])
    axA.set_title("A. Feature × activity-cutoff AUROC (oriented, higher=better)", fontsize=10)
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02); cb.set_label("AUROC", fontsize=8)

    # ---- B: enrichment vs global ----
    axB = fig.add_subplot(gs[0, 1])
    for r in rows:
        t5 = fnum(r["top5_auroc"]); rho = abs(fnum(r["spearman_all"]))
        if np.isnan(t5) or np.isnan(rho): continue
        lab, col = catcol(r["feature"])
        axB.scatter(rho, t5, c=col, s=32, alpha=0.75, edgecolor="none")
    # label a few notable features
    notable = {"preorg:posn_rmsf","preorg:angle","mg:mgP_all_median","bat:mean_inline_angle","preorg:approach_std","hyp:H7_adenineBurial"}
    for r in rows:
        if r["feature"] in notable:
            t5=fnum(r["top5_auroc"]); rho=abs(fnum(r["spearman_all"]))
            if not (np.isnan(t5) or np.isnan(rho)):
                axB.annotate(r["feature"].split(":")[1], (rho,t5), fontsize=7,
                             xytext=(4,3), textcoords="offset points")
    axB.axhline(0.5, ls=":", c="k", lw=0.8)
    axB.set_xlabel("global rank-correlation  |Spearman ρ|  (all 48)")
    axB.set_ylabel("enrichment of top actives  (top-5 AUROC)")
    axB.set_title("B. Enrich-the-top vs rank-the-panel", fontsize=10)
    axB.legend(handles=[Patch(color=c, label=l) for l,c,_ in CATS], fontsize=6.5, loc="lower right", framealpha=0.9)

    fig.suptitle("Structural features surfacing synthetase catalysts — Boltz-2 step-1, 50 samples", fontsize=12, y=1.01)
    out = os.path.join(base, "figures", "synthetase_feature_surfacing.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)

if __name__ == "__main__":
    main()
