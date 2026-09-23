#!/usr/bin/env python3
"""Holistic view: which structural feature surfaces top synthetase catalysts?

Form = scatter (the data's job is a 2-D tradeoff, not a 1-D ranking):
  x = global rank-ordering  (Spearman rho vs continuous activity, oriented + = active)
  y = enrichment of the very top  (top-3 AUROC)
Each point = one representative feature (redundant families collapsed), coloured by
mechanism category (CVD-validated categorical palette). Marker dark-edge ring = the
contrast relief the palette check flagged. Quadrants read: top-left = enriches the
few best only; top-right = ranks the best AND the whole panel; bottom = uninformative.

Usage: python make_feature_surfacing_figure.py [--base exp_data/amide_synthetase]
"""
import os, csv, argparse, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# CVD-validated categorical hues (dataviz reference palette, light mode)
CAT = {
    "near-attack angle": "#2a78d6",
    "preorganization":   "#eb6834",
    "Mg coordination":   "#1baf7a",
    "electrostatics":    "#eda100",
    "burial / SASA":     "#e87ba4",
    "tunnel geometry":   "#008300",
    "confidence/other":  "#7f8c8d",
}
# curated: feature key in ranking CSV -> (nice label, category). One rep per family.
CUR = {
    "preorg:angle":            ("in-line attack angle", "near-attack angle"),
    "bat:inline_frac_gated":   ("in-line pose fraction", "near-attack angle"),
    "boltz:boltz2_min_d":      ("adipate→Pα distance", "near-attack angle"),
    "preorg:posn_rmsf":        ("reactive-center RMSF", "preorganization"),
    "preorg:approach_std":     ("approach-angle spread", "preorganization"),
    "mg:mgP_all_median":       ("Mg–phosphate distance", "Mg coordination"),
    "bat:mg_pa_dist":          ("Mg–Pα distance", "Mg coordination"),
    "hyp:H7_adenineBurial":    ("adenine burial", "burial / SASA"),
    "sasa:distalC_sasa":       ("distal-carboxyl SASA", "burial / SASA"),
    "bat:reactive_sasa":       ("reactive-center SASA", "burial / SASA"),
    "elec:nearBase_6":         ("near-base residues", "electrostatics"),
    "elec:acidO_8":            ("acidic-O residues", "electrostatics"),
    "tunnel:enc_tight":        ("acceptor-tunnel width", "tunnel geometry"),
    "psiptm:es_mean":          ("ps-iPTM confidence", "confidence/other"),
}

def fnum(v):
    try: return float(v)
    except: return math.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="exp_data/amide_synthetase")
    base = os.path.abspath(ap.parse_args().base)
    rk = {r["feature"]: r for r in csv.DictReader(open(os.path.join(base, "analysis/feature_surfacing_ranking.csv")))}

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.set_facecolor("#fcfcfb")
    # quadrant guides
    ax.axhline(0.70, ls=":", c="#9a9a95", lw=1); ax.axvline(0.15, ls=":", c="#9a9a95", lw=1)
    ax.text(0.005, 0.985, "enrich top only", transform=ax.transAxes, fontsize=8, color="#777", va="top")
    ax.text(0.995, 0.985, "ranks top + panel", transform=ax.transAxes, fontsize=8, color="#777", va="top", ha="right")

    pts = []
    for key, (label, cat) in CUR.items():
        r = rk.get(key)
        if not r: continue
        y = fnum(r["top3_auroc"]); rho = fnum(r["spearman_all"]); d = r["dir"]
        if math.isnan(y) or math.isnan(rho): continue
        x = rho if d == "+" else -rho          # oriented: higher = more active
        pts.append((x, y, label, cat, d))
        ax.scatter(x, y, s=110, c=CAT[cat], edgecolor="#2b2b2b", linewidth=0.8, zorder=3)
    # direct labels (offset, small), collision-avoided by hand-nudge dict
    nudge = {"in-line attack angle": (6, 4), "reactive-center RMSF": (6, -12),
             "approach-angle spread": (-6, 8), "Mg–phosphate distance": (6, 4),
             "Mg–Pα distance": (6, -10), "in-line pose fraction": (6, 4),
             "adenine burial": (6, 4), "distal-carboxyl SASA": (6, -4),
             "near-base residues": (6, 4), "adipate→Pα distance": (6, -10),
             "ps-iPTM confidence": (6, 4), "acceptor-tunnel width": (6, 4),
             "acidic-O residues": (6, -10), "reactive-center SASA": (-6, -12)}
    for x, y, label, cat, d in pts:
        dx, dy = nudge.get(label, (6, 4))
        ax.annotate(label, (x, y), fontsize=7.5, color="#2b2b2b",
                    xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left")

    ax.set_xlabel("global rank-ordering  ·  Spearman ρ (feature vs activity, all 48)", fontsize=10)
    ax.set_ylabel("enrichment of the top actives  ·  top-3 AUROC", fontsize=10)
    ax.set_ylim(0.48, 0.98); ax.set_title(
        "Which structural features surface the top synthetase catalysts?\n"
        "Boltz-2 step-1, 50 samples · near-attack angle enriches the very top; Mg/preorg rank the panel",
        fontsize=11)
    ax.legend(handles=[Patch(color=c, label=l) for l, c in CAT.items() if any(p[3]==l for p in pts)],
              fontsize=8, loc="lower right", framealpha=0.95, title="feature category", title_fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(base, "figures", "synthetase_feature_surfacing.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); print("wrote", out, "|", len(pts), "features")

if __name__ == "__main__":
    main()
