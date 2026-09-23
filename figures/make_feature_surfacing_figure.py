#!/usr/bin/env python3
"""Rank structural features by top-3-catalyst surfacing (Boltz-2 synthetase panel).

Reads exp_data/amide_synthetase/analysis/feature_surfacing_ranking.csv and draws a
horizontal bar of top-3 discrimination (|AUROC-0.5|+0.5, direction-agnostic) for the
top-N features, colored by feature category, annotated with AS9 rank and direction.
Usage: python make_feature_surfacing_figure.py [--base exp_data/amide_synthetase] [--top 18]
"""
import os, csv, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CATS = [
    ("near-attack angle", "#2E6E9E", ("inline_angle", "gated_angle", "inline_frac", "near_attack", "preorg:angle")),
    ("preorganization",   "#27ae60", ("posn_rmsf", "approach_std", "clamp")),
    ("Mg coordination",   "#8e44ad", ("mg", "bridge")),
    ("electrostatics",    "#e67e22", ("base", "netchg", "acid", "elec:")),
    ("burial / SASA",     "#c0392b", ("sasa", "burial", "distalsasa", "oxyanion")),
    ("tunnel geometry",   "#16a085", ("tunnel", "channel", "enc_", "exit", "l2.", "l3.")),
    ("confidence/other",  "#7f8c8d", ()),
]
def cat(name):
    n = name.lower()
    for label, col, keys in CATS[:-1]:
        if any(k in n for k in keys): return label, col
    return CATS[-1][0], CATS[-1][1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="exp_data/amide_synthetase"); ap.add_argument("--top", type=int, default=18)
    a = ap.parse_args(); base = os.path.abspath(a.base)
    rows = [r for r in csv.DictReader(open(os.path.join(base, "analysis/feature_surfacing_ranking.csv"))) if r["top3_disc"]]
    rows = sorted(rows, key=lambda r: float(r["top3_disc"]), reverse=True)[:a.top][::-1]
    labels = [r["feature"] for r in rows]
    disc = [float(r["top3_disc"]) for r in rows]
    cols = [cat(r["feature"])[1] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.42*len(rows))))
    y = range(len(rows))
    ax.barh(list(y), disc, color=cols)
    ax.axvline(0.5, ls=":", c="k", lw=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("top-3 surfacing discrimination  (|AUROC−0.5|+0.5)")
    ax.set_title("Which structural feature surfaces the top synthetase catalysts?\nBoltz-2 step-1 panel, 50 samples · label = 3 most active (AS9/AS17/AS23)", fontsize=10)
    for i, r in enumerate(rows):
        ax.text(disc[i]+0.004, i, f"AS9 #{r['as9_rank']} ({r['dir']})", va="center", fontsize=7)
    seen=set(); handles=[]
    for lab,col,_ in CATS:
        handles.append(Patch(color=col, label=lab))
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(base, "figures", "synthetase_feature_surfacing.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight"); print("wrote", out)

if __name__ == "__main__":
    main()
