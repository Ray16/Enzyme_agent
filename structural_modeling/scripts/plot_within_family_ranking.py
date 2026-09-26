#!/usr/bin/env python
"""Two-axis structural screen of discovery candidates for run 20260917_135330.

Replaces the 1-D lollipop that ranked only by mean in-line angle. The angle-only
view collapses the two-axis near-attack screen (CANDIDATE_RANKING_POLICY.md) into
one dimension and hides the reliability axis -- how many of the 50 Boltz-2 poses
actually reached near-attack (d < 4 A). Here we plot both:

    x = mean in-line attack angle (deg), over d<4A poses  [geometry quality]
    y = near-attack pose count out of 50 (n_distance_gated) [reliability]

Colour = mechanistic family. The >=150 deg in-line threshold and the n>=5
conclusive threshold split the plane into quadrants; top-right = geometrically
in-line AND reliably reaching near-attack. Inconclusive candidates (n<5, angle
undefined) -- including the ground-truth ddaG, which never docks productively --
are listed explicitly rather than dropped as blank rows.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUN = "20260917_135330_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl"
ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / "structural_modeling/cofold" / RUN / "results/candidate_prioritization.csv"
OUTDIR = ROOT / "exp_data/amide_synthetase/figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 50
ANGLE_THRESH = 150.0
CONCLUSIVE_MIN = 5
GROUND_TRUTH = {"E2JA32"}  # ddaG

# Okabe-Ito colourblind-safe qualitative palette, ordered by family frequency.
FAM_COLORS = {
    "Glutamine synthetase": "#0072B2",
    "Aerobactin NIS":       "#E69F00",
    "ATP-grasp":            "#009E73",
    "RimK/bS6":             "#CC79A7",
    "PurD/GAR":             "#56B4E9",
    "Dapdiamide":           "#D55E00",
    "phosphoribosylformylglycinamidine cyclo-ligase": "#7F7F7F",
    "carbamoyl-phosphate synthase": "#000000",
}

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
    "savefig.dpi": 300, "figure.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def label_of(row):
    g = str(row.gene).strip()
    return g if g and g.lower() != "nan" else str(row.accession)


def main():
    df = pd.read_csv(CP)
    concl = df[df.n_distance_gated >= CONCLUSIVE_MIN].copy()
    incon = df[df.n_distance_gated < CONCLUSIVE_MIN].copy()

    fig, ax = plt.subplots(figsize=(12, 8.5))

    # quadrant shading: reliable (y>=5) AND in-line (x>=150)
    ax.axhspan(CONCLUSIVE_MIN, N_SAMPLES + 3, xmin=0, xmax=1, color="none")
    ax.axvspan(ANGLE_THRESH, 180, color="#009E73", alpha=0.04, zorder=0)
    ax.axhline(CONCLUSIVE_MIN, color="#888", ls=":", lw=1.2, zorder=1)
    ax.axvline(ANGLE_THRESH, color="#2a7f62", ls="--", lw=1.4, zorder=1)

    # scatter conclusive candidates
    for fam, sub in concl.groupby("family"):
        ax.scatter(sub.mean_inline_angle, sub.n_distance_gated,
                   s=90, c=FAM_COLORS.get(fam, "#7F7F7F"),
                   edgecolors="white", linewidths=0.8, alpha=0.9,
                   zorder=3, label=fam)

    # annotate: ground truth, and the strongest / notable per family
    concl["_lab"] = concl.apply(label_of, axis=1)
    # label the top-3 by pose-count-weighted angle within the two dominant families,
    # plus any high-consistency (n>=20) point, plus ground truth if present.
    to_label = set()
    for fam, sub in concl.groupby("family"):
        top = sub.sort_values(["n_distance_gated", "mean_inline_angle"],
                              ascending=False).head(2)
        to_label.update(top.index)
    to_label.update(concl[concl.n_distance_gated >= 20].index)

    for i in to_label:
        r = concl.loc[i]
        ax.annotate(r._lab, (r.mean_inline_angle, r.n_distance_gated),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color="#333", zorder=4)

    ax.set_xlim(100, 175)
    ax.set_ylim(-2, N_SAMPLES + 3)
    ax.set_xlabel("mean in-line attack angle (°)  —  geometry quality  "
                  "(dashed = 150° in-line threshold)")
    ax.set_ylabel("near-attack poses (d < 4 Å) out of 50  —  reliability")
    ax.set_title("Structural screen of discovery candidates — two axes\n"
                 f"run {RUN[:15]} · Boltz-2 50-sample cofold · "
                 "top-right = in-line AND reliable", loc="left")

    # quadrant guide text
    ax.text(174, N_SAMPLES + 1.5, "in-line & reliable", ha="right", va="top",
            fontsize=10, style="italic", color="#2a7f62")
    ax.text(101, CONCLUSIVE_MIN - 0.5, f"n < {CONCLUSIVE_MIN}: inconclusive",
            ha="left", va="top", fontsize=9, color="#888")

    # legend: families present, ordered by count
    fams_present = concl.family.value_counts().index.tolist()
    handles = [Line2D([0], [0], marker="o", ls="", markersize=9,
                      markerfacecolor=FAM_COLORS.get(f, "#7F7F7F"),
                      markeredgecolor="white",
                      label=f"{f} (n={int((concl.family==f).sum())})")
               for f in fams_present]
    leg = ax.legend(handles=handles, title="mechanistic family",
                    loc="lower right", frameon=True, fontsize=9, title_fontsize=10)
    leg.get_frame().set_alpha(0.95)

    # inconclusive strip (incl. ground truth) as a text box
    inc = incon.copy()
    inc["_lab"] = inc.apply(label_of, axis=1)
    inc = inc.sort_values("n_distance_gated", ascending=False)
    lines = []
    for _, r in inc.iterrows():
        star = " ★GT" if r.accession in GROUND_TRUTH else ""
        lines.append(f"{r._lab} ({int(r.n_distance_gated)}){star}")
    box = (f"Inconclusive — <{CONCLUSIVE_MIN} near-attack poses "
           f"({len(inc)} candidates), angle undefined:\n"
           + "  ".join(lines))
    fig.text(0.012, -0.02, box, ha="left", va="top", fontsize=8, color="#555",
             wrap=True)
    fig.text(0.012, -0.10, "★GT = ground-truth ddaG (E2JA32): never docks "
             "productively → known cofold false-negative for diacids.",
             ha="left", va="top", fontsize=8, style="italic", color="#a33")

    out_png = OUTDIR / "within_family_structural_screen_20260917_135330.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)
    print("wrote", out_pdf)
    print(f"conclusive={len(concl)}  inconclusive={len(incon)}")


if __name__ == "__main__":
    main()
