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

    fig, ax = plt.subplots(figsize=(13.5, 8.5))

    # in-line & reliable quadrant shading (x>=150 and y>=5)
    ax.axvspan(ANGLE_THRESH, 180, ymin=(CONCLUSIVE_MIN + 2) / (N_SAMPLES + 5),
               ymax=1.0, color="#009E73", alpha=0.06, zorder=0)
    ax.axhline(CONCLUSIVE_MIN, color="#888", ls=":", lw=1.2, zorder=1)
    ax.axvline(ANGLE_THRESH, color="#2a7f62", ls="--", lw=1.4, zorder=1)

    # scatter conclusive candidates
    for fam, sub in concl.groupby("family"):
        ax.scatter(sub.mean_inline_angle, sub.n_distance_gated,
                   s=95, c=FAM_COLORS.get(fam, "#7F7F7F"),
                   edgecolors="white", linewidths=0.9, alpha=0.92,
                   zorder=3, label=fam)

    ax.set_xlim(100, 175)
    ax.set_ylim(-3, N_SAMPLES + 5)

    # ---- curated labels: the in-line & reliable shortlist + a few outliers ----
    concl["_lab"] = concl.apply(label_of, axis=1)
    shortlist = concl[(concl.mean_inline_angle >= ANGLE_THRESH) &
                      (concl.n_distance_gated >= CONCLUSIVE_MIN)]
    outliers = concl[(concl.n_distance_gated >= 30) &
                     (concl.mean_inline_angle < ANGLE_THRESH)]
    lo = concl.sort_values("mean_inline_angle").head(1)  # weakest geometry (purD)
    to_label = pd.concat([shortlist, outliers, lo]).drop_duplicates()

    # lightweight greedy repel in axis (data) coords
    placed = []  # (x, y) of placed label anchors
    xr, yr = 75.0, 58.0  # axis spans, for spacing scale
    def free(x, y):
        for px, py in placed:
            if abs(x - px) / xr < 0.045 and abs(y - py) / yr < 0.055:
                return False
        return True
    cands = [(6, 6), (6, -12), (-6, 6), (-6, -12), (10, 14), (10, -20),
             (-12, 14), (-12, -20), (14, 2), (-16, 2)]
    for _, r in to_label.sort_values(["n_distance_gated", "mean_inline_angle"],
                                     ascending=False).iterrows():
        x0, y0 = r.mean_inline_angle, r.n_distance_gated
        best = None
        for dx, dy in cands:
            lx = x0 + dx * xr / 300.0
            ly = y0 + dy * yr / 170.0
            if free(lx, ly):
                best = (dx, dy, lx, ly)
                break
        if best is None:
            best = (6, 6, x0 + 6 * xr / 300.0, y0 + 6 * yr / 170.0)
        dx, dy, lx, ly = best
        placed.append((lx, ly))
        ax.annotate(r._lab, (x0, y0), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, color="#222",
                    zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6,
                                    shrinkA=0, shrinkB=3))
    ax.set_xlabel("mean in-line attack angle (°)  —  geometry quality  "
                  "(dashed = 150° in-line threshold)")
    ax.set_ylabel("near-attack poses (d < 4 Å) out of 50  —  reliability")
    ax.set_title("Structural screen of discovery candidates — two axes\n"
                 f"run {RUN[:15]} · Boltz-2 50-sample cofold · "
                 "top-right = in-line AND reliable", loc="left")

    # quadrant guide text
    ax.text(174.5, N_SAMPLES + 4, "in-line & reliable", ha="right", va="top",
            fontsize=11, style="italic", color="#2a7f62")
    ax.text(100.5, CONCLUSIVE_MIN - 0.8, f"n < {CONCLUSIVE_MIN}: inconclusive",
            ha="left", va="top", fontsize=9, color="#888")

    # legend outside, to the right (families present, ordered by count)
    fams_present = concl.family.value_counts().index.tolist()
    handles = [Line2D([0], [0], marker="o", ls="", markersize=9,
                      markerfacecolor=FAM_COLORS.get(f, "#7F7F7F"),
                      markeredgecolor="white",
                      label=f"{f} (n={int((concl.family==f).sum())})")
               for f in fams_present]
    leg = ax.legend(handles=handles, title="mechanistic family",
                    loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    frameon=True, fontsize=9.5, title_fontsize=10.5)
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
