#!/usr/bin/env python3
"""
Boxplot of hydrolase pocket chemistry by functional family,
with NylC (autocleavage form) as the validated reference.

Shows pocket hydrophobic and charged fractions for three major
hydrolase families: Polyglutamate hydrolase, Cyanophycinase,
Pseudomurein hydrolase.

Outputs:
    analysis_results/hydrolase_pocket_boxplot.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(BASE_DIR, "analysis_results")

PFAM_TO_FAMILY = {
    "Gamma_PGA_hydro": "Polyglutamate\nhydrolase",
    "Peptidase_S51": "Cyanophycinase",
    "Transglut_core": "Pseudomurein\nhydrolase",
    "Peptidase_C71": "Pseudomurein\nhydrolase",
    "Pseudomurein endoisopeptidase": "Pseudomurein\nhydrolase",
}

FAMILY_COLORS = {
    "Polyglutamate\nhydrolase": "#e74c3c",
    "Cyanophycinase": "#3498db",
    "Pseudomurein\nhydrolase": "#2ecc71",
}

FAMILY_ORDER = [
    "Polyglutamate\nhydrolase",
    "Cyanophycinase",
    "Pseudomurein\nhydrolase",
]

NYLC_AUTOCLEAVAGE_UID = "Q1EPR4_autocleavage"


def main():
    # Load family assignments
    fam_map = {}
    fam_path = os.path.join(ANALYSIS, "enzyme_family_classification.csv")
    with open(fam_path) as f:
        for r in csv.DictReader(f):
            if r["project"] == "hydrolase":
                pfam = r.get("assigned_family", "")
                if pfam in PFAM_TO_FAMILY:
                    fam_map[r["uniprot_id"]] = PFAM_TO_FAMILY[pfam]

    # Load holo pocket chemistry
    data = {fam: {"hydrophobic": [], "charged": []} for fam in FAMILY_ORDER}
    nylc_vals = {}

    holo_path = os.path.join(ANALYSIS, "holo_pocket_chemistry.csv")
    with open(holo_path) as f:
        for r in csv.DictReader(f):
            if r["project"] != "hydrolase":
                continue
            uid = r["uid"]

            if uid == NYLC_AUTOCLEAVAGE_UID:
                nylc_vals["hydrophobic"] = float(r["holo_frac_hydrophobic"]) * 100
                nylc_vals["charged"] = float(r["holo_frac_charged"]) * 100
                continue

            if uid in fam_map:
                fam = fam_map[uid]
                h = float(r["holo_frac_hydrophobic"]) * 100
                c = float(r["holo_frac_charged"]) * 100
                data[fam]["hydrophobic"].append(h)
                data[fam]["charged"].append(c)

    for fam in FAMILY_ORDER:
        print(f"{fam.replace(chr(10), ' ')}: n={len(data[fam]['hydrophobic'])}")
    print(f"NylC (autocleavage): hydrophobic={nylc_vals.get('hydrophobic', '?')}%, "
          f"charged={nylc_vals.get('charged', '?')}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    positions = np.arange(len(FAMILY_ORDER))
    box_width = 0.55

    # Panel A: Hydrophobic fraction
    bp1 = ax1.boxplot(
        [data[fam]["hydrophobic"] for fam in FAMILY_ORDER],
        positions=positions, widths=box_width, patch_artist=True,
        showfliers=True, flierprops=dict(marker="o", markersize=4, alpha=0.5),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, fam in zip(bp1["boxes"], FAMILY_ORDER):
        patch.set_facecolor(FAMILY_COLORS[fam])
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, fam in enumerate(FAMILY_ORDER):
        vals = data[fam]["hydrophobic"]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax1.scatter(positions[i] + jitter, vals, color=FAMILY_COLORS[fam],
                    s=20, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3)

    ax1.axhline(nylc_vals["hydrophobic"], color="#2c3e50", linestyle="--",
                linewidth=2, zorder=4)
    ax1.text(len(FAMILY_ORDER) - 0.5, nylc_vals["hydrophobic"] + 1,
             f'NylC ({nylc_vals["hydrophobic"]:.1f}%)',
             ha="right", va="bottom", fontsize=9, fontweight="bold",
             color="#2c3e50")

    ax1.set_xticks(positions)
    ax1.set_xticklabels(FAMILY_ORDER, fontsize=9)
    ax1.set_ylabel("Pocket hydrophobic fraction (%)", fontsize=11)
    ax1.set_title("(A) Hydrophobic fraction", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Charged fraction
    bp2 = ax2.boxplot(
        [data[fam]["charged"] for fam in FAMILY_ORDER],
        positions=positions, widths=box_width, patch_artist=True,
        showfliers=True, flierprops=dict(marker="o", markersize=4, alpha=0.5),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, fam in zip(bp2["boxes"], FAMILY_ORDER):
        patch.set_facecolor(FAMILY_COLORS[fam])
        patch.set_alpha(0.7)

    for i, fam in enumerate(FAMILY_ORDER):
        vals = data[fam]["charged"]
        jitter = np.random.default_rng(43).uniform(-0.15, 0.15, len(vals))
        ax2.scatter(positions[i] + jitter, vals, color=FAMILY_COLORS[fam],
                    s=20, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3)

    ax2.axhline(nylc_vals["charged"], color="#2c3e50", linestyle="--",
                linewidth=2, zorder=4)
    ax2.text(len(FAMILY_ORDER) - 0.5, nylc_vals["charged"] + 1,
             f'NylC ({nylc_vals["charged"]:.1f}%)',
             ha="right", va="bottom", fontsize=9, fontweight="bold",
             color="#2c3e50")

    ax2.set_xticks(positions)
    ax2.set_xticklabels(FAMILY_ORDER, fontsize=9)
    ax2.set_ylabel("Pocket charged fraction (%)", fontsize=11)
    ax2.set_title("(B) Charged fraction", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    out_path = os.path.join(ANALYSIS, "hydrolase_pocket_boxplot.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
