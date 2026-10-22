#!/usr/bin/env python3
"""Docking-validity diagnostic -- plot this BEFORE believing any A ranking.

`A` is only interpretable for an ensemble where the acid donor actually reaches P-alpha. If
n_gated == 0 the donor never came within the 4 A validity gate in any of 50 poses, and `A` is then
an angle measured at 6-13 A -- geometrically meaningless for catalysis.

This figure answers one question: does this campaign have the dynamic range to test anything, or is
the whole panel sitting outside the gate? It is the same failure that made the hydrolase campaign's
apparent signal turn out to be nothing but "did the ligand dock at all".

Left block: A vs harmonic distance D_h, coloured by n_gated, with the 4 A gate drawn, FACETED BY
         ENZYME. It was one axes with 58 two-part labels and was unreadable; faceting moves the
         enzyme name into the facet title so each point needs only its 1-3 character acid code.
Right:   n_gated distribution for this campaign beside the prior campaign's ensembles (same step-1
         setup, different substrate panel) -- establishes whether low docking is inherited from the
         setup or introduced here.
"""
import csv, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
from plot_panels import INK, INK2, INK3, SURFACE  # noqa: E402
from declutter import place_labels  # noqa: E402

PRIOR = (CAMP.parents[1] / "exp_data/amide_synthetase/results/modeling/literature_cofolding"
         / "all_step1_nac_summary.csv")
OUT = CAMP / "analysis"

# sequential, ONE hue light->dark (magnitude). viridis is multi-hue and reads as a rainbow here.
RAMP = LinearSegmentedColormap.from_list("gated", ["#dce9f8", "#2a78d6", "#0d2b4e"])
C_GATE = "#b03030"
ENZ_ORDER = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA", "DesD"]   # ANL fold first, then NIS


def main():
    rows = list(csv.DictReader((OUT / "step1_scores.csv").open()))
    if not rows:
        raise SystemExit("no scores yet")

    fig = plt.figure(figsize=(17.2, 7.4))
    fig.patch.set_facecolor(SURFACE)
    # column 3 is a DEDICATED slot for the colorbar. Floating it in the inter-column gap put its
    # tick labels on top of the histogram's y-axis.
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.30, 1.30],
                          left=0.052, right=0.975, top=0.845, bottom=0.135,
                          hspace=0.34, wspace=0.22)

    # The facets show the 1-Mg panel -- the campaign's actual design. The three DesD 2-Mg
    # sensitivity folds re-use donors A/Z/H, so plotting them here puts two identically-labelled
    # points in the most crowded facet; they are reported in MECHANISM.md section 2 against their
    # pre-declared ΔA threshold instead, which is where that comparison belongs.
    facet_rows = [r for r in rows if r["n_mg"] == "1"]
    n_2mg = len(rows) - len(facet_rows)
    A = [float(r["A_deg"]) for r in facet_rows]
    D = [float(r["D_h_A"]) for r in facet_rows]
    G = [int(r["n_gated"]) for r in facet_rows]
    gmax = max(5, max(G))
    xlo, xhi = min(D) - 0.7, max(D) + 1.4
    ylo, yhi = min(A) - 9, max(A) + 11

    enzymes = [e for e in ENZ_ORDER if any(r["enzyme"] == e for r in facet_rows)]
    axes = []
    for i, enz in enumerate(enzymes):
        ax = fig.add_subplot(gs[i // 3, i % 3],
                             sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.set_facecolor(SURFACE)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.axhspan(150, 180, color="#2a78d6", alpha=0.08, zorder=0)
        ax.axvline(4.0, color=C_GATE, ls="--", lw=1.4, zorder=1)

        mine = [r for r in facet_rows if r["enzyme"] == enz]
        pts = []
        for r in mine:
            d, a, g = float(r["D_h_A"]), float(r["A_deg"]), int(r["n_gated"])
            ax.scatter(d, a, s=150, zorder=3, linewidths=1.6,
                       facecolor=RAMP(g / gmax) if g else SURFACE,
                       edgecolor=RAMP(g / gmax) if g else INK3)
            pts.append((d, a, r["acid_code"], INK2 if g else INK3))
        place_labels(ax, pts, fontsize=7.8, marker_pad=6.5)

        n0 = sum(1 for r in mine if int(r["n_gated"]) == 0)
        ax.set_title(f"{enz}   ·   {n0}/{len(mine)} outside the gate",
                     fontsize=10.5, color=INK, pad=7, loc="left")
        ax.grid(True, color=INK3, alpha=0.22, ls=(0, (1, 4)), lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)

    axes[0].text(4.25, yhi - 2, "4 Å validity gate", color=C_GATE, fontsize=7.6,
                 va="top", ha="left", rotation=90)
    axes[0].text(0.975, 0.975, "in-line attack band 150–180°", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=7.4, color=INK3)

    fig.text(0.052 + 0.33, 0.055,
             "harmonic mean distance $D_h$ (Å)  —  donor carboxylate O → ATP Pα",
             ha="center", fontsize=10.5, color=INK2)
    fig.text(0.013, 0.49, "A — proximity-weighted attack angle (°)",
             rotation=90, va="center", fontsize=10.5, color=INK2)

    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, gmax))
    cax = fig.add_subplot(gs[1, 3])
    cax.set_box_aspect(14)          # a slim bar inside a wide slot
    cb = fig.colorbar(sm, cax=cax)
    # ticks and label on the LEFT of the bar: on the right they ran into the histogram's own
    # "ensembles" axis label, which sits immediately beside this slot.
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cb.set_label("$n_{gated}$  (poses with d < 4 Å, of 50)", fontsize=8.6, color=INK2,
                 labelpad=6)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=2.5); cb.outline.set_visible(False)

    # ---- right: n_gated, this campaign vs the prior one -----------------------------------
    ax2 = fig.add_subplot(gs[:, 4])
    ax2.set_facecolor(SURFACE)
    mine = G          # 1-Mg panel, matching the facets
    prior = []
    if PRIOR.exists():
        for r in csv.DictReader(PRIOR.open()):
            try:
                prior.append(int(float(r["n_gated"])))
            except (ValueError, KeyError):
                pass
    # 5-pose bins: unit bins over 0-50 produced a comb of 1-count spikes that read as noise
    bins = list(range(0, 55, 5))
    ax2.hist([mine, prior], bins=bins,
             label=[f"this campaign (n={len(mine)}, 1 Mg²⁺ panel)",
                    f"prior step-1 campaign (n={len(prior)})"],
             color=["#2a78d6", "#c3c7ce"], edgecolor=SURFACE, linewidth=1.6)
    ax2.set_xlabel("$n_{gated}$ — poses within 4 Å of Pα (of 50)", fontsize=10.5, color=INK2)
    ax2.set_ylabel("ensembles", fontsize=10.5, color=INK2, labelpad=7)
    ax2.set_title("Low docking is inherited from the setup,\nnot introduced here",
                  fontsize=10.5, color=INK, loc="left", pad=7, linespacing=1.4)
    ax2.legend(fontsize=9, frameon=False, labelcolor=INK2)
    ax2.grid(True, axis="y", color=INK3, alpha=0.22, ls=(0, (1, 4)), lw=0.8)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color(INK3)
    ax2.tick_params(colors=INK2, labelsize=8.5)

    n0 = sum(1 for g in mine if g == 0)
    fig.text(0.052, 0.985,
             "Is A interpretable? Hollow points never reached Pα in any of 50 poses",
             fontsize=14.5, color=INK, va="top", fontweight="medium")
    fig.text(0.052, 0.945,
             f"{n0}/{len(mine)} step-1 ensembles have $n_{{gated}}$ = 0 → A is NOT interpretable "
             f"for them: it is an angle measured at 6–16 Å, in empty space. Ensembles are reported,\n"
             f'never dropped — dropping non-dockers is how "did the ligand dock at all" masquerades '
             f"as a catalytic signal. Facets show the 1 Mg²⁺ panel; the {n_2mg} DesD 2 Mg²⁺ "
             f"sensitivity folds are reported in MECHANISM.md §2.",
             fontsize=9.2, color=INK2, va="top", linespacing=1.6)

    f = OUT / "docking_validity_diagnostic.png"
    fig.savefig(f, dpi=200, facecolor=SURFACE)
    fig.savefig(f.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"wrote {f}")
    print(f"  n_gated==0: {n0}/{len(mine)} this campaign; "
          f"{sum(1 for g in prior if g == 0)}/{len(prior)} prior campaign")


if __name__ == "__main__":
    main()
