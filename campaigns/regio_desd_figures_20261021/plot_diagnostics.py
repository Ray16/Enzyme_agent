#!/usr/bin/env python3
"""Docking-validity diagnostic -- plot this BEFORE believing any A ranking.

`A` is only interpretable for an ensemble where the acid donor actually reaches P-alpha. If
n_gated == 0 the donor never came within the 4 A validity gate in any of 50 poses, and `A` is then
an angle measured at 6-13 A -- geometrically meaningless for catalysis.

This figure answers one question: does this campaign have the dynamic range to test anything, or is
the whole panel sitting outside the gate? It is the same failure that made the hydrolase campaign's
apparent signal turn out to be nothing but "did the ligand dock at all".

Panel 1: A vs harmonic distance D_h, coloured by n_gated, with the 4 A gate drawn.
Panel 2: n_gated distribution for this campaign beside the prior campaign's 28 ensembles
         (same step-1 setup, different substrate panel) -- establishes whether low docking is
         inherited from the setup or introduced here.
"""
import csv, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMP = pathlib.Path(__file__).resolve().parent
PRIOR = (CAMP.parents[1] / "exp_data/amide_synthetase/results/modeling/literature_cofolding"
         / "all_step1_nac_summary.csv")
OUT = CAMP / "analysis"


def main():
    rows = list(csv.DictReader((OUT / "step1_scores.csv").open()))
    if not rows:
        raise SystemExit("no scores yet")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # ---- panel 1: A vs D_h --------------------------------------------------------------
    A = [float(r["A_deg"]) for r in rows]
    D = [float(r["D_h_A"]) for r in rows]
    G = [int(r["n_gated"]) for r in rows]
    sc = ax1.scatter(D, A, c=G, cmap="viridis", s=170, edgecolor="k", linewidths=0.8,
                     vmin=0, vmax=max(5, max(G)), zorder=3)
    for r, d, a in zip(rows, D, A):
        ax1.annotate(f"{r['enzyme']}·{r['acid_code']}", (d, a), fontsize=7.4,
                     textcoords="offset points", xytext=(7, -3), color="#333")
    ax1.axvline(4.0, color="#b03030", ls="--", lw=1.6)
    ax1.text(4.0, ax1.get_ylim()[1], " 4 Å validity gate", color="#b03030", fontsize=8.4,
             va="top", ha="left", rotation=90)
    ax1.axhspan(150, 180, color="#2b7bba", alpha=0.10)
    ax1.text(0.98, 0.03, "in-line attack band (150–180°)", transform=ax1.transAxes,
             ha="right", fontsize=8, color="#2b7bba")
    ax1.set_xlabel("harmonic mean distance $D_h$ (Å)  —  donor carboxylate O → ATP Pα")
    ax1.set_ylabel("A — proximity-weighted attack angle (deg)")
    ax1.set_title("Is A interpretable? Points right of the gate never reached Pα", fontsize=10.5)
    ax1.grid(alpha=0.25, ls="--")
    plt.colorbar(sc, ax=ax1, label="n_gated (poses with d < 4 Å, of 50)")

    # ---- panel 2: n_gated, this campaign vs the prior one ---------------------------------
    mine = [int(r["n_gated"]) for r in rows]
    prior = []
    if PRIOR.exists():
        for r in csv.DictReader(PRIOR.open()):
            try:
                prior.append(int(float(r["n_gated"])))
            except (ValueError, KeyError):
                pass
    bins = range(0, max(mine + prior + [1]) + 2)
    ax2.hist([mine, prior], bins=bins, label=[f"this campaign (n={len(mine)}, in progress)",
                                              f"prior step-1 campaign (n={len(prior)})"],
             color=["#2b7bba", "#bbbbbb"], edgecolor="k", linewidth=0.7)
    ax2.set_xlabel("n_gated — poses within 4 Å of Pα (of 50)")
    ax2.set_ylabel("ensembles")
    ax2.set_title("Low docking is inherited from the setup, not introduced here", fontsize=10.5)
    ax2.legend(fontsize=8.6)
    ax2.grid(alpha=0.25, ls="--", axis="y")

    n0 = sum(1 for g in mine if g == 0)
    fig.text(0.5, -0.02,
             f"{n0}/{len(mine)} ensembles scored so far have n_gated = 0 → A NOT interpretable for "
             f"them. Ensembles are reported, never dropped:\ndropping non-dockers is how "
             f'"did the ligand dock at all" masquerades as a catalytic signal.',
             ha="center", va="top", fontsize=8.4, color="#555")
    fig.tight_layout()
    f = OUT / "docking_validity_diagnostic.png"
    fig.savefig(f, dpi=185, bbox_inches="tight")
    print(f"wrote {f}")
    print(f"  n_gated==0: {n0}/{len(mine)} this campaign; "
          f"{sum(1 for g in prior if g == 0)}/{len(prior)} prior campaign")


if __name__ == "__main__":
    main()
