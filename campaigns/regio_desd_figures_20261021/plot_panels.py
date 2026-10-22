#!/usr/bin/env python3
"""Plot modelled `A` against the measured activity, one figure per experimental panel.

Fires as soon as a panel's enzymes are all scored -- run it repeatedly while the fleet works.

Reads:
  analysis/step1_scores.csv                       (from analyze_step1.py)
  ../../exp_data/.../regio_fig2/regio_fig2_normalized.csv   (verified digitization, is_mean rows)

Honesty rules baked into every panel:
  * non-dockers (n_gated == 0) are drawn as HOLLOW markers -- `A` is not interpretable for them,
    and they are shown rather than dropped, because dropping them turns "did the ligand dock"
    into a fake catalytic signal
  * below-LOD enzymes are drawn at the axis floor and labelled, not as a real low number
  * the pre-registered within-acid rank test and its EXACT p-value are printed into the panel,
    together with the power ceiling, so a "nice-looking" plot cannot be mistaken for significance
  * n_gated is annotated per point (validity precondition, never a ranking input)
"""
import csv, math, pathlib, sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
FIG2 = ROOT / "exp_data/amide_synthetase/literature_activity/regio_fig2/regio_fig2_normalized.csv"
OUT = CAMP / "fig2_diads"

# panel -> the step-1 acid donor it activates (see MECHANISM.md sec 4)
PANEL_ACID = {"Figure 2A": "A", "Figure 2B": "Z", "Figure 2C": "H", "Figure 2D": "H"}
ENZ_ORDER = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA", "DesD"]


def load_labels():
    """is_mean rows only -- replicate rows are partly reconstructed (README caveat 1)."""
    lab = {}
    for r in csv.DictReader(FIG2.open()):
        if r["is_mean"].strip().lower() != "true":
            continue
        if r["enzyme"] == "Control":
            continue
        lab.setdefault(r["figure"].strip(), {})[r["enzyme"]] = dict(
            value=float(r["value"]),
            below_LOD=r["below_LOD"].strip().lower() == "true",
            readout=r["readout_type"].strip())
    return lab


def rank_test(scores, actives):
    """One-tailed exact p that the actives occupy the top |actives| ranks by A."""
    enzymes = list(scores)
    k = len(actives)
    if k == 0 or k == len(enzymes):
        return None, None
    top = {e for e in sorted(enzymes, key=lambda e: -scores[e])[:k]}
    perfect = top == set(actives)
    p = 1 / math.comb(len(enzymes), k)
    return perfect, p


def main():
    scores_csv = CAMP / "analysis/step1_scores.csv"
    if not scores_csv.exists():
        sys.exit("no scores yet -- run analyze_step1.py first")
    sc = {}
    for r in csv.DictReader(scores_csv.open()):
        if int(r["n_mg"]) != 1:          # keep the 2-Mg sensitivity set out of the panels
            continue
        sc[(r["enzyme"], r["acid_code"])] = r

    labels = load_labels()
    OUT.mkdir(exist_ok=True)
    made = []

    for panel, acid in PANEL_ACID.items():
        lab = labels.get(panel)
        if not lab:
            continue
        have = [e for e in ENZ_ORDER if (e, acid) in sc and e in lab]
        if len(have) < len(ENZ_ORDER):
            print(f"{panel}: {len(have)}/{len(ENZ_ORDER)} enzymes scored -- waiting")
            continue

        A = {e: float(sc[(e, acid)]["A_deg"]) for e in have}
        gated = {e: int(sc[(e, acid)]["n_gated"]) for e in have}
        docked = {e: sc[(e, acid)]["docked"] == "True" for e in have}
        actives = [e for e in have if not lab[e]["below_LOD"]]

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        floor = min(v["value"] for v in lab.values() if not v["below_LOD"]) * 0.12 or 0.05
        for e in have:
            y = floor if lab[e]["below_LOD"] else lab[e]["value"]
            is_act = e in actives
            ax.scatter(A[e], y, s=190,
                       facecolor=("#2b7bba" if is_act else "white") if docked[e] else "white",
                       edgecolor="#2b7bba" if is_act else "#b03030",
                       linewidths=2.2, zorder=3,
                       marker="o" if docked[e] else "s")
            ax.annotate(f"{e}\nn_gated={gated[e]}", (A[e], y), textcoords="offset points",
                        xytext=(9, 6), fontsize=8.5, color="#333")
        ax.axhline(floor, ls=":", c="#b03030", lw=1.2)
        ax.text(0.015, floor, " below LOD (not a real low value)", color="#b03030",
                fontsize=8, va="bottom", transform=ax.get_yaxis_transform())

        perfect, p = rank_test(A, actives)
        if perfect is None:
            verdict = f"no contrast in this panel ({len(actives)}/{len(have)} active)"
        else:
            verdict = (f"pre-registered rank test: actives in top-{len(actives)} by A = "
                       f"{'YES' if perfect else 'NO'}   exact p = {p:.4f}"
                       f"{'' if perfect else ' (not attained)'}")
        ax.set_title(f"{panel} — acid donor {acid}\n{verdict}", fontsize=10.5)
        ax.set_xlabel("modelled A — proximity-weighted attack angle (deg), step 1 acid side")
        ax.set_ylabel(f"measured activity ({lab[have[0]]['readout']})")
        if lab[have[0]]["readout"] == "log10_ms":
            pass
        else:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, ls="--")
        ax.text(0.5, -0.17,
                "filled = active & docked · hollow circle = active, n_gated=0 (A not interpretable) · "
                "square = inactive\npower ceiling for this design: best attainable p = 0.048 "
                "(6 enzymes/stratum) — a null here is uninformative, not refuting",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.6, color="#555")
        fig.tight_layout()
        f = OUT / f"{panel.replace(' ', '_').lower()}_A_vs_activity.png"
        fig.savefig(f, dpi=185, bbox_inches="tight")
        plt.close(fig)
        made.append(f)
        print(f"{panel}: WROTE {f.name}  |  {verdict}")

    if not made:
        print("no panel complete yet")


if __name__ == "__main__":
    main()
