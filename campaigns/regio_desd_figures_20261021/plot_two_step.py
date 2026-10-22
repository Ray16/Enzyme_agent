#!/usr/bin/env python3
"""THE two-step figure: step 1 on x, step 2 on y, measured activity as hue.

This is the right frame for a two-step enzyme. The previous figures plotted one modelled quantity
against activity, which forces activity onto an axis and can only ever show one half-reaction. Here
BOTH modelled steps occupy the axes and the experiment is the encoded outcome, so the question
becomes the natural one: does a cell being good at BOTH steps correspond to it being active?

  x  step-1 A        proximity-weighted attack angle, donor carboxylate O -> ATP Pa, ideal ~180 deg
  y  step-2 BD       Burgi-Dunitz angle, amine N -> acyl-adenylate carbonyl C, ideal ~105 deg
  hue  measured product signal, ranked WITHIN its experimental panel (readouts are not comparable
       across panels: 2A/2B are normalised MS, the rest are log10 relative MS)
  below-LOD cells get a distinct status colour, never a position on the activity ramp -- a censored
  value is "not detected", not a small number.

Validity is shown, never hidden: a cell whose step-1 ensemble never reached the 4 A gate
(n_gated = 0) is drawn with a hollow x-facing marker, because its A is an angle measured in empty
space -- DdaG on donor MA reads a beautiful 141.8 deg from 0/50 docked poses.
"""
import csv, math, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP))
from plot_panels import SOURCES, REACTION, load, INK, INK2, INK3, SURFACE  # noqa: E402

C_LOD = "#eb6834"
# sequential ramp, one hue light->dark (magnitude); NOT a rainbow
RAMP = LinearSegmentedColormap.from_list("act", ["#cfe3f7", "#2a78d6", "#10325c"])

# panel -> (step-1 acid donor, step-2 acceptor)
PAIRS = {
    "Figure 2A": ("A", "X"), "Figure 2B": ("Z", "M"),
    "Figure 2C": ("H", "H"), "Figure 2D": ("H", "M"),
    "Figure 4A": ("G", "MA"), "Figure 4B": ("MA", "X"), "Figure 4C": ("MA", "H"),
    "Figure 5A": ("A", "MAM"), "Figure 5B": ("AMA", "M"), "Figure 5C": ("MA", "MA"),
}


def main():
    s1 = {}
    f1 = CAMP / "analysis/step1_scores.csv"
    if f1.exists():
        for r in csv.DictReader(f1.open()):
            if int(r["n_mg"]) == 1:
                s1[(r["enzyme"], r["acid_code"])] = r
    s2 = {}
    f2 = CAMP / "analysis/step2_scores.csv"
    if f2.exists():
        for r in csv.DictReader(f2.open()):
            s2[(r["enzyme"], r["donor_code"], r["acceptor_code"])] = r
    if not s2:
        print("no step-2 ensembles scored yet -- two-step figure not drawn")
        return

    labels = {}
    for _n, (path, _p, _o) in SOURCES.items():
        if path.exists():
            labels.update(load(path))

    pts = []
    for panel, (donor, acc) in PAIRS.items():
        lab = labels.get(panel)
        if not lab:
            continue
        # rank activity WITHIN panel; readouts are not comparable across panels
        det = {e: v["value"] for e, v in lab.items() if not v["below_LOD"]}
        lo, hi = (min(det.values()), max(det.values())) if det else (0, 1)
        for e, v in lab.items():
            a, b = s1.get((e, donor)), s2.get((e, donor, acc))
            if not a or not b:
                continue
            frac = None if v["below_LOD"] else (
                0.5 if hi == lo else (v["value"] - lo) / (hi - lo))
            pts.append(dict(panel=panel, enz=e, x=float(a["A_deg"]), y=float(b["BD_deg"]),
                            x_ok=int(a["n_gated"]) > 0, y_ok=int(b["n_gated"]) > 0,
                            lod=v["below_LOD"], frac=frac, value=v["value"]))
    if not pts:
        print("no cell has BOTH steps scored yet -- two-step figure not drawn")
        return

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    # ideal bands: step 1 wants ~180 (in-line), step 2 wants ~105 (Burgi-Dunitz)
    ax.axvspan(150, 185, color="#2a78d6", alpha=0.06, zorder=0)
    ax.axhspan(95, 115, color="#2a78d6", alpha=0.06, zorder=0)
    ax.text(0.995, 0.012, "step-1 in-line band 150–180°", transform=ax.transAxes,
            ha="right", fontsize=7.8, color=INK3)
    ax.text(0.012, 0.985, "step-2 Bürgi–Dunitz band 95–115°", transform=ax.transAxes,
            va="top", fontsize=7.8, color=INK3)

    for p in pts:
        col = C_LOD if p["lod"] else RAMP(0.18 + 0.82 * p["frac"])
        interp = p["x_ok"] and p["y_ok"]
        ax.scatter(p["x"], p["y"], s=215, marker="o", zorder=4,
                   facecolor=col if interp else SURFACE, edgecolor=col, linewidths=2.5)
        if not interp:
            bad = "step 1" if not p["x_ok"] else ("step 2" if not p["y_ok"] else "")
            ax.annotate(f"{p['enz']}\n{p['panel'].replace('Figure ','')}  ($n_{{gated}}$=0, {bad})",
                        (p["x"], p["y"]), textcoords="offset points", xytext=(0, 15),
                        ha="center", fontsize=7.4, color=INK2, linespacing=1.3, zorder=5)
        else:
            ax.annotate(f"{p['enz']}\n{p['panel'].replace('Figure ','')}", (p["x"], p["y"]),
                        textcoords="offset points", xytext=(0, 15), ha="center",
                        fontsize=7.6, color=INK2, linespacing=1.3, zorder=5)

    ax.set_xlabel("step 1 — A, proximity-weighted attack angle (°)\n"
                  "donor carboxylate O → ATP Pα   ·   ideal ≈ 180° (in-line)",
                  fontsize=10, color=INK2, labelpad=9)
    ax.set_ylabel("step 2 — Bürgi–Dunitz angle (°)\n"
                  "amine N → acyl-adenylate carbonyl C   ·   ideal ≈ 105°",
                  fontsize=10, color=INK2, labelpad=9)
    n_done = len(pts)
    ax.text(0, 1.085, "Both half-reactions modelled; measured activity as hue",
            transform=ax.transAxes, fontsize=12.5, color=INK, va="bottom", fontweight="medium")
    ax.text(0, 1.028,
            f"{n_done} (enzyme × reaction) cells with BOTH steps scored. Activity is ranked within "
            f"its own panel —\nreadouts are not comparable across figures. Hollow = $n_{{gated}}$ = 0, "
            f"that step's angle measured outside the gate.",
            transform=ax.transAxes, fontsize=8.8, color=INK2, va="bottom", linespacing=1.45)

    ax.grid(True, color=INK3, alpha=0.22, ls=(0, (1, 4)), lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=9)

    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("measured activity (rank within panel: low → high)", fontsize=8.8, color=INK2)
    cb.ax.tick_params(labelsize=0, length=0)
    cb.outline.set_visible(False)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=C_LOD, mec=C_LOD, ms=9, label="below LOD"),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK3, ms=9, mew=2.2,
               label="hollow: that step's $n_{gated}$ = 0"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
        fontsize=8.6, labelcolor=INK2)

    out = CAMP / "analysis/FIGURE_two_step_activity.png"
    fig.savefig(out, dpi=330, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n_done} cells with both steps)")


if __name__ == "__main__":
    main()
