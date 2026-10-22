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
from declutter import place_labels  # noqa: E402

# fixed facet order: ANL fold first, then NIS -- identity never depends on the data
ENZ_ORDER = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA", "DesD"]

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

    # ---- FACET BY ENZYME -------------------------------------------------------------------
    # One axes per enzyme instead of one axes for everything. This is the fix for the overlap:
    # the enzyme name moves into the facet title, so each point needs only its 2-character panel
    # tag, and each facet holds <=10 points instead of 48. Shared axes keep every facet directly
    # comparable -- the whole point of a small-multiple.
    enzymes = [e for e in ENZ_ORDER if any(p["enz"] == e for p in pts)]
    ncol = 3
    nrow = -(-len(enzymes) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.15 * ncol, 3.75 * nrow),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
    xlo, xhi = min(xs) - 9, max(xs) + 9
    ylo, yhi = min(ys) - 9, max(ys) + 9

    for ax, enz in zip(axes, enzymes):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        # ideal bands: step 1 wants ~180 (in-line), step 2 wants ~105 (Burgi-Dunitz)
        ax.axvspan(150, max(185, xhi), color="#2a78d6", alpha=0.06, zorder=0)
        ax.axhspan(95, 115, color="#2a78d6", alpha=0.06, zorder=0)

        mine = [p for p in pts if p["enz"] == enz]
        n_act = sum(1 for p in mine if not p["lod"])
        for p in mine:
            col = C_LOD if p["lod"] else RAMP(0.18 + 0.82 * p["frac"])
            interp = p["x_ok"] and p["y_ok"]
            ax.scatter(p["x"], p["y"], s=150, marker="o", zorder=4,
                       facecolor=col if interp else SURFACE, edgecolor=col, linewidths=2.2)
        place_labels(ax, [(p["x"], p["y"], p["panel"].replace("Figure ", ""),
                           C_LOD if p["lod"] else "#1d4e8f") for p in mine],
                     fontsize=7.8, marker_pad=6.5)

        ax.set_title(f"{enz}   ·   {n_act}/{len(mine)} active",
                     fontsize=10.5, color=INK, pad=7, loc="left")
        ax.grid(True, color=INK3, alpha=0.22, ls=(0, (1, 4)), lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)

    for ax in axes[len(enzymes):]:
        ax.set_visible(False)

    # band annotations once, on the first facet only -- repeating them 6x is noise
    axes[0].text(0.03, 0.97, "Bürgi–Dunitz band 95–115°", transform=axes[0].transAxes,
                 va="top", fontsize=7.4, color=INK3)
    axes[0].text(0.97, 0.03, "in-line band\n150–180°", transform=axes[0].transAxes,
                 ha="right", fontsize=7.4, color=INK3, linespacing=1.3)

    fig.supxlabel("step 1 — A, proximity-weighted attack angle (°)\n"
                  "donor carboxylate O → ATP Pα   ·   ideal ≈ 180° (in-line)",
                  fontsize=10.5, color=INK2, y=0.055)
    fig.supylabel("step 2 — Bürgi–Dunitz angle (°)\n"
                  "amine N → acyl-adenylate carbonyl C   ·   ideal ≈ 105°",
                  fontsize=10.5, color=INK2, x=0.022)

    n_done = len(pts)
    fig.text(0.072, 0.985, "Both half-reactions modelled; measured activity as hue",
             fontsize=14.5, color=INK, va="top", fontweight="medium")
    fig.text(0.072, 0.949,
             f"{n_done} (enzyme × reaction) cells with BOTH steps scored · one facet per enzyme · "
             f"point labels are the figure panel\n"
             f"Activity is ranked within its own panel (readouts are not comparable across "
             f"figures). Hollow = that step's $n_{{gated}}$ = 0 — angle measured outside the "
             f"4 Å gate, in empty space.",
             fontsize=9.2, color=INK2, va="top", linespacing=1.6)

    fig.subplots_adjust(top=0.855, bottom=0.145, left=0.085, right=0.885,
                        hspace=0.33, wspace=0.11)

    # colorbar on its OWN axes, outside the facet grid. Letting matplotlib steal space from a
    # shared-axes grid put the bar on top of two facets.
    cax = fig.add_axes([0.915, 0.30, 0.0125, 0.40])
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("measured activity\n(rank within panel: low → high)", fontsize=9, color=INK2,
                 labelpad=10)
    cb.ax.tick_params(labelsize=0, length=0)
    cb.outline.set_visible(False)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=C_LOD, mec=C_LOD, ms=9, label="below LOD"),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK3, ms=9, mew=2.2,
               label="hollow: that step's $n_{gated}$ = 0 (angle measured outside the gate)"),
    ], loc="lower center", bbox_to_anchor=(0.49, 0.005), ncol=2, frameon=False,
        fontsize=9.2, labelcolor=INK2)

    out = CAMP / "analysis/FIGURE_two_step_activity.png"
    fig.savefig(out, dpi=330, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}  ({n_done} cells with both steps)")


if __name__ == "__main__":
    main()
