#!/usr/bin/env python3
"""Publication figures: modelled `A` vs measured activity, ONE experimental panel per figure.

Design follows the house dataviz procedure (form -> encoding -> validated color -> marks):
  form      scatter -- a relationship between two measures over 6 named entities
  color     2-slot categorical, VALIDATED (`validate_palette.js`, light surface):
            blue #2a78d6 = product detected, orange #eb6834 = below LOD
            worst adjacent CVD dE 24.7 (protan) / 33.6 (normal) -> PASS on all six checks
  fill      solid = `A` is interpretable (n_gated > 0); HOLLOW = n_gated 0, i.e. the donor never
            reached the 4 A validity gate in any of 50 poses, so `A` is an angle measured in
            empty space. Fill is the validity flag -- never dropped from the plot.
  censoring below-LOD points sit on a labelled floor with a down arrow (standard notation), never
            as a real small number
  labels    every enzyme direct-labelled with its n_gated; collisions resolved by alternating
            offsets; no number printed on every axis tick

Earlier versions keyed marker SHAPE to docking while the caption read "square = inactive", which
mislabelled SfaB -- an ACTIVE enzyme with n_gated 0. Colour now carries activity and fill carries
validity, and the legend states exactly that.
"""
import csv, math, pathlib, sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CAMP = pathlib.Path(__file__).resolve().parent
LIT = CAMP.parents[1] / "exp_data/amide_synthetase/literature_activity"

# validated 2-slot categorical palette + recessive ink
C_DET, C_LOD = "#2a78d6", "#eb6834"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"

SOURCES = {
    "regio_fig2": (LIT / "regio_fig2/regio_fig2_normalized.csv",
                   {"Figure 2A": "A", "Figure 2B": "Z", "Figure 2C": "H", "Figure 2D": "H"},
                   CAMP / "fig2_diads"),
    "regio_fig4": (LIT / "regio_fig4/regio_fig4_quantified.csv",
                   {"Figure 4A": "G", "Figure 4B": "MA", "Figure 4C": "MA"},
                   CAMP / "fig4_triads"),
    "regio_fig5": (LIT / "regio_fig5/regio_fig5_quantified.csv",
                   {"Figure 5A": "A", "Figure 5B": "AMA", "Figure 5C": "MA"},
                   CAMP / "fig5_tetrads"),
}
REACTION = {
    "Figure 2A": "X + A → XA", "Figure 2B": "M + Z → MZ",
    "Figure 2C": "H + H → HH", "Figure 2D": "M + H → MH",
    "Figure 4A": "MA + G → GMA", "Figure 4B": "MA + X → MAX", "Figure 4C": "MA + H → MAH",
    "Figure 5A": "MAM + A → MAMA", "Figure 5B": "AMA + M → MAMA", "Figure 5C": "MA + MA → MAMA",
}


def spearman(x, y):
    n = len(x)
    if n < 3:
        return None, None
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return None, None
    rho = num / den
    # Exact two-sided permutation p enumerates n! orderings -- fine at n<=8 (<=40320), ruinous
    # beyond it (n=20 never returns). Above the cutoff use the asymptotic t approximation.
    if n <= 8:
        from itertools import permutations
        target = abs(rho) - 1e-12
        cnt = tot = 0
        for perm in permutations(ry):
            num2 = sum((a - mx) * (b - my) for a, b in zip(rx, perm))
            tot += 1
            if abs(num2 / den) >= target:
                cnt += 1
        return rho, cnt / tot
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # two-sided Student-t survival via the regularised incomplete beta
    df = n - 2
    x = df / (df + t * t)
    return rho, _betainc(df / 2, 0.5, x)


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz).

    The continued fraction only converges well for x below (a+1)/(a+b+2); above it the identity
    I_x(a,b) = 1 - I_{1-x}(b,a) must be applied by RECURSING on the swapped parameters. An earlier
    version merely returned `1 - res` from the same (a,b,x) evaluation, which silently returned the
    complement of the correct p -- caught by validating against scipy (n=12: 0.735 vs 0.265).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def rank_test(scores, actives):
    enzymes = list(scores)
    k = len(actives)
    if k == 0 or k == len(enzymes):
        return None, None
    top = set(sorted(enzymes, key=lambda e: -scores[e])[:k])
    return top == set(actives), 1 / math.comb(len(enzymes), k)


def load(csv_path):
    lab = {}
    for r in csv.DictReader(csv_path.open()):
        if r["is_mean"].strip().lower() != "true" or r["enzyme"] == "Control":
            continue
        lab.setdefault(r["figure"].strip(), {})[r["enzyme"]] = dict(
            value=float(r["value"]),
            below_LOD=r["below_LOD"].strip().lower() == "true",
            readout=r["readout_type"].strip())
    return lab


def draw(panel, acid, lab, sc, outdir):
    have = [e for e in lab if (e, acid) in sc]
    if len(have) < len(lab):
        return None, f"{len(have)}/{len(lab)} enzymes scored"

    A = {e: float(sc[(e, acid)]["A_deg"]) for e in have}
    gated = {e: int(sc[(e, acid)]["n_gated"]) for e in have}
    detected = [e for e in have if not lab[e]["below_LOD"]]
    readout = lab[have[0]]["readout"]
    logy = readout != "log10_ms"

    vals = [lab[e]["value"] for e in detected]
    n_cens = sum(1 for e in have if lab[e]["below_LOD"])
    # Only reserve room for an LOD floor when something is ACTUALLY censored -- otherwise the
    # floor line and its headroom are pure dead space (Fig 2A has 0 censored enzymes).
    floor = ((min(vals) * 0.16) if logy else (min(vals) - 0.8)) if (n_cens and vals) else None

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    if logy:
        ax.set_yscale("log")

    order = sorted(have, key=lambda e: A[e])
    pts = {}
    for e in order:
        cens = lab[e]["below_LOD"]
        y = (floor if cens else lab[e]["value"])
        col = C_LOD if cens else C_DET
        ok = gated[e] > 0
        ax.scatter(A[e], y, s=225, marker="o", zorder=4,
                   facecolor=col if ok else SURFACE, edgecolor=col, linewidths=2.6)
        if cens:
            ax.annotate("", xy=(A[e], y * 0.62 if logy else y - 0.4), xytext=(A[e], y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9,
                                        shrinkA=9, shrinkB=0), zorder=3)
        pts[e] = (A[e], y)

    perfect, p_rank = rank_test(A, detected)
    rho, p_rho = spearman([A[e] for e in detected],
                          [lab[e]["value"] for e in detected]) if len(detected) >= 3 else (None, None)

    if perfect is None:
        verdict = f"no active/inactive contrast — {len(detected)}/{len(have)} enzymes detected"
    else:
        verdict = (f"detected enzymes occupy the top {len(detected)} by A: "
                   f"{'YES' if perfect else 'NO'}  (exact p = {p_rank:.3f})")
    stats = verdict
    if rho is not None:
        stats += f"\namong detected: Spearman ρ(A, activity) = {rho:+.2f}, exact p = {p_rho:.3f}"

    # title and verdict stacked ABOVE the axes with explicit, non-overlapping anchors
    n_lines = stats.count("\n") + 1
    ax.text(0, 1.045 + 0.055 * n_lines, f"{panel}   ·   {REACTION.get(panel, '')}"
            f"   ·   acid donor {acid}", transform=ax.transAxes, fontsize=12.5,
            color=INK, va="bottom", ha="left", fontweight="medium")
    _stats_txt = ax.text(0, 1.035, stats, transform=ax.transAxes, fontsize=9.2, color=INK2,
                         va="bottom", ha="left", linespacing=1.45)

    # --- label placement by real collision detection in DISPLAY coords -----------------------
    # An alternating up/down heuristic ignores y and collided (AsbB sits low, DesD high, so
    # AsbB's "up" label landed inside DesD's "down" label). Try candidate offsets in order and
    # keep the first that hits neither a placed label nor any data point.
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())   # freeze before measuring
    fig.canvas.draw()
    CANDIDATES = [(0, 21), (0, -34), (34, 9), (-34, 9), (34, -16), (-34, -16),
                  (0, 40), (0, -53), (52, 9), (-52, 9)]
    fig.canvas.draw()
    _sb = _stats_txt.get_window_extent()
    placed = [(_sb.x0, _sb.y0, _sb.x1 + 1e4, _sb.y1)]   # the whole subtitle band is off-limits
    pt_boxes = []
    for e in order:
        x_d, y_d = ax.transData.transform(pts[e])
        pt_boxes.append((x_d - 13, y_d - 13, x_d + 13, y_d + 13))

    def hits(box, others, pad=3):
        x0, y0, x1, y1 = box
        return any(not (x1 + pad < o[0] or o[2] < x0 - pad or
                        y1 + pad < o[1] or o[3] < y0 - pad) for o in others)

    for e in sorted(order, key=lambda e: -gated[e]):   # best-docked labelled first
        txt = f"{e}\n$n_{{gated}}$ = {gated[e]}/50"
        chosen = None
        for dx, dy in CANDIDATES:
            ann = ax.annotate(txt, pts[e], textcoords="offset points", xytext=(dx, dy),
                              ha="center", fontsize=8.7, color=INK2, linespacing=1.35, zorder=5)
            fig.canvas.draw()
            bb = ann.get_window_extent()
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            if not hits(box, placed + pt_boxes):
                chosen = (ann, box); break
            ann.remove()
        if chosen is None:   # nothing clean -- take the first candidate rather than drop a label
            ann = ax.annotate(txt, pts[e], textcoords="offset points", xytext=CANDIDATES[0],
                              ha="center", fontsize=8.7, color=INK2, linespacing=1.35, zorder=5)
            fig.canvas.draw()
            bb = ann.get_window_extent()
            chosen = (ann, (bb.x0, bb.y0, bb.x1, bb.y1))
        placed.append(chosen[1])

    if floor is not None:
        ax.axhline(floor, color=C_LOD, ls=(0, (2, 3)), lw=1.3, zorder=1)
        # Put the caption on whichever side is emptier: right-aligning fixed a collision with a
        # left-side marker but then hit DdaG on the right in 4B.
        _mid = (min(A.values()) + max(A.values())) / 2
        _right_busy = any(A[e] > _mid for e in have if lab[e]["below_LOD"])
        ax.text(0.004 if _right_busy else 0.996, floor,
                " below limit of detection" if _right_busy else "below limit of detection ",
                color=C_LOD, fontsize=8.2, va="bottom",
                ha="left" if _right_busy else "right",
                transform=ax.get_yaxis_transform())

    ax.set_xlabel("modelled A — proximity-weighted attack angle (°),  step 1 acid side",
                  fontsize=10, color=INK2, labelpad=8)
    ax.set_ylabel("measured product signal"
                  + ("\n(normalised MS intensity)" if readout == "norm_ms"
                     else "\n(log₁₀ relative MS intensity)"),
                  fontsize=10, color=INK2, labelpad=8)
    ax.grid(True, color=INK3, alpha=0.22, ls=(0, (1, 4)), lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.margins(x=0.19, y=0.30)

    # legend BELOW the axes in one row -- never over the data (it previously covered AcsA).
    # Only show the entries this panel actually uses.
    handles = [Line2D([], [], marker="o", ls="", mfc=C_DET, mec=C_DET, ms=9,
                      label="product detected")]
    if n_cens:
        handles.append(Line2D([], [], marker="o", ls="", mfc=C_LOD, mec=C_LOD, ms=9,
                              label="below LOD (arrow: true value lower)"))
    if any(gated[e] == 0 for e in have):
        handles.append(Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=C_DET, ms=9, mew=2.2,
                              label="hollow: $n_{gated}$ = 0, A not interpretable"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.145),
              ncol=len(handles), frameon=False, fontsize=8.7, labelcolor=INK2,
              handletextpad=0.5, columnspacing=1.8)

    fig.text(0.5, -0.115,
             "A is scored over all 50 poses, no threshold. $n_{gated}$ (poses within 4 Å of Pα) is a "
             "VALIDITY PRECONDITION, never a ranking input.\nNon-docking ensembles are shown, not "
             "dropped — dropping them turns “did the ligand dock at all” into a false catalytic signal. "
             f"Power ceiling of this design: best attainable p = 0.048.",
             ha="center", va="top", fontsize=7.7, color=INK3, linespacing=1.5)

    outdir.mkdir(exist_ok=True)
    f = outdir / f"{panel.replace(' ', '_').lower()}_A_vs_activity.png"
    fig.savefig(f, dpi=330, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return f, stats.replace("\n", " | ")


def main():
    scores_csv = CAMP / "analysis/step1_scores.csv"
    if not scores_csv.exists():
        sys.exit("no scores yet -- run analyze_step1.py first")
    sc = {}
    for r in csv.DictReader(scores_csv.open()):
        if int(r["n_mg"]) == 1:
            sc[(r["enzyme"], r["acid_code"])] = r

    for name, (path, panels, outdir) in SOURCES.items():
        if not path.exists():
            continue
        labels = load(path)
        for panel, acid in panels.items():
            if panel not in labels:
                continue
            f, msg = draw(panel, acid, labels[panel], sc, outdir)
            print(f"{panel}: {'WROTE ' + f.name if f else 'waiting'}  |  {msg}")


if __name__ == "__main__":
    main()
