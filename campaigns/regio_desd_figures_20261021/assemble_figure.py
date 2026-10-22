#!/usr/bin/env python3
"""Assemble the campaign into ONE publication figure.

    (a) small multiples -- modelled A vs measured activity, one mini-panel per experimental panel
    (b) the confound    -- A vs n_gated across every ensemble
    (c) positive control -- predicted DesD+HSC pose vs crystal 7TGM
    (d) pre-registered test summary -- exact p per panel against the design's power ceiling

Encoding is identical to the per-panel figures (validated 2-slot categorical palette; colour =
measured activity, FILL = whether A is interpretable). Degrades gracefully: any sub-panel whose
data is not yet on disk is drawn as an explicit "pending" placeholder rather than silently omitted.
"""
import csv, math, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plot_panels import (CAMP, LIT, SOURCES, REACTION, C_DET, C_LOD, INK, INK2, INK3, SURFACE,
                         load, rank_test, spearman)

PANEL_ORDER = ["Figure 2A", "Figure 2B", "Figure 2C", "Figure 2D",
               "Figure 4A", "Figure 4B", "Figure 4C",
               "Figure 5A", "Figure 5B", "Figure 5C"]
PANEL_ACID = {}
for _n, (_p, _panels, _o) in SOURCES.items():
    PANEL_ACID.update(_panels)


def load_scores():
    f = CAMP / "analysis/step1_scores.csv"
    sc = {}
    if f.exists():
        for r in csv.DictReader(f.open()):
            if int(r["n_mg"]) == 1:
                sc[(r["enzyme"], r["acid_code"])] = r
    return sc


def main():
    sc = load_scores()
    labels = {}
    for _n, (path, panels, _o) in SOURCES.items():
        if path.exists():
            labels.update(load(path))

    xall = [float(r["A_deg"]) for r in sc.values()]
    XLIM = (min(xall) - 6, max(xall) + 6) if xall else (80, 170)

    fig = plt.figure(figsize=(15.2, 13.4))
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(4, 5, figure=fig, height_ratios=[1.0, 1.0, 1.18, 0.80],
                  hspace=0.42, wspace=0.40,
                  left=0.062, right=0.985, top=0.885, bottom=0.050)

    # ---------- (a) small multiples ------------------------------------------------------
    summary = []
    for i, panel in enumerate(PANEL_ORDER):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.set_facecolor(SURFACE)
        acid = PANEL_ACID.get(panel)
        lab = labels.get(panel)
        have = [e for e in (lab or {}) if (e, acid) in sc]
        ax.set_title(f"{panel.replace('Figure ', '')}   {REACTION.get(panel, '')}",
                     fontsize=9.2, color=INK, pad=5, loc="left")
        _ro = (lab or {}).get(next(iter(lab), ""), {}).get("readout") if lab else None
        ax.set_ylabel("norm. MS int.\n(SfaB = 1)" if _ro == "norm_ms"
                      else ("log$_{10}$ rel. MS int." if _ro else ""),
                      fontsize=7.4, color=INK3, labelpad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=7.4)
        ax.grid(True, color=INK3, alpha=0.2, ls=(0, (1, 4)), lw=0.7)
        ax.set_axisbelow(True)

        if not lab or len(have) < len(lab):
            ax.text(0.5, 0.5, f"pending\n{len(have)}/{len(lab or [])} folded",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.4, color=INK3)
            ax.set_xlim(*XLIM); ax.set_yticks([])
            summary.append((panel, None, None, None, None, False))
            continue

        A = {e: float(sc[(e, acid)]["A_deg"]) for e in have}
        gated = {e: int(sc[(e, acid)]["n_gated"]) for e in have}
        det = [e for e in have if not lab[e]["below_LOD"]]
        logy = lab[have[0]]["readout"] != "log10_ms"
        vals = [lab[e]["value"] for e in det]
        n_cens = len(have) - len(det)
        floor = ((min(vals) * 0.3) if logy else (min(vals) - 0.55)) if (n_cens and vals) else None

        if logy:
            ax.set_yscale("log")
        for e in sorted(have, key=lambda e: A[e]):
            cens = lab[e]["below_LOD"]
            y = floor if cens else lab[e]["value"]
            col = C_LOD if cens else C_DET
            ax.scatter(A[e], y, s=78, marker="o", zorder=4,
                       facecolor=col if gated[e] > 0 else SURFACE,
                       edgecolor=col, linewidths=1.7)
            ax.annotate(e, (A[e], y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.0, color=INK2, zorder=5)
        if floor is not None:
            ax.axhline(floor, color=C_LOD, ls=(0, (2, 3)), lw=1.0, zorder=1)
        ax.set_xlim(*XLIM)          # SHARED across all cells -- enables direct comparison
        ax.margins(y=0.34)

        perfect, p_rank = rank_test(A, det)
        rho, p_rho = (spearman([A[e] for e in det], [lab[e]["value"] for e in det])
                      if len(det) >= 3 else (None, None))
        if perfect is None:
            msg = f"no contrast ({len(det)}/{len(have)} active)"
        else:
            msg = f"top-{len(det)} by A: {'YES' if perfect else 'NO'}  p={p_rank:.3f}"
        note = msg + (f"\nρ={rho:+.2f}, p={p_rho:.3f}" if rho is not None else "")
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7.0,
                va="bottom", ha="left", linespacing=1.45,
                color=INK2 if (perfect is None or not perfect) else "#1b7a4b")
        summary.append((panel, perfect, p_rank, rho, p_rho, True))

    fig.text(0.062, 0.922, "a", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.082, 0.922, "modelled A (x, °) vs measured product signal (y) — one experimental "
             "panel per cell", fontsize=10.2, color=INK2)

    # ---------- (b) the confound: A vs n_gated -------------------------------------------
    axb = fig.add_subplot(gs[2, 0:2]); axb.set_facecolor(SURFACE)
    allr = [r for r in sc.values()]
    if allr:
        xs = [int(r["n_gated"]) for r in allr]
        ys = [float(r["A_deg"]) for r in allr]
        axb.scatter(xs, ys, s=66, facecolor=C_DET, edgecolor=C_DET, alpha=0.62, linewidths=0)
        rho, p = spearman(xs, ys) if len(xs) >= 3 else (None, None)
        if rho is not None:
            axb.text(0.98, 0.05, f"Spearman ρ = {rho:+.2f}" + (f"\np = {p:.3g}" if p else ""),
                     transform=axb.transAxes, ha="right", va="bottom",
                     fontsize=9.4, color=INK2)
    axb.set_xlabel("$n_{gated}$ — poses within 4 Å of Pα (of 50)", fontsize=9.4, color=INK2)
    axb.set_ylabel("A (°)", fontsize=9.4, color=INK2)
    axb.set_title("is A measuring catalysis, or just how well the ligand docked?",
                  fontsize=9.6, color=INK, loc="left", pad=12)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axb.spines[s].set_color(INK3)
    axb.tick_params(colors=INK2, labelsize=8)
    axb.grid(True, color=INK3, alpha=0.2, ls=(0, (1, 4)), lw=0.7); axb.set_axisbelow(True)
    fig.text(0.062, 0.398, "b", fontsize=15, fontweight="bold", color=INK)

    # ---------- (c) positive control ------------------------------------------------------
    axc = fig.add_subplot(gs[2, 2:4]); axc.set_facecolor(SURFACE)
    cf = CAMP / "analysis/control_7tgm_vs_predicted.csv"
    if cf.exists():
        cr = list(csv.DictReader(cf.open()))
        d = sorted(float(r["centroid_d"]) for r in cr)
        axc.hist(d, bins=14, color=C_DET, edgecolor="white", linewidth=0.8)
        axc.axvline(3.0, color=C_LOD, ls="--", lw=1.6)
        near = sum(1 for v in d if v <= 3.0)
        axc.text(0.97, 0.93, f"{near}/{len(d)} poses ≤ 3 Å\nbest {d[0]:.2f} Å",
                 transform=axc.transAxes, ha="right", va="top", fontsize=9.2, color=INK2)
        axc.set_xlabel("ligand centroid displacement from crystallographic HSC (Å)",
                       fontsize=9.4, color=INK2)
        axc.set_ylabel("poses", fontsize=9.4, color=INK2)
    else:
        axc.text(0.5, 0.5, "positive control pending\n(DesD + native HSC vs PDB 7TGM)",
                 transform=axc.transAxes, ha="center", va="center", fontsize=9.4, color=INK3)
        axc.set_xticks([]); axc.set_yticks([])
    axc.set_title("positive control — does the modelled chemistry land in the right place?",
                  fontsize=9.6, color=INK, loc="left", pad=12)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axc.spines[s].set_color(INK3)
    axc.tick_params(colors=INK2, labelsize=8)
    fig.text(0.452, 0.398, "c", fontsize=15, fontweight="bold", color=INK)

    # ---------- (d) pre-registered test summary ------------------------------------------
    axd = fig.add_subplot(gs[3, :]); axd.set_facecolor(SURFACE); axd.axis("off")
    hdr = f"{'panel':<11}{'reaction':<20}{'active/total':<14}{'actives top-ranked by A':<26}{'exact p':<10}{'ρ(A, activity)':<16}"
    lines = [hdr, "─" * len(hdr)]
    for panel, perfect, p_rank, rho, p_rho, has_data in summary:
        lab = labels.get(panel, {})
        det = sum(1 for e in lab if not lab[e]["below_LOD"])
        if not has_data:
            verdict, ps = "pending", "—"
        elif perfect is None:
            verdict, ps = "no contrast (all detected)", "—"
        else:
            verdict = "YES" if perfect else ("NO" if perfect is not None else "no contrast")
            ps = f"{p_rank:.3f}" if p_rank else "—"
        rs = f"{rho:+.2f} (p={p_rho:.3f})" if rho is not None else "—"
        lines.append(f"{panel.replace('Figure ', ''):<11}{REACTION.get(panel, ''):<20}"
                     f"{f'{det}/{len(lab)}':<14}{verdict:<26}{ps:<10}{rs:<16}")
    axd.text(0, 1.0, "\n".join(lines), transform=axd.transAxes, va="top", ha="left",
             family="monospace", fontsize=8.3, color=INK2, linespacing=1.52)
    fig.text(0.062, 0.148, "d", fontsize=15, fontweight="bold", color=INK)

    # ---------- titles & legend ------------------------------------------------------------
    fig.text(0.062, 0.963,
             "Does a step-1 proximity-weighted attack angle separate active from inactive "
             "amide synthetases?", fontsize=15.2, color=INK, fontweight="medium")
    fig.text(0.062, 0.938,
             "6 enzymes × every quantitative panel of Regioselective biosynthesis of oligoamides "
             "(bioRxiv 2025.11.03.685623).\nMetric frozen a priori "
             "(inverse_distance_attack_angle_v1); hypotheses and exact p-values pre-registered.",
             fontsize=9.6, color=INK2)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=C_DET, mec=C_DET, ms=8, label="product detected"),
        Line2D([], [], marker="o", ls="", mfc=C_LOD, mec=C_LOD, ms=8, label="below LOD"),
        Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=C_DET, ms=8, mew=1.9,
               label="hollow: $n_{gated}$ = 0, A not interpretable"),
    ], loc="upper right", bbox_to_anchor=(0.985, 0.952), ncol=3, frameon=False,
        fontsize=8.8, labelcolor=INK2, handletextpad=0.5, columnspacing=1.6)

    fig.text(0.5, 0.012,
             "A is scored over all 50 poses with no threshold; $n_{gated}$ is a validity "
             "precondition, never a ranking input. Non-docking ensembles are shown, not dropped. "
             "Power ceiling of this design: best attainable p = 0.048 (6 enzymes per stratum).",
             ha="center", fontsize=8.2, color=INK3)

    out = CAMP / "analysis/FIGURE_regio_A_validation.png"
    fig.savefig(out, dpi=330, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    done = sum(1 for s in summary if s[5])
    print(f"wrote {out}\n      {out.with_suffix('.pdf')}")
    print(f"  panels with data: {done}/{len(PANEL_ORDER)}")


if __name__ == "__main__":
    main()
