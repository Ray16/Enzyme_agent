#!/usr/bin/env python3
"""Refined, reproducible figures for the amide-synthetase diversity panel.

Reads the CURRENT result layout under exp_data/amide_synthetase/ (the older
make_figures.py pointed at pre-Sep-10 paths that no longer exist). Produces an
honest set that shows both the negative result (no single proxy ranks activity,
n=3 actives) and the one usable finding (a combined step-1 gate as a ~6x-enriching
negative filter). Run from anywhere:  python analysis/make_synthetase_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
AS = f"{ROOT}/exp_data/amide_synthetase"
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

ACTIVES = ["AS9", "AS17", "AS23"]          # the three clear hits (>0.2 ug/ml)
ACT_THR = 0.2
RED, GREY, BLUE, GREEN = "#d62728", "#9aa0a6", "#3060c0", "#2e8b57"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
})


def load():
    act = pd.read_csv(f"{AS}/assay_data/panel_activity.csv")
    act = act[["id", "activity_N66L1_ugml_avg", "activity_N66L1_ugml_std"]].rename(
        columns={"activity_N66L1_ugml_avg": "act", "activity_N66L1_ugml_std": "act_std"})
    s1 = pd.read_csv(f"{AS}/boltz2/step1/results/near_attack_geometry.csv")
    s1 = s1[["id", "nac_frac", "best_angle_at_min_d", "min_d"]].rename(
        columns={"best_angle_at_min_d": "s1_ang", "min_d": "s1_min_d"})
    df = act.merge(s1, on="id", how="inner")
    df["is_active"] = (df["act"] > ACT_THR).astype(int)
    return df


def boot_roc_ci(y, s, n=2000, seed=0):
    """Bootstrap 95% CI band for the ROC curve (stratified resample)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    curves, aucs = [], []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], s[idx])
        curves.append(np.interp(grid, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    C = np.vstack(curves)
    return grid, np.percentile(C, 2.5, 0), np.percentile(C, 97.5, 0), np.percentile(aucs, [2.5, 97.5])


# ---------------------------------------------------------------------------
def fig_landscape(df):
    """Fig 1 - the activity landscape: why this is a hard prediction problem."""
    d = df.sort_values("act", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(d))
    colors = [RED if a else "#c8cdd4" for a in d.is_active]
    floor = 5e-4
    ax.bar(x, d.act.clip(lower=floor), color=colors, edgecolor="k", linewidth=0.3, width=0.8)
    ax.errorbar(x, d.act.clip(lower=floor), yerr=[np.zeros(len(d)), d.act_std.fillna(0)],
                fmt="none", ecolor="#555", elinewidth=0.6, capsize=1.5, alpha=0.6)
    ax.set_yscale("log")
    ax.axhline(ACT_THR, color=GREEN, ls="--", lw=1.2)
    ax.text(len(d) - 0.5, ACT_THR * 1.2, f"  active threshold  {ACT_THR} µg/ml",
            color=GREEN, ha="right", va="bottom", fontsize=9)
    for i, r in d.iterrows():
        if r.is_active:
            ax.text(i, r.act * 1.35, r.id, ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_xlabel("panel member (sorted by activity)")
    ax.set_ylabel("N66-L1 activity (µg/ml, log)")
    ax.set_title("The panel is 3 hits over a flat tail — 3/49 actives, median 0.009 µg/ml\n"
                 "(a useful screen must ENRICH for these, not rank them)")
    ax.set_xlim(-1, len(d))
    ax.legend(handles=[Patch(color=RED, label="active (>0.2)"),
                       Patch(color="#c8cdd4", label="inactive")], loc="upper right", frameon=False)
    plt.tight_layout()
    p = f"{FIG}/synthetase_activity_landscape.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_classifier(df):
    """Fig 2 - best single proxy: ranked bar + ROC with bootstrap CI (honest)."""
    d = df.sort_values("nac_frac", ascending=False).reset_index(drop=True)
    y = df.is_active.values
    s = df.nac_frac.values
    rho, pval = spearmanr(df.nac_frac, np.log10(df.act + 1e-3))
    fpr, tpr, _ = roc_curve(y, s); a = auc(fpr, tpr)
    grid, lo, hi, aci = boot_roc_ci(y, s)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Best single proxy (step-1 NAC fraction): ranks weakly, does not top-rank the hits",
                 fontsize=13, fontweight="bold")
    # A. ranked bar
    x = np.arange(len(d))
    colors = [RED if a_ else "#cfe0f5" for a_ in d.is_active]
    axA.bar(x, d.nac_frac, color=colors, edgecolor="k", linewidth=0.3)
    for i, r in d.iterrows():
        if r.is_active:
            axA.text(i, r.nac_frac + 0.02, f"{r.id}\n#{i+1}", ha="center", va="bottom",
                     fontsize=8, color=RED, fontweight="bold")
    axA.set_xlabel("rank by step-1 NAC fraction (best → worst)")
    axA.set_ylabel("step-1 NAC fraction")
    axA.set_title("A. Top-5 by NAC are all inactive (P@3 = 0/3)")
    axA.set_ylim(0, 1.08)
    axA.legend(handles=[Patch(color=RED, label="active"), Patch(color="#cfe0f5", label="inactive")],
               loc="upper right", frameon=False)
    # B. ROC + CI
    axB.fill_between(grid, lo, hi, color=BLUE, alpha=0.15, label="95% CI (bootstrap)")
    axB.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"ROC  AUROC={a:.2f}")
    axB.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="random")
    axB.set_xlabel("false positive rate"); axB.set_ylabel("true positive rate")
    axB.set_title(f"B. AUROC={a:.2f} but CI [{aci[0]:.2f}, {aci[1]:.2f}] (only 3 positives)")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right", frameon=False)
    axB.text(0.5, 0.07, f"Spearman(NAC, log-act) = {rho:+.2f}  (p = {pval:.2f}, n.s.)",
             ha="center", fontsize=9, color="#333",
             bbox=dict(boxstyle="round", fc="#fff6e6", ec="#e0a000"))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{FIG}/synthetase_step1_nac_refined.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_gate(df):
    """Fig 3 - the usable finding: combined step-1 gate as a negative filter."""
    t_nac = df.set_index("id").loc[ACTIVES, "nac_frac"].min()
    t_ang = df.set_index("id").loc[ACTIVES, "s1_ang"].min()
    df = df.copy()
    df["kept"] = (df.nac_frac >= t_nac) & (df.s1_ang >= t_ang)
    kept = df[df.kept]
    base_rate = df.is_active.mean()
    gate_rate = kept.is_active.mean()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.suptitle("A combined step-1 gate works as a negative filter (not a ranker)",
                 fontsize=13, fontweight="bold")
    # A. 2D scatter with gate box
    axA.add_patch(Rectangle((t_nac, t_ang), 1.05 - t_nac, 185 - t_ang,
                            color=GREEN, alpha=0.10, zorder=0))
    inact = df[df.is_active == 0]
    axA.scatter(inact.nac_frac, inact.s1_ang, s=45, c="#b8bec7", edgecolors="k",
                linewidths=0.3, label="inactive", zorder=2)
    axA.scatter(df[df.is_active == 1].nac_frac, df[df.is_active == 1].s1_ang, s=140, c=RED,
                edgecolors="k", linewidths=0.6, label="active", zorder=4)
    for _, r in df.iterrows():
        if r.is_active or (r.kept and not r.is_active):
            axA.annotate(r.id, (r.nac_frac, r.s1_ang), fontsize=8,
                         xytext=(4, 4), textcoords="offset points",
                         color=RED if r.is_active else "#444")
    axA.axvline(t_nac, color=GREEN, ls="--", lw=1); axA.axhline(t_ang, color=GREEN, ls="--", lw=1)
    axA.text(t_nac + 0.01, t_ang + 1, f"gate: NAC≥{t_nac:.2f} & angle≥{t_ang:.0f}°",
             color=GREEN, fontsize=9, va="bottom")
    axA.set_xlabel("step-1 NAC fraction"); axA.set_ylabel("step-1 in-line angle (°)")
    axA.set_title("A. Gate keeps all 3 hits + 5 others (8/49)")
    axA.set_xlim(-0.02, 1.02)
    axA.legend(loc="lower left", frameon=False)
    # B. enrichment bars
    cats = ["whole panel", "gated set"]
    rates = [base_rate * 100, gate_rate * 100]
    ns = [len(df), len(kept)]
    bars = axB.bar(cats, rates, color=["#b8bec7", GREEN], edgecolor="k", width=0.6)
    for b, rate, n, na in zip(bars, rates, ns, [df.is_active.sum(), kept.is_active.sum()]):
        axB.text(b.get_x() + b.get_width() / 2, rate + 1.2,
                 f"{rate:.0f}%\n{na}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axB.set_ylabel("hit rate (% active)")
    axB.set_title(f"B. {gate_rate/base_rate:.0f}× enrichment,\n100% recall of strong hits")
    axB.set_ylim(0, max(rates) * 1.35)
    axB.text(0.5, -0.22, "threshold set on the same 3 actives → descriptive, not validated",
             transform=axB.transAxes, ha="center", fontsize=8.5, color="#a05000")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    p = f"{FIG}/synthetase_step1_gate_enrichment.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, sorted(kept.id.tolist())


def fig_screening(df):
    """Fig 4 - THE screening story: hit recovery vs experimental budget.

    Frames the proxy for its actual use (triage candidates for the wet lab):
    if you rank/gate by the proxy and only test the top slice, what fraction of
    the true hits do you still catch, and how many experiments do you save?
    """
    N = len(df)
    H = int(df.is_active.sum())

    def recall_curve(score, higher=True):
        d = df.sort_values("score" if False else score, ascending=not higher).reset_index(drop=True)
        rec = np.cumsum(d.is_active.values) / H
        budget = (np.arange(N) + 1) / N
        return np.concatenate([[0], budget]), np.concatenate([[0], rec])

    # combined step-1 gate as an operating point
    t_nac = df.set_index("id").loc[ACTIVES, "nac_frac"].min()
    t_ang = df.set_index("id").loc[ACTIVES, "s1_ang"].min()
    kept = df[(df.nac_frac >= t_nac) & (df.s1_ang >= t_ang)]
    g_budget, g_recall = len(kept) / N, kept.is_active.mean() * len(kept) / H

    fig, ax = plt.subplots(figsize=(8, 6.2))
    xr, yr = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    ax.plot(xr, yr, ls="--", color="grey", lw=1.2, label="random triage")
    for key, lab, col in [("nac_frac", "rank by NAC fraction", BLUE),
                          ("s1_ang", "rank by in-line angle", "#8e44ad")]:
        b, r = recall_curve(key)
        ax.step(b, r, where="post", lw=2, color=col, label=lab, alpha=0.9)
    ax.scatter([g_budget], [g_recall], s=220, marker="*", color=GREEN, edgecolors="k",
               zorder=6, label="combined step-1 gate")
    ax.annotate(f"test {len(kept)}/{N} candidates ({g_budget*100:.0f}%)\n"
                f"catch {int(g_recall*H)}/{H} hits  →  {(1-g_budget)*100:.0f}% fewer experiments",
                (g_budget, g_recall), xytext=(g_budget + 0.06, g_recall - 0.22),
                fontsize=10, color="#155724", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax.set_xlabel("fraction of candidates sent to the wet lab (experimental budget)")
    ax.set_ylabel("fraction of true hits recovered (recall)")
    ax.set_title("Screening utility: recover every strong hit while testing ~1/6 of candidates",
                 fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.02, 0.97, "up-and-to-the-left = better triage", transform=ax.transAxes,
            fontsize=9, va="top", color="#555", style="italic")
    plt.tight_layout()
    p = f"{FIG}/synthetase_screening_operating_curve.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    df = load()
    assert set(ACTIVES) <= set(df.id), "actives missing from merged data"
    outs = [fig_landscape(df), fig_classifier(df)]
    gp, kept = fig_gate(df)
    outs.append(gp)
    outs.append(fig_screening(df))
    print(f"n={len(df)} members, {df.is_active.sum()} actives ({ACTIVES})")
    print(f"gated set ({len(kept)}): {kept}")
    print("wrote:")
    for o in outs:
        print("  ", o)


if __name__ == "__main__":
    main()
