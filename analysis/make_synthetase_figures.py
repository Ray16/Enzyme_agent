#!/usr/bin/env python3
"""Refined, reproducible figures for the amide-synthetase diversity panel.

Uses the VALIDATED static-cofold proxy (structural_modeling/scripts/angle_gate_proxy.py):
distance <4A is a GATE (keep docked poses; >=5 required or the enzyme is INCONCLUSIVE),
then the score is the MEAN in-line attack angle among the gated poses (mechanism phosphate
= Pa for this ANL/AMP-binding panel). Raw NAC (d AND angle) and best-single-pose angle are
NOT used - they were shown to fail (rho ~0.09) because the hard distance gate cancels the
angle signal.

Regenerate the proxy CSV first:
  python structural_modeling/scripts/angle_gate_proxy.py \
      exp_data/amide_synthetase/boltz2/step1/results/output_configs \
      --phosphate pa --out exp_data/amide_synthetase/analysis/angle_gate_step1_pa.csv
then:  python analysis/make_synthetase_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
AS = f"{ROOT}/exp_data/amide_synthetase"
PROXY_CSV = f"{AS}/analysis/angle_gate_step1_pa.csv"
FIG = f"{ROOT}/figures"
os.makedirs(FIG, exist_ok=True)

ACTIVES = ["AS9", "AS17", "AS23"]          # the three clear hits (>0.2 ug/ml)
ACT_THR = 0.2
A_CUT = 150.0                              # in-line angle cut used by the proxy (deg)
RED, BLUE, GREEN, GREY = "#d62728", "#3060c0", "#2e8b57", "#9aa0a6"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
})


def load():
    act = pd.read_csv(f"{AS}/assay_data/panel_activity.csv")
    act = act[["id", "activity_N66L1_ugml_avg", "activity_N66L1_ugml_std"]].rename(
        columns={"activity_N66L1_ugml_avg": "act", "activity_N66L1_ugml_std": "act_std"})
    prox = pd.read_csv(PROXY_CSV)
    df = act.merge(prox, on="id", how="inner")
    df["is_active"] = (df["act"] > ACT_THR).astype(int)
    df["concl"] = df["docking_ok"].astype(bool)
    return df


def boot_auc_ci(y, s, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 101)
    curves, aucs = [], []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if len(np.unique(y[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[idx], s[idx])
        curves.append(np.interp(grid, fpr, tpr)); aucs.append(auc(fpr, tpr))
    C = np.vstack(curves)
    return grid, np.percentile(C, 2.5, 0), np.percentile(C, 97.5, 0), np.percentile(aucs, [2.5, 97.5])


# ---------------------------------------------------------------------------
def fig_landscape(df):
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
                 "(a useful screen must ENRICH for these, not rank the tail)")
    ax.set_xlim(-1, len(d))
    ax.legend(handles=[Patch(color=RED, label="active (>0.2)"),
                       Patch(color="#c8cdd4", label="inactive")], loc="upper right", frameon=False)
    plt.tight_layout()
    p = f"{FIG}/synthetase_activity_landscape.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_proxy_ranking(df):
    """The validated proxy: ranked bar of mean in-line angle + ROC (conclusive enzymes)."""
    c = df[df.concl].copy()
    d = c.sort_values("mean_inline_angle", ascending=False).reset_index(drop=True)
    y = c.is_active.values
    s = c.mean_inline_angle.values
    rho, pval = spearmanr(c.mean_inline_angle, np.log10(c.act + 1e-3))
    fpr, tpr, _ = roc_curve(y, s); a = auc(fpr, tpr)
    grid, lo, hi, aci = boot_auc_ci(y, s)
    ranks = {r.id: i + 1 for i, r in d.iterrows() if r.is_active}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Validated proxy — mean in-line attack angle (distance-gated poses): "
                 "ranks the hits to the top", fontsize=13, fontweight="bold")
    # A. ranked bar
    x = np.arange(len(d))
    colors = [RED if a_ else "#cfe0f5" for a_ in d.is_active]
    axA.bar(x, d.mean_inline_angle, color=colors, edgecolor="k", linewidth=0.3)
    axA.axhline(A_CUT, color=GREY, ls=":", lw=1)
    axA.text(len(d) - 0.5, A_CUT - 1.5, f"{A_CUT:.0f}° in-line cut", color="#666",
             ha="right", va="top", fontsize=8)
    callout = "actives by angle rank:\n" + "\n".join(f"  {k}  →  #{ranks[k]}" for k in ACTIVES)
    axA.text(0.97, 0.96, callout, transform=axA.transAxes, ha="right", va="top",
             fontsize=9.5, color=RED, fontweight="bold",
             bbox=dict(boxstyle="round", fc="#fff0f0", ec=RED, alpha=0.9))
    axA.set_xlabel(f"rank by mean in-line angle ({len(d)} conclusive enzymes)")
    axA.set_ylabel("mean in-line attack angle (°)")
    axA.set_title("A. All 3 hits in the top 7 (AS9 = #1)")
    axA.set_ylim(min(d.mean_inline_angle) - 4, 180)
    axA.legend(handles=[Patch(color=RED, label="active"), Patch(color="#cfe0f5", label="inactive")],
               loc="lower left", frameon=False)
    # B. ROC + CI
    axB.fill_between(grid, lo, hi, color=BLUE, alpha=0.15, label="95% CI (bootstrap)")
    axB.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"ROC  AUROC={a:.2f}")
    axB.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="random")
    axB.set_xlabel("false positive rate"); axB.set_ylabel("true positive rate")
    axB.set_title(f"B. AUROC={a:.2f}  (95% CI [{aci[0]:.2f}, {aci[1]:.2f}], 3 positives)")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right", frameon=False)
    axB.text(0.97, 0.42, f"Spearman(angle, log-act)\n= {rho:+.2f}  (p = {pval:.2f})",
             transform=axB.transAxes, ha="right", va="top", fontsize=9, color="#333",
             bbox=dict(boxstyle="round", fc="#eafaef", ec=GREEN))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{FIG}/synthetase_angle_proxy_ranking.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_gate_mechanism(df):
    """How the two-stage recipe works: docking gate (x) then angle score (y)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    inc = df[~df.concl]; con = df[df.concl]
    ax.axvspan(0, 5, color="#f2c94c", alpha=0.18, zorder=0)
    ax.axhline(A_CUT, color=GREY, ls=":", lw=1)
    ax.scatter(inc.n_gated, inc.mean_inline_angle.fillna(120), s=55, facecolors="none",
               edgecolors="#999", linewidths=1.2, label="INCONCLUSIVE (<5 docked)", zorder=3)
    ci = con[con.is_active == 0]
    ax.scatter(ci.n_gated, ci.mean_inline_angle, s=55, c="#b8bec7", edgecolors="k",
               linewidths=0.3, label="inactive (scored)", zorder=3)
    ca = con[con.is_active == 1]
    ax.scatter(ca.n_gated, ca.mean_inline_angle, s=150, c=RED, edgecolors="k",
               linewidths=0.6, label="active (scored)", zorder=5)
    for _, r in df.iterrows():
        if r.is_active or (not r.concl and r.id in ("SfaB", "AS8")):
            yv = r.mean_inline_angle if not np.isnan(r.mean_inline_angle) else 120
            ax.annotate(r.id, (r.n_gated, yv), fontsize=9, xytext=(5, 4),
                        textcoords="offset points",
                        color=RED if r.is_active else "#555",
                        fontweight="bold" if r.is_active else "normal")
    ax.axvline(5, color="#c99700", ls="--", lw=1.2)
    ax.text(5.3, ax.get_ylim()[0] + 2, "docking gate: ≥5 poses within 4 Å", color="#8a6d00", fontsize=9)
    ax.set_xlabel("# poses reaching attack distance (<4 Å to Pα)  →  distance is a GATE")
    ax.set_ylabel("mean in-line attack angle (°)  →  angle is the SCORE")
    ax.set_title("The recipe: gate on docking success, then score by in-line angle\n"
                 "(distance never enters the score — it only decides scored vs. inconclusive)")
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    p = f"{FIG}/synthetase_angle_gate_mechanism.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_screening(df):
    """Screening story: hit recovery vs experimental budget, ranking by the validated proxy."""
    c = df[df.concl].copy()
    N, H = len(c), int(c.is_active.sum())
    d = c.sort_values("mean_inline_angle", ascending=False).reset_index(drop=True)
    rec = np.concatenate([[0], np.cumsum(d.is_active.values) / H])
    budget = np.concatenate([[0], (np.arange(N) + 1) / N])
    k = int(np.argmax(np.cumsum(d.is_active.values) == H)) + 1

    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.2, label="random triage")
    ax.step(budget, rec, where="post", lw=2.4, color=GREEN, label="rank by mean in-line angle")
    ax.scatter([k / N], [1.0], s=160, marker="o", color=GREEN, edgecolors="k", zorder=6)
    ax.annotate(f"test top {k}/{N} conclusive ({k/N*100:.0f}%)\n"
                f"catch {H}/{H} hits  →  {(1-k/N)*100:.0f}% fewer experiments",
                (k / N, 1.0), xytext=(k / N + 0.08, 0.80), fontsize=10,
                color="#155724", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax.set_xlabel("fraction of conclusive candidates sent to the wet lab (budget)")
    ax.set_ylabel("fraction of true hits recovered (recall)")
    ax.set_title("Screening utility: recover every strong hit while testing ~1/6 of candidates",
                 fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.02, 0.02, f"note: {int((~df.concl).sum())} enzymes were INCONCLUSIVE (docking failure, "
            "incl. SfaB) and are set aside, not scored", transform=ax.transAxes,
            fontsize=8.5, color="#a05000", style="italic")
    plt.tight_layout()
    p = f"{FIG}/synthetase_screening_operating_curve.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    df = load()
    assert set(ACTIVES) <= set(df.id), "actives missing from merged data"
    assert df[df.is_active == 1].concl.all(), "an active is inconclusive — check phosphate/gate"
    outs = [fig_landscape(df), fig_proxy_ranking(df), fig_gate_mechanism(df), fig_screening(df)]
    print(f"n={len(df)}  conclusive={int(df.concl.sum())}  inconclusive={int((~df.concl).sum())} "
          f"({df[~df.concl].id.tolist()})")
    print("wrote:")
    for o in outs:
        print("  ", o)


if __name__ == "__main__":
    main()
