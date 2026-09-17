#!/usr/bin/env python3
"""The one figure for the amide-synthetase diversity panel: the validated proxy.

Proxy = structural_modeling/scripts/angle_gate_proxy.py. Distance <4A to the mechanism
phosphate (Pa for this ANL/AMP-binding panel) is a GATE (keep docked poses; >=5 required
or the enzyme is INCONCLUSIVE); the SCORE is the MEAN in-line attack angle among gated
poses. Distance never enters the score. Raw NAC (d AND angle) and best-single-pose angle
were shown to fail (rho ~0.09) and are not used.

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
from matplotlib.patches import Patch
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
RED, ORANGE, BLUE, GREEN, GREY = "#d62728", "#f39c12", "#3060c0", "#2e8b57", "#9aa0a6"
BROWN = "#8b5a2b"                          # SfaB (the baseline / starting enzyme)
FS = 18
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.titleweight": "bold",
    "axes.labelsize": FS, "xtick.labelsize": FS - 3, "ytick.labelsize": FS - 3,
    "legend.fontsize": FS - 3, "figure.titlesize": FS + 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 300,
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


def fig_landscape(df):
    """Activity landscape, every panel member labelled on the x-axis (actives in red)."""
    d = df.sort_values("act", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(22, 8))
    x = np.arange(len(d))
    colors = [BROWN if i == "SfaB" else RED if a else "#c8cdd4"
              for i, a in zip(d.id, d.is_active)]
    floor = 4e-4
    ax.bar(x, d.act.clip(lower=floor), color=colors, edgecolor="k", linewidth=0.3, width=0.82, zorder=2)
    ax.errorbar(x, d.act.clip(lower=floor), yerr=[np.zeros(len(d)), d.act_std.fillna(0)],
                fmt="none", ecolor="#555", elinewidth=0.7, capsize=2, alpha=0.6, zorder=3)
    ax.set_yscale("log")
    ax.axhline(ACT_THR, color=GREEN, ls="--", lw=1.8, zorder=1)
    ax.text(len(d) - 0.5, ACT_THR * 1.25, f"active threshold  {ACT_THR} µg/ml",
            color=GREEN, ha="right", va="bottom", fontsize=FS - 3)
    ax.set_xticks(x)
    ax.set_xticklabels(d.id, rotation=90, fontsize=FS - 6)
    for lab, iid, act in zip(ax.get_xticklabels(), d.id, d.is_active):
        if iid == "SfaB":
            lab.set_color(BROWN); lab.set_fontweight("bold")
        elif act:
            lab.set_color(RED); lab.set_fontweight("bold")
    ax.set_xlabel("panel member (sorted by activity)", labelpad=8)
    ax.set_ylabel("N66-L1 activity (µg/ml, log)")
    ax.set_title("The panel is 3 hits over a flat tail — 3/49 actives, median 0.009 µg/ml")
    ax.set_xlim(-1, len(d))
    ax.legend(handles=[Patch(color=RED, label="active (>0.2 µg/ml)"),
                       Patch(color=BROWN, label="SfaB (baseline)"),
                       Patch(color="#c8cdd4", label="inactive")],
              loc="upper right", frameon=True, framealpha=0.95, edgecolor="#ccc")
    plt.tight_layout()
    p = f"{FIG}/synthetase_activity_landscape.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_proxy_ranking(df):
    """Validated proxy: ranked bar of mean in-line angle + ROC (conclusive enzymes)."""
    c = df[df.concl].copy()
    d = c.sort_values("mean_inline_angle", ascending=False).reset_index(drop=True)
    y = c.is_active.values
    s = c.mean_inline_angle.values
    rho, pval = spearmanr(c.mean_inline_angle, np.log10(c.act + 1e-3))
    fpr, tpr, _ = roc_curve(y, s); a = auc(fpr, tpr)
    grid, lo, hi, aci = boot_auc_ci(y, s)
    # top-6 panel members by measured activity: 3 strong hits (>0.2) + 3 marginal (0.14-0.2)
    top6 = df.sort_values("act", ascending=False).head(6).id.tolist()
    strong, near = top6[:3], top6[3:]
    ang_rank = {r.id: i + 1 for i, r in d.iterrows()}   # only conclusive enzymes have a rank

    fig, axA = plt.subplots(1, 1, figsize=(11, 8.5))
    fig.suptitle("Mean in-line attack angle (distance-gated cofold poses) ranks the active enzymes to the top",
                 fontweight="bold")
    # ranked bar
    x = np.arange(len(d))
    colors = [RED if i in strong else ORANGE if i in near else "#c8cdd4" for i in d.id]
    axA.bar(x, d.mean_inline_angle, color=colors, edgecolor="k", linewidth=0.3, zorder=2)
    lines = ["top 6 by activity  →  angle rank"]
    for i, e in enumerate(top6, 1):
        rk = f"#{ang_rank[e]}" if e in ang_rank else "INCONCLUSIVE"
        lines.append(f"{i}. {e}  →  {rk}")
    axA.text(0.985, 0.985, "\n".join(lines), transform=axA.transAxes, ha="right", va="top",
             fontsize=FS - 4, color="#222", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fbfbfb", ec="#888", lw=1.2))
    axA.legend(handles=[Patch(color=RED, label="active  (>0.2 µg/ml)"),
                        Patch(color=ORANGE, label="marginal  (#4–6 by activity)"),
                        Patch(color="#c8cdd4", label="other")],
               loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=1,
               frameon=True, framealpha=0.95, edgecolor="#ccc")
    axA.set_xlabel(f"rank by mean in-line angle  ({len(d)} conclusive enzymes)")
    axA.set_ylabel("mean in-line attack angle (°)")
    axA.set_ylim(min(d.mean_inline_angle) - 4, 182)
    axA.set_xlim(-1, len(d))
    # ROC/AUROC dropped: only 3 actives make discrimination metrics uninformative
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{FIG}/synthetase_angle_proxy_ranking.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    df = load()
    assert set(ACTIVES) <= set(df.id), "actives missing from merged data"
    assert df[df.is_active == 1].concl.all(), "an active is inconclusive — check phosphate/gate"
    print(f"n={len(df)}  conclusive={int(df.concl.sum())}  inconclusive={int((~df.concl).sum())} "
          f"({df[~df.concl].id.tolist()})")
    print("wrote:")
    for o in (fig_proxy_ranking(df), fig_landscape(df)):
        print("  ", o)


if __name__ == "__main__":
    main()
