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
D_GATE = 4.0                               # distance gate (A) the PROXY_CSV was built with
                                           # (angle_gate_proxy.py D_GATE default; validated)
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


def _tier(iid, a, SF):
    """Color by activity tier: SfaB=brown, >0.2=red, >SfaB&<0.2=orange, else grey."""
    if iid == "SfaB":
        return BROWN
    if a > ACT_THR:
        return RED
    if a > SF:
        return ORANGE
    return "#c8cdd4"


def fig_landscape(df):
    """Activity landscape as a vertical ranking (horizontal bars, most active on top)."""
    d = df.sort_values("act", ascending=False).reset_index(drop=True)
    SF = float(d.loc[d.id == "SfaB", "act"].iloc[0]) if (d.id == "SfaB").any() else 0.0489
    n = len(d)
    fig, ax = plt.subplots(figsize=(11, max(9, 0.30 * n)))
    y = np.arange(n)
    floor = 4e-4
    colors = [_tier(i, a, SF) for i, a in zip(d.id, d.act)]
    vals = d.act.clip(lower=floor)
    ax.barh(y, vals, color=colors, edgecolor="k", linewidth=0.3, height=0.82, zorder=2)
    ax.errorbar(vals, y, xerr=[np.zeros(n), d.act_std.fillna(0)], fmt="none",
                ecolor="#555", elinewidth=0.7, capsize=2, alpha=0.6, zorder=3)
    ax.set_xscale("log")
    ax.invert_yaxis()                                     # most active on top
    tr = ax.get_xaxis_transform()
    ax.axvline(SF, color=BROWN, ls=":", lw=1.8, zorder=1)
    ax.text(SF, 0.5, f"  SfaB baseline ({SF:.3g})", transform=tr, color=BROWN, rotation=90,
            ha="left", va="center", fontsize=FS - 5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(d.id, fontsize=FS - 7)
    for lab, iid, a in zip(ax.get_yticklabels(), d.id, d.act):
        col = _tier(iid, a, SF)
        if col != "#c8cdd4":
            lab.set_color(BROWN if iid == "SfaB" else col); lab.set_fontweight("bold")
    ax.set_ylabel("panel member (ranked by activity)")
    ax.set_xlabel("N66-L1 activity (µg/ml, log)")
    ax.set_title("Activity landscape — hits (>0.2), enzymes beating SfaB, and the tail")
    ax.set_ylim(n - 0.5, -0.5)
    ax.legend(handles=[Patch(color=RED, label="active (>0.2 µg/ml)"),
                       Patch(color=ORANGE, label="marginal (> SfaB, < 0.2)"),
                       Patch(color=BROWN, label="SfaB (baseline)"),
                       Patch(color="#c8cdd4", label="≤ SfaB")],
              loc="lower right", frameon=True, framealpha=0.95, edgecolor="#ccc")
    plt.tight_layout()
    p = f"{FIG}/synthetase_activity_landscape.png"
    plt.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_proxy_ranking(df):
    """Validated proxy as a vertical ranking (horizontal bars, highest angle on top)."""
    c = df[df.concl].copy()
    d = c.sort_values("mean_inline_angle", ascending=False).reset_index(drop=True)
    SF = float(df.loc[df.id == "SfaB", "act"].iloc[0]) if (df.id == "SfaB").any() else 0.0489
    ang_rank = {r.id: i + 1 for i, r in d.iterrows()}     # only conclusive enzymes have a rank
    beat = df[df.act > SF].sort_values("act", ascending=False)   # enzymes beating SfaB
    n = len(d)

    fig, axA = plt.subplots(1, 1, figsize=(10, max(9, 0.26 * n)))
    fig.suptitle(f"Mean in-line attack angle (d < {D_GATE:.1f} Å–gated poses) ranks the active enzymes to the top",
                 fontweight="bold")
    y = np.arange(n)
    colors = [_tier(i, a, SF) for i, a in zip(d.id, d.act)]
    axA.barh(y, d.mean_inline_angle, color=colors, edgecolor="k", linewidth=0.3, zorder=2)
    axA.invert_yaxis()                                    # rank #1 (highest angle) on top
    for i, (iid, a) in enumerate(zip(d.id, d.act)):       # label the enzymes that beat SfaB
        if a > SF:
            axA.text(d.mean_inline_angle[i] + 0.4, i, iid, va="center", ha="left",
                     fontsize=FS - 7, fontweight="bold", color=_tier(iid, a, SF))
    lines = ["above SfaB baseline  →  angle rank"]
    for k, (_, r) in enumerate(beat.iterrows(), 1):
        rk = f"#{ang_rank[r.id]}" if r.id in ang_rank else "INCONCLUSIVE"
        lines.append(f"{k}. {r.id}  →  {rk}")
    axA.text(0.985, 0.02, "\n".join(lines), transform=axA.transAxes, ha="right", va="bottom",
             fontsize=FS - 5, color="#222", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fbfbfb", ec="#888", lw=1.2))
    axA.legend(handles=[Patch(color=RED, label="active (>0.2 µg/ml)"),
                        Patch(color=ORANGE, label="marginal (> SfaB, < 0.2)"),
                        Patch(color="#c8cdd4", label="≤ SfaB")],
               loc="lower right", bbox_to_anchor=(0.985, 0.30),
               frameon=True, framealpha=0.95, edgecolor="#ccc")
    axA.set_ylabel(f"rank by mean in-line angle  ({n} conclusive enzymes)")
    axA.set_xlabel("mean in-line attack angle (°)")
    axA.set_xlim(min(d.mean_inline_angle) - 4, 184)
    axA.set_ylim(n - 0.5, -0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
