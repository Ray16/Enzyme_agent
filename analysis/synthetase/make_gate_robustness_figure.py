#!/usr/bin/env python3
"""Robustness of the angle proxy to the distance-gate cutoff, shown as the RANKING
(not AUROC): one ranked mean-in-line-angle bar chart per cutoff, actives highlighted.
The actives stay pinned to the left (top) across 3.5-6.0 A -> ranking is cutoff-robust.
Companion to figures/synthetase_angle_proxy_ranking.png."""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
sys.path.insert(0, f"{ROOT}/structural_modeling/scripts")
import step1_nac as s
assert hasattr(s, "geom_model")
RES = f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs"
FIG = f"{ROOT}/figures/synthetase_gate_robustness_ranking.png"
CUTS = [3.5, 4.0, 4.5, 5.0, 6.0]
RED, ORANGE, GREY = "#d62728", "#f39c12", "#c8cdd4"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 300})

# per-pose (angle, distance-to-Pa) for every enzyme
data = {}
for cfg in sorted(glob.glob(f"{RES}/step1_*")):
    pred = glob.glob(f"{cfg}/boltz_results_*/predictions/*")
    if not pred:
        continue
    per = [(g["ang_pa"], g["d_pa"]) for pdb in sorted(glob.glob(f"{pred[0]}/*model_*.pdb"))
           for g in [s.geom_model(pdb)] if g]
    if per:
        data[os.path.basename(cfg)[len("step1_"):]] = np.array(per)

act = pd.read_csv(f"{ROOT}/exp_data/amide_synthetase/assay_data/panel_activity.csv")[
    ["id", "activity_N66L1_ugml_avg"]].rename(columns={"activity_N66L1_ugml_avg": "act"})
act = act[act.id != "SfaB"].set_index("id")["act"]
top6 = list(act.sort_values(ascending=False).head(6).index)
ACTIVES, MARGINAL = top6[:3], top6[3:]     # AS9/17/23 ; AS41/AS8/AS3

fig, axes = plt.subplots(1, len(CUTS), figsize=(3.6 * len(CUTS), 11), sharex=True)
fig.suptitle("Angle-proxy ranking is robust to the distance-gate cutoff — actives stay on top",
             fontsize=17, fontweight="bold", y=0.995)
for ax, cut in zip(axes, CUTS):
    score = {i: arr[arr[:, 1] < cut][:, 0].mean() for i, arr in data.items()
             if (arr[:, 1] < cut).sum() >= 5}
    order = sorted(score, key=lambda i: -score[i])          # rank 1 = highest angle
    rank = {i: r + 1 for r, i in enumerate(order)}
    y = np.arange(len(order))                               # 0 = top after invert
    vals = [score[i] for i in order]
    colors = [RED if i in ACTIVES else ORANGE if i in MARGINAL else GREY for i in order]
    ax.barh(y, vals, color=colors, edgecolor="k", linewidth=0.3)
    ax.invert_yaxis()                                       # rank #1 on top
    for i in order:                                         # label the highlighted enzymes
        if i in ACTIVES or i in MARGINAL:
            ax.text(vals[rank[i] - 1] + 0.6, rank[i] - 1, i, va="center", ha="left",
                    fontsize=8.5, fontweight="bold",
                    color=RED if i in ACTIVES else ORANGE)
    hitranks = ", ".join(f"#{rank[a]}" for a in ACTIVES if a in rank)
    ax.set_title(f"d < {cut} Å\n{len(order)} conclusive\nhits at {hitranks}", fontsize=13)
    ax.set_xlim(122, 186)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_ylabel("rank by mean in-line angle" if cut == CUTS[0] else "")
    ax.set_xlabel("mean angle (°)")
axes[0].legend(handles=[Patch(color=RED, label="active (>0.2 µg/ml)"),
                        Patch(color=ORANGE, label="marginal (#4–6)"),
                        Patch(color=GREY, label="other")],
               loc="lower left", frameon=True, framealpha=0.95, edgecolor="#ccc", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print("wrote", FIG)
for cut in CUTS:
    score = {i: arr[arr[:, 1] < cut][:, 0].mean() for i, arr in data.items() if (arr[:, 1] < cut).sum() >= 5}
    order = sorted(score, key=lambda i: -score[i])
    print(f"d<{cut}: n={len(order)}  hit ranks={[order.index(a)+1 for a in ACTIVES if a in order]}  AS9=#{order.index('AS9')+1}")
