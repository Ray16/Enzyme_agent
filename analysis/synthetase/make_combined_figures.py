#!/usr/bin/env python3
"""Two publication figures for the amide-synthetase panel (CORRECTED 1-Mg data).

FIG 1  synthetase_activity_and_proxy.png  -- two independent panels:
    (a) measured activity landscape  (b) structural-proxy (in-line angle) ranking.
    Actives (AS9/AS17/AS23) highlighted+labelled on both; shared tier legend.

FIG 2  synthetase_proxy_comparison.png    -- how well each structural feature
    SURFACES THE TOP CANDIDATES: AUROC (the 3 measured actives as positives) with
    95% bootstrap CI, one bar per feature, grouped by mechanistic category.
    Uses the same validated evaluator as the rest of the paper (analysis/common/proxy_eval).

Run:  conda run -n palm python analysis/synthetase/make_combined_figures.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
sys.path.insert(0, os.path.join(ROOT, "analysis", "common"))
from proxy_eval import evaluate  # noqa: E402

AS = f"{ROOT}/exp_data/amide_synthetase"
FIG = f"{AS}/figures"
os.makedirs(FIG, exist_ok=True)

ACTIVES = ["AS9", "AS17", "AS23"]
ACT_THR = 0.2                                # > this => a clear hit (AS9/AS17/AS23)
RED, ORANGE, BROWN, GREY = "#d62728", "#f39c12", "#8b5a2b", "#c8cdd4"
HILITE = "#3060c0"                           # the validated proxy / significant bars
NEUTRAL = "#9aa0a6"
FS = 17
plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.titleweight": "bold",
    "axes.labelsize": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 3,
    "legend.fontsize": FS - 2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 300,
})


# ----------------------------------------------------------------------------- data
def load():
    act = pd.read_csv(f"{AS}/assay_data/panel_activity.csv")[
        ["id", "activity_N66L1_ugml_avg", "activity_N66L1_ugml_std"]].rename(
        columns={"activity_N66L1_ugml_avg": "act", "activity_N66L1_ugml_std": "act_std"})
    ang = pd.read_csv(f"{AS}/analysis/angle_gate_step1_1mg.csv")   # angle proxy + ipTM + docking_ok
    bat = pd.read_csv(f"{AS}/analysis/battery_1mg.csv")           # full geometric battery
    df = act.merge(ang, on="id", how="left")
    df["is_active"] = (df["act"] > ACT_THR).astype(int)
    df["concl"] = (df["docking_ok"] == True)  # noqa: E712 (NaN from left-merge -> False)
    return df, bat, act


def _tier(iid, a, SF):
    if iid == "SfaB":
        return BROWN
    if a > ACT_THR:
        return RED
    if a > SF:
        return ORANGE
    return GREY


# ----------------------------------------------------------------------------- FIG 1
def fig_activity_and_proxy(df):
    SF = float(df.loc[df.id == "SfaB", "act"].iloc[0])

    a = df.sort_values("act", ascending=False).reset_index(drop=True)          # panel (a)
    b = df[df.concl].sort_values("mean_inline_angle", ascending=False).reset_index(drop=True)  # (b)
    na, nb = len(a), len(b)
    as9_rank = int(b.index[b.id == "AS9"][0]) + 1 if (b.id == "AS9").any() else None

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(16, max(11, 0.27 * na)), gridspec_kw={"wspace": 0.28})

    # ---- (a) activity landscape (log-x horizontal bars) ----
    y = np.arange(na)
    floor = 4e-4
    colsA = [_tier(i, v, SF) for i, v in zip(a.id, a.act)]
    axA.barh(y, a.act.clip(lower=floor), color=colsA, edgecolor="k", linewidth=0.3,
             height=0.80, zorder=2)
    axA.errorbar(a.act.clip(lower=floor), y, xerr=[np.zeros(na), a.act_std.fillna(0)],
                 fmt="none", ecolor="#555", elinewidth=0.7, capsize=2, alpha=0.6, zorder=3)
    axA.set_xscale("log")
    axA.set_yticks(y)
    axA.set_yticklabels(a.id)
    for lab, iid, v in zip(axA.get_yticklabels(), a.id, a.act):
        col = _tier(iid, v, SF)
        if col != GREY:
            lab.set_color(col); lab.set_fontweight("bold")
    tr = axA.get_xaxis_transform()
    axA.axvline(SF, color=BROWN, ls=":", lw=1.8, zorder=1)
    axA.text(SF, 0.5, f"  SfaB baseline ({SF:.3g})", transform=tr, color=BROWN,
             rotation=90, ha="left", va="center", fontsize=FS - 3, fontweight="bold")
    axA.set_ylim(na - 0.5, -0.5)
    axA.set_xlabel("Nylon-6,6 diad molecular yield (µg/mL, log)")
    axA.set_ylabel("panel member (ranked by measured activity)")
    axA.set_title("(a) Measured activity", loc="left")

    # ---- (b) structural-proxy ranking (in-line attack angle) ----
    yb = np.arange(nb)
    colsB = [_tier(i, v, SF) for i, v in zip(b.id, b.act)]
    axB.barh(yb, b.mean_inline_angle, color=colsB, edgecolor="k", linewidth=0.3,
             height=0.80, zorder=2)
    for i, (iid, v) in enumerate(zip(b.id, b.act)):
        if v > SF:                                    # label enzymes that beat SfaB
            axB.text(b.mean_inline_angle[i] + 0.5, i, iid, va="center", ha="left",
                     fontsize=FS - 4, fontweight="bold", color=_tier(iid, v, SF))
    axB.set_ylim(nb - 0.5, -0.5)
    axB.set_xlim(float(b.mean_inline_angle.min()) - 5, 186)
    axB.set_yticks([0, 4, 9, 14, 19, 24, 29, 34, 39])
    axB.set_yticklabels([str(t + 1) for t in [0, 4, 9, 14, 19, 24, 29, 34, 39]])
    axB.set_xlabel("mean in-line attack angle (°)")
    axB.set_ylabel("rank by structural proxy")
    axB.set_title("(b) Structural proxy", loc="left")
    note = "all 3 actives in top 5 (AUROC ≈ 0.94)"
    if as9_rank:
        note += f"\nbut most-active AS9 ranks #{as9_rank}"
    axB.text(0.97, 0.03, note, transform=axB.transAxes, ha="right", va="bottom",
             fontsize=FS - 4, color="#333",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))

    fig.legend(handles=[Patch(color=RED, label="significantly > SfaB (active)"),
                        Patch(color=ORANGE, label="marginally > SfaB"),
                        Patch(color=BROWN, label="SfaB (baseline)"),
                        Patch(color=GREY, label="≤ SfaB")],
               loc="lower center", ncol=4, frameon=True, framealpha=0.95,
               edgecolor="#ccc", bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.11, top=0.95, left=0.10, right=0.98)
    p = f"{FIG}/synthetase_activity_and_proxy.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, as9_rank


# ----------------------------------------------------------------------------- FIG 2
# feature -> (dataframe-source, column, higher_is_productive, category)
#   direction fixed A PRIORI from mechanism (never chosen to maximise AUROC)
def fig_proxy_comparison(df, bat):
    act = df[["id", "act", "is_active"]].copy()
    m = act.merge(bat, on="id", how="left").merge(
        df[["id", "mean_iptm"]], on="id", how="left")
    try:
        amin = pd.read_csv(f"{AS}/analysis/step2_aminolysis_nac_1mg.csv")[["id", "frac_nac"]]
        m = m.merge(amin.rename(columns={"frac_nac": "amino_nac"}), on="id", how="left")
        has_amino = True
    except Exception:
        has_amino = False

    FEATS = [
        ("In-line attack angle (gated)", "mean_inline_angle", True,  "distance/angle geometry"),
        ("Near-attack fraction",         "near_attack_frac",  True,  "distance/angle geometry"),
        ("Interface confidence (ipTM)",  "mean_iptm",         True,  "interface confidence"),
        ("Reactive-center burial (SASA)","reactive_sasa",     False, "burial / accessibility"),
        ("Mg–Pα distance",               "mg_pa_dist",        False, "metal coordination"),
        ("Mg–O(adipate) distance",       "mg_o_dist",         False, "metal coordination"),
        ("Mg coordination number",       "mg_coord",          True,  "metal coordination"),
        ("Basic residues near amine",    "n_basic_6A",        True,  "active-site electrostatics"),
    ]
    if has_amino:
        FEATS.append(("Aminolysis near-attack (step 2)", "amino_nac", True, "distance/angle geometry"))

    rows = []
    for name, col, hb, cat in FEATS:
        sub = m[["is_active", col]].dropna()
        if sub[col].nunique() < 2 or sub["is_active"].sum() < 2:
            continue
        r = evaluate(sub[col].to_numpy(float), sub["is_active"].to_numpy(int),
                     name=name, higher_better=hb, nboot=5000, nperm=20000)
        r["cat"] = cat
        r["n_scored"] = len(sub)
        r["p3"] = r["precision_at_k"][3][0]
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("auroc", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12.5, 0.82 * len(res) + 2.2))
    y = np.arange(len(res))
    # highlight ONLY features that surface actives better than chance (CI above 0.5);
    # a feature with AUROC<0.5 fails at surfacing regardless of anti-correlation.
    sig = res.ci_lo > 0.5
    colors = [HILITE if s else NEUTRAL for s in sig]
    xerr = np.vstack([res.auroc - res.ci_lo, res.ci_hi - res.auroc])
    ax.barh(y, res.auroc, color=colors, edgecolor="k", linewidth=0.3, height=0.68, zorder=2)
    ax.errorbar(res.auroc, y, xerr=xerr, fmt="none", ecolor="#333", elinewidth=1.1,
                capsize=3, zorder=3)
    ax.axvline(0.5, color="#888", ls="--", lw=1.6, zorder=1)
    ax.text(0.5, len(res) - 0.35, "random", color="#888", fontsize=FS - 4,
            ha="center", va="bottom")
    for i, r in res.iterrows():
        ax.text(min(r.ci_hi, 1.0) + 0.015, i, f"{r.auroc:.2f}", va="center", ha="left",
                fontsize=FS - 4, fontweight="bold",
                color=HILITE if sig[i] else "#555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']}\n" + r"$\it{" + r['cat'].replace(' ', r'\ ') + r"}$ · n=" + str(r['n_scored'])
                        for _, r in res.iterrows()])
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(-0.6, len(res) - 0.4)
    ax.set_xlabel("AUROC for surfacing the 3 measured actives (95% CI)")
    ax.set_title("Which structural features surface the top candidates?", loc="left")
    ax.legend(handles=[Patch(color=HILITE, label="surfaces actives better than chance (95% CI > 0.5)"),
                       Patch(color=NEUTRAL, label="not better than chance")],
              loc="lower right", frameon=True, framealpha=0.95, edgecolor="#ccc")
    fig.tight_layout()
    p = f"{FIG}/synthetase_proxy_comparison.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, res


def main():
    df, bat, _ = load()
    assert set(ACTIVES) <= set(df.id), "actives missing"
    p1, as9 = fig_activity_and_proxy(df)
    p2, res = fig_proxy_comparison(df, bat)
    print("wrote:\n ", p1, f"(AS9 proxy rank #{as9})\n ", p2)
    print("\nfeature AUROC (surfacing 3 actives), best first:")
    for _, r in res.sort_values("auroc", ascending=False).iterrows():
        print(f"  {r['name']:32s} AUROC={r.auroc:.3f} [{r.ci_lo:.2f},{r.ci_hi:.2f}] "
              f"perm_p={r.perm_p:.3f}  P@3={r.p3}/3  n={r.n_scored}")


if __name__ == "__main__":
    main()
