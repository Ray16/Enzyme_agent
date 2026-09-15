#!/usr/bin/env python3
"""Regenerate all analysis figures from the result CSVs (+ step-1 PDBs for one panel).

Previously these plots were produced ad-hoc; this consolidates them so every figure
in ../figures/ is reproducible. Run from anywhere:  python analysis/make_figures.py
"""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from sklearn.metrics import roc_curve, auc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from geometry import iter_atoms, angle_deg, carboxylate_oxygens

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
EXP = f"{ROOT}/exp_data"
FIG = f"{ROOT}/figures"; os.makedirs(FIG, exist_ok=True)
SA = {"AS9", "AS17", "AS23"}


def load(p): return list(csv.DictReader(open(p)))
def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def rank_roc(rows, score_key, label_fn, title, fname, score_label, higher_better=True,
             annotate_active=False):
    """Ranked-bar + ROC panel (the standard classifier view)."""
    d = [(r["id"], fnum(r[score_key]), label_fn(r)) for r in rows if fnum(r[score_key]) is not None]
    d.sort(key=lambda t: -t[1] if higher_better else t[1])
    ids = [t[0] for t in d]; s = np.array([t[1] for t in d]); lab = np.array([t[2] for t in d])
    fig = plt.figure(figsize=(16, 6)); fig.suptitle(title, fontsize=13, fontweight="bold")
    ax = fig.add_subplot(1, 2, 1)
    colors = ["#d62728" if l == 1 else "#c6dbef" for l in lab]
    ax.bar(range(len(ids)), s, color=colors, edgecolor="k", linewidth=0.3)
    if annotate_active:
        for j, i in enumerate(ids):
            if i in SA:
                ax.text(j, s[j] + 0.02, i, ha="center", fontsize=8, color="#d62728", fontweight="bold", rotation=90)
    ax.set_xlabel(f"enzyme rank by {score_label} (best → worst)"); ax.set_ylabel(score_label)
    ax.set_title("A. Ranked (actives red)")
    ax.legend(handles=[Patch(color="#d62728", label="active"), Patch(color="#c6dbef", label="inactive")], loc="upper right")
    axB = fig.add_subplot(1, 2, 2)
    sc = s if higher_better else -s
    fpr, tpr, _ = roc_curve(lab, sc); a = auc(fpr, tpr)
    axB.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUROC={a:.2f})")
    axB.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="random")
    axB.set_xlabel("false positive rate"); axB.set_ylabel("true positive rate"); axB.set_title("B. ROC")
    axB.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{FIG}/{fname}", dpi=130, bbox_inches="tight"); plt.close(fig)
    return a


def fig_summary():
    s1 = load(f"{EXP}/boltz2_amide_synthetase_step1_results/nac_step1.csv")
    s2 = load(f"{EXP}/boltz2_amide_synthetase_step2_results/nac_step2.csv")
    hy = load(f"{EXP}/nylon_hydrolase_results/hydrolase_nac.csv")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Cofold geometry proxies vs experimental activity", fontsize=15, fontweight="bold")
    def scat(a, rows, xk, yk, xl, yl, t, logy=True):
        pts = [(fnum(r[xk]), fnum(r[yk]), r["id"]) for r in rows if fnum(r[xk]) is not None and fnum(r[yk]) is not None]
        a.scatter([p[0] for p in pts], [p[1] for p in pts], s=28, c="#888", alpha=0.7, edgecolors="none")
        for x, y, i in pts:
            if i in SA: a.scatter([x], [y], s=85, c="#d62728", edgecolors="k", zorder=5); a.annotate(i, (x, y), fontsize=8)
        if logy: a.set_yscale("log")
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(t)
    scat(ax[0, 0], s1, "nac_frac", "activity_N66L1", "step-1 NAC fraction", "N66 L1 activity", "Synthetase step-1 NAC")
    scat(ax[0, 1], s2, "min_d", "activity_N66L1", "step-2 min attack dist (Å)", "N66 L1 activity", "Synthetase step-2 attack dist")
    scat(ax[1, 0], hy, "nac_frac", "L2", "hydrolase NAC fraction", "L2 release", "Hydrolase NAC vs L2")
    md = [fnum(r["median_d"]) for r in s2 if fnum(r["median_d"]) is not None]
    ax[1, 1].hist(md, bins=20, color="#4c72b0", edgecolor="k", alpha=0.8)
    for i in SA:
        v = [fnum(r["median_d"]) for r in s2 if r["id"] == i][0]; ax[1, 1].axvline(v, color="#d62728", lw=2)
    ax[1, 1].axvline(3.5, color="green", ls="--"); ax[1, 1].set_xlabel("step-2 median amine→carbonyl dist (Å)")
    ax[1, 1].set_ylabel("# enzymes"); ax[1, 1].set_title("Step-2 nucleophile not poised (~8–10Å)")
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(f"{FIG}/summary_geometry_vs_activity.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def _step1_per_model(cd):
    """Per-model (attack distance, in-line angle) for one step-1 enzyme (module-level for Pool)."""
    b = os.path.basename(cd); name = b[len("step1_"):]; pred = f"{cd}/boltz_results_{b}/predictions/{b}"
    rows = []
    for pdb in glob.glob(f"{pred}/{b}_model_*.pdb"):
        at = {}
        for ch, het, an, el, xyz in iter_atoms(pdb):
            at.setdefault(het, []).append((an, el, xyz))
        pa = o3a = None
        for lst in at.values():
            nm = {a: x for a, e, x in lst}
            if "PA" in nm and "O3A" in nm: pa, o3a = nm["PA"], nm["O3A"]; break
        het = at.get(True, [])
        cb = carboxylate_oxygens([(e, x) for a, e, x in het])
        if pa is None or not cb: continue
        o = min(cb, key=lambda x: np.linalg.norm(x - pa))
        rows.append((float(np.linalg.norm(o - pa)), angle_deg(o, pa, o3a)))
    return name, np.array(rows) if rows else None


def fig_step1_scatter():
    """NAC fraction vs activity + per-model geometry scatter (parses step-1 PDBs)."""
    from multiprocessing import Pool
    rc = {r["id"]: r for r in load(f"{EXP}/boltz2_amide_synthetase_step1_results/nac_step1.csv")}
    RES = f"{EXP}/boltz2_amide_synthetase_step1_results/output_configs"
    with Pool(32) as p: data = {n: a for n, a in p.map(_step1_per_model, sorted(glob.glob(f"{RES}/*"))) if a is not None}
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Step-1 adenylation NAC (amide synthetase) — the signal-carrying proxy", fontsize=13, fontweight="bold")
    ids = list(data)
    ax[0].scatter([fnum(rc[i]["nac_frac"]) for i in ids], [fnum(rc[i]["activity_N66L1"]) for i in ids], s=50, c="#888", edgecolors="k", linewidths=0.4)
    for i in SA:
        ax[0].scatter([fnum(rc[i]["nac_frac"])], [fnum(rc[i]["activity_N66L1"])], s=130, c="#d62728", edgecolors="k", zorder=5)
        ax[0].annotate(i, (fnum(rc[i]["nac_frac"]), fnum(rc[i]["activity_N66L1"])), fontsize=10)
    ax[0].set_yscale("log"); ax[0].set_xlabel("step-1 NAC fraction"); ax[0].set_ylabel("N66 L1 activity"); ax[0].set_title("A. NAC fraction vs activity")
    alld = np.concatenate([data[i] for i in ids]); am = np.concatenate([data[i] for i in SA])
    ax[1].scatter(alld[:, 0], alld[:, 1], s=4, c="#bbb", alpha=0.25, label="inactive models")
    ax[1].scatter(am[:, 0], am[:, 1], s=10, c="#d62728", alpha=0.6, label="active models")
    ax[1].add_patch(Rectangle((0, 150), 4.0, 40, color="green", alpha=0.12))
    ax[1].axvline(4.0, color="green", ls="--", lw=1); ax[1].axhline(150, color="green", ls="--", lw=1)
    ax[1].text(1.2, 183, "NAC region\n(d<4Å, angle>150°)", color="green", fontsize=9)
    ax[1].set_xlim(2, 7); ax[1].set_ylim(40, 185)
    ax[1].set_xlabel("carboxylate O → ATP Pα (Å)"); ax[1].set_ylabel("in-line angle O–Pα–O3A (°)")
    ax[1].set_title("B. Per-model geometry (actives cluster in NAC region)"); ax[1].legend(loc="lower right", markerscale=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{FIG}/synthetase_step1_nac_scatter.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    a1 = rank_roc(load(f"{EXP}/boltz2_amide_synthetase_step1_results/nac_step1.csv"), "nac_frac",
                  lambda r: 1 if (fnum(r["activity_N66L1"]) or 0) >= 0.5 else 0,
                  "Step-1 NAC as an activity classifier (amide synthetase)",
                  "synthetase_step1_nac_rank_roc.png", "step-1 NAC fraction", True, annotate_active=True)
    a2 = rank_roc(load(f"{EXP}/boltz2_amide_synthetase_step2_results/nac_step2.csv"), "frac_lt5",
                  lambda r: 1 if (fnum(r["activity_N66L1"]) or 0) >= 0.5 else 0,
                  "Step-2 attack proximity as a classifier (loose run)",
                  "synthetase_step2_attack_rank_roc.png", "step-2 near-attack fraction (<5Å)", True, annotate_active=True)
    a3 = rank_roc(load(f"{EXP}/nylon_hydrolase_results/hydrolase_nac.csv"), "nac_frac",
                  lambda r: 1 if (r.get("L1sig") == "1" or r.get("L2sig") == "1") else 0,
                  "Hydrolase NAC as an activity classifier (nylon hydrolases)",
                  "hydrolase_nac_rank_roc.png", "hydrolase NAC fraction", True)
    fig_summary()
    fig_step1_scatter()
    print(f"Wrote figures to {FIG}/  (AUROC: step1={a1:.2f}, step2={a2:.2f}, hydrolase={a3:.2f})")
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
