#!/usr/bin/env python3
"""Substrate binding-mode analysis (colleague's point): is the diacid failing to reach the
reactive phosphorus because it is chelated onto the catalytic Mg2+ instead?

Per pose we locate the adipate carboxylate O closest to the reactive phosphorus P and ask
two competing distances:
  d_P  = closest carboxylate O -> P            (phosphate-attack mode)
  d_Mg = closest carboxylate O -> nearest Mg   (Mg-chelation mode)
plus the in-line angle O-P-O(bridge). Each pose is classified:
  productive     : d_P < 4  AND angle > 150            (attack-competent)
  in-site off-axis: d_P < 4  AND angle <= 150          (at phosphate, wrong geometry)
  Mg-trapped     : d_Mg < 2.6 AND d_P >= 4             (chelated to Mg, away from P)
  unbound/other  : neither

Compares working (AS9 active; pubC conclusive) vs failing (glnA2, DdaG headline leads),
all with the TARGET substrate adipate, mechanism-correct phosphate per enzyme.
Geometry is measured; this is a diagnostic of WHY the in-line proxy fails.
"""
import glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
# enzyme -> (predictions dir, reactive phosphate, bridge O, label)  -- mechanism validated by rxn products
ENZ = {
    "AS9 (active, panel)":      (f"{ROOT}/exp_data/amide_synthetase/boltz2/step1/results/output_configs/step1_AS9/boltz_results_step1_AS9/predictions/step1_AS9", "PA", "O3A"),
    "pubC P0DXD6 (NIS, 163°)":  (f"{ROOT}/structural_modeling/results/output_configs/step1_P0DXD6/boltz_results_step1_P0DXD6/predictions/step1_P0DXD6", "PA", "O3A"),
    "glnA2 Q9RDS6 (run #1)":    (f"{ROOT}/structural_modeling/results/output_configs/step1_Q9RDS6/boltz_results_step1_Q9RDS6/predictions/step1_Q9RDS6", "PG", "O3B"),
    "DdaG E2JA32 (adenylate)":  (f"{ROOT}/structural_modeling/results/output_configs/step1_E2JA32/boltz_results_step1_E2JA32/predictions/step1_E2JA32", "PA", "O3A"),
}
DP, DMG, ACUT = 4.0, 2.6, 150.0


def carbox_O(lst):
    Cs = [x for n, e, x in lst if e == "C"]; Os = [x for n, e, x in lst if e == "O"]
    out = []
    for cx in Cs:
        nb = [o for o in Os if np.linalg.norm(cx - o) < 1.6]
        if len(nb) >= 2:
            out.extend(nb)
    return out


def angle(a, b, c):
    v1, v2 = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))


def pose(pdb, pk, bk):
    atoms = {}
    for ln in open(pdb):
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        nm = ln[12:16].strip(); rn = ln[17:20].strip(); ch = ln[21]
        e = (ln[76:78].strip() or nm[0]).upper()
        x = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        atoms.setdefault((ch, rn), []).append((nm, e, x))
    atp = None; mgs = []; cbo = []
    for (ch, rn), lst in atoms.items():
        names = {a[0] for a in lst}
        if "PA" in names and "PG" in names:
            atp = {a[0]: a[2] for a in lst}
        for nm, e, x in lst:
            if e == "MG":
                mgs.append(x)
        if "PA" not in names:
            cb = carbox_O(lst)
            if cb:
                cbo = cb
    if not atp or not mgs or not cbo or pk not in atp:
        return None
    P = atp[pk]; B = atp.get(bk)
    on = min(cbo, key=lambda o: np.linalg.norm(o - P))
    d_P = float(np.linalg.norm(on - P))
    d_Mg = min(np.linalg.norm(o - m) for o in cbo for m in mgs)
    ang = angle(on, P, B) if B is not None else np.nan
    return d_P, float(d_Mg), ang


def classify(d_P, d_Mg, ang):
    if d_P < DP and ang > ACUT:
        return "productive"
    if d_P < DP:
        return "in-site off-axis"
    if d_Mg < DMG:
        return "Mg-trapped"
    return "unbound/other"


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "figure.dpi": 150})
    MODE_COL = {"productive": "#2e8b57", "in-site off-axis": "#e08a14", "Mg-trapped": "#3060c0", "unbound/other": "#c8cdd4"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    summary = {}
    for ax, (name, (pred, pk, bk)) in zip(axes.ravel(), ENZ.items()):
        rows = [r for p in sorted(glob.glob(f"{pred}/*model_*.pdb")) if (r := pose(p, pk, bk))]
        modes = [classify(*r) for r in rows]
        summary[name] = {m: modes.count(m) for m in MODE_COL}
        for r, m in zip(rows, modes):
            ax.scatter(r[1], r[0], c=MODE_COL[m], s=55, edgecolor="k", linewidth=0.3, alpha=0.85)
        ax.axhline(DP, ls="--", color="#888", lw=1.2)
        ax.axvline(DMG, ls="--", color="#888", lw=1.2)
        ax.axhspan(DP, 20, xmin=0, xmax=(DMG - 0) / 12, color="#3060c0", alpha=0.06)  # Mg-trap corner
        ax.text(DMG - 0.1, 19, "Mg-trapped\n(on Mg, off P)", color="#3060c0", fontsize=9, ha="right", va="top")
        ax.text(11.5, DP - 0.3, "reaches P (d<4Å)", color="#2e8b57", fontsize=9, ha="right", va="top")
        n = len(rows)
        prod = summary[name]["productive"]; trap = summary[name]["Mg-trapped"]
        ax.set_title(f"{name}\nproductive {prod}/{n} · Mg-trapped {trap}/{n}", fontweight="bold", fontsize=11)
        ax.set_xlabel("closest carboxylate O → nearest Mg²⁺ (Å)")
        ax.set_ylabel("closest carboxylate O → reactive P (Å)")
        ax.set_xlim(1.5, 12); ax.set_ylim(2, 20)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k', markersize=10, label=m)
               for m, c in MODE_COL.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, frameon=True, edgecolor="#ccc", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Substrate binding mode (adipate): does it sit on the phosphorus for attack, or chelate the catalytic Mg²⁺?",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{ROOT}/structural_modeling/figures/binding_mode_adipate.png"
    plt.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)
    for name, s in summary.items():
        n = sum(s.values())
        print(f"{name:26} " + "  ".join(f"{m}={s[m]}" for m in MODE_COL) + f"  (n={n})")


if __name__ == "__main__":
    main()
