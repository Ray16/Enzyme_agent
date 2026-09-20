#!/usr/bin/env python3
"""Native-substrate vs target-substrate cofold comparison for SfaB and DdaG.

Both are acyl-adenylate enzymes (Pa). For each enzyme we cofolded a substrate SERIES
(native + diacids incl. the nylon target adipate), same protein/MSA/ATP/2Mg, only the
substrate changes. This isolates substrate geometry and exposes the two DISTINCT cofold
failure modes:
  * SfaB: native (5-chlorovalerate) docks in-line, but the non-native diacids float ~10A
    from Pa  ->  a DISTANCE failure (substrate never reaches the catalytic subsite).
  * DdaG: every substrate (incl. native fumarate & 80%-yield succinate) sits in the site
    but is chelated onto the catalytic Mg2+  ->  an Mg-TRAP failure (in site, never in-line).

Geometry metrics are MEASURED here over the 50 diffusion samples. Yields are the measured
Fig 2C molar yields (paper), shown as reference only. SfaB native identity is from the
characterization paper (SfaB is unannotated in UniProt); DdaG native (fumarate) is the
reviewed UniProt reaction (E2JA32).
"""
import glob, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results", "output_configs")
FIGDIR = os.path.join(HERE, "..", "figures")
A_CUT, D_CUT, MG_CUT = 150.0, 4.0, 2.6

# substrate series (order: native first, then diacids); measured Fig 2C yields
SERIES = {
    "SfaB": [("chlorovalerate", "native"), ("glutarate", 34), ("succinate", 31), ("adipate", 6)],
    "DdaG": [("fumarate", "native"), ("succinate", 80), ("glutarate", 19), ("adipate", 1)],
}


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


def pose_metrics(pdb):
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
    if not atp or not mgs or not cbo or "PA" not in atp:
        return None
    pa = atp["PA"]; o3a = atp.get("O3A")
    on = min(cbo, key=lambda o: np.linalg.norm(o - pa))
    d = float(np.linalg.norm(on - pa))
    a = angle(on, pa, o3a) if o3a is not None else np.nan
    mgd = min(np.linalg.norm(o - m) for o in cbo for m in mgs)
    return d, a, float(mgd)


def config_metrics(name):
    inner = glob.glob(f"{RES}/{name}/boltz_results_*/predictions/*")
    if not inner:
        return None
    rows = [m for p in sorted(glob.glob(f"{inner[0]}/*model_*.pdb")) if (m := pose_metrics(p))]
    if not rows:
        return None
    d = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows]); mg = np.array([r[2] for r in rows])
    near = d < D_CUT
    return dict(
        n=len(rows),
        median_d=float(np.median(d)),
        near_frac=float(np.mean(near)),
        inline_frac=float(np.mean(near & (a > A_CUT))),
        mgchel_frac=float(np.mean(mg < MG_CUT)),
    )


def main():
    data = {}
    for enz, subs in SERIES.items():
        data[enz] = []
        for sub, yld in subs:
            m = config_metrics(f"nat_{enz}_{sub}")
            if m:
                m.update(sub=sub, yld=yld)
                data[enz].append(m)

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    GREEN, RED, GREY, BLUE = "#2e8b57", "#d62728", "#9aa0a6", "#3060c0"

    for col, enz in enumerate(("SfaB", "DdaG")):
        rows = data[enz]
        labs = [f"{r['sub']}\n({'native' if r['yld']=='native' else str(r['yld'])+'%'})" for r in rows]
        x = np.arange(len(rows))

        # ROW 1: does the substrate reach the catalytic subsite? (median carboxylate O -> Pa)
        ax = axes[0][col]
        md = [r["median_d"] for r in rows]
        cols = [GREEN if r["inline_frac"] > 0.05 else (RED if r["median_d"] > 6 else "#e08a14") for r in rows]
        ax.bar(x, md, color=cols, edgecolor="k", linewidth=0.4)
        ax.axhline(D_CUT, ls="--", color=GREY, lw=1.5)
        ax.text(len(rows) - 0.5, D_CUT + 0.2, "4 Å near-attack gate", ha="right", va="bottom", fontsize=9, color="#555")
        for xi, r in zip(x, rows):
            ax.text(xi, r["median_d"] + 0.2, f"{r['median_d']:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylabel("median carboxylate O → Pα (Å)")
        ax.set_ylim(0, max(12, max(md) + 1.5))
        ax.set_title(f"{enz} — does the substrate reach Pα?", fontweight="bold")

        # ROW 2: when in the site, is it in-line (productive) or Mg-chelated (trapped)?
        ax = axes[1][col]
        w = 0.38
        ax.bar(x - w / 2, [r["inline_frac"] for r in rows], w, label="in-line (productive)", color=GREEN, edgecolor="k", linewidth=0.4)
        ax.bar(x + w / 2, [r["mgchel_frac"] for r in rows], w, label="Mg-chelated (trapped)", color=BLUE, edgecolor="k", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylabel("fraction of 50 poses")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{enz} — productive vs Mg-trapped", fontweight="bold")
        ax.legend(fontsize=9, frameon=True, edgecolor="#ccc")

    fig.suptitle("Native vs target substrate — cofold reproduces neither enzyme's diacid activity\n"
                 "SfaB: only the native substrate reaches Pα (diacids float ~10 Å)  ·  "
                 "DdaG: every substrate sits at Pα but none reach in-line (incl. 80%-yield succinate); "
                 "Mg-chelation worsens with chain length",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIGDIR, "native_vs_target_SfaB_DdaG.png")
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.abspath(out))
    for enz in SERIES:
        print(f"\n{enz}:")
        for r in data[enz]:
            print(f"  {r['sub']:15} yield={str(r['yld']):7} median_d={r['median_d']:.2f}  "
                  f"near_frac={r['near_frac']:.2f}  inline_frac={r['inline_frac']:.2f}  mgchel_frac={r['mgchel_frac']:.2f}")


if __name__ == "__main__":
    main()
