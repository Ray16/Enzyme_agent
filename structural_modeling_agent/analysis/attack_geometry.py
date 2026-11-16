"""Distribution-aware step-1 attack-geometry ranking across candidates/replicates.

For each candidate's MD replicate(s) it reads the per-frame geometry log (forming distance = nucleophile
-> electrophilic P, attack angle = nucleophile-P-leaving) and computes that replicate's NEAR-ATTACK
FRACTION -- the fraction of frames with d < near_d AND angle > near_ang, i.e. how OFTEN the pose visits
reactive geometry. This is the distribution-aware summary: unlike mean(distance)/mean(angle), it is not
dragged into no-man's-land by a skewed/bimodal trajectory (distance is bounded below ~3 A but unbounded
above, so drift excursions inflate the mean; a sharp reactive basin + a drifted basin average to a
mediocre single point). The plot is a dumbbell ranking: one dot per REPLICATE (colour = that replicate's
near-attack fraction) with a min-max bar, sorted by the MEDIAN replicate so reproducible candidates --
not one lucky seed -- rank highest; the per-replicate spread shows seed reproducibility directly.

CLI:
  python -m structural_modeling_agent.analysis.attack_geometry --run-dirs DIR1 DIR2 ... \
      --out ranking.png [--near-d 4.0 --near-ang 150]
Each candidate = an immediate subdir of a run-dir; each replicate = a subdir matching 'md*' holding
geometry.jsonl.
"""
from __future__ import annotations
import os, json, glob, re, argparse
import numpy as np


def _seed_stats(gj, near_d, near_ang):
    rows = [json.loads(l) for l in open(gj) if l.strip()]
    if not rows:
        return None
    d = np.array([r["forming_A"] for r in rows])
    a = np.array([r["attack_angle_deg"] for r in rows])
    na = float(np.mean((d < near_d) & (a > near_ang)))
    return {"n": len(rows), "mean_d": float(d.mean()), "mean_ang": float(a.mean()),
            "near_attack": na}


def collect(run_dirs, near_d=4.0, near_ang=150.0):
    """-> {candidate_label: [per-replicate stats...]} (candidate = subdir; replicate = md*/geometry.jsonl)."""
    out = {}
    for rd in run_dirs:
        for cand in sorted(glob.glob(f"{rd}/*")):
            if not os.path.isdir(cand) or os.path.basename(cand) in ("analysis", "archive", "figures",
                                                                      "panel_prep", "_inputs"):
                continue
            seeds = []
            for gj in sorted(glob.glob(f"{cand}/md*/geometry.jsonl")):
                rep = os.path.basename(os.path.dirname(gj))
                # skip non-canonical / superseded replicate dirs (bare 'md', salt/failed/test variants)
                if rep == "md" or any(t in rep for t in ("salt", "failed", "test", "equil", "noc4")):
                    continue
                s = _seed_stats(gj, near_d, near_ang)
                if s:
                    s["seed_dir"] = rep; seeds.append(s)
            if seeds:
                out.setdefault(os.path.basename(cand), []).extend(seeds)
    return out


def _pretty_label(cand):
    """'GlnA3_Q9KZC7' -> 'GlnA3 (Q9KZC7)'. Keep the accession so homologs sharing a gene name
    (desD x11, asnB x4) stay distinguishable and no row is a bare accession."""
    parts = cand.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} ({'_'.join(parts[1:])})"
    return cand


def plot(cand_stats, out_png, near_d=4.0, near_ang=150.0, label_all=False):
    """Distribution-aware dumbbell ranking: one dot per MD replicate at its NEAR-ATTACK FRACTION
    (fraction of frames with d < near_d AND angle > near_ang), sorted by the MEDIAN replicate so
    reproducible candidates outrank one-lucky-seed ones. The per-replicate spread shows seed
    reproducibility; the shaded band marks the >= 0.5 'majority-of-frames poised' zone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []   # (label, nas, median, max) per candidate
    for cand, seeds in cand_stats.items():
        nas = [s["near_attack"] for s in seeds]
        rows.append((_pretty_label(cand), nas, float(np.median(nas)), float(np.max(nas))))
    # sort ascending by (median, max, mean) so the most reproducibly-poised candidate ends up on TOP
    rows.sort(key=lambda r: (r[2], r[3], float(np.mean(r[1]))))
    cmap = plt.cm.RdYlGn
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.34 * n + 1.8)))
    ax.axvspan(0.5, 1.03, color="#1b7837", alpha=0.08, zorder=0)
    ax.axvline(0.5, ls=":", color="gray", lw=1, zorder=1)
    ax.text(0.5, n - 0.3, " majority of frames poised →", ha="left", va="center",
            fontsize=9, color="#1b7837", fontweight="bold")
    for i, (lab, nas, med, mx) in enumerate(rows):
        if len(nas) > 1:
            ax.plot([min(nas), max(nas)], [i, i], color="#cccccc", lw=1.6, zorder=1)
        ax.scatter(nas, [i] * len(nas), c=nas, cmap=cmap, vmin=0, vmax=1, s=70,
                   edgecolor="black", linewidth=0.5, zorder=3)
        # black tick = the MEDIAN replicate (the sort key), so the ranking metric is visible
        ax.scatter([med], [i], marker="|", s=360, color="black", linewidth=1.3, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for i, r in enumerate(rows):            # bold the reproducibly-poised candidates (median >= 0.5)
        if r[2] >= 0.5:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#1b5e20")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("near-attack fraction  (frames with d < %.0f Å AND ∠ > %.0f°)  —  one dot per MD replicate"
                  % (near_d, near_ang), fontsize=10.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("near-attack fraction", fontsize=10)
    ax.set_title("Step-1 attack geometry — distribution-aware ranking\n"
                 "(how OFTEN each candidate visits reactive geometry; black tick = median seed, "
                 "spread = reproducibility)", fontsize=12, fontweight="bold")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near-d", type=float, default=4.0)
    ap.add_argument("--near-ang", type=float, default=150.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--label-all", action="store_true")
    a = ap.parse_args()
    cs = collect(a.run_dirs, a.near_d, a.near_ang)
    if a.json:
        json.dump(cs, open(a.json, "w"), indent=1)
    plot(cs, a.out, a.near_d, a.near_ang, a.label_all)


if __name__ == "__main__":
    _cli()
