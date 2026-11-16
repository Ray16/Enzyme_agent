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


def collect_boltz(run_dirs, cand_labels, near_d=4.0, near_ang=150.0):
    """Boltz-cofold near-attack fraction per candidate: run the SAME generic forming-distance + attack-angle
    calc (render_contacts._attack_geometry) over each candidate's cofold pose ensemble. Keyed to match
    collect()'s candidate labels by accession (the part after the first underscore), since the run-dir
    folder names differ in case (e.g. MD 'PubC1_P0DXD6' vs cofold dir 'pubC_P0DXD6')."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("render_contacts", os.path.join(here, "render_contacts.py"))
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    acc2lab = {}
    for lab in cand_labels:
        p = lab.split("_", 1)
        if len(p) == 2:
            acc2lab[p[1]] = lab
    out = {}
    for rd in run_dirs:
        for pj in glob.glob(f"{rd}/*/*_boltz_poses.json"):
            name = os.path.basename(pj).replace("_boltz_poses.json", "")
            acc = name.split("_", 1)[1] if "_" in name else name
            lab = acc2lab.get(acc)
            if not lab:
                continue
            poses = json.load(open(pj))
            n = na = 0
            for p in poses:
                if not os.path.exists(p):
                    continue
                ag = rc._attack_geometry(p)
                if not ag or ag["angle_deg"] is None:
                    continue
                n += 1
                if ag["dist"] < near_d and ag["angle_deg"] > near_ang:
                    na += 1
            if n:
                out[lab] = na / n
    return out


def _pretty_label(cand):
    """'GlnA3_Q9KZC7' -> 'GlnA3 (Q9KZC7)'. Keep the accession so homologs sharing a gene name
    (desD x11, asnB x4) stay distinguishable and no row is a bare accession."""
    parts = cand.split("_")
    if len(parts) >= 2:
        return f"{parts[0]} ({'_'.join(parts[1:])})"
    return cand


def plot(cand_stats, out_png, near_d=4.0, near_ang=150.0, label_all=False, boltz_na=None):
    """Publication dumbbell: one dot per MD replicate at its NEAR-ATTACK FRACTION (fraction of frames
    with d < near_d AND angle > near_ang), sorted by the MEDIAN replicate. Design follows an EMPHASIS
    scheme (not a value-ramp): the reproducibly-poised candidates (median >= 0.5) carry the single
    accent colour and are directly value-labelled; every other candidate is kept for completeness but
    de-emphasised in gray. All candidates are shown; nulls collapse onto the left axis without clutter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    ACCENT, ACCENT_DK = "#1b7837", "#0f5023"      # poised (emphasis)
    MUTE, MUTE_DK = "#bdbdbd", "#8a8a8a"          # not poised (de-emphasised)
    POISED = 0.5

    rows = []   # (label, nas, median, max, poised, cand) per candidate
    for cand, seeds in cand_stats.items():
        nas = [s["near_attack"] for s in seeds]
        med = float(np.median(nas))
        rows.append((_pretty_label(cand), nas, med, float(np.max(nas)), med >= POISED, cand))
    # sort ascending by (median, max, mean) so the most reproducibly-poised candidate lands on TOP
    rows.sort(key=lambda r: (r[2], r[3], float(np.mean(r[1]))))
    n = len(rows)

    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.32 * n + 1.9)))
    # poised zone: a quiet band, no dashed rule (hairline anti-pattern)
    ax.axvspan(POISED, 1.04, color=ACCENT, alpha=0.06, zorder=0)
    for gx in (0.25, 0.5, 0.75, 1.0):             # recessive solid hairline grid
        ax.axvline(gx, color="#ededed", lw=0.8, zorder=0)

    BOLTZ = "#2166ac"                             # cofold-static series (distinct hue + diamond shape)
    for i, (lab, nas, med, mx, poised, cand) in enumerate(rows):
        c, cdk = (ACCENT, ACCENT_DK) if poised else (MUTE, MUTE_DK)
        if poised and len(nas) > 1:               # spread bar only where it carries signal
            ax.plot([min(nas), max(nas)], [i, i], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter(nas, [i] * len(nas), s=(84 if poised else 34), color=c,
                   edgecolor="white", linewidth=(1.1 if poised else 0.6),
                   zorder=4 if poised else 3)
        ax.scatter([med], [i], marker="|", s=(300 if poised else 120),
                   color=(cdk if poised else MUTE_DK), linewidth=(2.0 if poised else 1.0), zorder=5)
        if boltz_na and cand in boltz_na:         # Boltz-cofold (static) near-attack, same metric/axis
            ax.scatter([boltz_na[cand]], [i], marker="D", s=50, facecolor=BOLTZ,
                       edgecolor="white", linewidth=0.9, zorder=6)
        if poised:                                # selective direct labels — only the emphasis rows
            ax.text(min(mx + 0.035, 1.02), i, f"{med:.2f}", va="center", ha="left",
                    fontsize=10, fontweight="bold", color=ACCENT_DK)

    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows])
    for i, r in enumerate(rows):
        t = ax.get_yticklabels()[i]
        if r[4]:
            t.set_fontweight("bold"); t.set_color(ACCENT_DK); t.set_fontsize(10.5)
        else:
            t.set_color(MUTE_DK); t.set_fontsize(8.3)

    ax.set_xlim(-0.03, 1.08); ax.set_ylim(-0.8, n + 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(f"near-attack fraction   (d < {near_d:.0f} Å  and  ∠ > {near_ang:.0f}°)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#cccccc")
    # band header, above every row so it never collides with a data label
    ax.text((POISED + 1.04) / 2, n + 0.15, "← majority of frames poised →", ha="center", va="center",
            fontsize=9.5, color=ACCENT_DK, fontweight="bold")

    ax.set_title("Step-1 attack geometry across candidates", fontsize=14, fontweight="bold", pad=18)
    # compact emphasis legend (dot = one of 3 MD seeds; tick = median)
    leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor=ACCENT, markeredgecolor="white",
                  markersize=9, label="reproducibly poised (median ≥ 0.5)"),
           Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTE, markeredgecolor="white",
                  markersize=7, label="not poised"),
           Line2D([0], [0], marker="|", color=ACCENT_DK, markersize=12, markeredgewidth=2,
                  linestyle="none", label="median seed  ·  dots = 3 MD seeds"),
           Line2D([0], [0], marker="D", color="none", markerfacecolor="#2166ac", markeredgecolor="white",
                  markersize=8, label="Boltz cofold pose (static, n=50)")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False,
              fontsize=9, ncol=2, handletextpad=0.4, columnspacing=2.2)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
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
    bz = collect_boltz(a.run_dirs, list(cs), a.near_d, a.near_ang)
    plot(cs, a.out, a.near_d, a.near_ang, a.label_all, boltz_na=bz)


if __name__ == "__main__":
    _cli()
