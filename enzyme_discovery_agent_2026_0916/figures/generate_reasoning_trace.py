#!/usr/bin/env python3
"""Visualise one run's REASONING TRACE in a single figure.

Reads a run's ``events.jsonl`` + ``family_decisions.json`` + ``family_landscape.json``
and renders two stacked panels:

  (top)    Decision timeline  -- phase bands, distinct-candidate growth, and the
           milestone decisions the agent made, across its iterations.
  (bottom) Mechanism-family triage -- the core reasoning step: which candidate
           folds the agent chose to PURSUE (expand) vs REJECT (stop), and why.

This is a run-specific artifact (it shows the real target and the folds the agent
reasoned about), not a generic pipeline diagram.

Usage:  python figures/generate_reasoning_trace.py [RUN_DIR]
Outputs: figures/reasoning_trace.png
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = (
    FIG_DIR.parent / "runs"
    / "20260916_120414_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl"
)

INK, MUTED = "#0f172a", "#475569"
BLUE, PURPLE, TEAL = "#2563eb", "#7c3aed", "#0d9488"
GREEN, RED = "#15803d", "#dc2626"
PHASE_COLOR = {"reaction_analysis": BLUE, "discovery": PURPLE, "ranking_and_report": TEAL}
PHASE_NAME = {
    "reaction_analysis": "Phase 1 · Reaction analysis",
    "discovery": "Phase 2 · Enzyme discovery",
    "ranking_and_report": "Phase 3 · Evaluation & ranking",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#cbd5e1", "svg.fonttype": "none",
})


def _tint(hex_color: str, amount: float) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def load(run: Path):
    evs = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    land = json.loads((run / "family_landscape.json").read_text())
    names = {x["fold_id"]: x.get("fold_name", "") for x in land}
    dec = json.loads((run / "family_decisions.json").read_text())["decisions"]
    return evs, names, dec


def _why(reason: str) -> str:
    """A short 'why' clause from a decision reason (text after the em dash)."""
    r = reason.replace("—", "-")
    tail = r.split("-", 1)[1] if "-" in r else r
    tail = tail.split("(")[0].split(";")[0].strip().rstrip(".,")
    return tail[0].upper() + tail[1:] if tail else ""


def build(run: Path):
    evs, names, dec = load(run)
    by = lambda t: [e for e in evs if e["type"] == t]

    # ---- timeline series: max distinct-candidate count seen per iteration ----
    per_iter, phase_of = {}, {}
    for e in by("iteration_start"):
        it = e["iteration"]
        per_iter[it] = max(per_iter.get(it, 0), e.get("candidate_count", 0))
        phase_of.setdefault(it, e.get("phase", ""))
    iters = sorted(per_iter)
    counts = [per_iter[i] for i in iters]
    final = by("run_complete")[0] if by("run_complete") else {}
    contract = by("contract_derived")[0] if by("contract_derived") else {}
    n_eval = final.get("candidates", 0)

    # phase spans (contiguous ranges of the same phase label)
    spans = []
    for i in iters:
        p = phase_of[i]
        if spans and spans[-1][0] == p:
            spans[-1][2] = i
        else:
            spans.append([p, i, i])

    # ---- triage: expanded vs stopped families ----
    expand, stop = [], []
    seen = set()
    for ipr, info in dec.items():
        if ipr in seen:
            continue
        seen.add(ipr)
        label = names.get(ipr) or ipr
        row = (label, _why(info.get("reason", "")))
        (expand if info["decision"] == "expand" else stop).append(row)

    # =========================== figure ===========================
    fig = plt.figure(figsize=(15.5, 11.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.15], hspace=0.28,
                          left=0.065, right=0.975, top=0.9, bottom=0.04)

    # ---------- top: decision timeline ----------
    ax = fig.add_subplot(gs[0])
    for p, lo, hi in spans:
        ax.axvspan(lo - 0.5, hi + 0.5, color=_tint(PHASE_COLOR.get(p, "#94a3b8"), 0.86), zorder=0)
        ax.text((lo + hi) / 2, 1.04, PHASE_NAME.get(p, p), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=PHASE_COLOR.get(p, MUTED))
    ax.fill_between(iters, counts, step="post", color=_tint(TEAL, 0.55), zorder=1)
    ax.step(iters, counts, where="post", color=TEAL, lw=2.4, zorder=2)

    # milestone annotations (iteration, text, y-offset in candidate units)
    def firstiter(t, default):
        xs = [e.get("iteration") for e in by(t) if isinstance(e.get("iteration"), int)]
        return min(xs) if xs else default
    milestones = [
        (firstiter("gate", 6) if any(e["gate"] == "literature_harvest" for e in by("gate")) else 6,
         "Read literature", 6),
        (next((e["iteration"] for e in by("gate") if e["gate"] == "mechanism_family"), 6),
         "Enumerate mechanisms", 15),
        (7, f"Triage families\n{len(expand)} pursue · {len(stop)} reject", 22),
        (9, "Register\ncandidates", 9),
        (next((e["iteration"] for e in by("coverage_checkpoint")), 10), "Coverage check", 15),
        (max(iters), f"Evaluate {n_eval}\n→ rank → report", 20),
    ]
    for x, txt, y in milestones:
        ax.annotate(txt, xy=(x, min(y, max(counts) + 1)), xytext=(x, y + 3.5),
                    ha="center", va="bottom", fontsize=9.2, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=1.1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", lw=0.8))
    ax.set_xlim(0.5, max(iters) + 0.5)
    ax.set_ylim(0, max(counts) + 12)
    ax.set_xlabel("agent iteration", fontsize=10.5, color=MUTED)
    ax.set_ylabel("distinct candidates", fontsize=10.5, color=MUTED)
    ax.set_yticks([0, 10, 20])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)

    # contract subtitle
    bt = ", ".join(contract.get("bond_types", []) or [])
    rd = contract.get("reaction_direction", "")
    rg = " + ".join(contract.get("reactive_groups", []) or [])
    sub = f"derived reaction contract:  bond = {bt}   ·   direction = {rd}   ·   groups = {rg}"
    ax.set_title(sub, fontsize=10, color=MUTED, loc="left", pad=26)

    # ---------- bottom: mechanism-family triage ----------
    axb = fig.add_subplot(gs[1]); axb.axis("off")
    axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    axb.text(0.5, 1.0, "Mechanism-family triage  —  which folds the agent pursued vs rejected",
             ha="center", va="top", fontsize=13.5, fontweight="bold", color=INK)

    def column(x0, x1, header, rows, accent):
        axb.add_patch(FancyBboxPatch((x0, 0.905), x1 - x0, 0.05,
                      boxstyle="round,pad=0.006", fc=accent, ec="none",
                      transform=axb.transAxes, zorder=2))
        axb.text((x0 + x1) / 2, 0.93, header, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color="white", zorder=3)
        n = max(len(rows), 1)
        top, bot, gap = 0.87, 0.02, 0.018
        rh = (top - bot - gap * (n - 1)) / n
        rh = min(rh, 0.115)
        for i, (name, why) in enumerate(rows):
            y = top - i * (rh + gap) - rh
            axb.add_patch(FancyBboxPatch((x0, y), x1 - x0, rh,
                          boxstyle="round,pad=0.004", fc=_tint(accent, 0.9),
                          ec=accent, lw=1.4, transform=axb.transAxes, zorder=1))
            axb.text(x0 + 0.012, y + rh * 0.64, textwrap.shorten(name, 52, placeholder="…"),
                     ha="left", va="center", fontsize=10.3, fontweight="bold", color=INK, zorder=2)
            if why:
                axb.text(x0 + 0.012, y + rh * 0.24,
                         textwrap.shorten(why, 60, placeholder="…"),
                         ha="left", va="center", fontsize=8.8, color=MUTED, zorder=2)

    column(0.0, 0.485, f"PURSUED  ·  expanded ({len(expand)})", expand, GREEN)
    column(0.515, 1.0, f"REJECTED  ·  stopped ({len(stop)})", stop, RED)

    # header
    q = final.get("query") or ""
    fig.suptitle("Enzyme Discovery Agent — reasoning trace of one run",
                 x=0.065, ha="left", fontsize=17, fontweight="bold", color=INK, y=0.975)

    out = FIG_DIR / "reasoning_trace.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    build(run)
