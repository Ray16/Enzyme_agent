#!/usr/bin/env python3
"""Reasoning trace attached ALONG the pipeline for one run (dense version).

Reads a run's events + family decisions + ranked candidates and draws a single
top-to-bottom flow where each pipeline stage is annotated with what the agent
actually did:  parse the reaction (with the WHY behind each field) -> enumerate
mechanisms -> the key expand/stop family decision (each family shown with its
representative's native REACTION) -> candidates -> the real top leads.

TARGET-AGNOSTIC.  Every label, molecule and family in this figure is read from
the run directory: the target from ``target_reasoning.json``, the families and
the agent's own stated reasons from ``family_decisions.json``, and each
family's representative plus its native substrates from
``ranked_candidates.csv``.  Nothing about any particular campaign's chemistry,
substrates or enzyme families is written into this file -- pointing it at a run
for a different reaction produces that reaction's figure with no code change.

Usage:  python figures/generate_reasoning_trace.py RUN_DIR [-o OUT.png]
        (with no RUN_DIR, the most recent run holding the required artifacts)
Outputs: figures/reasoning_trace.png, or -o
"""
import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

# Resolution controls.  The layout was tuned at BASE_DPI with SS=1; rendering the
# molecule bitmaps at SS× and the whole figure at DPI, while shrinking each image's
# zoom by SS*(BASE_DPI/DPI), keeps the physical layout pixel-identical (so no new
# overlaps) but gives the 2D structures much higher native resolution.
BASE_DPI, DPI, SS = 200, 300, 4

FIG_DIR = Path(__file__).resolve().parent
RUNS_DIR = FIG_DIR.parent / "runs"

# Artifacts this figure cannot be drawn without.
REQUIRED = ("events.jsonl", "family_decisions.json", "ranked_candidates.csv")


def latest_run(runs_dir: Path = RUNS_DIR) -> Path:
    """Most recently modified run directory that has the required artifacts."""
    usable = [d for d in runs_dir.iterdir()
              if d.is_dir() and all((d / f).exists() for f in REQUIRED)]
    if not usable:
        raise SystemExit(f"No run under {runs_dir} has all of {REQUIRED}; pass RUN_DIR explicitly.")
    return max(usable, key=lambda d: d.stat().st_mtime)

INK, MUTED, LINE = "#0f172a", "#475569", "#94a3b8"
BLUE, PURPLE, TEAL = "#2563eb", "#7c3aed", "#0d9488"
GREEN, RED = "#15803d", "#b91c1c"

N_PER_SIDE = 4          # family cards drawn per column (2x2 grid each side)


def _tint(hex_color: str, amount: float) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _shorten(text: str, limit: int) -> str:
    """Trim to ``limit`` chars on a word boundary, with an ellipsis."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _short_mech(name: str) -> str:
    """Abbreviate a mechanism-family name without assuming any chemistry.

    Drops a trailing generic noun and any parenthetical, so e.g.
    "Diphosphomevalonate decarboxylase-like N-terminal domain" stays readable
    while nothing about a particular reaction is baked in.
    """
    name = re.sub(r"\s*\([^)]*\)", "", name or "").strip()
    name = re.sub(r",?\s+(catalytic|N-terminal|C-terminal)?\s*(domain|superfamily|family|fold)\s*$",
                  "", name, flags=re.I).strip()
    return _shorten(name, 34) or "unnamed"


def _png_to_rgb(png_bytes):
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def pair_img(sa, sb, la, lb, size=(210, 150)):
    mols = [Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)]
    pw, ph = size[0] * SS, size[1] * SS
    d = rdMolDraw2D.MolDraw2DCairo(pw * 2, ph, pw, ph)
    o = d.drawOptions()
    o.legendFontSize = int(15 * SS)
    o.bondLineWidth = max(1, int(1.3 * SS))
    o.padding = 0.03
    d.DrawMolecules(mols, legends=[la, lb])
    d.FinishDrawing()
    return _png_to_rgb(d.GetDrawingText())


def mol_img(smiles, size=(220, 150)):
    pw, ph = size[0] * SS, size[1] * SS
    d = rdMolDraw2D.MolDraw2DCairo(pw, ph)
    o = d.drawOptions()
    o.bondLineWidth = max(1, int(1.3 * SS))
    o.padding = 0.04
    d.DrawMolecule(Chem.MolFromSmiles(smiles))
    d.FinishDrawing()
    return _png_to_rgb(d.GetDrawingText())


def _json_or(run: Path, name: str, default):
    path = run / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return default


def load(run: Path) -> dict:
    """Collect everything the figure draws, entirely from the run directory."""
    evs = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    contract = next((e for e in evs if e["type"] == "contract_derived"), {})
    complete = next((e for e in evs if e["type"] == "run_complete"), {})
    screened = [e for e in evs if e["type"] == "candidates_screened"]

    # target_reasoning.json is the authoritative record of what was asked for.
    target = _json_or(run, "target_reasoning.json", {})
    for field in ("reaction_direction", "bond_types", "reactive_groups", "substrate_topology"):
        if not contract.get(field) and target.get(field):
            contract[field] = target[field]

    decisions = _json_or(run, "family_decisions.json", {}).get("decisions", {})
    mech = _json_or(run, "mechanism_families.json", [])
    mech_names = [_short_mech(m.get("name", "")) for m in mech if m.get("name")]

    rows = {}
    with open(run / "ranked_candidates.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["accession"], r)

    return {
        "contract": contract, "complete": complete, "target": target,
        "decisions": decisions, "mech_names": mech_names, "rows": rows,
        "n_expand": sum(1 for v in decisions.values() if v.get("decision") == "expand"),
        "n_stop": sum(1 for v in decisions.values() if v.get("decision") == "stop"),
        "n_folds": max((s.get("fold_families", 0) for s in screened), default=0),
    }


_STOPWORDS = {"domain", "family", "superfamily", "terminal", "related",
              "containing", "like", "protein", "subunit", "binding", "putative"}


def _family_rep(family_name: str, ranked: list, taken: set | None = None) -> dict | None:
    """Best candidate whose InterPro / mechanism family matches ``family_name``.

    Scored on how many informative words the candidate's family strings share
    with this family name, so sibling families that differ by only one word do
    not all collapse onto the same representative.  Candidates already shown on
    another card are only reused when nothing else matches, which keeps the
    cards distinct without inventing a match.
    """
    taken = taken or set()
    words = {w for w in re.findall(r"[A-Za-z0-9-]{4,}", (family_name or "").lower())
             if w not in _STOPWORDS}
    if not words:
        return None
    best, best_key = None, (0, 0)
    for r in ranked:
        haystack = f"{r.get('interpro_family','')} {r.get('mechanism_family','')}".lower()
        hits = sum(1 for w in words if w in haystack)
        if not hits:
            continue
        # prefer: more shared words, then an unused candidate, then better rank
        key = (hits, r.get("accession") not in taken)
        if key > best_key:
            best, best_key = r, key
    return best


def family_cards(decisions: dict, ranked: list, decision: str, limit: int) -> list:
    """(label, reason, representative-row) for families with the given decision.

    Families whose representative can be identified are listed first, since
    those cards can show a real native reaction.
    """
    named = [(info.get("name") or "unnamed family", info.get("reason") or "")
             for info in decisions.values() if info.get("decision") == decision]
    items, taken = [], set()
    for name, reason in named:
        rep = _family_rep(name, ranked, taken)
        if rep is not None:
            taken.add(rep.get("accession"))
        items.append((name, reason, rep))
    items.sort(key=lambda t: t[2] is None)
    return items[:limit]


def native_pair(row: dict | None) -> tuple:
    """Up to two (label, SMILES) native substrates of a candidate row."""
    if not row:
        return ()
    names = [n.strip() for n in (row.get("native_substrates") or "").split(";") if n.strip()]
    smis = [s.strip() for s in (row.get("native_substrate_smiles") or "").split(";") if s.strip()]
    pairs = []
    for name, smi in zip(names, smis):
        if Chem.MolFromSmiles(smi) is not None:
            pairs.append((_shorten(name, 24), smi))
        if len(pairs) == 2:
            break
    return tuple(pairs)


def build(run: Path, out: Path):
    data = load(run)
    contract, complete, target = data["contract"], data["complete"], data["target"]
    decisions, mech_names, rows = data["decisions"], data["mech_names"], data["rows"]
    n_expand, n_stop, n_folds = data["n_expand"], data["n_stop"], data["n_folds"]
    leads = list(rows.values())[:5]

    fig = plt.figure(figsize=(16.5, 13.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def card(x, y, w, h, fc, ec, lw=1.6, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                     fc=fc, ec=ec, lw=lw, zorder=z))

    def arrow(y0, y1, x=0.5):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=16, color=LINE, lw=2.0, zorder=1))

    def tag(x, y, text, color, fs=13):
        ax.text(x, y, text, ha="left", va="center", fontsize=fs, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.34", fc=color, ec="none"), zorder=4)

    def img_at(arr, x, y, zoom):
        # arr is rendered at SS× native px and the figure at DPI; shrink the zoom so
        # the on-page size matches the SS=1/BASE_DPI layout exactly (no new overlaps).
        z = zoom / SS * (BASE_DPI / DPI)
        ax.add_artist(AnnotationBbox(OffsetImage(arr, zoom=z), (x, y), frameon=False, zorder=5))

    # ---- title ----
    ax.text(0.5, 0.988, "Enzyme Discovery Agent — reasoning trace along the pipeline",
            ha="center", va="top", fontsize=22, fontweight="bold", color=INK)
    subtitle = _shorten(target.get("substrate_name") or target.get("query") or "", 120)
    ax.text(0.5, 0.966, f"one run · target: {subtitle}" if subtitle else "one run",
            ha="center", va="top", fontsize=13.5, color=MUTED, style="italic")

    # ---- 1. INPUT (target substrates, read from the run) ----
    smis = [s for s in (target.get("substrate_smiles") or "").split(".") if s.strip()]
    smis = [s for s in smis if Chem.MolFromSmiles(s) is not None][:3]
    names = [n.strip() for n in re.split(r"\s*(?:\+|;|,)\s*",
                                         target.get("substrate_name") or "") if n.strip()]
    card(0.24, 0.862, 0.52, 0.076, "white", BLUE)
    tag(0.253, 0.930, "INPUT  ·  target substrates", BLUE)
    if smis:
        xs = [0.5] if len(smis) == 1 else [0.38, 0.62] if len(smis) == 2 else [0.33, 0.5, 0.67]
        for i, (sx, smi) in enumerate(zip(xs, smis)):
            img_at(mol_img(smi, (260, 118)), sx, 0.901, 0.58 if len(smis) < 3 else 0.42)
            label = _shorten(names[i], 26) if i < len(names) and len(names) == len(smis) else ""
            if label:
                ax.text(sx, 0.868, label, ha="center", fontsize=10, color=MUTED)
        for sx0, sx1 in zip(xs, xs[1:]):
            ax.text((sx0 + sx1) / 2, 0.901, "+", ha="center", va="center", fontsize=20, color=INK)
    else:
        ax.text(0.5, 0.901, _shorten(target.get("query") or "target reaction", 90),
                ha="center", va="center", fontsize=12, color=INK)
    arrow(0.862, 0.852)

    # ---- 2. PARSE (4 columns, each with the why) ----
    card(0.04, 0.752, 0.92, 0.096, _tint(BLUE, 0.94), BLUE)
    tag(0.055, 0.840, "PARSE REACTION  →  contract", BLUE)
    bt = ", ".join(contract.get("bond_types", []) or []) or "—"
    rd = contract.get("reaction_direction", "") or "—"
    rg = " + ".join(contract.get("reactive_groups", []) or []) or "—"
    topo = (contract.get("substrate_topology") or "—").replace("_", " ")
    cells = [
        ("bond", bt, "bond type the\ncandidate must act on"),
        ("direction", rd, "rejects the opposite\ndirection (gate)"),
        ("reactive groups", rg, "seed search terms\n& substrate match"),
        ("substrates", topo, "constrains which\nfolds can bind"),
    ]
    cw = 0.212
    for i, (k, v, why) in enumerate(cells):
        cx = 0.06 + i * (cw + 0.018)
        card(cx, 0.760, cw, 0.070, "white", _tint(BLUE, 0.4), lw=1.2)
        ax.text(cx + cw / 2, 0.821, k, ha="center", va="center", fontsize=11,
                color=BLUE, fontweight="bold")
        ax.text(cx + cw / 2, 0.803, v, ha="center", va="center", fontsize=12,
                color=INK, fontweight="bold", linespacing=0.95)
        ax.text(cx + cw / 2, 0.780, why, ha="center", va="center", fontsize=9.5,
                color=MUTED, linespacing=0.95)
    arrow(0.752, 0.744)

    # ---- 3. ENUMERATE ----
    card(0.10, 0.658, 0.80, 0.078, _tint(PURPLE, 0.94), PURPLE)
    tag(0.115, 0.726, f"ENUMERATE MECHANISMS  ·  {len(mech_names)} distinct folds",
        PURPLE, fs=12.5)
    ax.text(0.5, 0.697, "   ·   ".join(mech_names[:4]), ha="center", va="center",
            fontsize=13, color=INK)
    ax.text(0.5, 0.673, "   ·   ".join(mech_names[4:]), ha="center", va="center",
            fontsize=13, color=INK)
    arrow(0.658, 0.632)

    # ---- 4. TRIAGE ----
    card(0.02, 0.150, 0.96, 0.478, "white", INK, lw=1.2)
    tag(0.035, 0.615, f"TRIAGE FAMILIES  →  expand vs stop   "
        f"(key decision:  {n_expand} expand · {n_stop} stop)", INK)

    cw2, ch2, gx, gy = 0.222, 0.150, 0.013, 0.013
    gtop = 0.556
    ranked = list(rows.values())  # already in rank order (CSV is rank-sorted)

    def fam_card(cx, cy, title, badge, badge_color, mol_arr, caption, accent,
                 subcaption="", glyph_label=None):
        card(cx, cy, cw2, ch2, "white", accent, lw=1.4)
        # header band inside the card
        ax.add_patch(FancyBboxPatch((cx + 0.006, cy + ch2 - 0.030), cw2 - 0.012, 0.026,
                     boxstyle="round,pad=0.002", fc=_tint(accent, 0.9), ec="none", zorder=3))
        ax.text(cx + 0.016, cy + ch2 - 0.017, title, ha="left", va="center",
                fontsize=10.8, fontweight="bold", color=INK, zorder=4)
        if badge:
            ax.text(cx + cw2 - 0.016, cy + ch2 - 0.017, badge, ha="right", va="center",
                    fontsize=10, color=badge_color, fontweight="bold", zorder=4)
        if glyph_label:  # one substrate is a macromolecule: mol + protein glyph
            img_at(mol_arr, cx + cw2 * 0.30, cy + ch2 * 0.52, 0.34)
            ax.text(cx + cw2 * 0.545, cy + ch2 * 0.52, "+", ha="center", va="center",
                    fontsize=13, color=INK)
            bw, bh = cw2 * 0.30, ch2 * 0.36
            bx, by = cx + cw2 * 0.63, cy + ch2 * 0.52 - bh / 2
            ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.006",
                         fc="#e5e7eb", ec=MUTED, lw=1.2, zorder=5))
            ax.text(bx + bw / 2, by + bh / 2, glyph_label, ha="center", va="center",
                    fontsize=9.0, color=MUTED, zorder=6, linespacing=1.1)
        elif mol_arr is not None:
            img_at(mol_arr, cx + cw2 / 2, cy + ch2 * 0.50, 0.46)
        else:
            ax.text(cx + cw2 / 2, cy + ch2 * 0.52, "structure\nnot recorded", ha="center",
                    va="center", fontsize=9.5, color=MUTED, style="italic", zorder=4)
        ax.text(cx + cw2 / 2, cy + 0.026, caption, ha="center", va="center",
                fontsize=10.5, color=INK)
        if subcaption:
            ax.text(cx + cw2 / 2, cy + 0.011, subcaption, ha="center", va="center",
                    fontsize=9.2, color=MUTED, style="italic")

    def draw_column(x0, accent, heading, decision, badge_for):
        ax.add_patch(FancyBboxPatch((x0, 0.570), 2 * cw2 + gx, 0.032,
                     boxstyle="round,pad=0.004", fc=accent, ec="none", zorder=3))
        ax.text(x0 + cw2 + gx / 2, 0.586, heading, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=4)
        for i, (fam, reason, rep) in enumerate(
                family_cards(decisions, ranked, decision, N_PER_SIDE)):
            cx = x0 + (i % 2) * (cw2 + gx)
            cy = gtop - ch2 - (i // 2) * (ch2 + gy)
            pair = native_pair(rep)
            if len(pair) == 2:
                arr = pair_img(pair[0][1], pair[1][1], pair[0][0], pair[1][0])
                caption = f"{pair[0][0]} + {pair[1][0]}"
            elif len(pair) == 1:
                arr = mol_img(pair[0][1], (200, 150))
                caption = pair[0][0]
            else:
                arr = None
                caption = "no native substrate recorded"
            badge = badge_for(rep)
            fam_card(cx, cy, _shorten(_short_mech(fam), 30 - len(badge)), badge, accent, arr,
                     _shorten(caption, 34), accent, subcaption=_shorten(reason, 58))

    # pursued (left half) ------------------------------------------------------
    lx = 0.035
    draw_column(lx, GREEN, "PURSUED  —  native chemistry matches the target", "expand",
                lambda r: (" · ".join(x for x in (r.get("gene") or "",
                                                   f"#{r.get('rank')}" if r.get("rank") else "")
                                      if x) or "expand") if r else "expand")

    # stopped (right half) -----------------------------------------------------
    rx = lx + 2 * cw2 + gx + 0.036
    draw_column(rx, RED, "STOPPED  —  off-target native chemistry", "stop",
                lambda r: "✗ off-target")

    # full-width decision-rule strip -------------------------------------------
    card(0.035, 0.162, 2 * (2 * cw2 + gx) + 0.036, 0.048, _tint(INK, 0.96), INK, lw=1.2)
    ax.text(0.5, 0.186, "decision rule:  same fold ≠ same substrate — pursue only the members "
            "whose native reaction matches the target chemistry",
            ha="center", va="center", fontsize=11.5, color=INK)
    arrow(0.150, 0.132)

    # ---- 5. SEARCH / SCREEN ----
    n_final = complete.get("candidates", 0)
    card(0.10, 0.082, 0.80, 0.048, _tint(TEAL, 0.94), TEAL)
    tag(0.115, 0.122, "SEARCH · SCREEN · EVALUATE", TEAL)
    ax.text(0.5, 0.100, f"pursued families → resolve members + orthologs across {n_folds} folds  "
            f"→  {n_final} candidates evaluated, deduplicated to distinct enzymes",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    arrow(0.082, 0.066)

    # ---- 6. RANK -> top leads (real, from CSV) ----
    card(0.06, 0.006, 0.88, 0.058, _tint(TEAL, 0.88), TEAL, lw=1.8)
    tag(0.075, 0.056, "RANK  →  top leads (deterministic priority spine)", TEAL)
    n = max(len(leads), 1)
    cwl = 0.86 / n
    for i, r in enumerate(leads):
        cx = 0.07 + i * cwl
        tier = r.get("tier_label", "") or f"tier {r.get('tier','')}"
        fam = (r.get("interpro_family") or "").split(",")[0]
        if len(fam) > 26:
            fam = fam[:25].rsplit(" ", 1)[0] + "…"
        ax.text(cx + cwl / 2, 0.041, f"{i+1}. {r.get('gene','?')}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(cx + cwl / 2, 0.025, fam, ha="center", va="center", fontsize=9.3, color=MUTED)
        ax.text(cx + cwl / 2, 0.012, tier, ha="center", va="center", fontsize=10.5,
                color=TEAL, fontweight="bold")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  (run: {run.name})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="discovery run directory; default = most recent usable run")
    ap.add_argument("-o", "--out", default=None, help="output PNG path")
    args = ap.parse_args(argv)
    run = Path(args.run_dir).resolve() if args.run_dir else latest_run()
    missing = [f for f in REQUIRED if not (run / f).exists()]
    if missing:
        raise SystemExit(f"{run} is missing required artifact(s): {missing}")
    build(run, Path(args.out) if args.out else FIG_DIR / "reasoning_trace.png")


if __name__ == "__main__":
    main()
