#!/usr/bin/env python3
"""Reasoning trace attached ALONG the pipeline for one run.

Reads a run's events + family decisions + ranked candidates and draws a single
top-to-bottom flow where each pipeline stage is annotated with what the agent
actually did: parse the reaction -> enumerate mechanisms -> the key expand/stop
family decision (with example native substrates of the pursued folds) ->
candidates -> top leads.

Run-specific results figure (shows the real target and folds), not a generic
pipeline diagram.

Usage:  python figures/generate_reasoning_trace.py [RUN_DIR]
Outputs: figures/reasoning_trace.png
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

FIG_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = (
    FIG_DIR.parent / "runs"
    / "20260916_120414_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl"
)

INK, MUTED, LINE = "#0f172a", "#475569", "#94a3b8"
BLUE, PURPLE, TEAL = "#2563eb", "#7c3aed", "#0d9488"
GREEN, RED = "#15803d", "#b91c1c"

# Distinct pursued mechanism families (the 8 expanded InterPro entries collapse
# into these): (family label, representative gene, representative accession,
# native-substrate name to illustrate, partner clause).
PURSUED = [
    ("NIS / IucA-IucC synthetase", "AsbA", "A0A0F7RJ52", "spermidine", "+ citrate"),
    ("Glutamine-synthetase fold", "PuuA", "P78061", "1,4-butanediammonium", "+ L-glutamate"),
    ("ATP-grasp ligase (PylC)", "PylC", "Q8PWY3", "(3R)-3-methyl-D-ornithine", "+ L-lysine"),
    ("Asparagine synthetase B", "AsnB", "O24338", "L-aspartate", "+ L-glutamine"),
]
# Rejected: same folds, off-target native chemistry (grouped).
REJECTED = [
    "Purine / nucleotide biosynthesis (PurK/PurT, PRGA synthetase)",
    "Prenyltransferases (NIS-fold, isoprenoid — not amide)",
    "LysW-carrier-tethered glutamate ligase (RimK / ATP-grasp)",
]


def _tint(hex_color: str, amount: float) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def mol_img(smiles: str, size=(240, 150)):
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    import io
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(d.GetDrawingText())))


def load(run: Path):
    evs = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    contract = next((e for e in evs if e["type"] == "contract_derived"), {})
    complete = next((e for e in evs if e["type"] == "run_complete"), {})
    dec = json.loads((run / "family_decisions.json").read_text())["decisions"]
    n_expand = sum(1 for v in dec.values() if v["decision"] == "expand")
    n_stop = sum(1 for v in dec.values() if v["decision"] == "stop")
    mech = json.loads((run / "mechanism_families.json").read_text())
    mech_names = [_short_mech(m["name"]) for m in mech]
    rows = {}
    with open(run / "ranked_candidates.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["accession"], r)
    return contract, complete, n_expand, n_stop, mech_names, rows


def sub_smiles(rows, acc, name):
    r = rows.get(acc, {})
    names = [n.strip() for n in (r.get("native_substrates") or "").split(";")]
    smis = [s.strip() for s in (r.get("native_substrate_smiles") or "").split(";")]
    for n, s in zip(names, smis):
        if name.lower() in n.lower():
            return s
    return smis[0] if smis else ""


def build(run: Path):
    contract, complete, n_expand, n_stop, mech_names, rows = load(run)

    fig = plt.figure(figsize=(13.5, 15.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def card(x, y, w, h, fc, ec, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                     fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(y0, y1, x=0.5):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=18, color=LINE, lw=2.2, zorder=1))

    def stage_tag(x, y, text, color):
        ax.text(x, y, text, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"), zorder=4)

    def molecule(smiles, x, y, zoom=0.62):
        try:
            im = OffsetImage(mol_img(smiles), zoom=zoom)
            ax.add_artist(AnnotationBbox(im, (x, y), frameon=False, zorder=5))
        except Exception:
            pass

    # title
    ax.text(0.5, 0.982, "Enzyme Discovery Agent — reasoning trace along the pipeline",
            ha="center", va="top", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.958, "one run · target: adipic acid + hexamethylenediamine (nylon-6,6 amide ligation)",
            ha="center", va="top", fontsize=10.5, color=MUTED, style="italic")

    # ---------- 1. INPUT ----------
    y = 0.945
    card(0.14, 0.855, 0.72, 0.088, "white", BLUE)
    stage_tag(0.155, 0.930, "INPUT  ·  target reaction", BLUE)
    ax.text(0.30, 0.900, "diacid", ha="center", fontsize=9.5, color=MUTED)
    ax.text(0.70, 0.900, "diamine", ha="center", fontsize=9.5, color=MUTED)
    molecule("O=C(O)CCCCC(=O)O", 0.30, 0.878, zoom=0.55)
    molecule("NCCCCCCN", 0.70, 0.878, zoom=0.55)
    ax.text(0.5, 0.878, "+", ha="center", va="center", fontsize=20, color=INK)

    arrow(0.855, 0.828)

    # ---------- 2. PARSE ----------
    card(0.10, 0.762, 0.80, 0.064, _tint(BLUE, 0.93), BLUE)
    stage_tag(0.115, 0.814, "PARSE REACTION  →  contract", BLUE)
    bt = ", ".join(contract.get("bond_types", []) or [])
    rd = contract.get("reaction_direction", "")
    rg = " + ".join(contract.get("reactive_groups", []) or [])
    ax.text(0.5, 0.783, f"bond = {bt}      direction = {rd}      groups = {rg}      "
            f"substrates = free small molecules",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

    arrow(0.762, 0.735)

    # ---------- 3. ENUMERATE ----------
    card(0.10, 0.655, 0.80, 0.078, _tint(PURPLE, 0.94), PURPLE)
    stage_tag(0.115, 0.722, f"ENUMERATE MECHANISMS  ·  {len(mech_names)} candidate folds", PURPLE)
    ax.text(0.5, 0.688, "  ·  ".join(mech_names[:4]), ha="center", va="center",
            fontsize=10, color=INK)
    ax.text(0.5, 0.668, "  ·  ".join(mech_names[4:]), ha="center", va="center",
            fontsize=10, color=INK)

    arrow(0.655, 0.628)

    # ---------- 4. TRIAGE (key decision) ----------
    card(0.04, 0.243, 0.92, 0.382, "white", INK, lw=1.2)
    stage_tag(0.055, 0.612, f"TRIAGE FAMILIES  →  expand vs stop   (key decision:  "
              f"{n_expand} expand · {n_stop} stop)", INK)

    # pursued (left)
    lx = 0.065
    ax.add_patch(FancyBboxPatch((lx, 0.545), 0.55, 0.038, boxstyle="round,pad=0.004",
                 fc=GREEN, ec="none", zorder=3))
    ax.text(lx + 0.275, 0.564, "PURSUED  —  fold condenses a free amine onto a carboxylate",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=4)
    # 2x2 grid of pursued example cards
    cw, ch, gx, gy = 0.267, 0.128, 0.016, 0.014
    x0, y0 = lx, 0.398
    for i, (fam, gene, acc, subname, partner) in enumerate(PURSUED):
        cx = x0 + (i % 2) * (cw + gx)
        cy = y0 - (i // 2) * (ch + gy)
        card(cx, cy, cw, ch, _tint(GREEN, 0.93), GREEN, lw=1.4)
        ax.text(cx + 0.012, cy + ch - 0.020, fam, ha="left", va="center",
                fontsize=9.6, fontweight="bold", color=INK)
        ax.text(cx + 0.012, cy + ch - 0.040, f"e.g. {gene}", ha="left", va="center",
                fontsize=8.6, color=GREEN, fontweight="bold")
        molecule(sub_smiles(rows, acc, subname), cx + cw * 0.5, cy + ch * 0.40, zoom=0.5)
        ax.text(cx + cw * 0.5, cy + 0.010, f"{subname.split('(')[0].strip()} {partner}",
                ha="center", va="center", fontsize=8.2, color=MUTED)

    # rejected (right)
    rx = 0.645
    ax.add_patch(FancyBboxPatch((rx, 0.545), 0.30, 0.038, boxstyle="round,pad=0.004",
                 fc=RED, ec="none", zorder=3))
    ax.text(rx + 0.15, 0.564, "STOPPED  —  off-target native chemistry",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=4)
    for i, txt in enumerate(REJECTED):
        cy = 0.500 - i * 0.058
        card(rx, cy, 0.30, 0.046, _tint(RED, 0.94), RED, lw=1.3)
        ax.text(rx + 0.012, cy + 0.023, txt, ha="left", va="center", fontsize=8.7,
                color=INK, wrap=True)
    ax.text(rx + 0.15, 0.300, "same folds — rejected because their\nmembers act on the wrong substrates",
            ha="center", va="center", fontsize=8.6, color=MUTED, style="italic")

    arrow(0.243, 0.216)

    # ---------- 5. SEARCH & SCREEN ----------
    n_final = complete.get("candidates", 0)
    card(0.10, 0.150, 0.80, 0.062, _tint(TEAL, 0.93), TEAL)
    stage_tag(0.115, 0.200, "SEARCH · SCREEN · EVALUATE", TEAL)
    ax.text(0.5, 0.170, f"pursued families → resolve members + orthologs  "
            f"→  {n_final} candidates evaluated (deduplicated to distinct enzymes)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=INK)

    arrow(0.150, 0.123)

    # ---------- 6. RANK -> LEADS ----------
    card(0.10, 0.038, 0.80, 0.078, _tint(TEAL, 0.86), TEAL, lw=1.8)
    stage_tag(0.115, 0.105, "RANK  →  top leads (Tier 1)", TEAL)
    ax.text(0.5, 0.074, "GlnA2 · PuuA  (γ-glutamyl-polyamine synthetases, GS-fold)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.5, 0.052, "AsbA · IucA  (NIS siderophore synthetases)  —  ranked by the "
            "deterministic priority spine",
            ha="center", va="center", fontsize=9.5, color=MUTED)

    out = FIG_DIR / "reasoning_trace.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    build(run)
