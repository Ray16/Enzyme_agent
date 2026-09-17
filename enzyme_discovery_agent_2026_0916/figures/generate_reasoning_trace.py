#!/usr/bin/env python3
"""Reasoning trace attached ALONG the pipeline for one run (dense version).

Reads a run's events + family decisions + ranked candidates and draws a single
top-to-bottom flow where each pipeline stage is annotated with what the agent
actually did:  parse the reaction (with the WHY behind each field) -> enumerate
mechanisms -> the key expand/stop family decision (each pursued fold shown with
its native REACTION, i.e. the amine + carboxylate it condenses) -> candidates ->
the real top leads.

Run-specific results figure (shows the real target and folds), not a generic
pipeline diagram.

Usage:  python figures/generate_reasoning_trace.py [RUN_DIR]
Outputs: figures/reasoning_trace.png
"""
import csv
import io
import json
import sys
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
RDLogger.DisableLog("rdApp.*")

FIG_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = (
    FIG_DIR.parent / "runs"
    / "20260916_120414_find_amide_synthetases_that_ligates_adipic_acid_and_hexamethylenediamine_for_nyl"
)

INK, MUTED, LINE = "#0f172a", "#475569", "#94a3b8"
BLUE, PURPLE, TEAL = "#2563eb", "#7c3aed", "#0d9488"
GREEN, RED = "#15803d", "#b91c1c"

# Distinct pursued mechanism families.  The representative enzyme is NOT hardcoded:
# it is resolved at draw time to the TOP-RANKED member of that family in
# ranked_candidates.csv.  Each entry: (family label, InterPro-match substrings,
# amine substrate to draw, acid substrate to draw).
# (family, InterPro-match substrings, amine-hint, amine-label, acid-hint,
#  acid-label, subcaption).  Hints select the substrate from the run's data;
#  labels are what we print (verified names).
PURSUED = [
    ("Glutamine-synthetase fold", ["glutamine synthetase", "glutamylputrescine"],
     "cadaverine", "cadaverine", "L-glutamate", "L-glutamate",
     "✓ free amine + carboxylate"),
    ("ATP-grasp ligase (PylC)", ["atp-grasp", "pylc"],
     "ornithine", "3-methyl-D-ornithine", "lysin", "L-lysine",
     "✓ free amine + carboxylate"),
    ("NIS / IucA-IucC synthetase", ["iuca", "aerobactin", "siderophore"],
     "spermidine", "spermidine", "citrate", "citrate",
     "✓ free amine + carboxylate"),
    ("Asparagine synthetase B", ["asparagine"],
     "L-glutamine", "L-glutamine", "L-aspartate", "L-aspartate",
     "amide via acyl-adenylate (N from Gln)"),
]
# Rejected folds -> the BOTH native substrates of a representative off-target
# reaction (SMILES from PubChem), shown like the pursued reactions so the
# off-target chemistry (phosphosugar nucleotides / isoprenoid diphosphates /
# poly-glutamate) is explicit.  (fold, subA name, subA SMILES, subB name,
# subB SMILES, why)
STOPPED = [
    ("ATP-grasp (PurT)", "glycinamide ribonucleotide",
     "C(C1C(C(C(O1)NC(=O)CN)O)O)OP(=O)(O)O", "formate", "C(=O)[O-]",
     "purine / nucleotide biosynthesis"),
    ("Polyprenyl synthetase", "dimethylallyl-PP",
     "CC(=CCOP(=O)(O)OP(=O)(O)O)C", "isopentenyl-PP", "CC(=C)CCOP(=O)(O)OP(=O)(O)O",
     "isoprenoid transfer — not amide"),
    ("ATP-grasp (PurD)", "5-phospho-D-ribosylamine",
     "C(C1C(C(C(O1)N)O)O)OP(=O)(O)O", "glycine", "C(C(=O)O)N",
     "glycinamide-ribonucleotide biosynthesis"),
    ("ATP-grasp (RimK-type)", "L-glutamate", "C(CC(=O)O)C(C(=O)O)N",
     "carrier protein", None, "substrate tethered to LysW carrier — not free"),
]


def _tint(hex_color: str, amount: float) -> str:
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    mix = lambda c: int(c + (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _short_mech(name: str) -> str:
    n = name.lower()
    for key, short in (
        ("siderophore", "NIS siderophore"), ("nis", "NIS siderophore"),
        ("atp-grasp", "ATP-grasp"), ("anl", "ANL adenylating"),
        ("glutamine synthetase", "Glutamine-synthetase"),
        ("asparagine synthetase", "Asparagine-synthetase B"),
        ("aminoacyl-trna", "class-II aaRS-like"), ("mur ligase", "Mur-ligase"),
    ):
        if key in n:
            return short
    return name.split()[0]


def pair_img(sa, sb, la, lb, size=(210, 150)):
    mols = [Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)]
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=size, legends=[la, lb])
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img.data))
    return np.asarray(img.convert("RGB"))


def mol_img(smiles, size=(220, 150)):
    return np.asarray(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=size).convert("RGB"))


def load(run: Path):
    evs = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    contract = next((e for e in evs if e["type"] == "contract_derived"), {})
    complete = next((e for e in evs if e["type"] == "run_complete"), {})
    screened = [e for e in evs if e["type"] == "candidates_screened"]
    dec = json.loads((run / "family_decisions.json").read_text())["decisions"]
    n_expand = sum(1 for v in dec.values() if v["decision"] == "expand")
    n_stop = sum(1 for v in dec.values() if v["decision"] == "stop")
    mech = json.loads((run / "mechanism_families.json").read_text())
    mech_names = [_short_mech(m["name"]) for m in mech]
    rows = {}
    with open(run / "ranked_candidates.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["accession"], r)
    n_folds = max((s.get("fold_families", 0) for s in screened), default=0)
    return contract, complete, n_expand, n_stop, mech_names, rows, n_folds


def sub_smiles(rows, acc, name):
    r = rows.get(acc, {})
    names = [n.strip() for n in (r.get("native_substrates") or "").split(";")]
    smis = [s.strip() for s in (r.get("native_substrate_smiles") or "").split(";")]
    for n, s in zip(names, smis):
        if name.lower() in n.lower():
            return s
    return smis[0] if smis else ""


def build(run: Path):
    contract, complete, n_expand, n_stop, mech_names, rows, n_folds = load(run)
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
        ax.add_artist(AnnotationBbox(OffsetImage(arr, zoom=zoom), (x, y), frameon=False, zorder=5))

    # ---- title ----
    ax.text(0.5, 0.988, "Enzyme Discovery Agent — reasoning trace along the pipeline",
            ha="center", va="top", fontsize=22, fontweight="bold", color=INK)
    ax.text(0.5, 0.966, "one run · target: adipic acid + hexamethylenediamine (nylon-6,6 amide ligation)",
            ha="center", va="top", fontsize=13.5, color=MUTED, style="italic")

    # ---- 1. INPUT ----
    card(0.24, 0.866, 0.52, 0.068, "white", BLUE)
    tag(0.253, 0.926, "INPUT  ·  target substrates", BLUE)
    ax.text(0.38, 0.908, "diacid", ha="center", fontsize=11.5, color=MUTED)
    ax.text(0.62, 0.908, "diamine", ha="center", fontsize=11.5, color=MUTED)
    img_at(mol_img("O=C(O)CCCCC(=O)O", (260, 120)), 0.38, 0.889, 0.62)
    img_at(mol_img("NCCCCCCN", (260, 120)), 0.62, 0.889, 0.62)
    ax.text(0.5, 0.889, "+", ha="center", va="center", fontsize=20, color=INK)
    arrow(0.866, 0.852)

    # ---- 2. PARSE (4 columns, each with the why) ----
    card(0.04, 0.752, 0.92, 0.096, _tint(BLUE, 0.94), BLUE)
    tag(0.055, 0.840, "PARSE REACTION  →  contract", BLUE)
    bt = ", ".join(contract.get("bond_types", []) or [])
    rd = contract.get("reaction_direction", "")
    rg = " + ".join(contract.get("reactive_groups", []) or [])
    cells = [
        ("bond", bt, "amide bond to form"),
        ("direction", rd, "rejects hydrolases\n(wrong-direction gate)"),
        ("reactive groups", rg, "seed search terms\n& substrate match"),
        ("substrates", "two free\nsmall molecules", "favours small-molecule\nligase folds"),
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
    tag(0.115, 0.726, f"ENUMERATE MECHANISMS  ·  {len(mech_names)} distinct folds "
        f"(acyl-phosphate & acyl-adenylate intermediates)", PURPLE, fs=12.5)
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
        else:
            img_at(mol_arr, cx + cw2 / 2, cy + ch2 * 0.50, 0.46)
        ax.text(cx + cw2 / 2, cy + 0.026, caption, ha="center", va="center",
                fontsize=10.5, color=INK)
        if subcaption:
            ax.text(cx + cw2 / 2, cy + 0.011, subcaption, ha="center", va="center",
                    fontsize=9.2, color=MUTED, style="italic")

    # pursued (left half) ------------------------------------------------------
    lx = 0.035
    ax.add_patch(FancyBboxPatch((lx, 0.570), 2 * cw2 + gx, 0.032, boxstyle="round,pad=0.004",
                 fc=GREEN, ec="none", zorder=3))
    ax.text(lx + cw2 + gx / 2, 0.586, "PURSUED  —  forms the target amide bond from small substrates",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=4)
    for i, (fam, substrs, amine_h, amine_l, acid_h, acid_l, subc) in enumerate(PURSUED):
        rep = next((r for r in ranked
                    if any(s in (r.get("interpro_family") or "").lower() for s in substrs)), None)
        if rep is None:
            continue
        gene, acc, rank = rep.get("gene", "?"), rep.get("accession", ""), rep.get("rank", "")
        cx = lx + (i % 2) * (cw2 + gx)
        cy = gtop - ch2 - (i // 2) * (ch2 + gy)
        arr = pair_img(sub_smiles(rows, acc, amine_h), sub_smiles(rows, acc, acid_h),
                       amine_l, acid_l)
        fam_card(cx, cy, fam, f"{gene} · #{rank}", GREEN, arr,
                 f"{amine_l} + {acid_l}", GREEN, subcaption=subc)

    # stopped (right half) -----------------------------------------------------
    rx = lx + 2 * cw2 + gx + 0.036
    ax.add_patch(FancyBboxPatch((rx, 0.570), 2 * cw2 + gx, 0.032, boxstyle="round,pad=0.004",
                 fc=RED, ec="none", zorder=3))
    ax.text(rx + cw2 + gx / 2, 0.586, "STOPPED  —  off-target native chemistry",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=4)
    for i, (fold, la, sa, lb, sb, why) in enumerate(STOPPED):
        cx = rx + (i % 2) * (cw2 + gx)
        cy = gtop - ch2 - (i // 2) * (ch2 + gy)
        if sb is None:  # second substrate is a carrier protein, not a small molecule
            fam_card(cx, cy, fold, "✗ off-target", RED, mol_img(sa, (200, 150)),
                     f"{la} + {lb}", RED, subcaption=why, glyph_label="LysW\ncarrier\nprotein")
        else:
            fam_card(cx, cy, fold, "✗ off-target", RED, pair_img(sa, sb, la, lb),
                     f"{la} + {lb}", RED, subcaption=why)

    # full-width decision-rule strip -------------------------------------------
    card(0.035, 0.162, 2 * (2 * cw2 + gx) + 0.036, 0.048, _tint(INK, 0.96), INK, lw=1.2)
    ax.text(0.5, 0.186, "decision rule:  same fold ≠ same substrate — pursue only the members "
            "whose native reaction matches the target chemistry (free amine + carboxylate)",
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

    out = FIG_DIR / "reasoning_trace.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    build(run)
