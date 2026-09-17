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
PURSUED = [
    ("Glutamine-synthetase fold", ["glutamine synthetase", "glutamylputrescine"],
     "cadaverine", "L-glutamate"),
    ("ATP-grasp ligase (PylC)", ["atp-grasp", "pylc"],
     "ornithine", "lysin"),
    ("NIS / IucA-IucC synthetase", ["iuca", "aerobactin", "siderophore"],
     "spermidine", "citrate"),
    ("Asparagine synthetase B", ["asparagine"],
     "L-glutamine", "L-aspartate"),
]
# Rejected folds -> a representative OFF-TARGET native substrate (SMILES from
# PubChem) that shows why the fold's members are wrong: they act on phosphosugar
# nucleotides / isoprenoid diphosphates / carrier-tethered amino acids, not a
# free diamine + diacid.  (fold, off-target substrate label, SMILES, why)
STOPPED = [
    ("ATP-grasp (PurK/PurT)", "AIR (phosphoribonucleotide)",
     "C1=C(N(C=N1)C2C(C(C(O2)COP(=O)(O)O)O)O)N", "purine / nucleotide biosynthesis"),
    ("NIS-fold prenyltransferase", "dimethylallyl-PP",
     "CC(=CCOP(=O)(O)OP(=O)(O)O)C", "isoprenoid transfer — not amide"),
    ("ANL adenylating (Pur)", "glycinamide ribonucleotide",
     "C(C1C(C(C(O1)NC(=O)CN)O)O)OP(=O)(O)O", "glycinamide-ribonucleotide biosynthesis"),
    ("ATP-grasp (RimK-type)", "L-glutamate*",
     "C(CC(=O)O)C(C(=O)O)N", "*polymerised / LysW-carrier-tethered"),
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

    fig = plt.figure(figsize=(16.5, 15.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def card(x, y, w, h, fc, ec, lw=1.6, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                     fc=fc, ec=ec, lw=lw, zorder=z))

    def arrow(y0, y1, x=0.5):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=16, color=LINE, lw=2.0, zorder=1))

    def tag(x, y, text, color, fs=10.5):
        ax.text(x, y, text, ha="left", va="center", fontsize=fs, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.34", fc=color, ec="none"), zorder=4)

    def img_at(arr, x, y, zoom):
        ax.add_artist(AnnotationBbox(OffsetImage(arr, zoom=zoom), (x, y), frameon=False, zorder=5))

    # ---- title ----
    ax.text(0.5, 0.986, "Enzyme Discovery Agent — reasoning trace along the pipeline",
            ha="center", va="top", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.963, "one run · target: adipic acid + hexamethylenediamine (nylon-6,6 amide ligation)",
            ha="center", va="top", fontsize=10.5, color=MUTED, style="italic")

    # ---- 1. INPUT ----
    card(0.20, 0.870, 0.60, 0.080, "white", BLUE)
    tag(0.215, 0.938, "INPUT  ·  target substrates", BLUE)
    ax.text(0.35, 0.908, "diacid", ha="center", fontsize=9, color=MUTED)
    ax.text(0.65, 0.908, "diamine", ha="center", fontsize=9, color=MUTED)
    img_at(mol_img("O=C(O)CCCCC(=O)O", (230, 120)), 0.35, 0.890, 0.5)
    img_at(mol_img("NCCCCCCN", (230, 120)), 0.65, 0.890, 0.5)
    ax.text(0.5, 0.890, "+", ha="center", va="center", fontsize=17, color=INK)
    arrow(0.870, 0.842)

    # ---- 2. PARSE (4 columns, each with the why) ----
    card(0.04, 0.744, 0.92, 0.090, _tint(BLUE, 0.94), BLUE)
    tag(0.055, 0.822, "PARSE REACTION  →  contract", BLUE)
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
        card(cx, 0.752, cw, 0.056, "white", _tint(BLUE, 0.4), lw=1.2)
        ax.text(cx + cw / 2, 0.797, k, ha="center", va="center", fontsize=8.6,
                color=BLUE, fontweight="bold")
        ax.text(cx + cw / 2, 0.781, v, ha="center", va="center", fontsize=9.6,
                color=INK, fontweight="bold", linespacing=0.95)
        ax.text(cx + cw / 2, 0.762, why, ha="center", va="center", fontsize=7.6,
                color=MUTED, linespacing=0.95)
    arrow(0.744, 0.716)

    # ---- 3. ENUMERATE ----
    card(0.10, 0.626, 0.80, 0.082, _tint(PURPLE, 0.94), PURPLE)
    tag(0.115, 0.698, f"ENUMERATE MECHANISMS  ·  {len(mech_names)} distinct folds "
        f"(each a different activated intermediate)", PURPLE, fs=10)
    ax.text(0.5, 0.668, "   ·   ".join(mech_names[:4]), ha="center", va="center",
            fontsize=10, color=INK)
    ax.text(0.5, 0.646, "   ·   ".join(mech_names[4:]), ha="center", va="center",
            fontsize=10, color=INK)
    arrow(0.626, 0.598)

    # ---- 4. TRIAGE ----
    card(0.02, 0.135, 0.96, 0.470, "white", INK, lw=1.2)
    tag(0.035, 0.592, f"TRIAGE FAMILIES  →  expand vs stop   "
        f"(key decision:  {n_expand} expand · {n_stop} stop)", INK)

    cw2, ch2, gx, gy = 0.222, 0.150, 0.013, 0.013
    gtop = 0.535
    ranked = list(rows.values())  # already in rank order (CSV is rank-sorted)

    def fam_card(cx, cy, title, rep_tag, rep_color, mol_arr, caption, accent):
        card(cx, cy, cw2, ch2, "white", accent, lw=1.5)
        ax.text(cx + 0.010, cy + ch2 - 0.017, title, ha="left", va="center",
                fontsize=8.8, fontweight="bold", color=INK)
        ax.text(cx + cw2 - 0.010, cy + ch2 - 0.017, rep_tag, ha="right", va="center",
                fontsize=8.4, color=rep_color, fontweight="bold")
        img_at(mol_arr, cx + cw2 / 2, cy + ch2 * 0.44, 0.42)
        ax.text(cx + cw2 / 2, cy + 0.012, caption, ha="center", va="center",
                fontsize=7.8, color=MUTED)

    # pursued (left half) ------------------------------------------------------
    lx = 0.035
    ax.add_patch(FancyBboxPatch((lx, 0.548), 2 * cw2 + gx, 0.032, boxstyle="round,pad=0.004",
                 fc=GREEN, ec="none", zorder=3))
    ax.text(lx + cw2 + gx / 2, 0.564, "PURSUED  —  native reaction = free amine + carboxylate",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="white", zorder=4)
    for i, (fam, substrs, amine, acid) in enumerate(PURSUED):
        rep = next((r for r in ranked
                    if any(s in (r.get("interpro_family") or "").lower() for s in substrs)), None)
        if rep is None:
            continue
        gene, acc, rank = rep.get("gene", "?"), rep.get("accession", ""), rep.get("rank", "")
        cx = lx + (i % 2) * (cw2 + gx)
        cy = gtop - ch2 - (i // 2) * (ch2 + gy)
        arr = pair_img(sub_smiles(rows, acc, amine), sub_smiles(rows, acc, acid),
                       amine.split("(")[0].strip(), acid)
        fam_card(cx, cy, fam, f"{gene} · #{rank}", GREEN, arr,
                 f"{amine.split('(')[0].strip()} + {acid}", GREEN)

    # stopped (right half) -----------------------------------------------------
    rx = lx + 2 * cw2 + gx + 0.036
    ax.add_patch(FancyBboxPatch((rx, 0.548), 2 * cw2 + gx, 0.032, boxstyle="round,pad=0.004",
                 fc=RED, ec="none", zorder=3))
    ax.text(rx + cw2 + gx / 2, 0.564, "STOPPED  —  same fold, off-target substrate",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="white", zorder=4)
    for i, (fold, sublabel, smi, why) in enumerate(STOPPED):
        cx = rx + (i % 2) * (cw2 + gx)
        cy = gtop - ch2 - (i // 2) * (ch2 + gy)
        fam_card(cx, cy, fold, why.replace("*", "").strip()[:26], RED,
                 mol_img(smi, (240, 150)), sublabel, RED)

    # full-width decision-rule strip -------------------------------------------
    card(0.035, 0.146, 2 * (2 * cw2 + gx) + 0.036, 0.048, _tint(INK, 0.96), INK, lw=1.2)
    ax.text(0.5, 0.170, "decision rule:  same fold ≠ same substrate — pursue only the members "
            "whose native reaction matches the target chemistry (free amine + carboxylate)",
            ha="center", va="center", fontsize=9, color=INK)
    arrow(0.135, 0.115)

    # ---- 5. SEARCH / SCREEN ----
    n_final = complete.get("candidates", 0)
    card(0.10, 0.096, 0.80, 0.052, _tint(TEAL, 0.94), TEAL)
    tag(0.115, 0.140, "SEARCH · SCREEN · EVALUATE", TEAL)
    ax.text(0.5, 0.114, f"pursued families → resolve members + orthologs across {n_folds} folds  "
            f"→  {n_final} candidates evaluated, deduplicated to distinct enzymes",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    arrow(0.096, 0.078)

    # ---- 6. RANK -> top leads (real, from CSV) ----
    card(0.06, 0.008, 0.88, 0.066, _tint(TEAL, 0.88), TEAL, lw=1.8)
    tag(0.075, 0.066, "RANK  →  top leads (deterministic priority spine)", TEAL)
    n = max(len(leads), 1)
    cwl = 0.86 / n
    for i, r in enumerate(leads):
        cx = 0.07 + i * cwl
        tier = r.get("tier_label", "") or f"tier {r.get('tier','')}"
        fam = (r.get("interpro_family") or "").split(",")[0][:26]
        ax.text(cx + cwl / 2, 0.048, f"{i+1}. {r.get('gene','?')}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK)
        ax.text(cx + cwl / 2, 0.030, fam, ha="center", va="center", fontsize=7.4, color=MUTED)
        ax.text(cx + cwl / 2, 0.015, tier, ha="center", va="center", fontsize=8.0,
                color=TEAL, fontweight="bold")

    out = FIG_DIR / "reasoning_trace.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    build(run)
