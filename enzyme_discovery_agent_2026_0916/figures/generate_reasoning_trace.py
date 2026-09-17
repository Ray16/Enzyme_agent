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

# Distinct pursued mechanism families (the 8 expanded InterPro entries collapse
# into these): (family, rep gene, rep accession, amine-substrate, acid-substrate).
PURSUED = [
    ("NIS / IucA-IucC synthetase", "AsbA", "A0A0F7RJ52", "spermidine", "citrate"),
    ("Glutamine-synthetase fold", "PuuA", "P78061", "1,4-butanediammonium", "L-glutamate"),
    ("ATP-grasp ligase (PylC)", "PylC", "Q8PWY3", "3-methyl-D-ornithine", "L-lysin"),
    ("Asparagine synthetase B", "AsnB", "O24338", "L-glutamine", "L-aspartate"),
]
# Rejected folds -> their off-target native chemistry (fold : reason).
REJECTED = [
    ("ATP-grasp (PurK/PurT)", "purine / nucleotide biosynthesis"),
    ("NIS-fold prenyltransferases", "isoprenoid transfer — not amide"),
    ("ATP-grasp (RimK-type)", "LysW-carrier-tethered glutamate ligase"),
    ("ANL adenylating (Pur)", "glycinamide-ribonucleotide biosynthesis"),
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

    fig = plt.figure(figsize=(14.0, 15.4))
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
    card(0.20, 0.855, 0.60, 0.088, "white", BLUE)
    tag(0.215, 0.930, "INPUT  ·  target substrates", BLUE)
    ax.text(0.35, 0.900, "diacid", ha="center", fontsize=9, color=MUTED)
    ax.text(0.65, 0.900, "diamine", ha="center", fontsize=9, color=MUTED)
    img_at(mol_img("O=C(O)CCCCC(=O)O", (230, 120)), 0.35, 0.884, 0.5)
    img_at(mol_img("NCCCCCCN", (230, 120)), 0.65, 0.884, 0.5)
    ax.text(0.5, 0.884, "+", ha="center", va="center", fontsize=17, color=INK)
    arrow(0.855, 0.833)

    # ---- 2. PARSE (4 columns, each with the why) ----
    card(0.04, 0.735, 0.92, 0.093, _tint(BLUE, 0.94), BLUE)
    tag(0.055, 0.815, "PARSE REACTION  →  contract", BLUE)
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
        card(cx, 0.745, cw, 0.058, "white", _tint(BLUE, 0.4), lw=1.2)
        ax.text(cx + cw / 2, 0.792, k, ha="center", va="center", fontsize=8.6,
                color=BLUE, fontweight="bold")
        ax.text(cx + cw / 2, 0.775, v, ha="center", va="center", fontsize=9.6,
                color=INK, fontweight="bold", linespacing=0.95)
        ax.text(cx + cw / 2, 0.756, why, ha="center", va="center", fontsize=7.6,
                color=MUTED, linespacing=0.95)
    arrow(0.735, 0.713)

    # ---- 3. ENUMERATE ----
    card(0.10, 0.622, 0.80, 0.088, _tint(PURPLE, 0.94), PURPLE)
    tag(0.115, 0.697, f"ENUMERATE MECHANISMS  ·  {len(mech_names)} distinct folds "
        f"(each a different activated intermediate)", PURPLE, fs=10)
    ax.text(0.5, 0.663, "   ·   ".join(mech_names[:4]), ha="center", va="center",
            fontsize=10, color=INK)
    ax.text(0.5, 0.641, "   ·   ".join(mech_names[4:]), ha="center", va="center",
            fontsize=10, color=INK)
    arrow(0.622, 0.600)

    # ---- 4. TRIAGE ----
    card(0.03, 0.235, 0.94, 0.362, "white", INK, lw=1.2)
    tag(0.045, 0.585, f"TRIAGE FAMILIES  →  expand vs stop   "
        f"(key decision:  {n_expand} expand · {n_stop} stop)", INK)

    # pursued (left) -----------------------------------------------------------
    lx, lw_ = 0.045, 0.585
    ax.add_patch(FancyBboxPatch((lx, 0.525), lw_, 0.034, boxstyle="round,pad=0.004",
                 fc=GREEN, ec="none", zorder=3))
    ax.text(lx + lw_ / 2, 0.542, "PURSUED  —  native reaction condenses a free amine + a carboxylate",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)
    cw2, ch2, gx, gy = 0.285, 0.145, 0.014, 0.014
    x0, y0 = lx, 0.360
    for i, (fam, gene, acc, amine, acid) in enumerate(PURSUED):
        cx = x0 + (i % 2) * (cw2 + gx)
        cy = y0 - (i // 2) * (ch2 + gy)
        card(cx, cy, cw2, ch2, "white", GREEN, lw=1.5)
        ax.text(cx + 0.012, cy + ch2 - 0.018, fam, ha="left", va="center",
                fontsize=9.4, fontweight="bold", color=INK)
        ax.text(cx + cw2 - 0.012, cy + ch2 - 0.018, gene, ha="right", va="center",
                fontsize=8.8, color=GREEN, fontweight="bold")
        arr = pair_img(sub_smiles(rows, acc, amine), sub_smiles(rows, acc, acid),
                       amine.split("(")[0].strip(), acid)
        img_at(arr, cx + cw2 / 2, cy + ch2 * 0.42, 0.44)
    ax.text(lx + lw_ / 2, 0.247, "→ same two-substrate amide ligation as the target",
            ha="center", va="center", fontsize=8.6, color=GREEN, style="italic")

    # stopped (right) ----------------------------------------------------------
    rx, rw = 0.645, 0.315
    ax.add_patch(FancyBboxPatch((rx, 0.525), rw, 0.034, boxstyle="round,pad=0.004",
                 fc=RED, ec="none", zorder=3))
    ax.text(rx + rw / 2, 0.542, "STOPPED  —  same folds, off-target chemistry",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)
    for i, (fold, why) in enumerate(REJECTED):
        cy = 0.470 - i * 0.052
        card(rx, cy, rw, 0.044, _tint(RED, 0.95), RED, lw=1.2)
        ax.text(rx + 0.012, cy + 0.029, fold, ha="left", va="center", fontsize=8.8,
                fontweight="bold", color=INK)
        ax.text(rx + 0.012, cy + 0.013, why, ha="left", va="center", fontsize=8.0, color=MUTED)
    # decision-rule box fills the bottom-right
    card(rx, 0.247, rw, 0.058, _tint(INK, 0.95), INK, lw=1.2)
    ax.text(rx + rw / 2, 0.290, "decision rule", ha="center", va="center", fontsize=8.4,
            color=INK, fontweight="bold")
    ax.text(rx + rw / 2, 0.266, "same fold ≠ same substrate — keep only members\n"
            "whose native reaction matches the target chemistry",
            ha="center", va="center", fontsize=8.0, color=MUTED, linespacing=1.1)
    arrow(0.235, 0.213)

    # ---- 5. SEARCH / SCREEN ----
    n_final = complete.get("candidates", 0)
    card(0.10, 0.150, 0.80, 0.058, _tint(TEAL, 0.94), TEAL)
    tag(0.115, 0.197, "SEARCH · SCREEN · EVALUATE", TEAL)
    ax.text(0.5, 0.168, f"pursued families → resolve members + orthologs across {n_folds} folds  "
            f"→  {n_final} candidates evaluated, deduplicated to distinct enzymes",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    arrow(0.150, 0.128)

    # ---- 6. RANK -> top leads (real, from CSV) ----
    card(0.06, 0.020, 0.88, 0.104, _tint(TEAL, 0.88), TEAL, lw=1.8)
    tag(0.075, 0.112, "RANK  →  top leads (deterministic priority spine)", TEAL)
    n = max(len(leads), 1)
    cwl = 0.86 / n
    for i, r in enumerate(leads):
        cx = 0.07 + i * cwl
        tier = r.get("tier_label", "") or f"tier {r.get('tier','')}"
        fam = (r.get("interpro_family") or "").split(",")[0][:22]
        ax.text(cx + cwl / 2, 0.086, f"{i+1}. {r.get('gene','?')}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(cx + cwl / 2, 0.062, fam, ha="center", va="center", fontsize=7.8, color=MUTED)
        ax.text(cx + cwl / 2, 0.040, tier, ha="center", va="center", fontsize=8.2,
                color=TEAL, fontweight="bold")

    out = FIG_DIR / "reasoning_trace.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_RUN
    build(run)
