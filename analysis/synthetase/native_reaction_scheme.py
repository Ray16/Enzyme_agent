#!/usr/bin/env python3
"""Publication-quality 2D reaction schemes for the top amide-synthetase leads.

Draws, for each of the top discovery candidates (DdaG, the gamma-glutamylputrescine
synthetases GlnA2/PuuA, and PylC) and the nylon-6,6 TARGET, the native reaction as
  donor (carboxylate)  +  acceptor (amine)  --ATP-->  amide product
with the newly formed amide bond highlighted, so a reader sees at a glance that all
four are the same carboxylate + primary-amine -> amide chemistry.

SMILES are the UniProt/Rhea-annotated substrates/products from the discovery run
(neutralised to their uncharged forms for a clean depiction); the DdaG and target
products are constructed as the amide of the annotated substrate pair. Every SMILES
is RDKit-validated before drawing.

Usage:  python analysis/synthetase/native_reaction_scheme.py [--out PNG]
"""
from __future__ import annotations
import argparse, io
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image

# Each entry: label, donor (acid), acceptor (amine), product (amide).
# Neutral forms; connectivity matches the run's UniProt/Rhea SMILES.
REACTIONS = [
    ("Target reaction",
     "OC(=O)CCCCC(=O)O",                  # adipic acid
     "NCCCCCCN",                          # hexamethylenediamine
     "OC(=O)CCCCC(=O)NCCCCCCN"),          # N-(6-aminohexyl) adipamide
    ("DdaG",
     "OC(=O)/C=C/C(=O)O",                 # fumarate
     "N[C@@H](CN)C(=O)O",                 # L-2,3-diaminopropanoate
     "OC(=O)/C=C/C(=O)NC[C@@H](N)C(=O)O"),  # N-fumaramoyl-DAP
    ("GlnA2 / PuuA",
     "N[C@@H](CCC(=O)O)C(=O)O",           # L-glutamate
     "NCCCCN",                            # putrescine
     "NCCCCNC(=O)CC[C@H](N)C(=O)O"),      # gamma-glutamylputrescine
    ("PylC",
     "C[C@H](CCN)[C@@H](N)C(=O)O",        # (3R)-3-methyl-D-ornithine
     "NCCCC[C@H](N)C(=O)O",               # L-lysine
     "C[C@H](CCN)[C@@H](N)C(=O)NCCCC[C@H](N)C(=O)O"),  # N6-(3-Me-D-ornithyl)-L-lysine
]

AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]")


def mol(sm):
    m = Chem.MolFromSmiles(sm)
    assert m is not None, f"bad SMILES: {sm}"
    AllChem.Compute2DCoords(m)
    return m


def draw(m, w, h, highlight_amide=False):
    """Render a molecule to a PIL image; optionally highlight the (formed) amide bond."""
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.bondLineWidth = 3
    o.minFontSize = 22
    o.maxFontSize = 30
    o.padding = 0.12
    o.clearBackground = True
    hb, ha = [], []
    if highlight_amide:
        matches = m.GetSubstructMatches(AMIDE)
        if matches:
            c, o_, n = matches[-1]            # last amide = the newly formed one
            ha = [c, o_, n]
            for a, b in ((c, n), (c, o_)):
                bd = m.GetBondBetweenAtoms(a, b)
                if bd is not None:
                    hb.append(bd.GetIdx())
    hlc = {i: (1.0, 0.80, 0.80) for i in ha}
    hbc = {i: (0.90, 0.20, 0.20) for i in hb}
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, m, highlightAtoms=ha, highlightBonds=hb,
        highlightAtomColors=hlc, highlightBondColors=hbc)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def main():
    ap = argparse.ArgumentParser()
    # Discovery/paper figure (not panel data): lives in the top-level figures/ dir.
    # exp_data/ is reserved for the experimental diversity panel only.
    ap.add_argument("--out", default="figures/synthetase_native_reactions.png")
    a = ap.parse_args()

    n = len(REACTIONS)
    # column x-centres (figure fraction): label, donor, +, acceptor, arrow, product
    cols = {"label": 0.075, "donor": 0.26, "plus": 0.40, "acc": 0.55, "arrow": 0.70, "prod": 0.88}
    fig_w, row_h = 13.5, 2.7
    fig = plt.figure(figsize=(fig_w, row_h * n), dpi=300)

    for i, (label, ds, acs, ps) in enumerate(REACTIONS):
        yc = 1 - (i + 0.5) / n                 # row centre in figure fraction
        # enzyme / reaction label
        fig.text(cols["label"], yc, label, ha="center", va="center",
                 fontsize=18, fontweight="bold", wrap=True)

        def put(cx, img, wfrac):
            ax = fig.add_axes([cx - wfrac / 2, yc - 0.5 / n * 0.9, wfrac, 0.9 / n])
            ax.imshow(img); ax.axis("off")

        put(cols["donor"], draw(mol(ds), 460, 360), 0.20)
        fig.text(cols["plus"], yc, "+", ha="center", va="center", fontsize=26)
        put(cols["acc"], draw(mol(acs), 460, 360), 0.20)
        # reaction arrow
        axr = fig.add_axes([cols["arrow"] - 0.06, yc - 0.02, 0.12, 0.05]); axr.axis("off")
        axr.annotate("", xy=(1, 0.5), xytext=(0, 0.5), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", lw=2.4, color="black"))
        put(cols["prod"], draw(mol(ps), 620, 360), 0.235)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
