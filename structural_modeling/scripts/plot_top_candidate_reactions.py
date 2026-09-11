#!/usr/bin/env python3
"""Top structural candidates (d<4A near-attack gate, high in-line angle) with their
NATIVE reaction drawn in 2D beneath each enzyme header, and the target reaction
(adipate + HMDA) as a reference banner for comparison.

Geometry stats come from results/discovery_angle_gate.csv; native-substrate SMILES
are hand-curated from the discovery run's ranked_candidates.csv native_substrate_smiles
column (one representative acceptor chosen per enzyme, noted in the caption).

Output: figures/top_candidates_native_reactions.png
"""
import csv, os, textwrap
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import ReactionToImage
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(HERE, "results", "discovery_angle_gate.csv")
OUT  = os.path.join(HERE, "figures", "top_candidates_native_reactions.png")

# --- geometry stats ---------------------------------------------------------
geo = {}
with open(GATE) as f:
    for r in csv.DictReader(f):
        geo[r["id"]] = r

def ang(acc):   return float(geo[acc]["mean_inline_angle"]) if geo[acc]["mean_inline_angle"] else None
def npos(acc):  return int(geo[acc]["n_gated"]) if geo[acc]["n_gated"] else 0

# --- target reference -------------------------------------------------------
TARGET = dict(
    title="TARGET  ·  nylon-6,6 amide diad",
    sub="adipic acid  +  hexamethylenediamine (C6)  →  amide  (+ATP, 2 Mg²⁺)",
    donor="[O-]C(=O)CCCCC(=O)[O-]",
    acceptor="[NH3+]CCCCCC[NH3+]",
    product="[NH3+]CCCCCCNC(=O)CCCCC(=O)[O-]",
)

# --- top candidates (curated), in geometry order ----------------------------
# donor = carboxylate/acyl substrate; acceptor = amine substrate (one representative);
# product = native amide product. note = why it matters for the target.
CANDS = [
    dict(acc="Q9KZC7", gene="glnA3", fam="Glutamine synthetase", mech="acyl-P (Pγ)",
         agent="#1 strong lead",
         donor="[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]",           # L-glutamate
         acceptor="[NH3+]CCCCC[NH3+]",                          # cadaverine (C5 diamine)
         product="[NH3+]CCCCCNC(=O)CC[C@H]([NH3+])C(=O)[O-]",   # gamma-glu-cadaverine
         note="Native amine scope = linear aliphatic diamines putrescine(C4)→cadaverine(C5); "
              "HMDA(C6) is a one-CH₂ extrapolation — same FG, same charge."),
    dict(acc="P78061", gene="puuA", fam="Glutamine synthetase", mech="acyl-P (Pγ)",
         agent="#2 strong lead",
         donor="[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]",           # L-glutamate
         acceptor="[NH3+]CCCC[NH3+]",                           # putrescine (C4)
         product="[NH3+]CCCCNC(=O)CC[C@H]([NH3+])C(=O)[O-]",    # gamma-glu-putrescine
         note="C4-diamine specialist — narrower amine scope than glnA3, the C4-only comparator."),
    dict(acc="Q5UR44", gene="(GS)", fam="Glutamine synthetase", mech="acyl-P (Pγ)",
         agent="#23 worth testing",
         donor="[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]",           # L-glutamate
         acceptor="[NH4+]",                                     # ammonia
         product="NC(=O)CC[C@H]([NH3+])C(=O)[O-]",             # glutamine
         note="Canonical NH₃-accepting GS — best raw angle but native acceptor is ammonia, "
              "no diamine precedent (geometry high, substrate fit weak)."),
    dict(acc="Q76BS7", gene="iucA", fam="Aerobactin NIS", mech="acyl-adenylate (Pα)",
         agent="#7 worth testing",
         donor="O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]",          # citrate
         acceptor="CC(=O)N(O)CCCC[C@H]([NH3+])C(=O)[O-]",      # N6-acetyl-N6-OH-L-Lys
         product="CC(=O)N(O)CCCC[C@H](NC(=O)CC(O)(CC(=O)[O-])C(=O)[O-])C(=O)[O-]",
         note="NIS acyl-adenylate route: ligates a poly-carboxylate (citrate) to an amine "
              "— diacid donor precedent, but native amine is an amino-acid, not a diamine."),
    dict(acc="Q8KWT3", gene="bacD", fam="ATP-grasp", mech="acyl-P (Pγ)",
         agent="#43 (surfaced by structure)",
         donor="C[C@H]([NH3+])C(=O)[O-]",                       # L-alanine
         acceptor="[NH3+][C@@H](C[C@@H]1CCC(=O)[C@@H]2O[C@H]12)C(=O)[O-]",  # anticapsin
         product="C[C@H]([NH3+])C(=O)N[C@@H](C[C@@H]1CCC(=O)[C@@H]2O[C@H]12)C(=O)[O-]",  # bacilysin
         note="L-amino-acid dipeptide ligase — cleanest geometry in the set yet agent rank 43; "
              "a genuine amide-bond former surfaced ONLY by structure."),
    dict(acc="P0DXD6", gene="pubC", fam="Aerobactin NIS", mech="acyl-adenylate (Pα)",
         agent="#25 worth testing",
         donor="[NH3+]CCCCN(O)C(=O)CCC(=O)[O-]",               # N-(3-carboxypropanoyl)-N-OH-putrescine
         acceptor="[NH3+]CCCCN(O)C(=O)CCC(=O)[O-]",            # same monomer (homo-condensation)
         product="O=C1CCC(=O)N(O)CCCCNC(=O)CCC(=O)N(O)CCCCN1", # putrebactin (macrocycle)
         note="Highest raw angle (164°). Macrocyclizes a succinyl-diamine monomer "
              "— amide-forming on a C4-diamine backbone; specialized siderophore synthase."),
]

def rxn_img(donor, acceptor, product, size=(560, 200)):
    smarts = f"{donor}.{acceptor}>>{product}"
    rxn = AllChem.ReactionFromSmarts(smarts, useSmiles=True)
    return ReactionToImage(rxn, subImgSize=(size[0]//3, size[1]))

# --- compose ----------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(16, 19))
# each candidate row is split into a header strip + image strip so text never
# collides with the drawing; target banner on top.
gs = fig.add_gridspec(7, 2, height_ratios=[1.15, 0.18, 1, 0.18, 1, 0.18, 1],
                      hspace=0.05, wspace=0.09,
                      left=0.015, right=0.985, top=0.93, bottom=0.015)

fig.suptitle("Top discovery candidates by near-attack geometry (d<4Å gate) — native reactions",
             fontsize=20, fontweight="bold", y=0.975)
fig.text(0.5, 0.945,
         "each panel: enzyme (family · mechanism · agent rank · mean in-line angle) over its NATIVE reaction "
         "— acyl donor + amine acceptor → amide product",
         ha="center", fontsize=11.5, color="#555")

# target banner (spans both columns)
axT = fig.add_subplot(gs[0, :])
axT.imshow(rxn_img(TARGET["donor"], TARGET["acceptor"], TARGET["product"], size=(1050, 230)))
axT.set_xticks([]); axT.set_yticks([])
axT.set_title(f"{TARGET['title']}\n{TARGET['sub']}", fontsize=14, fontweight="bold",
              color="#8a1c1c", pad=8)
for s in ("top", "bottom", "left", "right"):
    axT.spines[s].set_visible(True); axT.spines[s].set_color("#8a1c1c"); axT.spines[s].set_linewidth(2.2)

def tier(a):
    if a is None: return ("inconclusive", "#999")
    if a >= 165: return ("strong ≥165°", "#1b7a5a")
    if a >= 150: return ("compatible", "#1b7a5a")
    return ("weak <150°", "#b8860b")

# candidate rows: header strip at gs rows 1/3/5(-hdr) is not enough for 6 panels,
# so lay 2 columns x 3 image-rows (gs rows 2,4,6) with header text drawn above each.
img_rows = [2, 4, 6]
positions = [(2,0),(2,1),(4,0),(4,1),(6,0),(6,1)]
for c, (rr, cc) in zip(CANDS, positions):
    ax = fig.add_subplot(gs[rr, cc])
    ax.imshow(rxn_img(c["donor"], c["acceptor"], c["product"]))
    ax.set_xticks([]); ax.set_yticks([])
    a = ang(c["acc"]); n = npos(c["acc"]); tlab, tcol = tier(a)
    astr = f"{a:.0f}°" if a is not None else "—"
    # header (bold) above the image
    ax.text(0.0, 1.20, f"{c['gene']}  ({c['acc']})   ·   {c['fam']}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")
    # stats line, colored by tier
    ax.text(0.0, 1.09,
            f"{c['mech']}  ·  agent {c['agent']}  ·  in-line {astr} ({tlab}, n={n} poses)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=tcol)
    # caption wrapped to the column width
    note = textwrap.fill(c["note"], width=74)
    ax.text(0.0, -0.06, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#333")
    for s in ("top","bottom","left","right"):
        ax.spines[s].set_visible(True); ax.spines[s].set_color("#ddd"); ax.spines[s].set_linewidth(1)

fig.savefig(OUT, dpi=145, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
