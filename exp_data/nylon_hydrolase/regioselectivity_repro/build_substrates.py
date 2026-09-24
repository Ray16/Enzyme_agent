#!/usr/bin/env python3
"""Step 1 of the regioselectivity reproduction (Dishner/Bourgery et al., JACS 2026,
148, 18285). Build the two truncated poly(X,AMA) tetrad substrates the paper docks
into Nyl10/12/50 and identify, for each, the SCISSILE amide carbonyl (the C/O that
Boltz-2 contact-restraints pin to the catalytic Thr and oxyanion Asn).

Monomers (alternating copolyamide poly(X,AMA); diacid is always adipate A):
  M = hexamethylenediamine   H2N-(CH2)6-NH2
  A = adipoyl                 -OC-(CH2)4-CO-
  X = p-xylylenediamine       H2N-CH2-C6H4(para)-CH2-NH2

Two neutrally capped (acetyl / N-methylamide) tetrads, differing in WHICH amide sits
at the nucleophilic Thr -> defines which tetrad product the cleavage would release:
  MAXA precursor: scissile bond = A-M amide  (cleavage between A and M monads)
  XAMA precursor: scissile bond = A-X amide  (cleavage between A and X monads)

Boltz-2 names a SMILES ligand's atoms <ELEMENT><CanonicalRankAtoms(AddHs(mol))+1>
(boltz/data/parse/schema.py). We reproduce that EXACTLY here so the contact restraint
in the YAML targets the right atom. MUST run under the boltz-2 env's RDKit.
"""
import json, os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "substrates"); os.makedirs(OUT, exist_ok=True)

# neutrally capped tetrads (acetyl on an amine end, N-methylamide on an acid end)
MAXA = "CC(=O)NCCCCCCNC(=O)CCCCC(=O)NCc1ccc(CNC(=O)CCCCC(=O)NC)cc1"   # M-A-X-A
XAMA = "CC(=O)NCc1ccc(CNC(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)NC)cc1"   # X-A-M-A

# SMARTS uniquely matching the scissile carbonyl [C:1](=[O:2]) in each framing.
# MAXA A-M amide: adipoyl carbonyl (bonded to a CH2 chain) whose amide N leads into
#   the hexamethylene (M) diamine -> N-(CH2)6-N.
SC_MAXA = "[CX3:1](=[OX1:2])([CH2])[NX3][CH2][CH2][CH2][CH2][CH2][CH2][NX3]"
# XAMA A-X amide: adipoyl carbonyl (bonded to a CH2 chain, NOT the acetyl CH3) whose
#   amide N leads into a benzylic CH2 of the p-xylylene (X) diamine.
SC_XAMA = "[CX3:1](=[OX1:2])([CH2])[NX3][CH2][c]"


def boltz_atom_names(smiles):
    """Return {rdkit_heavy_atom_idx: boltz_atom_name} exactly as Boltz assigns."""
    mol = Chem.MolFromSmiles(smiles)
    molH = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(molH))
    names = {}
    for atom in molH.GetAtoms():
        i = atom.GetIdx()
        names[i] = atom.GetSymbol().upper() + str(ranks[i] + 1)
    # heavy-atom idx is identical in mol and molH for the first N_heavy atoms
    return mol, molH, names


def scissile(smiles, smarts, label):
    mol, molH, names = boltz_atom_names(smiles)
    patt = Chem.MolFromSmarts(smarts)
    matches = mol.GetSubstructMatches(patt)
    assert len(matches) == 1, f"{label}: expected 1 scissile match, got {len(matches)}: {matches}"
    # map class 1 -> carbonyl C, class 2 -> carbonyl O
    cmap = {a.GetAtomMapNum(): i for a, i in
            [(patt.GetAtomWithIdx(j), matches[0][j]) for j in range(patt.GetNumAtoms())]}
    # SMARTS atom map nums are on the query; recover via GetAtomMapNum on query atoms
    c_idx = o_idx = None
    for qidx in range(patt.GetNumAtoms()):
        mn = patt.GetAtomWithIdx(qidx).GetAtomMapNum()
        if mn == 1: c_idx = matches[0][qidx]
        elif mn == 2: o_idx = matches[0][qidx]
    assert c_idx is not None and o_idx is not None
    can = Chem.MolToSmiles(mol)
    n_amide = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]")))
    info = {
        "label": label, "smiles": smiles, "canonical_smiles": can,
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(Chem.Descriptors.MolWt(mol), 1),
        "n_amide_bonds": n_amide,
        "scissile_C_idx": c_idx, "scissile_O_idx": o_idx,
        "scissile_C_name": names[c_idx], "scissile_O_name": names[o_idx],
    }
    # 2D depiction with the scissile carbonyl highlighted
    AllChem.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(900, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=[c_idx, o_idx],
        highlightAtomColors={c_idx: (1, .5, .5), o_idx: (1, .3, .3)},
        legend=f"{label}  scissile C={names[c_idx]} O={names[o_idx]}")
    d.FinishDrawing()
    open(os.path.join(OUT, f"{label}.png"), "wb").write(d.GetDrawingText())
    return info


def main():
    subs = [scissile(MAXA, SC_MAXA, "MAXA_precursor"),
            scissile(XAMA, SC_XAMA, "XAMA_precursor")]
    json.dump({s["label"]: s for s in subs},
              open(os.path.join(OUT, "substrates.json"), "w"), indent=2)
    for s in subs:
        print(f"[{s['label']}] {s['formula']} MW {s['mw']}  amides={s['n_amide_bonds']}"
              f"  scissile C={s['scissile_C_name']} O={s['scissile_O_name']}")
    print(f"\nwrote {OUT}/substrates.json + PNGs")


if __name__ == "__main__":
    main()
