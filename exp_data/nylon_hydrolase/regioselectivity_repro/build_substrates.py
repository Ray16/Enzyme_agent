#!/usr/bin/env python3
"""Step 1 (CORRECTED): build the two truncated poly(X,AMA) tetrad substrates for the
regioselectivity reproduction (Dishner/Bourgery et al., JACS 2026, 148, 18285).

Reasoning (see README + chat): poly(X,AMA) is a strict alternation …X-A-M-A-X-A-M-A…
(A = adipoyl diacid; every A is flanked by one X and one M). The enzyme's Thr attacks
an adipoyl carbonyl; the diamine is the leaving group. So:
  * cut the A-M amide  -> releases M-A-X-A = MAXA
  * cut the A-X amide  -> releases X-A-M-A = XAMA
The modeled substrate is a neutrally-capped TETRAD (4 monomer units, 3 internal
amides) threaded so the SCISSILE amide is the CENTRAL bond, giving ~2 units on each
side. The regioselectivity discriminant is where the aromatic X ring sits:
  MAXA precursor:  tetrad X-A-M-A, scissile = central A-M  -> leaving-adjacent M, X on acyl side
  XAMA precursor:  tetrad M-A-X-A, scissile = central A-X  -> leaving-adjacent X (S1')
The scissile carbonyl is atom-mapped [C:1](=[O:2]) so its identity is unambiguous;
maps are stripped before Boltz sees the SMILES. Boltz names ligand atoms
<ELEM><CanonicalRankAtoms(AddHs)+1> (canonical -> invariant to atom order), so the
names computed here match Boltz's. Run under the boltz-2 env's RDKit.
"""
import json, os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "substrates"); os.makedirs(OUT, exist_ok=True)

# central scissile carbonyl marked [C:1](=[O:2])
# MAXA precursor = tetrad X-A-M-A, scissile = central A-M amide (carbonyl -> M's N)
MAXA = "CC(=O)NCc1ccc(CNC(=O)CCCC[C:1](=[O:2])NCCCCCCNC(=O)CCCCC(=O)NC)cc1"
# XAMA precursor = tetrad M-A-X-A, scissile = central A-X amide (carbonyl -> X's benzylic N)
XAMA = "CC(=O)NCCCCCCNC(=O)CCCC[C:1](=[O:2])NCc1ccc(CNC(=O)CCCCC(=O)NC)cc1"


def build(smiles_mapped, label, expect_leaving):
    mol = Chem.MolFromSmiles(smiles_mapped)
    assert mol is not None, f"{label}: bad SMILES"
    c_idx = o_idx = None
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == 1: c_idx = a.GetIdx()
        elif a.GetAtomMapNum() == 2: o_idx = a.GetIdx()
    assert c_idx is not None and o_idx is not None, f"{label}: scissile atoms not marked"
    # sanity: the scissile amide's leaving N-side neighbour (aromatic->X, aliphatic->M)
    cat = mol.GetAtomWithIdx(c_idx)
    amideN = [nb for nb in cat.GetNeighbors() if nb.GetSymbol() == "N"]
    assert len(amideN) == 1, f"{label}: scissile C must have exactly one amide N"
    nN = amideN[0]
    # leaving-group side: the non-carbonyl C neighbour of the amide N; benzylic (bonded
    # to an aromatic C) -> X (p-xylylene), purely aliphatic -> M (hexamethylene)
    ncar = [nb for nb in nN.GetNeighbors() if nb.GetSymbol() == "C" and nb.GetIdx() != c_idx][0]
    leaving = "X" if any(nb.GetIsAromatic() for nb in ncar.GetNeighbors()) else "M"
    assert leaving == expect_leaving, f"{label}: leaving-adjacent is {leaving}, expected {expect_leaving}"

    # strip atom maps -> plain SMILES for Boltz; scissile atom indices are unchanged
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    plain = Chem.MolToSmiles(mol)
    # Boltz atom names = <ELEM><CanonicalRankAtoms(AddHs)+1> (canonical -> order-invariant)
    ranks = list(Chem.CanonicalRankAtoms(Chem.AddHs(mol)))
    c_name = "C" + str(ranks[c_idx] + 1)
    o_name = "O" + str(ranks[o_idx] + 1)

    info = {
        "label": label, "smiles": plain,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(Descriptors.MolWt(mol), 1),
        "n_amide_bonds": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))),
        "scissile_C_idx": c_idx, "scissile_O_idx": o_idx,
        "scissile_C_name": c_name, "scissile_O_name": o_name,
        "leaving_adjacent": leaving,
    }
    # 2D depiction, scissile carbonyl highlighted
    AllChem.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(950, 520)
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=[c_idx, o_idx],
        highlightAtomColors={c_idx: (1, .5, .5), o_idx: (1, .3, .3)},
        legend=f"{label}  scissile(central)=C {c_name}/O {o_name}  leaving-adjacent={leaving}")
    d.FinishDrawing()
    open(os.path.join(OUT, f"{label}.png"), "wb").write(d.GetDrawingText())
    return info


def main():
    subs = [build(MAXA, "MAXA_precursor", "M"),
            build(XAMA, "XAMA_precursor", "X")]
    json.dump({s["label"]: s for s in subs},
              open(os.path.join(OUT, "substrates.json"), "w"), indent=2)
    for s in subs:
        print(f"[{s['label']}] {s['formula']} MW {s['mw']}  amides={s['n_amide_bonds']}"
              f"  scissile C={s['scissile_C_name']} O={s['scissile_O_name']}"
              f"  leaving-adjacent={s['leaving_adjacent']}")
    print(f"\nwrote {OUT}/substrates.json + PNGs")


if __name__ == "__main__":
    main()
