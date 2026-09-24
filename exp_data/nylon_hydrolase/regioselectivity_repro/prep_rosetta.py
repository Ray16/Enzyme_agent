#!/usr/bin/env python3
"""Scoring Part A (palm env: rdkit + rdkit-to-params): from a GPU-minimized complex,
produce a Rosetta ligand .params + a complex PDB with the ligand on its own chain X
and atom names matching the params. Run per system.

The minimized PDB is single-chain 'A' protein (resid 1..N) + resname LIG (ligand).
We rebuild the ligand mol (bonds from the substrate SMILES) from the minimized ligand
coords, generate params with rdkit-to-params (name LIG), and emit a name-consistent
ligand PDB on chain X. interface_delta cancels the protein's internal terms, so the
concatenated-chain protein is fine.
Usage: prep_rosetta.py <system>
"""
import json, os, sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit_to_params import Params

HERE = os.path.dirname(os.path.abspath(__file__))
SUBS = json.load(open(f"{HERE}/substrates/substrates.json"))
MIN = f"{HERE}/minimized"
OUT = f"{HERE}/rosetta"; os.makedirs(OUT, exist_ok=True)


def main(system):
    frame = system.split("__", 1)[1]; smiles = SUBS[frame]["smiles"]
    minpdb = f"{MIN}/{system}.min.pdb"
    prot_lines, lig_lines = [], []
    for ln in open(minpdb):
        if ln.startswith(("ATOM", "HETATM")):
            (lig_lines if ln[17:20].strip() == "LIG" else prot_lines).append(ln)
    # ligand heavy atoms -> mol with bonds from SMILES, then Hs on min coords
    ligblock = "".join(l for l in lig_lines) + "END\n"
    open(f"{OUT}/{system}_lig_raw.pdb", "w").write(ligblock)
    raw = Chem.MolFromPDBFile(f"{OUT}/{system}_lig_raw.pdb", removeHs=True, sanitize=False)
    templ = Chem.MolFromSmiles(smiles)
    mol = AllChem.AssignBondOrdersFromTemplate(templ, raw)
    mol = Chem.AddHs(mol, addCoords=True)
    p = Params.from_mol(mol, name="LIG")
    params = f"{OUT}/{system}.params"; p.dump(params)
    ligpdb = f"{OUT}/{system}_lig.pdb"; p.dump_pdb(ligpdb)
    # reassign ligand to chain X, resname LIG, resid 1
    lig_out = []
    for ln in open(ligpdb):
        if ln.startswith(("ATOM", "HETATM")):
            ln = ln[:17] + "LIG X" + f"{1:>4}" + ln[26:]
            lig_out.append(ln)
    # protein on chain A as-is
    complexpdb = f"{OUT}/{system}_complex.pdb"
    with open(complexpdb, "w") as f:
        f.writelines(prot_lines)
        f.write("TER\n")
        f.writelines(lig_out)
        f.write("TER\nEND\n")
    natom_lig = sum(1 for l in lig_out)
    print(f"{system}: params + complex written ({len(prot_lines)} prot atoms, "
          f"{natom_lig} lig atoms) -> {complexpdb}")


if __name__ == "__main__":
    main(sys.argv[1])
