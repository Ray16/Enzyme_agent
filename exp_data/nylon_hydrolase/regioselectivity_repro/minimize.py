#!/usr/bin/env python3
"""Step (b) of the paper's protocol: classical-force-field energy minimization of the
top Boltz pose. Amber FF19SB (protein) + GAFF2 (substrate); substrate partial charges
via antechamber (AM1-BCC). Build with tleap, minimize with OpenMM. Run in the `fep`
env (antechamber+tleap+openmm+parmed all present).

Usage: minimize.py <system>   e.g. Nyl12__MAXA_precursor
Reads the selected top pose from selected_models.json, writes a minimized complex PDB.
"""
import json, os, sys, subprocess, tempfile
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = json.load(open(f"{HERE}/selected_models.json"))
SUBS = json.load(open(f"{HERE}/substrates/substrates.json"))
OUTD = f"{HERE}/minimized"; os.makedirs(OUTD, exist_ok=True)


def read_pdb_lines(pdb):
    prot, lig = [], []
    for ln in open(pdb):
        if ln.startswith(("ATOM", "TER")) and ln[21] != "I":
            prot.append(ln)
        elif ln.startswith("HETATM") and ln[21] == "I":
            lig.append(ln)
    return prot, lig


def ligand_mol2(lig_lines, smiles, work):
    """Assign bonds from the SMILES template onto the docked ligand coords, add Hs at
    a sensible geometry, write a mol2; antechamber then does GAFF2 + AM1-BCC charges."""
    # write heavy-atom ligand PDB
    ligpdb = f"{work}/lig_heavy.pdb"
    with open(ligpdb, "w") as f:
        for ln in lig_lines:
            f.write(ln)
        f.write("END\n")
    raw = Chem.MolFromPDBFile(ligpdb, removeHs=False, sanitize=False)
    templ = Chem.MolFromSmiles(smiles)
    mol = AllChem.AssignBondOrdersFromTemplate(templ, raw)
    mol = Chem.AddHs(mol, addCoords=True)                 # protons on docked heavy atoms
    ligsdf = f"{work}/lig.sdf"
    Chem.MolToMolFile(mol, ligsdf)                        # MDL molfile w/ 3D coords + bonds
    nc = Chem.GetFormalCharge(mol)
    ligmol2 = f"{work}/lig_bcc.mol2"
    r = subprocess.run(["antechamber", "-i", ligsdf, "-fi", "mdl", "-o", ligmol2,
                        "-fo", "mol2", "-c", "bcc", "-nc", str(nc), "-at", "gaff2",
                        "-rn", "LIG", "-pf", "y"], capture_output=True, text=True, cwd=work)
    if not os.path.exists(ligmol2):
        raise RuntimeError("antechamber failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    frcmod = f"{work}/lig.frcmod"
    subprocess.run(["parmchk2", "-i", ligmol2, "-f", "mol2", "-o", frcmod, "-s", "gaff2"],
                   check=True, capture_output=True, cwd=work)
    return ligmol2, frcmod


def build_and_min(system):
    pdb = SEL[system]; frame = system.split("__", 1)[1]; smiles = SUBS[frame]["smiles"]
    work = tempfile.mkdtemp(prefix=f"min_{system}_", dir=OUTD)
    prot_lines, lig_lines = read_pdb_lines(pdb)
    protpdb = f"{work}/prot.pdb"
    open(protpdb, "w").write("".join(prot_lines) + "END\n")
    ligmol2, frcmod = ligand_mol2(lig_lines, smiles, work)

    tleap = f"{work}/tleap.in"
    complex_top = f"{work}/complex.prmtop"; complex_crd = f"{work}/complex.inpcrd"
    open(tleap, "w").write(f"""\
source leaprc.protein.ff19SB
source leaprc.gaff2
lig = loadmol2 {ligmol2}
loadamberparams {frcmod}
prot = loadpdb {protpdb}
complex = combine {{prot lig}}
setBox complex vdw
saveamberparm complex {complex_top} {complex_crd}
quit
""")
    r = subprocess.run(["tleap", "-f", tleap], capture_output=True, text=True, cwd=work)
    if not os.path.exists(complex_top):
        raise RuntimeError("tleap failed:\n" + r.stdout[-3000:])

    # OpenMM minimization (implicit solvent GBN2)
    from openmm import app, unit, LangevinIntegrator, Platform
    import openmm as mm
    prm = app.AmberPrmtopFile(complex_top); crd = app.AmberInpcrdFile(complex_crd)
    system_mm = prm.createSystem(implicitSolvent=app.GBn2, nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds)
    integ = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    try:
        plat = Platform.getPlatformByName("CUDA")
    except Exception:
        plat = Platform.getPlatformByName("CPU")
    sim = app.Simulation(prm.topology, system_mm, integ, plat)
    sim.context.setPositions(crd.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy(maxIterations=5000)
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    minpdb = f"{OUTD}/{system}.min.pdb"
    with open(minpdb, "w") as f:
        app.PDBFile.writeFile(prm.topology,
                              sim.context.getState(getPositions=True).getPositions(), f)
    print(f"{system}: E {e0.value_in_unit(unit.kilocalorie_per_mole):.1f} -> "
          f"{e1.value_in_unit(unit.kilocalorie_per_mole):.1f} kcal/mol  -> {minpdb}")
    return minpdb


if __name__ == "__main__":
    build_and_min(sys.argv[1])
