#!/usr/bin/env python3
"""Stage B/C: MM interaction energy of the geometry-selected enzyme-substrate pose.

Faithful to the paper's minimization stack (FF19SB protein + GAFF2 ligand, AM1-BCC
charges) but scores the minimized complex with an MM interaction energy (implicit
solvent GBn2) instead of Rosetta REF2015 interface_delta (PyRosetta ligand-params
tooling is unavailable here). Per system:
  1. build the ligand IN THE BOLTZ POSE with GAFF2 params (index-matched to the
     antechamber mol2; H placed from heavy-atom geometry),
  2. tleap -> complex / receptor / ligand prmtops (ff19SB + gaff2),
  3. OpenMM minimize the complex (GBn2), then single-point E(complex), E(receptor),
     E(ligand) on the minimized coords,
  4. dG_int = E_complex - E_receptor - E_ligand  (kcal/mol).

Usage: score_energy.py <system>   e.g. Nyl50__XAMA_precursor
"""
import json, os, subprocess, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
SUBS = json.load(open(f"{HERE}/substrates_verified.json"))
SEL = json.load(open(f"{HERE}/selected_models.json"))
LIGDIR = "/tmp/claude-21574/-nfs-lambda-stor-01-homes-rzhu-Enzyme-agent/831c52d9-0140-4c6d-84de-412b3946de17/scratchpad/ligparams"
WORK = f"{HERE}/energy"; os.makedirs(WORK, exist_ok=True)
TLEAP = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/fep/bin/tleap"


def pose_ligand(system):
    """Write LIG.pdb (antechamber atom names, Boltz-pose coords, H from geometry)."""
    frame = system.split("__", 1)[1]
    sub = SUBS[frame]; smi = sub["smiles"]
    pdb = SEL[system]
    # Boltz pose ligand heavy-atom coords keyed by Boltz atom name (elem+canonRank+1)
    pose = {}
    for ln in open(pdb):
        if ln.startswith("HETATM") and ln[21] == "I":
            pose[ln[12:16].strip()] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    # heavy-atom mol in canonical order; assign Boltz names, set pose coords
    mh = Chem.MolFromSmiles(smi)
    ranks = list(Chem.CanonicalRankAtoms(Chem.AddHs(mh)))   # ranks over heavy indices too
    conf = Chem.Conformer(mh.GetNumAtoms())
    for a in mh.GetAtoms():
        nm = a.GetSymbol().upper() + str(ranks[a.GetIdx()] + 1)
        assert nm in pose, f"{system}: ligand atom {nm} missing in pose"
        x = pose[nm]; conf.SetAtomPosition(a.GetIdx(), (float(x[0]), float(x[1]), float(x[2])))
    mh.AddConformer(conf)
    mp = Chem.AddHs(mh, addCoords=True)          # H placed from heavy geometry; order = heavy+H
    # antechamber mol2 atom order (must equal AddHs(MolFromSmiles) order used to build it)
    mol2 = f"{LIGDIR}/{frame}.mol2"
    names, elems = [], []
    with open(mol2) as f:
        inatom = False
        for ln in f:
            if ln.startswith("@<TRIPOS>ATOM"): inatom = True; continue
            if ln.startswith("@<TRIPOS>BOND"): break
            if inatom:
                p = ln.split(); names.append(p[1]); elems.append(p[5].split(".")[0].upper())
    assert len(names) == mp.GetNumAtoms(), f"atom count mol2 {len(names)} vs pose {mp.GetNumAtoms()}"
    pc = mp.GetConformer()
    lig_lines = []
    for i, (nm, el) in enumerate(zip(names, elems)):
        assert mp.GetAtomWithIdx(i).GetSymbol().upper() == el, \
            f"order mismatch at {i}: mol2 {el} vs pose {mp.GetAtomWithIdx(i).GetSymbol()}"
        p = pc.GetAtomPosition(i)
        lig_lines.append(f"HETATM{i+1:5d} {nm:<4s} LIG X   1    "
                         f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00          {el:>2s}")
    open(f"{WORK}/{system}_LIG.pdb", "w").write("\n".join(lig_lines) + "\nEND\n")
    return frame


def protein_pdb(system):
    """Extract the 8 protein chains (heavy atoms) to a clean PDB with TER between chains."""
    pdb = SEL[system]; out = f"{WORK}/{system}_prot.pdb"
    prev = None; L = []
    for ln in open(pdb):
        if ln.startswith("ATOM"):
            ch = ln[21]
            if prev is not None and ch != prev:
                L.append("TER")
            L.append(ln.rstrip("\n")); prev = ch
    L.append("TER"); open(out, "w").write("\n".join(L) + "\nEND\n")
    return out


def run(cmd, cwd=WORK):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def build_prmtops(system, frame):
    lj = f"{WORK}/{system}_LIG.pdb"; pr = f"{WORK}/{system}_prot.pdb"
    mol2 = f"{LIGDIR}/{frame}.mol2"; frc = f"{LIGDIR}/{frame}.frcmod"
    script = f"""source leaprc.protein.ff19SB
source leaprc.gaff2
set default PBRadii mbondi3
LIG = loadmol2 {mol2}
loadamberparams {frc}
prot = loadpdb {pr}
lig = loadpdb {lj}
com = combine {{prot lig}}
saveamberparm lig  {WORK}/{system}_lig.prmtop  {WORK}/{system}_lig.inpcrd
saveamberparm prot {WORK}/{system}_rec.prmtop  {WORK}/{system}_rec.inpcrd
saveamberparm com  {WORK}/{system}_com.prmtop  {WORK}/{system}_com.inpcrd
savepdb com {WORK}/{system}_com.pdb
quit
"""
    open(f"{WORK}/{system}_tleap.in", "w").write(script)
    rc, log = run(f"{TLEAP} -f {system}_tleap.in")
    open(f"{WORK}/{system}_tleap.log", "w").write(log)
    ok = all(os.path.exists(f"{WORK}/{system}_{x}.prmtop") for x in ("com", "rec", "lig"))
    return ok, log


def energy(system):
    import openmm as mm, openmm.app as app
    from openmm import unit
    def E(prmtop, inpcrd, coords=None):
        p = app.AmberPrmtopFile(prmtop)
        sysm = p.createSystem(implicitSolvent=app.GBn2, nonbondedMethod=app.NoCutoff,
                              constraints=None)
        integ = mm.VerletIntegrator(1.0 * unit.femtosecond)
        sim = app.Simulation(p.topology, sysm, integ,
                             mm.Platform.getPlatformByName("CPU"))
        c = coords if coords is not None else app.AmberInpcrdFile(inpcrd).positions
        sim.context.setPositions(c)
        return sim, p
    # minimize complex
    sim, ptop = E(f"{WORK}/{system}_com.prmtop", f"{WORK}/{system}_com.inpcrd")
    sim.minimizeEnergy(maxIterations=2000)
    st = sim.context.getState(getPositions=True, getEnergy=True)
    Ecom = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    minpos = st.getPositions(asNumpy=True)
    nrec = app.AmberPrmtopFile(f"{WORK}/{system}_rec.prmtop").topology.getNumAtoms()
    # single-point receptor & ligand on the minimized complex coords
    recsim, _ = E(f"{WORK}/{system}_rec.prmtop", None, coords=minpos[:nrec])
    Erec = recsim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    ligsim, _ = E(f"{WORK}/{system}_lig.prmtop", None, coords=minpos[nrec:])
    Elig = ligsim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    return Ecom, Erec, Elig, Ecom - Erec - Elig


def main():
    system = sys.argv[1]
    print(f"=== {system} ===")
    frame = pose_ligand(system)
    protein_pdb(system)
    ok, log = build_prmtops(system, frame)
    if not ok:
        print("tleap FAILED; tail:"); print("\n".join(log.splitlines()[-25:])); sys.exit(1)
    print("tleap OK; minimizing + scoring...")
    Ecom, Erec, Elig, dG = energy(system)
    print(f"E_complex={Ecom:.1f}  E_receptor={Erec:.1f}  E_ligand={Elig:.1f}  "
          f"dG_int={dG:.2f} kcal/mol")
    json.dump({"system": system, "E_complex": Ecom, "E_receptor": Erec,
               "E_ligand": Elig, "dG_int": dG},
              open(f"{WORK}/{system}_energy.json", "w"), indent=2)


if __name__ == "__main__":
    main()
