#!/usr/bin/env python3
"""
Prepare MD simulation systems from Boltz-2 co-folded structures.

Splits co-folded PDB into protein + ligands, parametrizes each component
with AmberTools (ff14SB for protein, GAFF2 for ligands), and builds a
solvated system with tleap for OpenMM simulation.

Usage:
    python prepare_md_systems.py <project_dir>
    python prepare_md_systems.py <project_dir> --candidates P31663 P0A6W9
    python prepare_md_systems.py <project_dir> --output-dir md_prep
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

try:
    from rdkit import Chem
except ImportError:
    Chem = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA_ENV = os.environ.get("BOLTZ_CONDA_ENV", "boltz-2")

# Net formal charges for common CCD ligands
CCD_CHARGES = {
    "ATP": -4,
    "ADP": -3,
    "AMP": -2,
    "GTP": -4,
    "GDP": -3,
    "GMP": -2,
    "NAD": -1,
    "NADP": -3,
    "FAD": -2,
    "COA": -4,
    "MG": 2,
    "MN": 2,
    "ZN": 2,
    "CA": 2,
    "FE": 2,
    "FE2": 2,
}

# CCD SMILES for ligands where antechamber PDB parsing fails
# These are used to assign proper bond orders via RDKit template matching
CCD_SMILES = {
    "ATP": "c1nc(c2c(n1)n(cn2)[C@@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)N",
    "ADP": "c1nc(c2c(n1)n(cn2)[C@@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])[O-])O)O)N",
    "AMP": "c1nc(c2c(n1)n(cn2)[C@@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])[O-])O)O)N",
    "GTP": "c1nc2c(n1[C@@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)NC(=O)NC2=N",
    "NAD": "c1cc(c[n+]1[C@@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)([O-])OP(=O)([O-])OC[C@@H]3[C@H]([C@H]([C@@H](O3)n4cnc5c4ncnc5N)O)O)O)O)C(=O)N",
}

# CCD codes that are metal ions (handled by AMBER ion libraries, not antechamber)
METAL_IONS = {"MG", "MN", "ZN", "CA", "FE", "FE2", "NA", "K", "CL", "CU", "CO", "NI"}

# Mapping from CCD metal codes to AMBER residue names
METAL_RESNAMES = {
    "MG": "MG",
    "MN": "MN",
    "ZN": "ZN",
    "CA": "CA",
    "FE": "FE",
    "FE2": "FE2",
}


def find_cofold_pdb(project_dir, uid):
    """Locate the Boltz-2 output PDB for a given UID."""
    pdb_path = os.path.join(
        project_dir, "cofold_output", uid,
        f"boltz_results_{uid}", "predictions", uid,
        f"{uid}_model_0.pdb"
    )
    if os.path.exists(pdb_path):
        return pdb_path
    # Fallback: search for any model PDB
    pattern = os.path.join(
        project_dir, "cofold_output", uid, "**", "*.pdb"
    )
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def parse_chain_mapping(project_dir, uid):
    """Parse cofold_input/{uid}.yaml to build chain_id -> molecule_info mapping.

    Returns dict like:
      {'A': {'type': 'protein', 'id': 'A'},
       'B': {'type': 'ligand', 'source': 'ccd', 'ccd': 'ATP', 'id': 'B'},
       'C': {'type': 'metal', 'source': 'ccd', 'ccd': 'MG', 'id': 'C'},
       'D': {'type': 'ligand', 'source': 'smiles', 'smiles': '...', 'id': 'D'},
       ...}
    """
    yaml_path = os.path.join(project_dir, "cofold_input", f"{uid}.yaml")
    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    mapping = {}
    for entry in data.get("sequences", []):
        if "protein" in entry:
            info = entry["protein"]
            mapping[info["id"]] = {"type": "protein", "id": info["id"]}
        elif "ligand" in entry:
            info = entry["ligand"]
            chain_id = info["id"]
            if "ccd" in info:
                ccd = info["ccd"].upper()
                if ccd in METAL_IONS:
                    mapping[chain_id] = {
                        "type": "metal", "source": "ccd",
                        "ccd": ccd, "id": chain_id
                    }
                else:
                    mapping[chain_id] = {
                        "type": "ligand", "source": "ccd",
                        "ccd": ccd, "id": chain_id
                    }
            elif "smiles" in info:
                mapping[chain_id] = {
                    "type": "ligand", "source": "smiles",
                    "smiles": info["smiles"], "id": chain_id
                }

    return mapping


def determine_charge(chain_info):
    """Determine net formal charge for a ligand."""
    if chain_info.get("source") == "ccd":
        ccd = chain_info["ccd"]
        if ccd in CCD_CHARGES:
            return CCD_CHARGES[ccd]
        print(f"  Warning: Unknown CCD charge for {ccd}, assuming 0")
        return 0
    elif chain_info.get("source") == "smiles":
        smiles = chain_info["smiles"]
        if Chem is not None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.GetFormalCharge(mol)
        # Manual fallback for common cases
        plus = smiles.count("+")
        minus = smiles.count("-")
        return plus - minus
    return 0


def split_pdb_by_chain(pdb_path, output_dir, chain_mapping):
    """Split PDB into separate files per chain.

    Returns dict of {chain_id: output_path}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read all lines
    with open(pdb_path) as f:
        lines = f.readlines()

    # Group ATOM/HETATM by chain, and collect CONECT records
    chain_atoms = {}  # chain_id -> list of (line, original_serial)
    conect_records = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            chain_id = line[21]
            serial = int(line[6:11].strip())
            if chain_id not in chain_atoms:
                chain_atoms[chain_id] = []
            chain_atoms[chain_id].append((line, serial))
        elif line.startswith("CONECT"):
            conect_records.append(line)

    # Parse CONECT into serial -> [bonded_serials]
    conect_map = {}
    for line in conect_records:
        parts = line[6:].split()
        if len(parts) >= 2:
            atom_serial = int(parts[0])
            bonded = [int(p) for p in parts[1:]]
            if atom_serial not in conect_map:
                conect_map[atom_serial] = []
            conect_map[atom_serial].extend(bonded)

    # Write each chain to its own file
    output_paths = {}
    for chain_id, atoms in chain_atoms.items():
        info = chain_mapping.get(chain_id, {})
        ctype = info.get("type", "unknown")

        if ctype == "protein":
            filename = "protein.pdb"
        elif ctype == "metal":
            ccd = info.get("ccd", "ION")
            filename = f"chain_{chain_id}_{ccd.lower()}.pdb"
        elif ctype == "ligand":
            if info.get("source") == "ccd":
                name = info["ccd"].lower()
            else:
                name = f"lig{chain_id.lower()}"
            filename = f"chain_{chain_id}_{name}.pdb"
        else:
            filename = f"chain_{chain_id}.pdb"

        filepath = os.path.join(output_dir, filename)

        # Build serial renumbering map
        old_to_new = {}
        for new_serial, (line, old_serial) in enumerate(atoms, start=1):
            old_to_new[old_serial] = new_serial

        # Determine residue name override for metals
        resname_override = None
        if ctype == "metal":
            resname_override = METAL_RESNAMES.get(info.get("ccd"), None)

        with open(filepath, "w") as f:
            for new_serial, (line, old_serial) in enumerate(atoms, start=1):
                # Renumber atom serial
                record = line[:6]
                rest = line[11:]

                # Override residue name for metals
                if resname_override:
                    # Columns 17-19 are residue name (right-justified in 17-19)
                    rest = rest[:6] + f"{resname_override:>3s}" + rest[9:]

                f.write(f"{record}{new_serial:5d}{rest}")

            # Write CONECT records for this chain (renumbered)
            chain_serials = set(old_to_new.keys())
            for old_serial in sorted(chain_serials):
                if old_serial in conect_map:
                    bonded_in_chain = [
                        old_to_new[b] for b in conect_map[old_serial]
                        if b in chain_serials
                    ]
                    if bonded_in_chain:
                        conect_line = f"CONECT{old_to_new[old_serial]:5d}"
                        for b in bonded_in_chain:
                            conect_line += f"{b:5d}"
                        f.write(conect_line + "\n")

            f.write("END\n")

        output_paths[chain_id] = filepath

    return output_paths


def clean_protein_pdb(protein_pdb, output_dir):
    """Run pdb4amber on protein.pdb to clean for AMBER."""
    output_pdb = os.path.join(output_dir, "protein_amber.pdb")

    cmd = [
        "pdb4amber",
        "-i", protein_pdb,
        "-o", output_pdb,
        "--dry",
        "--nohyd",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        print(f"  Warning: pdb4amber returned code {result.returncode}")
        if result.stderr:
            # pdb4amber often writes warnings to stderr that are non-fatal
            for line in result.stderr.strip().split("\n")[:5]:
                print(f"    {line}")

    if not os.path.exists(output_pdb):
        print(f"  Warning: pdb4amber did not produce output, using original PDB")
        import shutil
        shutil.copy2(protein_pdb, output_pdb)

    return output_pdb


def convert_pdb_to_sdf(ligand_pdb, smiles, output_dir, ligand_name):
    """Convert ligand PDB to SDF using RDKit template-based bond assignment.

    Uses the known SMILES to assign proper bond orders to PDB coordinates.
    This solves antechamber's "more than one unit" error for complex ligands
    like ATP where PDB bond perception fails.

    Returns SDF path or None.
    """
    if Chem is None:
        return None

    from rdkit.Chem import AllChem

    # Read PDB as molecule (bond orders may be wrong)
    pdb_mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=True, sanitize=False)
    if pdb_mol is None:
        return None

    # Create template from SMILES
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None

    try:
        # Assign bond orders from template to PDB geometry
        mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
        mol = Chem.AddHs(mol, addCoords=True)

        sdf_path = os.path.join(output_dir, f"{ligand_name}.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        return sdf_path
    except Exception as e:
        print(f"    Warning: RDKit bond assignment failed: {e}")
        return None


def parametrize_ligand(ligand_pdb, ligand_name, charge, output_dir,
                       smiles=None, ccd_code=None, timeout_seconds=600):
    """Run antechamber + parmchk2 for a ligand.

    For CCD ligands with known SMILES, converts PDB -> SDF via RDKit first
    for proper bond perception. Falls back to direct PDB input.

    Returns (mol2_path, frcmod_path) or (None, None) on failure.
    """
    mol2_path = os.path.join(output_dir, f"{ligand_name}.mol2")
    frcmod_path = os.path.join(output_dir, f"{ligand_name}.frcmod")

    # Determine input file and format for antechamber
    input_file = ligand_pdb
    input_format = "pdb"

    # For CCD ligands, try RDKit conversion to SDF for better bond perception
    template_smiles = None
    if ccd_code and ccd_code in CCD_SMILES:
        template_smiles = CCD_SMILES[ccd_code]
    elif smiles:
        template_smiles = smiles

    if template_smiles:
        sdf_path = convert_pdb_to_sdf(ligand_pdb, template_smiles, output_dir,
                                       ligand_name)
        if sdf_path:
            input_file = sdf_path
            input_format = "sdf"
            print(f"    Converted {ligand_name} to SDF via RDKit")

    # antechamber with AM1-BCC charges (proper for charged species like ATP)
    cmd_ac = [
        "antechamber",
        "-i", input_file,
        "-fi", input_format,
        "-o", mol2_path,
        "-fo", "mol2",
        "-c", "bcc",
        "-at", "gaff2",
        "-nc", str(charge),
        "-rn", ligand_name[:3].upper(),
        "-pf", "y",  # suppress intermediate files
    ]

    print(f"    Running antechamber for {ligand_name} "
          f"(charge={charge}, format={input_format})...")
    try:
        result = subprocess.run(
            cmd_ac, capture_output=True, text=True,
            timeout=120, cwd=output_dir,
        )
        if result.returncode != 0 or not os.path.exists(mol2_path):
            print(f"    ERROR: antechamber failed for {ligand_name}")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:3]:
                    print(f"      {line}")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"    ERROR: antechamber timed out for {ligand_name}")
        return None, None

    # parmchk2 for missing parameters
    cmd_pc = [
        "parmchk2",
        "-i", mol2_path,
        "-f", "mol2",
        "-o", frcmod_path,
        "-s", "gaff2",
    ]

    result = subprocess.run(
        cmd_pc, capture_output=True, text=True,
        timeout=60, cwd=output_dir,
    )

    if result.returncode != 0 or not os.path.exists(frcmod_path):
        print(f"    Warning: parmchk2 failed for {ligand_name}, "
              f"creating empty frcmod")
        with open(frcmod_path, "w") as f:
            f.write("Remark: empty frcmod\n")

    return mol2_path, frcmod_path


def build_system_tleap(uid, output_dir, protein_pdb, chain_paths,
                       chain_mapping, ligand_params):
    """Generate tleap input script and run it to build the solvated system.

    Returns (prmtop_path, inpcrd_path) or (None, None) on failure.
    """
    prmtop_path = os.path.join(output_dir, "system.prmtop")
    inpcrd_path = os.path.join(output_dir, "system.inpcrd")
    tleap_in = os.path.join(output_dir, "tleap.in")
    tleap_log = os.path.join(output_dir, "tleap.log")

    # Build tleap script
    lines = [
        "# Auto-generated tleap input for MD system preparation",
        f"# UID: {uid}",
        "",
        "# Force fields (leaprc.water.tip3p loads ion params automatically)",
        "source leaprc.protein.ff14SB",
        "source leaprc.gaff2",
        "source leaprc.water.tip3p",
        "",
        "# Load protein",
        f"protein = loadpdb {protein_pdb}",
        "",
    ]

    # Load ligand parameters and structures
    unit_names = ["protein"]
    for chain_id in sorted(chain_mapping.keys()):
        info = chain_mapping[chain_id]
        if info["type"] == "protein":
            continue

        if info["type"] == "metal":
            # Metal ions loaded from PDB with AMBER residue name
            metal_pdb = chain_paths.get(chain_id)
            if metal_pdb:
                ccd = info["ccd"]
                unit_name = ccd.lower()
                lines.append(f"# Metal ion: {ccd} (chain {chain_id})")
                lines.append(f"{unit_name} = loadpdb {metal_pdb}")
                lines.append("")
                unit_names.append(unit_name)

        elif info["type"] == "ligand":
            params = ligand_params.get(chain_id)
            if params and params[0] is not None:
                mol2_path, frcmod_path = params
                if info.get("source") == "ccd":
                    unit_name = info["ccd"].lower()
                else:
                    unit_name = f"lig{chain_id.lower()}"

                lines.append(f"# Ligand: chain {chain_id}")
                lines.append(f"loadAmberParams {frcmod_path}")
                lines.append(f"{unit_name} = loadmol2 {mol2_path}")
                lines.append("")
                unit_names.append(unit_name)

    # Combine all units
    lines.append("# Combine into complex")
    unit_list = " ".join(unit_names)
    lines.append(f"complex = combine {{ {unit_list} }}")
    lines.append("")

    # Solvate
    lines.append("# Solvate with TIP3P (12 A buffer, truncated octahedron)")
    lines.append("solvateOct complex TIP3PBOX 12.0")
    lines.append("")

    # Add counterions for neutralization, then 0.15M NaCl
    lines.append("# Neutralize and add 0.15M NaCl")
    lines.append("addIonsRand complex Na+ 0")
    lines.append("addIonsRand complex Cl- 0")
    lines.append("addIonsRand complex Na+ 40 Cl- 40")
    lines.append("")

    # Save
    lines.append("# Check and save")
    lines.append("check complex")
    lines.append(f"saveAmberParm complex {prmtop_path} {inpcrd_path}")
    lines.append("quit")
    lines.append("")

    # Write tleap input
    with open(tleap_in, "w") as f:
        f.write("\n".join(lines))

    # Run tleap
    print(f"    Running tleap...")
    result = subprocess.run(
        ["tleap", "-f", tleap_in],
        capture_output=True, text=True,
        timeout=600, cwd=output_dir,
    )

    # Save tleap log
    with open(tleap_log, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    if not os.path.exists(prmtop_path) or not os.path.exists(inpcrd_path):
        print(f"    ERROR: tleap failed to produce output files")
        # Print relevant error lines from log
        for line in result.stdout.split("\n"):
            if "ERROR" in line.upper() or "FATAL" in line.upper():
                print(f"      {line.strip()}")
        return None, None

    return prmtop_path, inpcrd_path


def validate_system(prmtop_path, inpcrd_path):
    """Validate the AMBER system files with ParmEd."""
    try:
        import parmed
        parm = parmed.load_file(prmtop_path, inpcrd_path)
        n_atoms = len(parm.atoms)
        n_residues = len(parm.residues)

        # Count component types
        protein_res = sum(1 for r in parm.residues
                         if r.name in {"ALA", "ARG", "ASN", "ASP", "CYS",
                                       "GLN", "GLU", "GLY", "HIS", "HID",
                                       "HIE", "HIP", "ILE", "LEU", "LYS",
                                       "MET", "PHE", "PRO", "SER", "THR",
                                       "TRP", "TYR", "VAL", "ACE", "NME"})
        water_res = sum(1 for r in parm.residues if r.name == "WAT")
        ion_res = sum(1 for r in parm.residues
                      if r.name in {"Na+", "Cl-", "MG", "MN", "ZN", "K+"})
        ligand_res = n_residues - protein_res - water_res - ion_res

        print(f"    System validated: {n_atoms} atoms, {n_residues} residues")
        print(f"      Protein: {protein_res} residues")
        print(f"      Ligands: {ligand_res} residues")
        print(f"      Water: {water_res} molecules")
        print(f"      Ions: {ion_res}")

        return {
            "n_atoms": n_atoms,
            "n_residues": n_residues,
            "protein_residues": protein_res,
            "water_molecules": water_res,
            "ions": ion_res,
            "ligand_residues": ligand_res,
        }
    except Exception as e:
        print(f"    Warning: ParmEd validation failed: {e}")
        return None


def prepare_one_system(uid, project_dir, output_base_dir, param_cache=None):
    """End-to-end preparation for a single UID.

    param_cache: dict mapping CCD code or SMILES -> (mol2_path, frcmod_path)
        for reusing ligand parameters across candidates (e.g., ATP computed once).
    """
    if param_cache is None:
        param_cache = {}
    print(f"\n  Preparing {uid}...", flush=True)
    status = {"uid": uid, "success": False, "steps": {}}
    start_time = time.time()

    # 1. Find cofold PDB
    pdb_path = find_cofold_pdb(project_dir, uid)
    if not pdb_path:
        print(f"    ERROR: No cofold PDB found for {uid}")
        status["error"] = "No cofold PDB found"
        return status

    # 2. Parse chain mapping
    chain_mapping = parse_chain_mapping(project_dir, uid)
    if not chain_mapping:
        print(f"    ERROR: No cofold input YAML found for {uid}")
        status["error"] = "No cofold input YAML"
        return status

    print(f"    PDB: {os.path.basename(pdb_path)}")
    chain_desc = ", ".join(f"{k}={v.get('type','?')}" for k, v in sorted(chain_mapping.items()))
    print(f"    Chains: {chain_desc}")

    # 3. Create output directory
    output_dir = os.path.join(output_base_dir, uid)
    os.makedirs(output_dir, exist_ok=True)

    # 4. Split PDB by chain
    print(f"    Splitting PDB by chain...")
    chain_paths = split_pdb_by_chain(pdb_path, output_dir, chain_mapping)
    status["steps"]["split"] = True

    # 5. Clean protein
    protein_path = chain_paths.get("A")
    if not protein_path:
        print(f"    ERROR: No protein chain (A) found")
        status["error"] = "No protein chain"
        return status

    print(f"    Cleaning protein with pdb4amber...")
    protein_amber = clean_protein_pdb(protein_path, output_dir)
    status["steps"]["pdb4amber"] = True

    # 6. Parametrize ligands (with caching for identical molecules like ATP)
    ligand_params = {}
    for chain_id in sorted(chain_mapping.keys()):
        info = chain_mapping[chain_id]
        if info["type"] == "protein":
            continue
        if info["type"] == "metal":
            # Metals don't need antechamber, handled by AMBER ion libs
            continue

        ligand_pdb = chain_paths.get(chain_id)
        if not ligand_pdb:
            print(f"    Warning: No PDB for chain {chain_id}")
            continue

        charge = determine_charge(info)
        if info.get("source") == "ccd":
            lig_name = f"chain_{chain_id}_{info['ccd'].lower()}"
            cache_key = info["ccd"]
        else:
            lig_name = f"chain_{chain_id}_lig{chain_id.lower()}"
            cache_key = info.get("smiles", lig_name)

        # Check cache for previously computed parameters
        if cache_key in param_cache:
            cached_mol2, cached_frcmod = param_cache[cache_key]
            # Copy cached files to this candidate's directory
            mol2 = os.path.join(output_dir, f"{lig_name}.mol2")
            frcmod = os.path.join(output_dir, f"{lig_name}.frcmod")
            shutil.copy2(cached_mol2, mol2)
            shutil.copy2(cached_frcmod, frcmod)
            print(f"    Reusing cached parameters for {cache_key}")
        else:
            mol2, frcmod = parametrize_ligand(
                ligand_pdb, lig_name, charge, output_dir,
                smiles=info.get("smiles"),
                ccd_code=info.get("ccd"),
            )
            # Store in cache for subsequent candidates
            if mol2 is not None:
                param_cache[cache_key] = (mol2, frcmod)

        ligand_params[chain_id] = (mol2, frcmod)
        status["steps"][f"parametrize_{chain_id}"] = mol2 is not None

    # Check that at least the substrates are parametrized
    failed_ligands = [k for k, v in ligand_params.items() if v[0] is None]
    if failed_ligands:
        print(f"    Warning: Failed to parametrize chains: {failed_ligands}")

    # 7. Build system with tleap
    prmtop, inpcrd = build_system_tleap(
        uid, output_dir, protein_amber,
        chain_paths, chain_mapping, ligand_params,
    )

    if prmtop is None:
        status["error"] = "tleap failed"
        return status

    status["steps"]["tleap"] = True

    # 8. Validate
    validation = validate_system(prmtop, inpcrd)
    if validation:
        status["validation"] = validation

    status["success"] = True
    status["prmtop"] = prmtop
    status["inpcrd"] = inpcrd
    status["time_seconds"] = round(time.time() - start_time, 1)

    # Save status
    status_path = os.path.join(output_dir, "prep_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    print(f"    Done ({status['time_seconds']}s)")
    return status


def collect_candidates(project_dir, candidate_filter=None):
    """Collect UIDs that have cofold output."""
    cofold_dir = os.path.join(project_dir, "cofold_output")
    if not os.path.isdir(cofold_dir):
        return []

    uids = []
    for entry in sorted(os.listdir(cofold_dir)):
        entry_path = os.path.join(cofold_dir, entry)
        if os.path.isdir(entry_path):
            # Check for PDB output
            if find_cofold_pdb(project_dir, entry):
                uids.append(entry)

    if candidate_filter:
        uids = [u for u in uids if u in candidate_filter]

    return uids


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MD systems from co-folded structures")
    parser.add_argument("project_dir",
                        help="Project directory with cofold output")
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Specific UIDs to process (default: all)")
    parser.add_argument("-o", "--output-dir", default="md_prep",
                        help="Output directory name (default: md_prep)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for system prep (default: 4)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    output_base = os.path.join(project_dir, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Phase 6a: Prepare MD Systems")
    print(f"{'='*60}\n", flush=True)

    print(f"Project: {project_dir}")
    print(f"Output:  {output_base}")

    # Collect candidates
    uids = collect_candidates(project_dir, args.candidates)
    if not uids:
        sys.exit("ERROR: No candidates found with cofold output")

    print(f"Candidates: {len(uids)} ({', '.join(uids)})\n")

    # Process each candidate
    os.makedirs(output_base, exist_ok=True)
    results = []

    # Separate into skipped vs to-do
    todo_uids = []
    for uid in uids:
        prmtop = os.path.join(output_base, uid, "system.prmtop")
        if os.path.exists(prmtop):
            print(f"\n  {uid}: SKIPPED (system.prmtop exists)")
            results.append({"uid": uid, "success": True, "skipped": True})
        else:
            todo_uids.append(uid)

    if todo_uids:
        # Run first candidate sequentially to populate param cache
        param_cache = {}
        first_uid = todo_uids[0]
        print(f"\n  Running {first_uid} first (populates parameter cache)...")
        result = prepare_one_system(first_uid, project_dir, output_base,
                                    param_cache=param_cache)
        results.append(result)

        # Save cached params to disk so parallel workers can load them
        cache_dir = os.path.join(output_base, "_param_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_index = {}
        for key, (mol2_path, frcmod_path) in param_cache.items():
            safe_key = key.replace("/", "_").replace("\\", "_")
            cached_mol2 = os.path.join(cache_dir, f"{safe_key}.mol2")
            cached_frcmod = os.path.join(cache_dir, f"{safe_key}.frcmod")
            shutil.copy2(mol2_path, cached_mol2)
            shutil.copy2(frcmod_path, cached_frcmod)
            cache_index[key] = (cached_mol2, cached_frcmod)

        # Run remaining candidates in parallel
        remaining = todo_uids[1:]
        if remaining:
            n_workers = min(args.workers, len(remaining))
            print(f"\n  Running {len(remaining)} candidates "
                  f"with {n_workers} parallel workers...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        prepare_one_system, uid, project_dir,
                        output_base, param_cache=dict(cache_index),
                    ): uid
                    for uid in remaining
                }
                for future in as_completed(futures):
                    uid = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"  {uid}: FAILED ({e})")
                        results.append({"uid": uid, "success": False,
                                        "error": str(e)})

    # Summary
    succeeded = sum(1 for r in results if r.get("success"))
    failed = sum(1 for r in results if not r.get("success"))
    skipped = sum(1 for r in results if r.get("skipped"))

    print(f"\n{'='*60}")
    print(f"  MD System Preparation Complete")
    print(f"{'='*60}")
    print(f"  Succeeded: {succeeded} ({skipped} skipped)")
    if failed:
        print(f"  Failed:    {failed}")
        for r in results:
            if not r.get("success"):
                print(f"    {r['uid']}: {r.get('error', 'unknown error')}")
    print(f"  Output:    {output_base}\n")


if __name__ == "__main__":
    main()
