#!/usr/bin/env python3
"""
Pocket Analysis Pipeline
========================
Identifies binding pockets in enzyme structures using evidence-based methods:

1. Foldseek — find closest PDB structural homolog
2. Binding site extraction from homolog (UniProt annotations or PDB ligand contacts)
3. Structural alignment — map homolog binding residues to query
4. pyKVFinder — detect cavities
5. Pocket selection — pick cavity containing mapped active site residues

No composite scores. No motif heuristics. No composition-based ranking.

Usage:
    python pocket_analysis.py <project_dir>
    python pocket_analysis.py <project_dir> --skip-foldseek   # reuse existing foldseek results
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import yaml

try:
    import pyKVFinder
    HAS_PYKVFINDER = True
except ImportError:
    HAS_PYKVFINDER = False

try:
    import prody
    prody.confProDy(verbosity="none")
    HAS_PRODY = True
except ImportError:
    HAS_PRODY = False

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

FOLDSEEK = os.environ.get("FOLDSEEK_BIN", os.path.expanduser("~/tools/foldseek/bin/foldseek"))
FOLDSEEK_DB = os.environ.get("FOLDSEEK_DB", os.path.expanduser("~/tools/foldseek_db/pdb"))

# Known crystallization artifacts, buffer components, and cryoprotectants.
# These are never biologically relevant and should always be filtered out
# from AlphaFill transplanted ligands.
CRYSTALLIZATION_ARTIFACTS = {
    # Cryoprotectants
    "GOL", "EDO", "MPD", "PEG", "PGE", "P6G", "1PE", "2PE",
    # Buffer salts and ions
    "SO4", "PO4", "NO3", "CO3", "ACT", "FMT", "CIT", "TRS",
    # Buffer components
    "EPE", "MES", "HEP",
    # Solvents
    "MOH", "EOH", "DMS", "IMD",
    # Reducing agents
    "BME", "DTT",
    # Non-specific counter-ions (not catalytic in enzyme active sites)
    "NA", "K", "CL",
    # Unknown/unmodeled atoms
    "UNX", "UNL",
}

AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def fetch_url(url, timeout=20, retries=3):
    """GET request with exponential backoff for transient errors. Returns text or None."""
    import time as _time
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PocketAnalysis/1.0")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {e.code}"
                _time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # Retry on timeouts, connection resets, and network errors
            last_err = str(e)
            _time.sleep(2 ** attempt)
            continue
        except Exception:
            return None
    if last_err:
        print(f"  WARNING: {url} failed after {retries} retries: {last_err}", flush=True)
    return None


def fetch_uniprot_annotation(uniprot_id):
    """Fetch active sites, binding sites, EC, domains from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    raw = fetch_url(url)
    if not raw:
        return {"uniprot_id": uniprot_id, "source": "not_found"}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"uniprot_id": uniprot_id, "source": "parse_error"}

    info = {"uniprot_id": uniprot_id, "source": "uniprotkb"}

    # Protein name
    prot_desc = data.get("proteinDescription", {})
    rec_name = prot_desc.get("recommendedName", {})
    if rec_name:
        info["protein_name"] = rec_name.get("fullName", {}).get("value", "")
    else:
        sub_names = prot_desc.get("submissionNames", [])
        if sub_names:
            info["protein_name"] = sub_names[0].get("fullName", {}).get("value", "")

    # EC number
    ec_numbers = []
    for name_block in [rec_name] + prot_desc.get("alternativeNames", []):
        for ec in name_block.get("ecNumbers", []):
            ec_numbers.append(ec.get("value", ""))
    info["ec_numbers"] = ec_numbers

    # Active site features
    active_sites = []
    for feat in data.get("features", []):
        if feat.get("type") == "Active site":
            pos = feat.get("location", {}).get("start", {}).get("value")
            desc = feat.get("description", "")
            if pos:
                active_sites.append({"position": pos, "description": desc})
    info["active_sites"] = active_sites

    # Binding sites
    binding_sites = []
    for feat in data.get("features", []):
        if feat.get("type") == "Binding site":
            pos = feat.get("location", {}).get("start", {}).get("value")
            desc = feat.get("description", "")
            ligand = feat.get("ligand", {}).get("name", "")
            if pos:
                binding_sites.append({"position": pos, "description": desc, "ligand": ligand})
    info["binding_sites"] = binding_sites

    # Organism
    info["organism"] = data.get("organism", {}).get("scientificName", "")

    return info


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: COLLECT STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

def collect_structures(project_dir):
    """Find all structure files from a project directory."""
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: {config_path} not found.")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    structures = []

    # Reference structure
    ref = config.get("reference", {})
    ref_path = os.path.join(project_dir, ref.get("structure", ""))
    if os.path.exists(ref_path):
        structures.append({
            "uid": ref["uniprot_id"],
            "path": ref_path,
            "source": "alphafold",
            "role": "reference",
        })

    # AlphaFold structures
    af_dir = os.path.join(project_dir, "alphafold_structures")
    if os.path.isdir(af_dir):
        for f in sorted(glob.glob(os.path.join(af_dir, "*.pdb"))):
            uid = os.path.basename(f).replace("AF-", "").replace("-F1-model_v4.pdb", "")
            structures.append({"uid": uid, "path": f, "source": "alphafold", "role": "candidate"})

    # PDB structures — use UniProt ID as key when available
    # (PDB IDs cause key mismatches with cofolding, which uses UniProt IDs)
    pdb_dir = os.path.join(project_dir, "pdb_structures")
    if os.path.isdir(pdb_dir):
        # Build PDB-to-UniProt mapping from structure_sequence_summary.csv
        pdb_to_uniprot = {}
        summary_csv = os.path.join(project_dir, "structure_sequence_summary.csv")
        if os.path.exists(summary_csv):
            import csv as _csv
            with open(summary_csv) as _f:
                for row in _csv.DictReader(_f):
                    pdb_ids_field = row.get("pdb_ids", "")
                    pdb_files_field = row.get("pdb_files", "")
                    uid = row.get("uniprot_id", "")
                    if pdb_ids_field and pdb_ids_field != "none" and uid:
                        for pid in pdb_ids_field.split(","):
                            pdb_to_uniprot[pid.strip()] = uid

        for f in sorted(glob.glob(os.path.join(pdb_dir, "*.pdb"))):
            pdb_id = os.path.splitext(os.path.basename(f))[0]
            uid = pdb_to_uniprot.get(pdb_id, pdb_id)  # fall back to PDB ID if no mapping
            structures.append({"uid": uid, "path": f, "source": "pdb", "role": "candidate"})

    # Boltz structures
    boltz_dir = os.path.join(project_dir, "boltz_output")
    if os.path.isdir(boltz_dir):
        for f in sorted(glob.glob(os.path.join(boltz_dir, "**/*.pdb"), recursive=True)):
            parts = f.split(os.sep)
            boltz_idx = [i for i, p in enumerate(parts) if p == "boltz_output"]
            uid = parts[boltz_idx[0] + 1] if boltz_idx else os.path.splitext(os.path.basename(f))[0]
            structures.append({"uid": uid, "path": f, "source": "boltz", "role": "candidate"})

    return structures, config


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: FOLDSEEK BATCH SEARCH
# ═══════════════════════════════════════════════════════════════════════

def run_foldseek(structures, output_tsv):
    """Run foldseek easy-search for all structures against PDB.
    Returns dict: uid -> list of hits [{pdb_id, target, tmscore, fident}, ...]
    """
    if not os.path.exists(FOLDSEEK) or not os.path.exists(FOLDSEEK_DB):
        print("  WARNING: Foldseek not available. Skipping structural homology search.", flush=True)
        return {}

    # Create temp directory with symlinks to all structures
    batch_dir = tempfile.mkdtemp(prefix="foldseek_input_")
    for s in structures:
        dst = os.path.join(batch_dir, f"{s['uid']}.pdb")
        if not os.path.exists(dst):
            os.symlink(s["path"], dst)

    foldseek_tmp = tempfile.mkdtemp(prefix="foldseek_tmp_")
    print(f"  Searching {len(structures)} structures against PDB ...", flush=True)

    try:
        ret = subprocess.run(
            [FOLDSEEK, "easy-search", batch_dir, FOLDSEEK_DB, output_tsv, foldseek_tmp,
             "--format-output", "query,target,fident,alnlen,evalue,bits,alntmscore,qtmscore,ttmscore,lddt",
             "-e", "0.01", "--max-accept", "3"],
            capture_output=True, text=True, timeout=600,
        )
        if ret.returncode != 0:
            print(f"  WARNING: Foldseek failed: {ret.stderr[:200]}", flush=True)
            return {}
    except subprocess.TimeoutExpired:
        print("  WARNING: Foldseek timed out", flush=True)
        return {}
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)
        shutil.rmtree(foldseek_tmp, ignore_errors=True)

    return parse_foldseek_results(output_tsv)


def parse_foldseek_results(tsv_path):
    """Parse foldseek output TSV into dict: uid -> list of hits."""
    hits = {}
    if not os.path.exists(tsv_path):
        return hits
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            query = parts[0].replace(".pdb", "")
            target = parts[1]
            pdb_id = target.split("-")[0].upper()
            tmscore = float(parts[6])
            fident = float(parts[2])
            # Extract chain from foldseek target (e.g., "3axg-assembly4_M" -> "M")
            target_chain = "A"
            if "_" in target:
                chain_part = target.split("_")[-1]
                # Chain can be like "A", "B", "A-2" (for bio assemblies)
                target_chain = chain_part.split("-")[0] if chain_part else "A"
            hits.setdefault(query, []).append({
                "pdb_id": pdb_id,
                "target": target,
                "target_chain": target_chain,
                "tmscore": tmscore,
                "fident": fident,
            })
    return hits


# ═══════════════════════════════════════════════════════════════════════
# STEP 3a: BINDING SITE EXTRACTION (AlphaFill primary)
# ═══════════════════════════════════════════════════════════════════════

_alphafill_cache = {}

def fetch_alphafill(uniprot_id):
    """Query AlphaFill API for transplanted ligand contacts.

    Returns dict: {compound_id: [contact_residue_numbers]} or None if not available.
    AlphaFill transplants ligands from PDB homologs into AlphaFold structures
    and reports which protein residues are in contact with each ligand.
    """
    if uniprot_id in _alphafill_cache:
        return _alphafill_cache[uniprot_id]

    url = f"https://alphafill.eu/v1/aff/{uniprot_id}/json"
    raw = fetch_url(url, timeout=15)
    if not raw:
        _alphafill_cache[uniprot_id] = None
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _alphafill_cache[uniprot_id] = None
        return None

    hits = data.get("hits")
    if not hits:
        _alphafill_cache[uniprot_id] = None
        return None

    # Extract contacts per ligand, keeping the best (most contacts) for each type
    ligand_contacts = {}
    ligand_details = {}
    for hit in hits:
        pdb_id = hit.get("pdb_id", "?")
        global_rmsd = hit.get("global_rmsd", 999)
        identity = hit.get("alignment", {}).get("identity", 0)

        for t in (hit.get("transplants") or []):
            compound = t.get("compound_id", "?")
            local_rmsd = t.get("local_rmsd", 999)
            contacts = set()
            for c in (t.get("clash", {}).get("distances") or []):
                pa = c.get("poly_atom")
                if pa and pa.get("seq_id") is not None:
                    contacts.add(pa["seq_id"])

            if contacts:
                # Keep the transplant with the most contacts for each compound
                if compound not in ligand_contacts or len(contacts) > len(ligand_contacts[compound]):
                    ligand_contacts[compound] = sorted(contacts)
                    ligand_details[compound] = {
                        "pdb_id": pdb_id,
                        "global_rmsd": round(global_rmsd, 2),
                        "local_rmsd": round(local_rmsd, 2),
                        "identity": round(identity, 3),
                        "n_contacts": len(contacts),
                    }

    result = {"contacts": ligand_contacts, "details": ligand_details} if ligand_contacts else None
    _alphafill_cache[uniprot_id] = result
    return result


def extract_pdb_cofactor_contacts(pdb_id, chain, relevant_ligands, cofactor_map=None):
    """Extract cofactor contact residues directly from a PDB structure.

    Used as a fallback when AlphaFill doesn't cover the homolog's UniProt ID.
    Downloads the PDB, finds ligands matching relevant_ligands (or their analogs
    via cofactor_map), and computes protein residues within 6A of those ligands.

    Returns dict: {compound_id: {"contacts": [residues], "pdb_id": pdb_id}} or None.
    """
    if cofactor_map is None:
        cofactor_map = {}

    # Build set of CCD codes to look for (relevant_ligands + their analogs)
    target_ccds = set(relevant_ligands or [])
    for src, dst in cofactor_map.items():
        if dst in target_ccds:
            target_ccds.add(src)

    # Query RCSB for non-polymer components in this PDB entry
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    raw = fetch_url(entry_url, timeout=10)
    if not raw:
        return None
    try:
        entry_data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    pdb_ligands = entry_data.get("rcsb_entry_info", {}).get("nonpolymer_bound_components", [])
    matching = [lig for lig in pdb_ligands if lig in target_ccds]
    if not matching:
        return None

    # Download PDB and compute contacts
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_content = fetch_url(pdb_url, timeout=30)
    if not pdb_content:
        return None

    tmp_pdb = os.path.join(tempfile.gettempdir(), f"{pdb_id}_cofactor.pdb")
    try:
        with open(tmp_pdb, "w") as f:
            f.write(pdb_content)

        struct = prody.parsePDB(tmp_pdb)
        if struct is None:
            return None

        result = {}
        for lig_ccd in matching:
            # Select ligand atoms
            lig_sel = struct.select(f"resname {lig_ccd}")
            if lig_sel is None:
                continue

            # Select protein atoms in the target chain within 6A of ligand
            protein_sel = struct.select(f"protein and chain {chain}")
            if protein_sel is None:
                # Try without chain filter
                protein_sel = struct.select("protein")
            if protein_sel is None:
                continue

            # Compute contacts: protein residues within 6A of any ligand atom
            contacts = set()
            lig_coords = lig_sel.getCoords()
            for atom in protein_sel:
                atom_coord = atom.getCoords().reshape(1, -1)
                dists = ((lig_coords - atom_coord) ** 2).sum(axis=1) ** 0.5
                if dists.min() <= 6.0:
                    contacts.add(int(atom.getResnum()))

            if contacts:
                result[lig_ccd] = {
                    "contacts": sorted(contacts),
                    "pdb_id": pdb_id,
                    "n_contacts": len(contacts),
                    "source": "pdb_direct",
                }

        return result if result else None
    except Exception as e:
        print(f"    PDB cofactor extraction error for {pdb_id}: {e}", flush=True)
        return None
    finally:
        if os.path.exists(tmp_pdb):
            os.remove(tmp_pdb)


# Metal ions recognized for active-site transfer
_METAL_IONS = {"ZN", "MG", "MN", "FE", "CO", "NI", "CU", "CA", "CD", "HG"}

# Amino acids that typically coordinate metals
_METAL_COORD_AA = {"HIS", "GLU", "ASP", "CYS", "SER", "THR"}

# Functional sidechain atoms used for motif searching
_COORD_ATOMS = {
    "HIS": ["NE2", "ND1"], "GLU": ["OE1", "OE2"], "ASP": ["OD1", "OD2"],
    "CYS": ["SG"], "SER": ["OG"], "THR": ["OG1"],
}

# All amino acids considered catalytically relevant
_CATALYTIC_AA = {"HIS", "GLU", "ASP", "CYS", "SER", "THR", "LYS", "TYR"}


def _extract_nonpolymer_info(pdb_path, chain_id="A", contact_radius=5.0):
    """Extract non-polymer entities and their protein contacts from a PDB chain.

    Returns dict with 'metals', 'ligands', 'coord_aa_pattern', or None.
    Each entry has 'name', 'atoms' (np.array), 'contacts' [(resid, resname)].
    coord_aa_pattern is the set of AA types coordinating metals.
    """
    from Bio.PDB import PDBParser as _BioParser
    _parser = _BioParser(QUIET=True)
    try:
        structure = _parser.get_structure("ref", pdb_path)
    except Exception:
        return None
    model = structure[0]
    if chain_id not in model:
        chain_id = list(model.child_dict.keys())[0]
    chain = model[chain_id]

    protein_info = []  # (atom_pos, resid, resname)
    for res in chain:
        if res.id[0] != " ":
            continue
        for atom in res:
            protein_info.append(
                (atom.get_vector().get_array(), res.id[1], res.get_resname()))
    if not protein_info:
        return None
    prot_coords = np.array([p[0] for p in protein_info])

    metals, ligands = [], []
    for res in chain:
        if res.id[0] == " ":
            continue
        resname = res.get_resname().strip()
        if resname in CRYSTALLIZATION_ARTIFACTS or resname == "HOH":
            continue
        atoms = [atom.get_vector().get_array() for atom in res]
        if not atoms:
            continue
        contacts = set()
        for apos in atoms:
            dists = np.linalg.norm(prot_coords - apos, axis=1)
            for idx in np.where(dists < contact_radius)[0]:
                contacts.add((protein_info[idx][1], protein_info[idx][2]))
        entry = {"name": resname, "atoms": np.array(atoms),
                 "contacts": sorted(contacts)}
        (metals if resname in _METAL_IONS else ligands).append(entry)

    coord_aa = set()
    for m in metals:
        for _, aa in m["contacts"]:
            if aa in _METAL_COORD_AA:
                coord_aa.add(aa)
    return {"metals": metals, "ligands": ligands, "coord_aa_pattern": coord_aa}


def _spatial_transfer_contacts(query_pdb, homolog_pdb, homolog_chain="A",
                               contact_radius=5.0):
    """Transfer active-site contacts from homolog to query via superposition.

    1. Extracts all non-polymer atoms (metals + ligands) from the homolog.
    2. Structurally aligns homolog → query.
    3. Projects non-polymer atom positions into query space.
    4. Returns query residues near the projected positions.

    Returns dict with 'contacts', 'metal_contacts', 'coord_aa_pattern',
    'alignment_rmsd', 'method', or None on failure.
    """
    lig_info = _extract_nonpolymer_info(homolog_pdb, homolog_chain, contact_radius)
    if not lig_info:
        return None

    all_atoms, atom_src = [], []
    for m in lig_info["metals"]:
        for pos in m["atoms"]:
            all_atoms.append(pos); atom_src.append("metal")
    for l in lig_info["ligands"]:
        for pos in l["atoms"]:
            all_atoms.append(pos); atom_src.append("ligand")
    if not all_atoms:
        return None
    all_atoms = np.array(all_atoms)

    try:
        q_prody = prody.parsePDB(query_pdb)
        h_prody = prody.parsePDB(homolog_pdb, chain=homolog_chain)
        if h_prody is None:
            h_prody = prody.parsePDB(homolog_pdb)
        q_ca = q_prody.select("protein and calpha")
        h_ca = h_prody.select("protein and calpha")
        if q_ca is None or h_ca is None:
            return None
        matches = prody.matchChains(q_ca, h_ca, overlap=30, seqid=10)
        if not matches:
            return None
        qm, hm, _, _ = matches[0]
        t = prody.calcTransformation(hm, qm)
        rmsd = float(prody.calcRMSD(qm, t.apply(hm)))
        projected = t.apply(all_atoms)
    except Exception:
        return None

    from Bio.PDB import PDBParser as _BioParser
    _parser = _BioParser(QUIET=True)
    model = _parser.get_structure("q", query_pdb)[0]

    contacts, metal_contacts = [], []
    for res in model["A"]:
        if res.id[0] != " ":
            continue
        for atom in res:
            apos = atom.get_vector().get_array()
            dists = np.linalg.norm(projected - apos, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < contact_radius:
                entry = (res.id[1], res.get_resname(), float(dists[idx]))
                contacts.append(entry)
                if atom_src[idx] == "metal":
                    metal_contacts.append(entry)
                break

    return {
        "contacts": sorted(set(contacts)),
        "metal_contacts": sorted(set(metal_contacts)),
        "coord_aa_pattern": lig_info["coord_aa_pattern"],
        "alignment_rmsd": rmsd,
        "method": "spatial_projection",
    }


def _spatial_transfer_from_positions(query_pdb, homolog_pdb, homolog_chain,
                                     positions, contact_radius=5.0):
    """Transfer contacts by projecting specific residue positions from the homolog.

    Used when the homolog has no non-polymer entities but has UniProt active-
    site annotations.  The sidechain atoms of the annotated residues act as
    'virtual ligand atoms' for spatial projection.

    Returns dict with 'contacts', 'method', or None.
    """
    from Bio.PDB import PDBParser as _BioParser
    _parser = _BioParser(QUIET=True)

    # Collect sidechain atoms at specified positions
    try:
        h_struct = _parser.get_structure("h", homolog_pdb)
    except Exception:
        return None
    h_model = h_struct[0]
    if homolog_chain not in h_model:
        homolog_chain = list(h_model.child_dict.keys())[0]
    h_chain = h_model[homolog_chain]

    site_atoms = []
    for res in h_chain:
        if res.id[0] != " " or res.id[1] not in positions:
            continue
        for atom in res:
            site_atoms.append(atom.get_vector().get_array())
    if not site_atoms:
        return None
    site_atoms = np.array(site_atoms)

    try:
        q_prody = prody.parsePDB(query_pdb)
        h_prody = prody.parsePDB(homolog_pdb, chain=homolog_chain)
        if h_prody is None:
            h_prody = prody.parsePDB(homolog_pdb)
        q_ca = q_prody.select("protein and calpha")
        h_ca = h_prody.select("protein and calpha")
        if q_ca is None or h_ca is None:
            return None
        matches = prody.matchChains(q_ca, h_ca, overlap=30, seqid=10)
        if not matches:
            return None
        qm, hm, _, _ = matches[0]
        t = prody.calcTransformation(hm, qm)
        projected = t.apply(site_atoms)
    except Exception:
        return None

    q_model = _parser.get_structure("q", query_pdb)[0]
    contacts = []
    for res in q_model["A"]:
        if res.id[0] != " ":
            continue
        for atom in res:
            apos = atom.get_vector().get_array()
            if np.linalg.norm(projected - apos, axis=1).min() < contact_radius:
                contacts.append((res.id[1], res.get_resname(),
                                 float(np.linalg.norm(projected - apos, axis=1).min())))
                break

    if not contacts:
        return None
    return {"contacts": sorted(set(contacts)), "method": "active_site_projection"}


def _validate_metal_transfer(result, min_coord=3, min_his=2):
    """Check if transferred metal contacts have correct AA types."""
    mc = result.get("metal_contacts", [])
    coord_aa = result.get("coord_aa_pattern", set())
    coord_res = [(r, aa) for r, aa, _ in mc if aa in _METAL_COORD_AA]
    n_his = sum(1 for _, aa in coord_res if aa == "HIS")
    if "HIS" in coord_aa:
        return len(coord_res) >= min_coord and n_his >= min_his
    return len(coord_res) >= min_coord


def _find_catalytic_cluster(query_pdb, coord_aa_types=None, search_radius=5.0):
    """Find the best catalytic-residue cluster in a query structure.

    When *coord_aa_types* is provided (derived from a homolog's coordination
    pattern), only those AA types seed the search.  Otherwise all residues in
    ``_CATALYTIC_AA`` are considered.

    Returns dict with 'center', 'catalytic', 'all_nearby', 'score', or None.
    """
    if coord_aa_types is None:
        coord_aa_types = _CATALYTIC_AA

    from Bio.PDB import PDBParser as _BioParser
    _parser = _BioParser(QUIET=True)
    model = _parser.get_structure("q", query_pdb)[0]

    cat_atoms = []  # (resid, resname, pos)
    res_atoms = {}
    for res in model["A"]:
        if res.id[0] != " ":
            continue
        rid = res.id[1]
        rname = res.get_resname()
        res_atoms[rid] = (rname, {a.get_name(): a.get_vector().get_array() for a in res})
        if rname in coord_aa_types:
            for aname in _COORD_ATOMS.get(rname, []):
                if aname in res_atoms[rid][1]:
                    cat_atoms.append((rid, rname, res_atoms[rid][1][aname]))
                    break

    if len(cat_atoms) < 2:
        return None

    best = None
    seen_centers = []
    for i, (r1, _, p1) in enumerate(cat_atoms):
        for r2, _, p2 in cat_atoms[i + 1:]:
            if np.linalg.norm(p1 - p2) > 10:
                continue
            center = (p1 + p2) / 2
            if any(np.linalg.norm(center - sc) < 3 for sc in seen_centers):
                continue
            seen_centers.append(center)

            cluster = sorted({(rid, rn, float(np.linalg.norm(pos - center)))
                              for rid, rn, pos in cat_atoms
                              if np.linalg.norm(pos - center) < search_radius})
            aa_set = {rn for _, rn, _ in cluster}
            score = len(cluster) + len(aa_set) * 2

            all_nearby = []
            for rid, (rname, atoms) in res_atoms.items():
                for apos in atoms.values():
                    if np.linalg.norm(apos - center) < search_radius:
                        all_nearby.append((rid, rname, float(np.linalg.norm(apos - center))))
                        break

            if best is None or score > best["score"]:
                best = {"center": center, "catalytic": cluster,
                        "all_nearby": sorted(set(all_nearby)), "score": score}

    return best


def _transfer_active_site(query_pdb, homolog_pdb, homolog_chain="A",
                           homolog_uniprot=None, contact_radius=5.0):
    """Transfer active-site contacts from a homolog PDB to a query structure.

    Cascade:
    1. Spatial projection of non-polymer entities (metals, ligands).
    2. If no ligands in homolog PDB — project UniProt active-site residues.
    3. Validate metal-site transfer (AA type check).
    4. If validation fails — motif search using homolog's coordination pattern.

    Returns dict with 'contacts' [(resid, resname, dist)], 'method', or None.
    """
    # Step 1: try spatial projection of actual ligands/metals
    result = _spatial_transfer_contacts(query_pdb, homolog_pdb, homolog_chain,
                                        contact_radius)

    if result is not None and (result["contacts"] or result["metal_contacts"]):
        has_metals = bool(result["metal_contacts"])
        if has_metals:
            if _validate_metal_transfer(result):
                return result
            # Metal validation failed — try motif search
        else:
            if len(result["contacts"]) >= 5:
                return result

    # Step 2: no usable ligands — try UniProt active-site positions
    if homolog_uniprot:
        ann = fetch_uniprot_annotation(homolog_uniprot)
        positions = set()
        for site in ann.get("active_sites", []):
            p = site.get("position")
            if p:
                positions.add(p)
        if positions:
            r2 = _spatial_transfer_from_positions(
                query_pdb, homolog_pdb, homolog_chain, positions,
                contact_radius)
            if r2 and len(r2["contacts"]) >= 3:
                return r2

    # Step 3: motif search fallback
    coord_aa = (result or {}).get("coord_aa_pattern", None)
    cluster = _find_catalytic_cluster(query_pdb, coord_aa)
    if cluster and len(cluster["catalytic"]) >= 3:
        return {"contacts": cluster["all_nearby"],
                "method": "motif_search"}

    return None


def get_active_site_residues(uid, structure_path, foldseek_hits,
                             relevant_ligands=None, catalytic_metals=None):
    """Determine active site residues for a structure.

    Strategy (ordered by reliability):
    1. AlphaFill — transplanted ligand contacts, biologically filtered
    2. UniProt annotation — curated active/binding site features
    3. Foldseek homolog → AlphaFill/UniProt on homolog

    Filtering logic (hybrid whitelist + blacklist):
    - Compounds in relevant_ligands: always kept (primary cofactors)
    - Compounds in catalytic_metals: always kept (essential metal ions)
    - Compounds in CRYSTALLIZATION_ARTIFACTS: always removed
    - Other compounds: kept for pocket definition (may be biologically relevant)

    Args:
        relevant_ligands: list of ligand IDs for primary cofactors (e.g., ["ATP", "AMP"]).
            If empty list [], AlphaFill is skipped entirely.
            If None, all AlphaFill ligands are used (no filtering).
        catalytic_metals: list of metal CCD codes essential for catalysis
            (e.g., ["MG", "MN"]). These are always kept from AlphaFill and
            tracked separately for cofolding chain inclusion.

    Returns (residue_set, evidence_dict).
    """
    active_positions = set()
    evidence = {"type": "none"}
    _s1_fallback = None  # Saved if Strategy 1 finds contacts but no relevant cofactor
    if catalytic_metals is None:
        catalytic_metals = []

    # Strategy 1: AlphaFill (primary) — only if relevant_ligands is not empty
    use_alphafill = relevant_ligands is None or len(relevant_ligands) > 0
    if use_alphafill:
        af_data = fetch_alphafill(uid)
        if af_data:
            filtered_contacts = {}
            filtered_details = {}
            metal_contacts = {}
            metal_details = {}
            artifacts_removed = []
            for compound, residues in af_data["contacts"].items():
                # Always remove crystallization artifacts
                if compound in CRYSTALLIZATION_ARTIFACTS:
                    artifacts_removed.append(compound)
                    continue
                # If relevant_ligands specified, keep: relevant_ligands + catalytic_metals + other bio-relevant
                if relevant_ligands is not None:
                    if compound in relevant_ligands:
                        filtered_contacts[compound] = residues
                        if compound in af_data["details"]:
                            filtered_details[compound] = af_data["details"][compound]
                    elif compound in catalytic_metals:
                        metal_contacts[compound] = residues
                        if compound in af_data["details"]:
                            metal_details[compound] = af_data["details"][compound]
                    else:
                        # Skip irrelevant ligands for pocket definition
                        continue
                else:
                    # No filtering specified — keep everything except artifacts
                    filtered_contacts[compound] = residues
                    if compound in af_data["details"]:
                        filtered_details[compound] = af_data["details"][compound]

            all_contacts = set()
            for residues in filtered_contacts.values():
                all_contacts.update(residues)
            for residues in metal_contacts.values():
                all_contacts.update(residues)

            if all_contacts:
                # Check if any contacts are from relevant cofactors
                has_relevant_cofactor = False
                if relevant_ligands:
                    has_relevant_cofactor = any(
                        c in relevant_ligands for c in filtered_contacts)

                active_positions = all_contacts
                evidence = {
                    "type": "alphafill",
                    "ligands": filtered_details,
                    "ligand_contacts": filtered_contacts,
                }
                if metal_contacts:
                    evidence["metal_contacts"] = metal_contacts
                    evidence["metal_details"] = metal_details
                if artifacts_removed:
                    evidence["filtered_out_ligands"] = sorted(set(artifacts_removed))

                if has_relevant_cofactor or not relevant_ligands:
                    # Found relevant cofactor or no filtering — return immediately
                    return active_positions, evidence
                else:
                    # Found contacts (e.g., NAD) but no relevant cofactor (e.g., ATP).
                    # Save as fallback and continue to Strategy 3 (Foldseek) to look
                    # for relevant cofactors in PDB homolog structures.
                    _s1_fallback = (active_positions.copy(), evidence.copy())
                    print(f"  INFO: {uid}: AlphaFill found {list(filtered_contacts.keys())} "
                          f"but no relevant cofactors {relevant_ligands}. "
                          f"Checking Foldseek homologs for cofactor data...", flush=True)

    # Strategy 2: UniProt annotation
    if use_alphafill and relevant_ligands and _s1_fallback is None:
        print(f"  WARNING: {uid}: AlphaFill returned no usable cofactor contacts "
              f"(expected {relevant_ligands}), falling back to UniProt annotation. "
              f"Consider re-running if this is unexpected.", flush=True)
    ann = fetch_uniprot_annotation(uid)
    if ann.get("source") == "uniprotkb":
        for site in ann.get("active_sites", []):
            pos = site.get("position")
            if pos:
                active_positions.add(pos)
        for site in ann.get("binding_sites", []):
            pos = site.get("position")
            if pos:
                active_positions.add(pos)
        if active_positions:
            evidence = {
                "type": "uniprot_annotation",
                "protein_name": ann.get("protein_name", ""),
                "ec_numbers": ann.get("ec_numbers", []),
            }
            if _s1_fallback is None:
                # No pending cofactor search — return immediately
                return active_positions, evidence
            else:
                # Strategy 1 found non-relevant ligands; save Strategy 2 results
                # and continue to Strategy 3 to look for relevant cofactors
                _s2_fallback = (active_positions.copy(), evidence.copy())

    # Strategy 3: Foldseek homolog → try AlphaFill on the homolog's UniProt ID,
    # then map residues back via structural alignment
    for fs_hit in foldseek_hits:
        if fs_hit["tmscore"] < 0.5:
            continue

        pdb_id = fs_hit["pdb_id"]

        # Get the homolog's UniProt ID from PDB
        pdb_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        raw = fetch_url(pdb_url, timeout=10)
        homolog_uid = None
        if raw:
            try:
                pdb_data = json.loads(raw)
                for ref in pdb_data.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers", []):
                    if ref.get("database_name") == "UniProt":
                        homolog_uid = ref.get("database_accession")
                        break
            except json.JSONDecodeError:
                pass

        if not homolog_uid:
            continue

        # Try AlphaFill on the homolog (biologically filtered)
        homolog_af = fetch_alphafill(homolog_uid) if use_alphafill else None
        homolog_sites = set()
        homolog_ligands = {}

        # Check if AlphaFill on the homolog has relevant cofactor data
        homolog_has_relevant = False
        if homolog_af:
            for comp, residues in homolog_af["contacts"].items():
                if comp in CRYSTALLIZATION_ARTIFACTS:
                    continue
                if relevant_ligands is not None:
                    if comp in relevant_ligands:
                        homolog_has_relevant = True
                    elif comp in catalytic_metals:
                        pass  # keep catalytic metals
                    else:
                        continue  # skip irrelevant ligands
                homolog_sites.update(residues)
                if comp in homolog_af["details"]:
                    homolog_ligands[comp] = homolog_af["details"][comp]

        # Fallback: extract cofactor contacts directly from the PDB structure
        # when AlphaFill doesn't cover the homolog or has no relevant cofactors.
        pdb_cofactor_data = None
        if not homolog_has_relevant and relevant_ligands:
            target_chain = fs_hit.get("target_chain", "A")
            pdb_cofactor_data = extract_pdb_cofactor_contacts(
                pdb_id, target_chain, relevant_ligands)
            if pdb_cofactor_data:
                print(f"    {uid}: extracted cofactor {list(pdb_cofactor_data.keys())} "
                      f"directly from PDB {pdb_id} chain {target_chain}", flush=True)
                for comp, info in pdb_cofactor_data.items():
                    homolog_sites.update(info["contacts"])
                    homolog_ligands[comp] = {
                        "pdb_id": pdb_id,
                        "n_contacts": info["n_contacts"],
                        "source": "pdb_direct",
                    }
                    homolog_has_relevant = True

        # ── Spatial transfer of active-site contacts ──
        # Instead of mapping residue numbers (unreliable at low sequence
        # identity), superpose the homolog onto the query and project
        # ligand/metal positions into query space.  Falls back to UniProt
        # active-site projection and then motif search.
        pdb_content = fetch_url(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
        if not pdb_content:
            continue

        tmp_pdb = os.path.join(tempfile.gettempdir(), f"{pdb_id}_align.pdb")
        with open(tmp_pdb, "w") as pf:
            pf.write(pdb_content)

        try:
            target_chain = fs_hit.get("target_chain", "A")
            transfer = _transfer_active_site(
                structure_path, tmp_pdb, target_chain,
                homolog_uniprot=homolog_uid)

            if transfer and transfer.get("contacts"):
                transferred_resids = set(r for r, _, _ in transfer["contacts"])
                active_positions.update(transferred_resids)

            # Also include any residues from the old homolog_sites that
            # were obtained via AlphaFill/PDB-direct (these are residue
            # numbers in the *homolog*, so still need mapping for those —
            # but the spatial transfer already covers the same pocket).

            if active_positions:
                source = "alphafill" if homolog_af else "uniprot"
                transfer_method = transfer.get("method", "unknown") if transfer else "none"
                evidence = {
                    "type": f"foldseek+{source}",
                    "homolog_pdb": pdb_id,
                    "homolog_chain": target_chain,
                    "homolog_uniprot": homolog_uid,
                    "tmscore": fs_hit["tmscore"],
                    "seqid": fs_hit["fident"],
                    "transfer_method": transfer_method,
                    "n_transferred": len(active_positions),
                }
                if homolog_ligands:
                    evidence["homolog_ligands"] = homolog_ligands
                return active_positions, evidence

        except Exception as e:
            print(f"    Spatial transfer error for {pdb_id} chain {target_chain}: {e}",
                  flush=True)
        finally:
            if os.path.exists(tmp_pdb):
                os.remove(tmp_pdb)

    # No new evidence found from Strategy 3.
    # Return best available fallback: Strategy 2 (UniProt) > Strategy 1 (non-relevant AlphaFill)
    _s2_fallback = locals().get('_s2_fallback')
    if _s2_fallback is not None:
        return _s2_fallback
    if _s1_fallback is not None:
        return _s1_fallback

    if foldseek_hits:
        best = foldseek_hits[0]
        tm = best["tmscore"]
        evidence = {
            "type": "none",
            "note": (f"Foldseek best: PDB {best['pdb_id']} TM={tm:.2f}"
                     + (" (TM < 0.5, unreliable)" if tm < 0.5 else " (no binding sites found)")),
        }

    return active_positions, evidence


# ═══════════════════════════════════════════════════════════════════════
# STEP 3b: POCKET DETECTION (pyKVFinder)
# ═══════════════════════════════════════════════════════════════════════

def clean_pdb(pdb_path):
    """Remove waters, ligands, non-protein atoms. Returns temp file path."""
    struct = prody.parsePDB(pdb_path)
    if struct is None:
        return None
    protein = struct.select("protein")
    if protein is None or protein.numAtoms() < 50:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    prody.writePDB(tmp.name, protein)
    return tmp.name


def detect_pockets(pdb_path, source_type, top_n=10):
    """Detect cavities using pyKVFinder. Returns list of pocket dicts."""
    if not HAS_PYKVFINDER:
        return []

    # Clean PDB if experimental
    cleaned_path = None
    analysis_path = pdb_path
    if source_type == "pdb":
        cleaned_path = clean_pdb(pdb_path)
        if cleaned_path is None:
            return []
        analysis_path = cleaned_path

    try:
        atomic = pyKVFinder.read_pdb(analysis_path)
    except Exception as e:
        print(f" pyKVFinder read error: {e}", flush=True)
        if cleaned_path:
            os.remove(cleaned_path)
        return []

    # Adaptive parameters validated on NylC (355 aa) and SfaB (585 aa)
    n_atoms = len(atomic)
    est_residues = n_atoms / 8
    if est_residues < 400:
        probe_out, removal_distance = 8.0, 1.0
    else:
        probe_out, removal_distance = 4.0, 2.4

    try:
        vertices = pyKVFinder.get_vertices(atomic, step=0.6, probe_out=probe_out)
        ncav, cavities = pyKVFinder.detect(
            atomic, vertices, step=0.6,
            probe_in=1.4, probe_out=probe_out,
            removal_distance=removal_distance,
            volume_cutoff=5.0, verbose=False,
        )
        if ncav == 0:
            return []

        _, volumes, areas = pyKVFinder.spatial(cavities, step=0.6)
        # include backbone contacts — active site residues can contribute
        # via backbone atoms (e.g., catalytic amide NH, oxyanion hole)
        residues_dict = pyKVFinder.constitutional(
            cavities, atomic, vertices, step=0.6, ignore_backbone=False
        )

        pockets = []
        for name in volumes.keys():
            raw_residues = residues_dict.get(name, [])
            lining_residues = [(r[2], int(r[0]), r[1]) for r in raw_residues]
            if len(lining_residues) < 2:
                continue
            pockets.append({
                "name": name,
                "volume_A3": round(volumes[name], 1),
                "area_A2": round(areas[name], 1),
                "n_residues": len(lining_residues),
                "lining_residues": lining_residues,
            })

        # Default sort by volume
        pockets.sort(key=lambda p: -p["volume_A3"])
        return pockets[:top_n]

    except Exception as e:
        print(f" pyKVFinder detection error: {e}", flush=True)
        return []
    finally:
        if cleaned_path and os.path.exists(cleaned_path):
            os.remove(cleaned_path)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3c: POCKET SELECTION
# ═══════════════════════════════════════════════════════════════════════

def select_binding_pocket(pockets, active_site_residues):
    """Select the pocket most likely to be the binding pocket.

    If active_site_residues provided: pick pocket with highest overlap.
    Otherwise: return largest pocket (flagged as no evidence).

    Returns (selected_pocket, overlap_count, overlap_fraction).
    """
    if not pockets:
        return None, 0, 0.0

    if not active_site_residues:
        # No active site info — return largest pocket, clearly marked
        return pockets[0], 0, 0.0

    active_set = set(active_site_residues)
    best_pocket = None
    best_overlap = 0
    best_fraction = 0.0

    for p in pockets:
        pocket_resnums = {r[1] for r in p["lining_residues"]}
        overlap = len(pocket_resnums & active_set)
        fraction = overlap / len(active_set) if active_set else 0

        if overlap > best_overlap or (overlap == best_overlap and p["volume_A3"] > (best_pocket or {}).get("volume_A3", 0)):
            best_pocket = p
            best_overlap = overlap
            best_fraction = fraction

    # If no pocket contains any active site residue, return largest but with 0 overlap
    # (caller will flag this as a warning)
    if best_overlap == 0:
        return pockets[0], 0, 0.0

    return best_pocket, best_overlap, round(best_fraction, 3)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pocket Analysis Pipeline")
    parser.add_argument("project_dir", help="Project directory with config.yaml")
    parser.add_argument("--skip-foldseek", action="store_true",
                        help="Reuse existing foldseek_results.tsv")
    parser.add_argument("-o", "--output", default="pocket_analysis.json",
                        help="Output JSON filename (default: pocket_analysis.json)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    print(f"Pocket Analysis Pipeline")
    print(f"Project: {project_dir}")
    print(f"{'='*60}\n")

    if not HAS_PYKVFINDER:
        sys.exit("ERROR: pyKVFinder not installed (pip install pyKVFinder)")
    if not HAS_PRODY:
        sys.exit("ERROR: ProDy not installed")

    # ── Step 1: Collect structures ──
    structures, config = collect_structures(project_dir)
    n_af = sum(1 for s in structures if s["source"] == "alphafold")
    n_pdb = sum(1 for s in structures if s["source"] == "pdb")
    n_boltz = sum(1 for s in structures if s["source"] == "boltz")
    print(f"Found {len(structures)} structures ({n_af} AlphaFold, {n_pdb} PDB, {n_boltz} Boltz)\n")

    # ── Step 2: Foldseek batch search ──
    print("Step 1: Foldseek structural homology search", flush=True)
    foldseek_tsv = os.path.join(project_dir, "foldseek_results.tsv")

    if args.skip_foldseek and os.path.exists(foldseek_tsv):
        print(f"  Reusing existing {foldseek_tsv}", flush=True)
        foldseek_hits = parse_foldseek_results(foldseek_tsv)
    else:
        foldseek_hits = run_foldseek(structures, foldseek_tsv)

    n_with_hits = sum(1 for uid_hits in foldseek_hits.values() if uid_hits)
    print(f"  {n_with_hits}/{len(structures)} structures matched to PDB\n", flush=True)

    # ── Step 3: Per-structure analysis ──
    # Get relevant ligands from config (controls AlphaFill filtering)
    relevant_ligands = config.get("target_reaction", {}).get("relevant_ligands")
    # relevant_ligands: None = use all, [] = skip AlphaFill, ["ATP",...] = filter
    catalytic_metals = config.get("target_reaction", {}).get("catalytic_metals", [])

    if relevant_ligands is not None:
        if len(relevant_ligands) == 0:
            print(f"  relevant_ligands: [] — AlphaFill disabled, using UniProt annotations only", flush=True)
        else:
            print(f"  relevant_ligands: {relevant_ligands} — AlphaFill filtered to these", flush=True)
    else:
        print(f"  relevant_ligands: not specified — using all AlphaFill ligands", flush=True)
    if catalytic_metals:
        print(f"  catalytic_metals: {catalytic_metals} — will be included as cofolding chains", flush=True)

    print("\nStep 2: Pocket detection + active site mapping", flush=True)
    results = {}

    for i, s in enumerate(structures):
        uid = s["uid"]
        source = s["source"]
        role = s["role"]
        label = f"[{role}] " if role == "reference" else ""

        print(f"  {i+1}/{len(structures)} {label}{uid} ({source})", end="", flush=True)

        # 3a: Active site residues
        if source == "pdb":
            # PDB structures: map PDB ID → UniProt, then query AlphaFill/UniProt
            pdb_api_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{uid}/1"
            pdb_raw = fetch_url(pdb_api_url, timeout=10)
            pdb_uniprot_id = None
            if pdb_raw:
                try:
                    pdb_data = json.loads(pdb_raw)
                    for ref in pdb_data.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers", []):
                        if ref.get("database_name") == "UniProt":
                            pdb_uniprot_id = ref.get("database_accession")
                            break
                except json.JSONDecodeError:
                    pass

            if pdb_uniprot_id:
                active_residues, evidence = get_active_site_residues(
                    pdb_uniprot_id, s["path"], [],
                    relevant_ligands=relevant_ligands,
                    catalytic_metals=catalytic_metals)
                evidence["pdb_id"] = uid
                evidence["uniprot_id"] = pdb_uniprot_id
            else:
                active_residues = set()
                evidence = {"type": "pdb_experimental", "pdb_id": uid}
        else:
            fs_hits = foldseek_hits.get(uid, [])
            active_residues, evidence = get_active_site_residues(
                uid, s["path"], fs_hits,
                relevant_ligands=relevant_ligands,
                catalytic_metals=catalytic_metals)

        # Also check config for manually specified catalytic residues (reference only)
        if role == "reference":
            for pos in config.get("reference", {}).get("catalytic_residues", []):
                active_residues.add(pos)

        # 3b: Pocket detection
        pockets = detect_pockets(s["path"], source, top_n=10)

        # 3c: Pocket selection
        binding_pocket, overlap, fraction = select_binding_pocket(pockets, active_residues)

        # Build result
        result = {
            "uid": uid,
            "source": source,
            "role": role,
            "path": s["path"],
            "n_cavities": len(pockets),
            "binding_pocket": None,
            "evidence": evidence,
            "all_pockets": pockets,
        }

        evidence["active_site_residues"] = sorted(active_residues)
        evidence["active_site_total"] = len(active_residues)

        if binding_pocket:
            # If we have active site residues but none are in the pocket,
            # do NOT report the pocket — better no pocket than a wrong one
            if active_residues and overlap == 0:
                evidence["warning"] = ("Active site residues mapped but not found in any "
                                       "detected cavity — pocket detection failed for this structure")
                evidence["active_site_overlap"] = 0
            else:
                result["binding_pocket"] = {
                    "name": binding_pocket["name"],
                    "volume_A3": binding_pocket["volume_A3"],
                    "area_A2": binding_pocket["area_A2"],
                    "n_residues": binding_pocket["n_residues"],
                    "lining_residues": binding_pocket["lining_residues"],
                }
                evidence["active_site_overlap"] = overlap

        # 3d: Extract best cofactor for cofolding (if relevant_ligands specified)
        best_cofactor = None
        if relevant_ligands and evidence.get("ligand_contacts"):
            # Pick the cofactor from relevant_ligands with the most contacts
            best_ccd = None
            best_n = 0
            for lig_name, contacts in evidence["ligand_contacts"].items():
                if lig_name in relevant_ligands and len(contacts) > best_n:
                    best_ccd = lig_name
                    best_n = len(contacts)
            if best_ccd:
                details = evidence.get("ligands", {}).get(best_ccd, {})
                best_cofactor = {
                    "ccd_code": best_ccd,
                    "contact_residues": sorted(evidence["ligand_contacts"][best_ccd]),
                    "n_contacts": best_n,
                    "source_pdb": details.get("pdb_id", ""),
                    "identity": details.get("identity", 0),
                    "local_rmsd": details.get("local_rmsd", 0),
                }
        elif relevant_ligands and evidence.get("type", "").startswith("foldseek+"):
            # Foldseek homolog-based: check homolog_ligands
            homolog_ligs = evidence.get("homolog_ligands", {})
            best_ccd = None
            best_n = 0
            for lig_name, details in homolog_ligs.items():
                if lig_name in relevant_ligands:
                    n = details.get("n_contacts", 0)
                    if n > best_n:
                        best_ccd = lig_name
                        best_n = n
            if best_ccd and active_residues:
                best_cofactor = {
                    "ccd_code": best_ccd,
                    "contact_residues": sorted(active_residues),
                    "n_contacts": len(active_residues),
                    "source_pdb": evidence.get("homolog_pdb", ""),
                    "identity": evidence.get("seqid", 0),
                    "local_rmsd": 0,
                }
        result["best_cofactor"] = best_cofactor

        # 3e: Extract catalytic metals for cofolding
        found_metals = []
        if catalytic_metals:
            # Check direct AlphaFill metal contacts
            metal_cts = evidence.get("metal_contacts", {})
            metal_dts = evidence.get("metal_details", {})
            cofactor_residues = set()
            if best_cofactor:
                cofactor_residues = set(best_cofactor.get("contact_residues", []))

            for metal_code in catalytic_metals:
                if metal_code in metal_cts:
                    metal_residues = metal_cts[metal_code]
                    # Include metal if its contacts overlap with the active site
                    # (shares at least 1 residue with cofactor or active site)
                    metal_set = set(metal_residues)
                    near_active_site = bool(metal_set & (cofactor_residues | set(active_residues)))
                    if near_active_site or not cofactor_residues:
                        details = metal_dts.get(metal_code, {})
                        found_metals.append({
                            "ccd_code": metal_code,
                            "contact_residues": sorted(metal_residues),
                            "n_contacts": len(metal_residues),
                            "source_pdb": details.get("pdb_id", ""),
                            "identity": details.get("identity", 0),
                            "local_rmsd": details.get("local_rmsd", 0),
                        })
                # Also check ligand_contacts (metals may end up there via
                # non-artifact compounds that aren't in relevant_ligands)
                elif metal_code in evidence.get("ligand_contacts", {}):
                    metal_residues = evidence["ligand_contacts"][metal_code]
                    metal_set = set(metal_residues)
                    near_active_site = bool(metal_set & (cofactor_residues | set(active_residues)))
                    if near_active_site or not cofactor_residues:
                        details = evidence.get("ligands", {}).get(metal_code, {})
                        found_metals.append({
                            "ccd_code": metal_code,
                            "contact_residues": sorted(metal_residues),
                            "n_contacts": len(metal_residues),
                            "source_pdb": details.get("pdb_id", ""),
                            "identity": details.get("identity", 0),
                            "local_rmsd": details.get("local_rmsd", 0),
                        })
        result["catalytic_metals"] = found_metals

        results[uid] = result

        # Status output
        ev_type = evidence["type"]
        if binding_pocket:
            pocket_info = f"pocket {binding_pocket['name']} ({binding_pocket['volume_A3']:.0f}A3, {binding_pocket['n_residues']} res)"
        else:
            pocket_info = "no pockets"

        if ev_type == "alphafill":
            ligands = ", ".join(sorted(evidence.get("ligand_contacts", {}).keys()))
            metals = ", ".join(sorted(evidence.get("metal_contacts", {}).keys()))
            metal_str = f" + metals: {metals}" if metals else ""
            print(f" — alphafill ({len(active_residues)} contacts, ligands: {ligands}{metal_str}) → {pocket_info}")
        elif ev_type == "uniprot_annotation":
            print(f" — uniprot ({len(active_residues)} sites) → {pocket_info}")
        elif ev_type.startswith("foldseek+"):
            n_trans = evidence.get('n_transferred', evidence.get('n_mapped', '?'))
            method = evidence.get('transfer_method', 'residue_map')
            print(f" — {ev_type} PDB:{evidence['homolog_pdb']} TM={evidence['tmscore']:.2f} "
                  f"({n_trans} via {method}) → {pocket_info}")
        elif ev_type == "pdb_experimental":
            print(f" — experimental → {pocket_info}")
        else:
            note = evidence.get("note", "")
            print(f" — no evidence{' (' + note + ')' if note else ''} → {pocket_info}")

        if evidence.get("warning"):
            print(f"    WARNING: {evidence['warning']}")

    # ── Step 4: Save results ──
    print(f"\nSaving results ...", flush=True)

    # JSON
    json_path = os.path.join(project_dir, args.output)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # CSV summary
    csv_path = os.path.join(project_dir, args.output.replace(".json", "_summary.csv"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "structure_source", "role", "n_cavities_detected",
            "evidence_type", "evidence_homolog_pdb", "evidence_tmscore",
            "n_active_site_residues", "active_site_overlap_in_pocket",
            "transplanted_ligands", "catalytic_metals_found",
            "best_cofactor_ccd", "best_cofactor_n_contacts",
            "kvfinder_pocket_id", "pocket_volume_A3", "pocket_area_A2", "pocket_n_lining_residues",
            "warning",
        ])
        for uid, r in sorted(results.items()):
            ev = r.get("evidence", {})
            bp = r.get("binding_pocket")

            # Collect ligand info from alphafill evidence
            ligands = ""
            if ev.get("ligand_contacts"):
                ligands = "; ".join(sorted(ev["ligand_contacts"].keys()))
            elif ev.get("homolog_ligands"):
                ligands = "; ".join(sorted(ev["homolog_ligands"].keys()))

            # Collect metal info
            metals_found = "; ".join(m["ccd_code"] for m in r.get("catalytic_metals", []))

            cofactor = r.get("best_cofactor")
            writer.writerow([
                uid, r["source"], r["role"], r["n_cavities"],
                ev.get("type", "none"),
                ev.get("homolog_pdb", ""),
                f"{ev['tmscore']:.2f}" if "tmscore" in ev else "",
                ev.get("active_site_total", 0),
                ev.get("active_site_overlap", 0),
                ligands, metals_found,
                cofactor["ccd_code"] if cofactor else "",
                cofactor["n_contacts"] if cofactor else "",
                bp["name"] if bp else "",
                bp["volume_A3"] if bp else "",
                bp["area_A2"] if bp else "",
                bp["n_residues"] if bp else "",
                ev.get("warning", ""),
            ])
    print(f"  CSV: {csv_path}")

    # Summary
    from collections import Counter
    ev_counts = Counter(r["evidence"]["type"] for r in results.values())
    n_warned = sum(1 for r in results.values() if r["evidence"].get("warning"))

    print(f"\n{'='*60}")
    print(f"Analyzed {len(results)}/{len(structures)} structures")
    print(f"Evidence breakdown:")
    for ev_type, count in ev_counts.most_common():
        print(f"  {ev_type}: {count}")
    n_with_cofactor = sum(1 for r in results.values() if r.get("best_cofactor"))
    if n_with_cofactor:
        print(f"Cofactor binding sites: {n_with_cofactor} structures have cofactor info for cofolding")
    n_with_metals = sum(1 for r in results.values() if r.get("catalytic_metals"))
    if n_with_metals:
        print(f"Catalytic metals: {n_with_metals} structures have metal ions for cofolding")
    if n_warned:
        print(f"Warnings: {n_warned} structures have mapped residues outside detected pockets")


if __name__ == "__main__":
    main()
