#!/usr/bin/env python3
"""
Prepare Boltz-2 cofolding inputs: enzyme + substrate(s).

Uses pocket_analysis.json to add pocket constraints that guide
substrate placement into the identified binding pocket.

Usage:
    python prepare_cofold_inputs.py <project_dir>
    python prepare_cofold_inputs.py <project_dir> --no-pocket-constraint
    python prepare_cofold_inputs.py <project_dir> --affinity

Reads from config.yaml:
    target_reaction:
      substrates:
        - name: "hexamethylenediamine"
          smiles: "NCCCCCCN"
        - name: "adipic_acid"
          smiles: "OC(=O)CCCCC(=O)O"
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import yaml


# CCD codes for common cofactors used in enzyme reactions.
# When relevant_ligands match these, the cofactor is included as an explicit
# chain in the Boltz-2 YAML using its CCD code (more accurate than SMILES).
COFACTOR_CCD_CODES = {
    "ATP", "ADP", "AMP", "ANP", "ACP",  # adenine nucleotides
    "GTP", "GDP", "GMP", "GNP", "GCP",   # guanine nucleotides
    "NAD", "NAP", "NDP",                 # NAD+, NADP+, NADPH
    "COA", "ACO",                        # coenzyme A, acetyl-CoA
    "FAD", "FMN",                        # flavin cofactors
    "SAM", "SAH",                        # S-adenosylmethionine/homocysteine
    "PLP", "PMP",                        # pyridoxal/pyridoxamine phosphate
    "TPP",                               # thiamine pyrophosphate
    "HEM", "HEC",                        # heme
}

# Map crystallographic analogs and product-state cofactors to their
# physiological pre-catalytic form. AlphaFill often returns ANP (AMPPNP,
# a non-hydrolyzable ATP analog used in crystallography) or ADP/AMP
# (product states). We use the contact residues from these for pocket
# definition, but substitute the physiological cofactor for cofolding.
COFACTOR_PHYSIOLOGICAL_MAP = {
    "ANP": "ATP",  # AMPPNP → ATP (crystallographic analog)
    "ACP": "ATP",  # AMPPCP → ATP (crystallographic analog)
    "ADP": "ATP",  # product of phosphorylation → pre-catalytic ATP
    "AMP": "ATP",  # product of adenylation → pre-catalytic ATP
    "GDP": "GTP",
    "GMP": "GTP",
    "GNP": "GTP",  # GMPPNP → GTP (crystallographic analog)
    "GCP": "GTP",  # GMPPCP → GTP (crystallographic analog)
}

# Map crystallographic metal substitutes to physiological form.
METAL_PHYSIOLOGICAL_MAP = {
    "MN": "MG",  # Mn2+ is a common crystallographic substitute for Mg2+
}

# Amino acid types that commonly coordinate metal ions in enzyme active sites.
METAL_COORDINATING_3 = {"ASP", "GLU", "HIS", "SER", "THR", "CYS"}

# One-letter to three-letter amino acid mapping for residue type lookup.
AA_1TO3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def validate_template(structure_path, uid=None):
    """Check that a PDB template file is valid (contains ATOM records).

    If the file is corrupted (e.g., binary data, no ATOM lines), attempts
    to re-download from AlphaFold DB.

    Args:
        structure_path: path to the PDB file
        uid: UniProt ID (used for re-download if corrupted)

    Returns:
        Valid path (original or re-downloaded), or None if unrecoverable.
    """
    if not structure_path or not os.path.exists(structure_path):
        return structure_path  # missing file handled elsewhere

    # Quick check: scan first 500 lines for at least one ATOM record
    has_atom = False
    try:
        with open(structure_path) as f:
            for i, line in enumerate(f):
                if i > 500:
                    break
                if line.startswith("ATOM"):
                    has_atom = True
                    break
    except UnicodeDecodeError:
        # Binary data — definitely corrupted
        has_atom = False

    if has_atom:
        return structure_path

    print(f"  WARNING: template {structure_path} appears corrupted (no ATOM records)")

    # Attempt re-download from AlphaFold DB
    if uid:
        import urllib.request
        import urllib.error
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
        try:
            print(f"    Attempting re-download from {url}")
            urllib.request.urlretrieve(url, structure_path)
            # Verify the download
            with open(structure_path) as f:
                for line in f:
                    if line.startswith("ATOM"):
                        print(f"    Re-download successful")
                        return structure_path
            print(f"    Re-downloaded file also invalid")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"    Re-download failed: {e}")
        except Exception as e:
            print(f"    Re-download failed: {e}")

    print(f"    Skipping template for this structure")
    return None


def enforce_constraint_coherence(constraints, ca_map, seq,
                                 max_intercentroid=15.0):
    """Validate and correct spatial coherence of all pocket constraints.

    When constraints for different ligands (cofactor, metal, substrates)
    define spatially separate pockets (inter-centroid distance > threshold),
    this function re-derives the offending constraints to ensure all ligands
    are placed in the same catalytic pocket.

    Approach: iteratively identify the outlier constraint set (highest average
    distance to all other centroids) and re-derive its contacts near the
    majority pocket until all pairwise distances are within threshold.

    For metals with few contacts: replaces contacts with metal-coordinating
    residues (Asp, Glu, His, Ser, Thr, Cys) from the other constraint sets.

    For cofactors/substrates with many contacts: trims to residues near
    the majority pocket centroid, supplemented from other constraint sets.

    Args:
        constraints: list of constraint dicts from boltz_input["constraints"]
        ca_map: dict resnum -> np.array(3) for CA coordinates
        seq: protein sequence string
        max_intercentroid: max allowed inter-centroid distance (default 15.0)

    Returns:
        corrected constraints list
    """
    if not constraints or not ca_map:
        return constraints

    def _extract_pocket_info(constraint_list):
        """Extract pocket constraint indices, binders, and residue sets."""
        info = []
        for i, c in enumerate(constraint_list):
            if "pocket" not in c:
                continue
            pocket = c["pocket"]
            binder = pocket["binder"]
            contacts = pocket.get("contacts", [])
            resids = set()
            for contact in contacts:
                if len(contact) == 2 and contact[0] == "A":
                    resids.add(int(contact[1]))
            if resids:
                info.append((i, binder, resids))
        return info

    def _compute_centroids(info):
        centroids = {}
        for idx, binder, resids in info:
            coords = [ca_map[r] for r in resids if r in ca_map]
            if coords:
                centroids[binder] = np.mean(coords, axis=0)
        return centroids

    def _max_pairwise(centroids):
        binders = list(centroids.keys())
        max_d, max_pair = 0, ("", "")
        for i in range(len(binders)):
            for j in range(i + 1, len(binders)):
                d = float(np.linalg.norm(
                    centroids[binders[i]] - centroids[binders[j]]))
                if d > max_d:
                    max_d = d
                    max_pair = (binders[i], binders[j])
        return max_d, max_pair

    pocket_info = _extract_pocket_info(constraints)
    if len(pocket_info) < 2:
        return constraints

    centroids = _compute_centroids(pocket_info)
    if len(centroids) < 2:
        return constraints

    max_dist, _ = _max_pairwise(centroids)
    if max_dist <= max_intercentroid:
        return constraints  # all coherent

    corrected = list(constraints)  # shallow copy

    # Iteratively fix the worst outlier until all pairs are within threshold
    for _iteration in range(len(pocket_info)):
        pocket_info = _extract_pocket_info(corrected)
        centroids = _compute_centroids(pocket_info)
        max_dist, _ = _max_pairwise(centroids)
        if max_dist <= max_intercentroid:
            break

        # Identify the outlier: highest average distance to all other centroids
        binder_list = list(centroids.keys())
        avg_dists = {}
        for b in binder_list:
            others = [centroids[o] for o in binder_list if o != b]
            if others:
                avg_dists[b] = float(np.mean([
                    np.linalg.norm(centroids[b] - o) for o in others]))
        outlier = max(avg_dists, key=avg_dists.get)
        outlier_dist = avg_dists[outlier]

        # Find the outlier's info
        outlier_entry = None
        for idx, binder, resids in pocket_info:
            if binder == outlier:
                outlier_entry = (idx, binder, resids)
                break
        if outlier_entry is None:
            break

        idx, binder, resids = outlier_entry
        n_contacts = len(resids)

        # Compute the "majority pocket" centroid from all non-outlier constraints
        majority_residues = set()
        majority_coords = []
        for _, b, r in pocket_info:
            if b != outlier:
                majority_residues |= r
                for rr in r:
                    if rr in ca_map:
                        majority_coords.append(ca_map[rr])
        if not majority_coords:
            break
        majority_centroid = np.mean(majority_coords, axis=0)

        is_metal = n_contacts <= 5

        if is_metal:
            # Metal: re-derive from metal-coordinating residues near majority
            new_contacts = []
            for r in sorted(majority_residues):
                if r in ca_map and 1 <= r <= len(seq):
                    aa3 = AA_1TO3.get(seq[r - 1], "")
                    if aa3 in METAL_COORDINATING_3:
                        d = float(np.linalg.norm(ca_map[r] - majority_centroid))
                        if d <= 10.0:
                            new_contacts.append((r, d))

            if new_contacts:
                new_contacts.sort(key=lambda x: x[1])
                new_resids = [r for r, _ in new_contacts[:5]]
            else:
                # Fallback: closest residues from majority set
                dists = [(r, float(np.linalg.norm(ca_map[r] - majority_centroid)))
                         for r in majority_residues if r in ca_map]
                dists.sort(key=lambda x: x[1])
                new_resids = [r for r, _ in dists[:4]]

            print(f"    {binder}: metal constraints relocated "
                  f"({sorted(resids)} -> {new_resids}, "
                  f"was {outlier_dist:.0f}A avg from other ligands)")
        else:
            # Cofactor/substrate: keep residues near majority centroid,
            # supplement from majority set if too few remain
            resid_dists = [(r, float(np.linalg.norm(ca_map[r] - majority_centroid)))
                           for r in resids if r in ca_map]
            resid_dists.sort(key=lambda x: x[1])
            near = [r for r, rd in resid_dists if rd <= max_intercentroid]
            if len(near) < 3:
                supplement = [
                    (r, float(np.linalg.norm(ca_map[r] - majority_centroid)))
                    for r in majority_residues
                    if r in ca_map and r not in near]
                supplement.sort(key=lambda x: x[1])
                near.extend(r for r, _ in supplement[
                    :max(5, n_contacts) - len(near)])
            new_resids = sorted(set(near))
            removed = sorted(resids - set(new_resids))
            if removed:
                print(f"    {binder}: trimmed {len(removed)} distant constraint "
                      f"residues (was {outlier_dist:.0f}A avg from other ligands)")

        # Update the constraint
        new_constraint = dict(corrected[idx])
        new_pocket = dict(new_constraint["pocket"])
        new_pocket["contacts"] = [["A", r] for r in sorted(new_resids)]
        new_constraint["pocket"] = new_pocket
        corrected[idx] = new_constraint

    return corrected


def select_p2rank_pocket(p2rank_entry, reference_residues=None,
                         catalytic_residues=None, min_probability=0.15):
    """Select the best P2Rank pocket, prioritizing catalytic residue coverage.

    Selection priority:
    1. Pocket containing the most catalytic residues (if known)
    2. Pocket with best overlap to reference residues (Foldseek/UniProt)
    3. Highest-probability pocket

    Checks top 3 pockets.

    Args:
        p2rank_entry: dict from p2rank_pockets.json for one protein
        reference_residues: set of residue numbers from existing evidence
        catalytic_residues: set of known catalytic/active site residue numbers
        min_probability: minimum P2Rank probability to consider (default: 0.15)

    Returns:
        list of residue numbers for the selected pocket, or None
    """
    if not p2rank_entry or not p2rank_entry.get("top_pocket"):
        return None

    candidates = p2rank_entry.get("all_pockets", [])[:3]
    if not candidates:
        return None

    candidates = [p for p in candidates if p["probability"] >= min_probability]
    if not candidates:
        return None

    # Priority 1: pocket covering the most catalytic residues
    if catalytic_residues and len(catalytic_residues) > 0:
        cat_set = set(catalytic_residues)
        best_pocket = None
        best_cat_overlap = 0
        for pocket in candidates:
            p_res = set(pocket["residues"])
            cat_overlap = len(p_res & cat_set)
            if cat_overlap > best_cat_overlap:
                best_cat_overlap = cat_overlap
                best_pocket = pocket
        if best_cat_overlap > 0 and best_pocket:
            return best_pocket["residues"]

    # Priority 2: overlap with reference residues
    if reference_residues and len(reference_residues) > 0:
        best_pocket = None
        best_overlap = -1
        for pocket in candidates:
            p_res = set(pocket["residues"])
            overlap = len(p_res & reference_residues)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pocket = pocket
        if best_overlap > 0 and best_pocket:
            return best_pocket["residues"]

    # Priority 3: highest probability pocket
    return candidates[0]["residues"]


def load_sequences(project_dir):
    """Load protein sequences from FASTA and CSV."""
    sequences = {}

    # From FASTA
    fasta_path = os.path.join(project_dir, "all_sequences.fasta")
    if os.path.exists(fasta_path):
        uid, lines = None, []
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if uid and lines:
                        sequences[uid] = "".join(lines)
                    # Handle sp|UID|NAME format
                    header = line.split()[0][1:]
                    parts = header.split("|")
                    uid = parts[1] if len(parts) >= 2 else parts[0]
                    lines = []
                else:
                    lines.append(line)
            if uid and lines:
                sequences[uid] = "".join(lines)

    # From CSV (may have sequences inline)
    csv_path = os.path.join(project_dir, "structure_sequence_summary.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                uid = row["uniprot_id"]
                seq = row.get("sequence", "")
                if seq and seq != "none" and uid not in sequences:
                    sequences[uid] = seq

    return sequences


def load_ca_coords(structure_path):
    """Parse CA coordinates from a PDB file (chain A implicit).

    Returns dict resnum -> np.array(3). Empty dict if file missing/unreadable.
    """
    residue_ca = {}
    if not structure_path or not os.path.exists(structure_path):
        return residue_ca
    try:
        with open(structure_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    resnum = int(line[22:26].strip())
                    residue_ca[resnum] = np.array([
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ])
    except Exception:
        return {}
    return residue_ca


def residues_within_radius(anchor_residues, ca_map, radius):
    """Return set of residue numbers whose CA is within `radius` Å of any
    anchor residue's CA. Anchor residues themselves are included if present."""
    if not anchor_residues or not ca_map:
        return set(anchor_residues or [])
    anchor_coords = np.array([ca_map[r] for r in anchor_residues if r in ca_map])
    if len(anchor_coords) == 0:
        return set(anchor_residues)
    out = set(anchor_residues)
    for resnum, ca in ca_map.items():
        if np.linalg.norm(anchor_coords - ca, axis=1).min() < radius:
            out.add(resnum)
    return out


def densest_local_cluster(residues, ca_map, radius=10.0):
    """For a large residue set, find the spatially-densest sub-cluster.

    Picks the residue with the most neighbors (from the set) within `radius`,
    and returns all set members within `radius` of that anchor. Used to trim
    aggregated foldseek pocket_lining to the actual binding-site core.
    """
    candidates = [r for r in residues if r in ca_map]
    if not candidates:
        return set(residues)
    coords = np.array([ca_map[r] for r in candidates])
    res_set = set(residues)
    best_res, best_n = candidates[0], 0
    for i, r in enumerate(candidates):
        n_in_set = int(np.sum(np.linalg.norm(coords - coords[i], axis=1) <= radius))
        if n_in_set > best_n:
            best_n, best_res = n_in_set, r
    return residues_within_radius([best_res], ca_map, radius) & res_set


def expand_contacts_spatially(contact_residues, structure_path, radius=8.0):
    """Expand contact residues to include neighbors within `radius` Å.

    Kept for compatibility; new substrate-constraint logic uses
    residues_within_radius() directly with a cached ca_map.
    """
    if not contact_residues:
        return contact_residues
    ca_map = load_ca_coords(structure_path)
    if not ca_map:
        return contact_residues
    return sorted(residues_within_radius(contact_residues, ca_map, radius))


def select_tight_constraints(candidate_residues, ca_map, target=10,
                             max_spread=18.0, catalytic_residues=None,
                             pocket_centroid_ref=None,
                             catalytic_max_dist=12.0):
    """Select a tight, spatially coherent constraint set.

    1. If catalytic_residues provided and pocket_centroid_ref available,
       force-include only those within catalytic_max_dist of the centroid.
    2. Sort remaining candidates by distance to centroid.
    3. Fill up to target count.
    4. Trim if spread exceeds max_spread.

    Args:
        candidate_residues: list of residue numbers (from evidence pipeline)
        ca_map: dict resnum -> np.array(3) for CA coordinates
        target: target number of constraint residues (default 10)
        max_spread: max pairwise CA distance allowed (default 18.0)
        catalytic_residues: list of catalytic residue numbers to force-include
        pocket_centroid_ref: np.array(3) reference centroid for filtering
            catalytic residues (cofactor centroid for synthetases, None for hydrolases)
        catalytic_max_dist: max distance from pocket centroid for a catalytic
            residue to be force-included (default 12.0)

    Returns:
        (sorted_residues, spread, n_cat_kept, n_cat_dropped)
    """
    available = [r for r in candidate_residues if r in ca_map]
    if len(available) <= 3:
        return sorted(available), 0, 0, 0

    coords = np.array([ca_map[r] for r in available])
    centroid = np.mean(coords, axis=0)

    # Filter catalytic residues by distance to the consensus pocket centroid
    forced = set()
    n_cat_dropped = 0
    if catalytic_residues and pocket_centroid_ref is not None:
        for r in catalytic_residues:
            if r in ca_map:
                d = np.linalg.norm(ca_map[r] - pocket_centroid_ref)
                if d <= catalytic_max_dist:
                    forced.add(r)
                else:
                    n_cat_dropped += 1
    elif catalytic_residues:
        # No reference centroid — include all catalytic residues that are
        # in the candidate set (conservative: don't add external ones)
        for r in catalytic_residues:
            if r in ca_map and r in set(available):
                forced.add(r)

    # Sort non-forced candidates by distance to centroid
    non_forced = [r for r in available if r not in forced]
    dists = [(r, np.linalg.norm(ca_map[r] - centroid)) for r in non_forced]
    dists.sort(key=lambda x: x[1])

    # Build set: forced first, then fill with closest
    n_free = max(0, target - len(forced))
    selected = sorted(forced | set(r for r, _ in dists[:n_free]))

    # Compute spread and trim if needed
    def _spread(residues):
        if len(residues) < 2:
            return 0
        c = np.array([ca_map[r] for r in residues])
        return max(np.linalg.norm(c[i] - c[j])
                   for i in range(len(c)) for j in range(i + 1, len(c)))

    spread = _spread(selected)
    while spread > max_spread and len(selected) > max(5, len(forced)):
        # Remove the farthest non-forced residue from centroid
        non_f = [r for r in selected if r not in forced]
        if not non_f:
            break
        farthest = max(non_f, key=lambda r: np.linalg.norm(ca_map[r] - centroid))
        selected.remove(farthest)
        spread = _spread(selected)

    return sorted(selected), round(spread, 1), len(forced), n_cat_dropped


def build_cofold_yaml(uid, seq, substrates, pocket_info=None,
                      use_pocket=True, max_distance=6.0, predict_affinity=False,
                      use_msa_server=False, cofactor_info=None,
                      catalytic_metals=None, p2rank_residues=None,
                      catalytic_residues=None,
                      substrate_contact_residues=None,
                      structure_path=None):
    """Build a single Boltz-2 cofolding YAML dict.

    Supports two cofolding strategies:

    1. **Cofactor-guided** (when cofactor_info provided):
       Protein + cofactor (CCD code) + catalytic metals (CCD) + substrate(s).
       Pocket constraint anchors the cofactor and metals to their binding sites.
       Substrates bind freely near the cofactor — more physically realistic
       for cofactor-dependent reactions (e.g., ATP-dependent synthetases).

    2. **Blind cofolding** (no cofactor, no pocket):
       Protein + substrate(s) with no constraints.
       Appropriate for hydrolases and other cofactor-free reactions.

    3. **Pocket-constrained** (legacy, when use_pocket=True and no cofactor):
       Protein + substrate(s), substrate anchored to detected pocket.

    Args:
        uid: UniProt ID
        seq: protein sequence
        substrates: list of dicts with 'name' and 'smiles'
        pocket_info: dict from pocket_analysis.json (or None)
        use_pocket: whether to add pocket constraints (fallback when no cofactor)
        max_distance: max distance in Angstroms for pocket constraint (4-20, default 6)
        predict_affinity: whether to add affinity prediction
        use_msa_server: use MSA server instead of empty MSA
        cofactor_info: dict with cofactor data from pocket analysis, e.g.:
            {"ccd_code": "ATP", "contact_residues": [145, 147, ...],
             "n_contacts": 12, "source_pdb": "7TGM"}
        catalytic_metals: list of metal dicts from pocket analysis, e.g.:
            [{"ccd_code": "MG", "contact_residues": [161, 164], "n_contacts": 2}]

    Returns:
        (boltz_input_dict, metadata_dict)
    """
    # Chain IDs: protein is A, ligands use B-Z then 0-9 (max 35 ligand chains)
    _CHAIN_IDS = [chr(c) for c in range(ord("B"), ord("Z") + 1)] + [str(d) for d in range(10)]
    chain_id_idx = 0

    def next_chain_id():
        nonlocal chain_id_idx
        if chain_id_idx >= len(_CHAIN_IDS):
            raise ValueError(f"Too many ligand chains for {uid}: exceeded {len(_CHAIN_IDS)} chain limit")
        cid = _CHAIN_IDS[chain_id_idx]
        chain_id_idx += 1
        return cid

    msa_value = "empty" if not use_msa_server else None
    protein_entry = {"id": "A", "sequence": seq}
    if msa_value:
        protein_entry["msa"] = msa_value

    boltz_input = {
        "version": 1,
        "sequences": [{"protein": protein_entry}],
    }

    # Add AlphaFold structure as template to fix protein fold.
    # This lets Boltz-2 focus on ligand placement rather than re-folding.
    # Validate template first — corrupted PDBs (binary data, no ATOM lines)
    # cause Boltz-2 preprocessing to silently fail.
    if structure_path and os.path.exists(structure_path):
        validated_path = validate_template(structure_path, uid=uid)
        if validated_path:
            structure_path = validated_path
            boltz_input["templates"] = [{"pdb": structure_path}]
        else:
            structure_path = None  # skip template, proceed without it

    meta = {"has_constraint": False, "n_contacts": 0, "evidence_type": "none",
            "strategy": "blind", "cofactor_ccd": None, "metal_ccds": []}
    if catalytic_metals is None:
        catalytic_metals = []

    # ── Strategy 1: Cofactor-guided cofolding ──
    # Include cofactor as explicit CCD-based ligand chain, anchor it to binding site
    cofactor_chain_id = None
    if cofactor_info and cofactor_info.get("ccd_code"):
        source_ccd = cofactor_info["ccd_code"]
        # Map crystallographic analogs / product states to physiological cofactor.
        # Contact residues from AlphaFill are preserved (same binding pocket),
        # but we use the pre-catalytic cofactor for cofolding.
        ccd_code = COFACTOR_PHYSIOLOGICAL_MAP.get(source_ccd, source_ccd)
        cofactor_chain_id = next_chain_id()

        # Add cofactor via CCD code (Boltz-2 resolves this from its built-in dictionary)
        boltz_input["sequences"].append({
            "ligand": {
                "id": cofactor_chain_id,
                "ccd": ccd_code,
            }
        })

        # Add pocket constraint on the cofactor chain
        contact_residues = cofactor_info.get("contact_residues", [])
        if contact_residues:
            contacts = [["A", r] for r in contact_residues]
            boltz_input["constraints"] = [{
                "pocket": {
                    "binder": cofactor_chain_id,
                    "contacts": contacts,
                    "max_distance": max_distance,
                    "force": True,
                }
            }]
            meta["has_constraint"] = True
            meta["n_contacts"] = len(contacts)
            meta["evidence_type"] = "cofactor_transplant"

        meta["strategy"] = "cofactor_guided"
        meta["cofactor_ccd"] = ccd_code
        meta["cofactor_source_ccd"] = source_ccd  # what AlphaFill found

    # ── Add catalytic metal ions as CCD chains ──
    # Metals (e.g., Mg2+ for ATP-dependent enzymes) are included as separate
    # chains with pocket constraints to their AlphaFill-derived binding sites
    for metal in catalytic_metals:
        metal_ccd = metal.get("ccd_code")
        if not metal_ccd:
            continue
        metal_chain_id = next_chain_id()

        boltz_input["sequences"].append({
            "ligand": {
                "id": metal_chain_id,
                "ccd": metal_ccd,
            }
        })

        # Constrain metal to its known binding site
        metal_contacts = metal.get("contact_residues", [])
        if metal_contacts:
            if "constraints" not in boltz_input:
                boltz_input["constraints"] = []
            boltz_input["constraints"].append({
                "pocket": {
                    "binder": metal_chain_id,
                    "contacts": [["A", r] for r in metal_contacts],
                    "max_distance": max_distance,
                    "force": True,
                }
            })
            meta["has_constraint"] = True

        meta["metal_ccds"].append(metal_ccd)

    # Add substrate ligands with pocket constraints to guide placement
    # into the binding pocket. Without constraints, small substrates
    # (e.g., adipic acid) can land on random surface patches far from
    # the active site, especially for cofactor-guided mode where only
    # the cofactor is anchored.
    substrate_ids = []
    for s in substrates:
        lig_id = next_chain_id()
        boltz_input["sequences"].append({
            "ligand": {
                "id": lig_id,
                "smiles": s["smiles"],
            }
        })
        substrate_ids.append(lig_id)

    # ── Substrate pocket constraints ──
    #
    # Strategy: gather all candidate residues from available evidence,
    # then apply tight spatial selection (cap ~8-10 residues, max 18A spread)
    # with consensus-weighted pocket centroid to filter catalytic residues.
    #
    # Evidence sources (same as before, combined into candidate pool):
    #   1. Catalytic residues (UniProt/homolog-transferred)
    #   2. Alphafill/foldseek binding pocket / cofactor contacts
    #   3. P2Rank (ML) — last resort
    #
    # Key improvements over previous approach:
    #   - force: true on all pocket constraints (gradient-enforced)
    #   - Template provided (protein fold is fixed)
    #   - Tight spatial selection removes distant/allosteric residues
    #   - Consensus-weighted centroid avoids arbitrary evidence ordering
    # Load CA coordinates for spatial filtering (used by substrate constraint
    # selection and by the post-hoc coherence enforcement step)
    ca_map = load_ca_coords(structure_path) if structure_path else {}

    if use_pocket:
        sub_contacts = None
        sub_source = None

        # Get pocket lining if available
        pocket_lining = None
        pocket_ev_type = "none"
        if pocket_info:
            bp = pocket_info.get("binding_pocket")
            evidence = pocket_info.get("evidence", {})
            pocket_ev_type = evidence.get("type", "none")
            meta["evidence_type"] = pocket_ev_type
            if bp and not evidence.get("warning"):
                lining = bp.get("lining_residues", [])
                if 0 < len(lining) <= 150:
                    pocket_lining = [r[1] for r in lining]

        # pocket_ref is set later: cofactor centroid for synthetases,
        # or None for hydrolases (select_tight_constraints computes its
        # own centroid from the candidate pool).
        pocket_ref = None

        # Gather candidate residues from available evidence
        candidate_pool = set()

        # From catalytic residues (will be filtered by tight selection)
        cat_for_filter = None
        if catalytic_residues:
            cat_set = set(catalytic_residues)
            # Guard against mis-annotated large catalytic lists (>15 residues)
            if len(cat_set) > 15:
                pocket_set = set(pocket_lining) if pocket_lining else set()
                validated = cat_set & pocket_set
                if validated:
                    cat_set = validated
                if len(cat_set) > 15 and ca_map:
                    cat_set = densest_local_cluster(cat_set, ca_map, radius=8.0)
            cat_for_filter = sorted(cat_set)
            candidate_pool.update(cat_set)

        # From cofactor contacts (tight, ligand-derived)
        if cofactor_chain_id and cofactor_info:
            raw = cofactor_info.get("contact_residues", [])
            if raw:
                candidate_pool.update(raw)

        # From active site residues (homolog-transferred)
        if pocket_info:
            asr = (pocket_info.get("evidence") or {}).get("active_site_residues", [])
            if asr:
                candidate_pool.update(asr)

        # From pocket lining (geometry-based) — ONLY when no cofactor
        # evidence exists.  When cofactor contacts are available, pocket
        # lining adds 50-188 geometry-only residues that dilute the
        # experimentally validated active-site signal and can pull
        # substrate constraints far from the cofactor pocket.
        if pocket_lining and not cofactor_chain_id:
            candidate_pool.update(pocket_lining)

        # From P2Rank (ML fallback)
        if p2rank_residues and not candidate_pool:
            candidate_pool.update(p2rank_residues)

        # When cofactor contacts exist, use their centroid as the pocket
        # reference and filter all candidates to within 15 A.  This
        # ensures substrate constraints stay inside the experimentally
        # validated binding pocket.
        if cofactor_chain_id and cofactor_info and ca_map:
            cof_contacts = cofactor_info.get("contact_residues", [])
            cof_coords = [ca_map[r] for r in cof_contacts if r in ca_map]
            if cof_coords:
                cofactor_centroid = np.mean(cof_coords, axis=0)
                filtered = {r for r in candidate_pool
                            if r in ca_map
                            and np.linalg.norm(ca_map[r] - cofactor_centroid) <= 15.0}
                if len(filtered) >= 3:
                    n_removed = len(candidate_pool) - len(filtered)
                    if n_removed > 0:
                        candidate_pool = filtered
                pocket_ref = cofactor_centroid

        # Apply tight spatial selection
        if candidate_pool and ca_map:
            tight_target = 8 if cofactor_chain_id else 10
            sub_contacts, spread, n_cat_kept, n_cat_dropped = \
                select_tight_constraints(
                    sorted(candidate_pool), ca_map,
                    target=tight_target, max_spread=18.0,
                    catalytic_residues=cat_for_filter,
                    pocket_centroid_ref=pocket_ref,
                    catalytic_max_dist=12.0,
                )
            sub_source = f"tight({len(sub_contacts)}/{len(candidate_pool)},sp={spread}A)"
            if n_cat_kept > 0:
                sub_source += f",cat={n_cat_kept}"
        elif candidate_pool:
            # No CA map — use candidates as-is (can't do spatial selection)
            sub_contacts = sorted(candidate_pool)
            sub_source = "no_structure"

        # Apply substrate constraints with force: true
        if sub_contacts and substrate_ids:
            if "constraints" not in boltz_input:
                boltz_input["constraints"] = []
            contacts = [["A", r] for r in sub_contacts]
            for lig_id in substrate_ids:
                boltz_input["constraints"].append({
                    "pocket": {
                        "binder": lig_id,
                        "contacts": contacts,
                        "max_distance": max_distance,
                        "force": True,
                    }
                })

            # Update metadata
            if not cofactor_chain_id:
                meta["has_constraint"] = True
                meta["n_contacts"] = len(contacts)
                meta["strategy"] = "pocket_constrained"
                meta["pocket_source"] = sub_source

    # If still no constraint, it's blind cofolding
    if not meta["has_constraint"]:
        meta["strategy"] = "blind"

    # ── Enforce spatial coherence across all ligand constraints ──
    # Cofactor, metal, and substrate constraints are derived from different
    # evidence sources and can end up defining spatially separate pockets.
    # This step validates inter-centroid distances and re-derives any
    # incoherent constraints to ensure all ligands land in the same active site.
    if ca_map and boltz_input.get("constraints"):
        boltz_input["constraints"] = enforce_constraint_coherence(
            boltz_input["constraints"], ca_map, seq,
            max_intercentroid=15.0,
        )

    # Add affinity prediction
    if predict_affinity:
        boltz_input["properties"] = []
        # Predict affinity for substrates (not for cofactor)
        for lig_id in substrate_ids:
            boltz_input["properties"].append({
                "affinity": {"binder": lig_id}
            })

    return boltz_input, meta


def main():
    parser = argparse.ArgumentParser(description="Prepare Boltz-2 cofolding inputs")
    parser.add_argument("project_dir", help="Project directory with config.yaml")
    parser.add_argument("--no-pocket-constraint", action="store_true",
                        help="Don't add pocket constraints (blind cofolding)")
    parser.add_argument("--affinity", action="store_true",
                        help="Request Boltz-2 binding affinity prediction")
    parser.add_argument("--max-distance", type=float, default=6.0,
                        help="Max contact distance in Angstroms for pocket (default: 6.0)")
    parser.add_argument("--use-msa-server", action="store_true",
                        help="Use MSA server instead of single-sequence mode")
    parser.add_argument("-o", "--output-dir", default="cofold_input",
                        help="Output directory for YAML files (default: cofold_input)")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)

    # Load config
    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get substrates from config
    substrates = config.get("target_reaction", {}).get("substrates", [])
    if not substrates:
        sys.exit("ERROR: No substrates defined in config.yaml under target_reaction.substrates.\n"
                 "Example:\n"
                 "  target_reaction:\n"
                 "    substrates:\n"
                 "      - name: hexamethylenediamine\n"
                 "        smiles: \"NCCCCCCN\"")

    use_pocket = not args.no_pocket_constraint

    # Determine relevant ligands (cofactors) from config
    relevant_ligands = config.get("target_reaction", {}).get("relevant_ligands")
    has_cofactors = relevant_ligands is not None and len(relevant_ligands) > 0
    config_catalytic_metals = config.get("target_reaction", {}).get("catalytic_metals", [])

    # Auto-extend cofactor and metal maps from config (enables new enzyme families
    # without modifying Python code)
    target_rx = config.get("target_reaction", {})
    for ccd in target_rx.get("additional_cofactor_ccds", []):
        COFACTOR_CCD_CODES.add(ccd)
    for src, dst in target_rx.get("cofactor_mappings", {}).items():
        COFACTOR_PHYSIOLOGICAL_MAP[src] = dst
    for src, dst in target_rx.get("metal_mappings", {}).items():
        METAL_PHYSIOLOGICAL_MAP[src] = dst
    if relevant_ligands:
        for lig in relevant_ligands:
            COFACTOR_CCD_CODES.add(lig)

    print("Prepare Boltz-2 Cofolding Inputs")
    print(f"Project: {project_dir}")
    print(f"Substrates:")
    for s in substrates:
        print(f"  {s['name']}: {s['smiles']}")
    if has_cofactors:
        print(f"Cofactor-guided mode: cofactors {relevant_ligands} will be included as explicit chains")
    else:
        print(f"Blind cofolding mode: no cofactors (hydrolase/cofactor-free reaction)")
    if config_catalytic_metals:
        print(f"Catalytic metals: {config_catalytic_metals} will be included as explicit chains")
    print(f"Pocket constraints: {'disabled' if not use_pocket else f'enabled (max_distance={args.max_distance} A)'}")
    print(f"Affinity prediction: {'enabled' if args.affinity else 'disabled'}")
    print(f"MSA mode: {'server' if args.use_msa_server else 'single-sequence (empty)'}")
    print(f"{'='*60}\n")

    # Load pocket analysis
    pocket_path = os.path.join(project_dir, "pocket_analysis.json")
    pocket_data = {}
    if os.path.exists(pocket_path):
        with open(pocket_path) as f:
            pocket_data = json.load(f)
        print(f"Loaded pocket analysis for {len(pocket_data)} structures")
        n_with_cofactor = sum(1 for v in pocket_data.values() if v.get("best_cofactor"))
        if has_cofactors:
            print(f"  {n_with_cofactor} structures have cofactor binding site info")
        print()
    elif use_pocket or has_cofactors:
        print("WARNING: pocket_analysis.json not found. Run pocket_analysis.py first.")
        print("         Proceeding with blind cofolding (no pocket constraints).\n")

    # Load P2Rank pocket predictions (fallback for candidates without
    # AlphaFill cofactor data or with weak binding site evidence)
    p2rank_path = os.path.join(project_dir, "p2rank_pockets.json")
    p2rank_data = {}
    if os.path.exists(p2rank_path):
        with open(p2rank_path) as f:
            p2rank_data = json.load(f)
        print(f"Loaded P2Rank predictions for {len(p2rank_data)} structures")
        n_high_conf = sum(1 for v in p2rank_data.values()
                         if v.get("top_pocket") and v["top_pocket"]["probability"] > 0.5)
        print(f"  {n_high_conf} structures have high-confidence P2Rank pockets (prob > 0.5)")
        print()

    # Load catalytic site validation (for pocket quality checking).
    # Auto-generate if missing by running identify_catalytic_sites.py.
    cat_val_path = os.path.join(project_dir, "catalytic_site_validation.json")
    if not os.path.exists(cat_val_path):
        print("catalytic_site_validation.json not found — generating automatically...")
        cat_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "identify_catalytic_sites.py")
        if os.path.exists(cat_script):
            import subprocess as _sp
            _result = _sp.run(
                [sys.executable, cat_script, project_dir, "--skip-enhance"],
                cwd=os.path.dirname(cat_script),
            )
            if _result.returncode != 0:
                print("  Warning: catalytic site identification failed; "
                      "proceeding without catalytic data.", file=sys.stderr)
            elif os.path.exists(cat_val_path):
                print(f"  Generated {cat_val_path}")
        else:
            print(f"  Warning: {cat_script} not found; skipping.", file=sys.stderr)

    cat_val_data = {}
    if os.path.exists(cat_val_path):
        with open(cat_val_path) as f:
            cat_val_data = json.load(f)
        # Use core_residues from UniProt direct annotations only.
        # FoldSeek homology-transferred catalytic residues are excluded
        # because structural alignment at low sequence identity produces
        # unreliable residue mappings (wrong AA types, non-compact triads).
        # Ignore shell_residues which are Binding/Metal site features —
        # those belong in the pocket-evidence tier, not as the catalytic
        # anchor.
        n_foldseek_dropped = 0
        for uid_key, val in list(cat_val_data.items()):
            if (val.get("evidence_tier") == "foldseek_homology"
                    and val.get("core_residues")):
                val["core_residues"] = []
                val["catalytic_residues"] = []
                n_foldseek_dropped += 1
        n_with_cat = sum(1 for v in cat_val_data.values()
                         if v.get("core_residues"))
        print(f"Loaded catalytic site validation for {len(cat_val_data)} structures")
        print(f"  {n_with_cat} have known catalytic residues")
        if n_foldseek_dropped:
            print(f"  {n_foldseek_dropped} foldseek-transferred annotations dropped (unreliable)")
        print()

    # Load sequences
    sequences = load_sequences(project_dir)
    print(f"Loaded {len(sequences)} protein sequences\n")

    if not sequences:
        sys.exit("ERROR: No sequences found. Check all_sequences.fasta or "
                 "structure_sequence_summary.csv.")

    # Output directory
    output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Determine which UIDs to process:
    # - If pocket data available and constraints enabled: all UIDs with sequences
    #   (those with pocket data get constraints, others get blind cofolding)
    # - If no pocket data or constraints disabled: all UIDs with sequences
    uids_to_process = sorted(sequences.keys())

    n_written = 0
    n_cofactor_guided = 0
    n_pocket_constrained = 0
    n_blind = 0

    for uid in uids_to_process:
        seq = sequences[uid]
        if not seq:
            print(f"  Warning: {uid} has empty sequence, skipping", file=sys.stderr)
            continue

        pocket_info = pocket_data.get(uid)

        # Extract cofactor info from pocket analysis (if available)
        cofactor_info = None
        if has_cofactors and pocket_info:
            # Primary: use pre-computed best_cofactor field (from updated pocket_analysis.py)
            best_cof = pocket_info.get("best_cofactor")
            if best_cof and best_cof.get("ccd_code") in COFACTOR_CCD_CODES:
                cofactor_info = best_cof
            else:
                # Fallback: extract cofactor from evidence.ligand_contacts directly
                # (works with older pocket_analysis.json that lack best_cofactor)
                evidence = pocket_info.get("evidence", {})
                lig_contacts = evidence.get("ligand_contacts", {})
                if lig_contacts:
                    best_ccd = None
                    best_n = 0
                    for lig_name, contacts in lig_contacts.items():
                        if lig_name in relevant_ligands and lig_name in COFACTOR_CCD_CODES:
                            if len(contacts) > best_n:
                                best_ccd = lig_name
                                best_n = len(contacts)
                    if best_ccd:
                        details = evidence.get("ligands", {}).get(best_ccd, {})
                        cofactor_info = {
                            "ccd_code": best_ccd,
                            "contact_residues": sorted(lig_contacts[best_ccd]),
                            "n_contacts": best_n,
                            "source_pdb": details.get("pdb_id", ""),
                        }

        # Extract catalytic metals from pocket analysis.
        # When a cofactor is present: only include metals whose binding site
        # overlaps the cofactor site (e.g., Mg2+ coordinating ATP phosphates),
        # filtering out crystal packing / non-catalytic metal sites.
        # When no cofactor: include all metals from the pocket that match
        # configured catalytic_metals (cofactor-free metalloenzymes).
        metals_for_cofold = []
        if config_catalytic_metals and pocket_info:
            cofactor_contacts = set(cofactor_info.get("contact_residues", [])) if cofactor_info else set()
            seen_sites = set()  # track contact sets to remove redundant metals

            for metal in pocket_info.get("catalytic_metals", []):
                metal_contacts = set(metal.get("contact_residues", []))
                if not metal_contacts:
                    continue

                # If cofactor exists, require overlap with cofactor binding site
                if cofactor_contacts:
                    overlap = cofactor_contacts & metal_contacts
                    if not overlap:
                        continue

                # Skip redundant metals at the same site (e.g., MN occupying
                # the same position as MG — alternative occupancy in crystal)
                site_key = frozenset(metal_contacts)
                if site_key in seen_sites:
                    continue
                seen_sites.add(site_key)

                metals_for_cofold.append(metal)

            # Fallback: if cofactor exists but no AlphaFill metal overlaps it,
            # infer metal contacts from metal-coordinating residues (Asp, Glu,
            # His, Ser, Thr, Cys) among the cofactor anchor residues.
            # This ensures Mg2+ is placed near ATP phosphates even when
            # AlphaFill returns metals from a different domain or no metals.
            if not metals_for_cofold and cofactor_contacts and cofactor_info:
                inferred = []
                for resnum in sorted(cofactor_contacts):
                    if 1 <= resnum <= len(seq):
                        aa3 = AA_1TO3.get(seq[resnum - 1], "")
                        if aa3 in METAL_COORDINATING_3:
                            inferred.append(resnum)
                if inferred:
                    # Limit to 5 contacts, preferring acidic residues
                    if len(inferred) > 5:
                        def _priority(rn):
                            aa3 = AA_1TO3.get(seq[rn - 1], "")
                            return 0 if aa3 in ("ASP", "GLU") else (1 if aa3 == "HIS" else 2)
                        inferred.sort(key=_priority)
                        inferred = inferred[:5]
                    for target_metal in sorted(set(
                            METAL_PHYSIOLOGICAL_MAP.get(m, m) for m in config_catalytic_metals)):
                        metals_for_cofold.append({
                            "ccd_code": target_metal,
                            "contact_residues": sorted(inferred),
                        })
                        print(f"    {uid}: inferred {target_metal} contacts from cofactor "
                              f"pocket ({len(inferred)} metal-coordinating residues)")

        # Select P2Rank pocket as fallback for candidates without cofactor
        # or with weak pocket evidence. Prioritizes pockets that contain
        # known catalytic residues, then overlap with existing evidence.
        p2rank_residues = None
        if uid in p2rank_data and not cofactor_info:
            # Get existing pocket residues as reference
            ref_residues = None
            if pocket_info:
                bp = pocket_info.get("binding_pocket") or {}
                lining = bp.get("lining_residues", [])
                if lining:
                    ref_residues = set(r[1] for r in lining)

            # Get catalytic residues for prioritized pocket selection.
            # Use core_residues only (true Active site), not the mixed
            # catalytic_residues field.
            cat_residues = None
            if uid in cat_val_data:
                cat_res = cat_val_data[uid].get("core_residues", [])
                if cat_res:
                    cat_residues = set(cat_res)
            else:
                print(f"  Warning: {uid} has no catalytic site data; P2Rank pocket selection uses probability only",
                      file=sys.stderr)

            p2rank_residues = select_p2rank_pocket(
                p2rank_data[uid],
                reference_residues=ref_residues,
                catalytic_residues=cat_residues,
            )

        # Get catalytic residues for this candidate (core = true Active site
        # residues only; shell residues are excluded).
        uid_cat_residues = None
        if uid in cat_val_data:
            cr = cat_val_data[uid].get("core_residues", [])
            if cr:
                uid_cat_residues = cr

        # Locate structure file (used for spatial trimming of pocket constraints)
        struct_path = None
        for sdir in ["alphafold_structures", "pdb_structures"]:
            candidates = glob.glob(os.path.join(project_dir, sdir, f"*{uid}*.pdb"))
            if candidates:
                struct_path = candidates[0]
                break

        boltz_input, meta = build_cofold_yaml(
            uid=uid,
            seq=seq,
            substrates=substrates,
            pocket_info=pocket_info,
            use_pocket=use_pocket,
            max_distance=args.max_distance,
            predict_affinity=args.affinity,
            use_msa_server=args.use_msa_server,
            cofactor_info=cofactor_info,
            catalytic_metals=metals_for_cofold,
            p2rank_residues=p2rank_residues,
            catalytic_residues=uid_cat_residues,
            structure_path=struct_path,
        )

        # Write YAML
        yaml_path = os.path.join(output_dir, f"{uid}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(boltz_input, f, default_flow_style=False, sort_keys=False)
        n_written += 1

        strategy = meta["strategy"]
        metal_str = f" + metals {meta['metal_ccds']}" if meta["metal_ccds"] else ""
        if strategy == "cofactor_guided":
            n_cofactor_guided += 1
            source_ccd = meta.get("cofactor_source_ccd", meta["cofactor_ccd"])
            mapped_str = f" (from {source_ccd})" if source_ccd != meta["cofactor_ccd"] else ""
            print(f"  {uid}: {len(seq)} aa, cofactor {meta['cofactor_ccd']}{mapped_str}{metal_str} + "
                  f"{len(substrates)} substrates ({meta['n_contacts']} contacts)")
        elif strategy == "pocket_constrained":
            n_pocket_constrained += 1
            source = meta.get("pocket_source", meta.get("evidence_type", "unknown"))
            print(f"  {uid}: {len(seq)} aa, {len(substrates)} ligands, "
                  f"pocket constraint ({meta['n_contacts']} contacts, {source})")
        else:
            n_blind += 1
            print(f"  {uid}: {len(seq)} aa, {len(substrates)} ligands, blind cofolding")

    print(f"\n{'='*60}")
    print(f"Written: {n_written} YAML files to {output_dir}")
    if n_cofactor_guided:
        print(f"  Cofactor-guided:     {n_cofactor_guided}")
    if n_pocket_constrained:
        print(f"  Pocket-constrained:  {n_pocket_constrained}")
    print(f"  Blind cofolding:     {n_blind}")
    print(f"\nTo run cofolding:")
    print(f"  python run_cofold.py {project_dir}")


if __name__ == "__main__":
    main()
