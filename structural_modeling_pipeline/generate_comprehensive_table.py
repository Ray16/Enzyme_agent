#!/usr/bin/env python3
"""
Generate a comprehensive structural analysis table for all amide synthetase
and hydrolase candidates, merging data from:
  - structure_sequence_summary.csv  (organism, sequence length)
  - pocket_analysis.json            (pocket geometry, lining residues, evidence)
  - catalytic_site_validation.json  (catalytic residues, core/shell)
  - cofolding_metrics.csv           (Boltz-2 cofolding quality metrics)
  - apo_pocket_chemistry.csv        (residue chemistry fractions)
  - pocket_volume_analysis.csv      (volume ratios, flags)
  - holo_pocket_analysis.csv        (holo pocket geometry from cofolded structures)
  - holo_pocket_chemistry.csv       (holo pocket residue chemistry)
  - holo_vs_apo_comparison.csv      (apo vs holo volume/residue comparison)

Output: analysis_results/comprehensive_enzyme_table.csv
"""

import csv
import glob
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SYN_DIR = BASE / "amide_synthetase"
HYD_DIR = BASE / "amide_hydrolase"
ANALYSIS = BASE / "analysis_results"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def fmt_residues(residue_list: list, style: str = "compact") -> str:
    """Format residue positions.
    compact: 'H37, M30, G31' (3-letter + position)
    positions_only: '37, 30, 31'
    """
    if not residue_list:
        return ""
    if style == "compact" and isinstance(residue_list[0], (list, tuple)):
        # lining_residues format: [[RES_NAME, pos, chain], ...]
        return "; ".join(f"{r[0]}{r[1]}" for r in residue_list)
    # catalytic_residues format: [pos1, pos2, ...]
    return "; ".join(str(r) for r in sorted(set(residue_list)))


def fmt_pct(val):
    """Format a float as percentage string."""
    if val is None or val == "":
        return ""
    try:
        return f"{float(val)*100:.1f}%"
    except (ValueError, TypeError):
        return str(val)


def fmt_float(val, decimals=3):
    if val is None or val == "":
        return ""
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def classify_residue(three_letter: str) -> str:
    """Classify amino acid as charged/hydrophobic/polar."""
    charged = {"ARG", "LYS", "ASP", "GLU", "HIS"}
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"}
    # polar = everything else (SER, THR, ASN, GLN, TYR, CYS, GLY)
    if three_letter in charged:
        return "charged"
    elif three_letter in hydrophobic:
        return "hydrophobic"
    return "polar"


def compute_pocket_chemistry(lining_residues: list) -> dict:
    """Compute charged/hydrophobic/polar fractions from lining residues."""
    if not lining_residues:
        return {"frac_charged": "", "frac_hydrophobic": "", "frac_polar": ""}
    counts = {"charged": 0, "hydrophobic": 0, "polar": 0}
    for r in lining_residues:
        counts[classify_residue(r[0])] += 1
    total = len(lining_residues)
    return {
        "frac_charged": f"{counts['charged']/total*100:.1f}%",
        "frac_hydrophobic": f"{counts['hydrophobic']/total*100:.1f}%",
        "frac_polar": f"{counts['polar']/total*100:.1f}%",
    }


def pocket_confidence(ev: dict) -> str:
    """Classify pocket confidence based on evidence quality and pocket overlap.

    Two axes combined:
      - Evidence quality: how reliable is the active site identification?
        (direct source vs foldseek-mediated, TM-score)
      - Pocket match: does the detected pocket capture the active site residues?
        (overlap fraction, number of residues)

    Labels match generate_pocket_html.py exactly.
    """
    etype = ev.get("type", "none")
    if etype == "none":
        return "Unknown Confidence"

    tm = ev.get("tmscore")
    if tm is not None:
        try:
            tm = float(tm)
        except (ValueError, TypeError):
            tm = None

    total = ev.get("active_site_total", 0)
    overlap = ev.get("active_site_overlap", 0)
    try:
        total = int(total) if total != "" else 0
        overlap = int(overlap) if overlap != "" else 0
    except (ValueError, TypeError):
        total, overlap = 0, 0
    n_res = len(ev.get("active_site_residues", []))
    overlap_frac = overlap / total if total else 0.0

    is_direct = etype in ("alphafill", "uniprot_annotation")
    good_source = is_direct or (tm is not None and tm >= 0.8)
    moderate_source = not is_direct and tm is not None and 0.6 <= tm < 0.8

    if good_source and overlap_frac >= 0.5 and n_res >= 2:
        return "High Confidence"

    if (good_source and overlap_frac > 0 and n_res >= 2) or \
       (moderate_source and overlap_frac >= 0.5):
        return "Medium Confidence"

    return "Low Confidence"


def compute_contact_shell_chemistry(cofold_pdb: Path, contact_cutoff: float = 5.0) -> dict:
    """Compute chemistry of protein residues within contact_cutoff of substrate.

    Parses cofolded PDB: chain A = protein, chain B = substrate (nylon66 trimer).
    Returns chemistry fractions for the contact shell, or empty strings if no contacts.
    """
    empty = {"substrate_contact_n_residues": "", "substrate_contact_frac_charged": "",
             "substrate_contact_frac_hydrophobic": "", "substrate_contact_frac_polar": ""}
    if not cofold_pdb or not cofold_pdb.exists():
        return empty

    protein_atoms = []  # (coord, resnum, resname)
    substrate_coords = []
    seen_resi = set()

    with open(cofold_pdb) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            ch = line[21]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            ri = int(line[22:26])
            rn = line[17:20].strip()
            if ch == "A":
                protein_atoms.append((np.array([x, y, z]), ri, rn))
            elif ch == "B":
                substrate_coords.append(np.array([x, y, z]))

    if not substrate_coords or not protein_atoms:
        return empty

    sub = np.array(substrate_coords)

    # Find contact residues (any atom within cutoff)
    contact_resnames = []
    for coord, ri, rn in protein_atoms:
        if ri in seen_resi:
            continue
        dists = np.sqrt(np.sum((sub - coord) ** 2, axis=1))
        if dists.min() <= contact_cutoff:
            contact_resnames.append(rn)
            seen_resi.add(ri)

    if not contact_resnames:
        return empty

    counts = {"charged": 0, "hydrophobic": 0, "polar": 0}
    for rn in contact_resnames:
        counts[classify_residue(rn)] += 1
    n = len(contact_resnames)
    return {
        "substrate_contact_n_residues": str(n),
        "substrate_contact_frac_charged": f"{counts['charged']/n*100:.1f}%",
        "substrate_contact_frac_hydrophobic": f"{counts['hydrophobic']/n*100:.1f}%",
        "substrate_contact_frac_polar": f"{counts['polar']/n*100:.1f}%",
    }


def find_best_cofold_pdb(cofold_dir: Path, uid: str) -> Path | None:
    """Find the best-model PDB for a given UID in cofold output."""
    uid_dir = cofold_dir / uid
    if not uid_dir.is_dir():
        return None
    pdbs = sorted(uid_dir.glob("**/*_best_model.pdb"))
    if not pdbs:
        pdbs = sorted(uid_dir.glob("**/*_model_0.pdb"))
    return pdbs[0] if pdbs else None


# ──────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_csv_keyed(path: Path, key_col: str) -> dict:
    """Load CSV into dict keyed by key_col."""
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row[key_col].strip()
            if k:
                result[k] = row
    return result


def load_synthetase_seq_info() -> dict:
    """Returns {uid: {gene_name, organism, sequence_length}}"""
    info = {}
    with open(SYN_DIR / "structure_sequence_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"].strip()
            info[uid] = {
                "gene_name": row["gene_name"].strip(),
                "organism": row["organism"].strip(),
                "sequence_length": row["sequence_length"].strip(),
            }
    return info


def load_hydrolase_seq_info() -> dict:
    """Returns {uid: {organism, sequence_length}}"""
    info = {}
    with open(HYD_DIR / "structure_sequence_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"].strip()
            info[uid] = {
                "organism": row["organism"].strip(),
                "sequence_length": row["sequence_length"].strip(),
            }
    return info


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load all data sources ────────────────────────────────────────
    syn_pocket = load_json(SYN_DIR / "pocket_analysis.json")
    hyd_pocket = load_json(HYD_DIR / "pocket_analysis.json")
    syn_cat = load_json(SYN_DIR / "catalytic_site_validation.json")
    hyd_cat = load_json(HYD_DIR / "catalytic_site_validation.json")

    syn_seq = load_synthetase_seq_info()
    hyd_seq = load_hydrolase_seq_info()

    # Cofolding metrics keyed by uid
    cofold_data = {}
    with open(ANALYSIS / "cofolding_metrics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"].strip()
            # Skip variant entries like P31663_native_substrates
            if "_native_" in uid:
                continue
            cofold_data[uid] = row

    # Pocket volume analysis keyed by uid (candidates only)
    vol_data = {}
    with open(ANALYSIS / "pocket_volume_analysis.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"].strip()
            # Keep only candidate entries (not reference PDBs)
            if uid in syn_seq or uid in hyd_seq:
                vol_data[uid] = row

    # Pocket chemistry keyed by uid (candidates only) — unused, kept for compatibility
    chem_data = {}
    apo_chem_path = ANALYSIS / "apo_pocket_chemistry.csv"
    if apo_chem_path.exists():
        with open(apo_chem_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row["uid"].strip()
                if uid in syn_seq or uid in hyd_seq:
                    chem_data[uid] = row

    # ── Load holo pocket data ───────────────────────────────────────
    holo_pocket_data = {}
    holo_pocket_path = ANALYSIS / "holo_pocket_analysis.csv"
    if holo_pocket_path.exists():
        with open(holo_pocket_path) as f:
            for row in csv.DictReader(f):
                holo_pocket_data[row["uid"].strip()] = row

    holo_chem_data = {}
    holo_chem_path = ANALYSIS / "holo_pocket_chemistry.csv"
    if holo_chem_path.exists():
        with open(holo_chem_path) as f:
            for row in csv.DictReader(f):
                holo_chem_data[row["uid"].strip()] = row

    holo_comparison_data = {}
    holo_comp_path = ANALYSIS / "holo_vs_apo_comparison.csv"
    if holo_comp_path.exists():
        with open(holo_comp_path) as f:
            for row in csv.DictReader(f):
                holo_comparison_data[row["uid"].strip()] = row

    # ── Load native substrate data from enzyme family classification ──
    native_substrate_data = {}
    classification_path = ANALYSIS / "enzyme_family_classification.csv"
    if classification_path.exists():
        with open(classification_path) as f:
            for row in csv.DictReader(f):
                uid = row.get("uniprot_id", "").strip()
                if uid:
                    native_substrate_data[uid] = {
                        "native_reaction": row.get("native_reaction", ""),
                        "native_substrates": row.get("native_substrates", ""),
                    }

    # ── Define output columns ────────────────────────────────────────
    columns = [
        # Basic
        "project",
        "enzyme_name",
        "uniprot_id",
        "organism",
        "sequence_length",
        # Native substrates (from UniProt catalytic activity)
        "native_reaction",
        "native_substrates",
        # Apo pocket geometry
        "evidence_source",
        "evidence_homolog_pdb",
        "evidence_tmscore",
        "apo_pocket_volume_A3",
        "apo_largest_cavity_A3",
        "apo_pocket_area_A2",
        "apo_pocket_n_residues",
        "active_site_overlap",
        "pocket_confidence",
        # Apo pocket chemistry
        "apo_frac_charged",
        "apo_frac_hydrophobic",
        "apo_frac_polar",
        # Holo pocket geometry
        "holo_pocket_volume_A3",
        "holo_pocket_area_A2",
        "holo_pocket_n_residues",
        "holo_pocket_shape",
        # Holo pocket chemistry (cavity-lining)
        "holo_frac_charged",
        "holo_frac_hydrophobic",
        "holo_frac_polar",
        # Substrate contact chemistry (residues within 5A of substrate)
        "substrate_contact_n_residues",
        "substrate_contact_frac_charged",
        "substrate_contact_frac_hydrophobic",
        "substrate_contact_frac_polar",
        # Apo vs holo comparison
        "volume_change_pct",
        "residue_jaccard",
        "holo_ligand_contacts",
        "holo_contact_fraction",
        # Pocket lining residues (full list, apo)
        "binding_pocket_residues",
        # Catalytic residues
        "catalytic_evidence_tier",
        "n_catalytic_core",
        "n_catalytic_shell",
        "catalytic_center_score",
        "core_catalytic_residues",
        "shell_catalytic_residues",
        # Cofolding metrics
        "cofold_strategy",
        "cofold_confidence",
        "cofold_iptm",
        "cofold_ptm",
        "substrate_sasa_burial",
        "substrate_n_contacts",
        "substrate_min_distance_A",
        "substrate_iptm_to_protein",
        "mean_protein_plddt",
        "complex_plddt",
        # Synthetase-specific cofolding
        "cofactor",
        "cofactor_sasa_burial",
        "cofactor_iptm_to_protein",
        # Volume analysis
        "substrate_volume_A3",
        "apo_pocket_to_substrate_ratio",
        "apo_pocket_shape",
        "holo_pocket_to_substrate_ratio",
        # Flags
        "flags",
        "notes",
    ]

    rows = []

    # ── Process synthetases ──────────────────────────────────────────
    for uid, seq_info in syn_seq.items():
        row = {c: "" for c in columns}
        row["project"] = "synthetase"
        row["pocket_confidence"] = "Unknown Confidence"
        row["enzyme_name"] = seq_info["gene_name"]
        row["uniprot_id"] = uid
        row["organism"] = seq_info["organism"]
        row["sequence_length"] = seq_info["sequence_length"]

        # Native substrates
        ns = native_substrate_data.get(uid, {})
        row["native_reaction"] = ns.get("native_reaction", "")
        row["native_substrates"] = ns.get("native_substrates", "")

        # Pocket analysis (apo)
        pa = syn_pocket.get(uid, {})
        if pa and pa.get("role") == "candidate":
            bp = pa.get("binding_pocket") or {}
            ev = pa.get("evidence") or {}
            if bp:
                row["apo_pocket_volume_A3"] = fmt_float(bp.get("volume_A3"), 1)
                row["apo_pocket_area_A2"] = fmt_float(bp.get("area_A2"), 1)
                row["apo_pocket_n_residues"] = str(bp.get("n_residues", ""))

            # Find largest cavity
            all_pockets = pa.get("all_pockets") or []
            if all_pockets:
                largest = max(p.get("volume_A3", 0) for p in all_pockets)
                row["apo_largest_cavity_A3"] = fmt_float(largest, 1)
            else:
                row["apo_largest_cavity_A3"] = row["apo_pocket_volume_A3"]

            # Evidence
            etype = ev.get("type", "none")
            row["evidence_source"] = etype
            row["evidence_homolog_pdb"] = ev.get("homolog_pdb", "")
            row["evidence_tmscore"] = fmt_float(ev.get("tmscore"), 3)

            # Active site overlap
            total = ev.get("active_site_total", "")
            overlap = ev.get("active_site_overlap", "")
            if total != "" and overlap != "":
                overlap_str = f"{overlap}/{total}"
                pct = int(overlap) / int(total) * 100 if int(total) > 0 else 0
                row["active_site_overlap"] = f"{overlap_str} ({pct:.0f}%)"
            else:
                row["active_site_overlap"] = ""

            row["pocket_confidence"] = pocket_confidence(ev)

            # Apo pocket chemistry from lining residues
            lining = bp.get("lining_residues", []) if bp else []
            chem = compute_pocket_chemistry(lining)
            row["apo_frac_charged"] = chem["frac_charged"]
            row["apo_frac_hydrophobic"] = chem["frac_hydrophobic"]
            row["apo_frac_polar"] = chem["frac_polar"]
            row["binding_pocket_residues"] = fmt_residues(lining, "compact")

        # Holo pocket data
        hp = holo_pocket_data.get(uid, {})
        if hp:
            row["holo_pocket_volume_A3"] = hp.get("holo_volume_A3", "")
            row["holo_pocket_area_A2"] = hp.get("holo_area_A2", "")
            row["holo_pocket_n_residues"] = hp.get("holo_n_residues", "")
            row["holo_pocket_shape"] = hp.get("holo_shape", "")
            row["holo_ligand_contacts"] = hp.get("holo_ligand_contacts", "")
            row["holo_contact_fraction"] = hp.get("holo_contact_fraction", "")

        hc = holo_chem_data.get(uid, {})
        if hc:
            row["holo_frac_charged"] = fmt_pct(hc.get("holo_frac_charged"))
            row["holo_frac_hydrophobic"] = fmt_pct(hc.get("holo_frac_hydrophobic"))
            row["holo_frac_polar"] = fmt_pct(hc.get("holo_frac_polar"))

        # Substrate contact chemistry (residues within 5A of substrate)
        cofold_pdb = find_best_cofold_pdb(SYN_DIR / "cofold_output", uid)
        cs = compute_contact_shell_chemistry(cofold_pdb)
        row.update(cs)

        hcomp = holo_comparison_data.get(uid, {})
        if hcomp:
            row["volume_change_pct"] = hcomp.get("volume_change_pct", "")
            row["residue_jaccard"] = hcomp.get("residue_jaccard", "")

        # Catalytic site validation
        cat = syn_cat.get(uid, {})
        if cat:
            row["catalytic_evidence_tier"] = cat.get("evidence_tier", "")
            row["n_catalytic_core"] = str(cat.get("n_core", ""))
            row["n_catalytic_shell"] = str(cat.get("n_shell", ""))
            row["catalytic_center_score"] = fmt_float(
                cat.get("catalytic_center_score"), 3
            )
            row["core_catalytic_residues"] = fmt_residues(
                cat.get("core_residues", [])
            )
            row["shell_catalytic_residues"] = fmt_residues(
                cat.get("shell_residues", [])
            )

        # Cofolding metrics
        cf = cofold_data.get(uid, {})
        if cf:
            row["cofold_strategy"] = cf.get("strategy", "")
            row["cofold_confidence"] = fmt_float(cf.get("confidence_score"), 3)
            row["cofold_iptm"] = fmt_float(cf.get("iptm"), 3)
            row["cofold_ptm"] = fmt_float(cf.get("ptm"), 3)
            row["substrate_sasa_burial"] = fmt_pct(cf.get("substrate_sasa_burial"))
            row["substrate_n_contacts"] = cf.get("substrate_n_contact_residues", "")
            row["substrate_min_distance_A"] = fmt_float(
                cf.get("substrate_min_distance"), 2
            )
            row["substrate_iptm_to_protein"] = fmt_float(
                cf.get("substrate_iptm_to_protein"), 3
            )
            row["mean_protein_plddt"] = fmt_float(cf.get("mean_protein_plddt"), 1)
            row["complex_plddt"] = fmt_float(cf.get("complex_plddt"), 3)
            row["cofactor"] = cf.get("cofactor_ccd", "")
            row["cofactor_sasa_burial"] = fmt_pct(cf.get("cofactor_sasa_burial"))
            row["cofactor_iptm_to_protein"] = fmt_float(
                cf.get("cofactor_iptm_to_protein"), 3
            )

        # Volume analysis
        va = vol_data.get(uid, {})
        if va:
            row["substrate_volume_A3"] = fmt_float(
                va.get("total_substrate_volume_A3"), 1
            )
            row["apo_pocket_to_substrate_ratio"] = fmt_float(
                va.get("volume_ratio"), 2
            )
            row["apo_pocket_shape"] = va.get("pocket_shape", "")
            row["flags"] = va.get("flags", "")

        # Compute holo pocket-to-substrate ratio
        try:
            holo_vol = float(row["holo_pocket_volume_A3"]) if row["holo_pocket_volume_A3"] else 0
            sub_vol = float(row["substrate_volume_A3"]) if row["substrate_volume_A3"] else 0
            if holo_vol > 0 and sub_vol > 0:
                row["holo_pocket_to_substrate_ratio"] = f"{holo_vol / sub_vol:.2f}"
        except (ValueError, ZeroDivisionError):
            pass

        # Notes
        notes = []
        try:
            sel_vol = float(row["apo_pocket_volume_A3"]) if row["apo_pocket_volume_A3"] else 0
            lrg_vol = float(row["apo_largest_cavity_A3"]) if row["apo_largest_cavity_A3"] else 0
            if sel_vol > 0 and lrg_vol > 0 and lrg_vol > sel_vol * 2:
                notes.append(
                    f"Evidence-selected pocket ({sel_vol:.0f} A3) much smaller "
                    f"than largest cavity ({lrg_vol:.0f} A3); likely conformational change"
                )
        except ValueError:
            pass
        row["notes"] = "; ".join(notes)

        rows.append(row)

    # ── Process hydrolases ───────────────────────────────────────────
    # Build name map from cofolding metrics
    hyd_names = {}
    for uid, cf in cofold_data.items():
        if cf.get("project", "").strip() == "hydrolase":
            name = cf.get("enzyme_name", "").strip()
            import re
            name = re.sub(r'\s*status=\w+', '', name).strip()
            hyd_names[uid] = name

    for uid, seq_info in hyd_seq.items():
        row = {c: "" for c in columns}
        row["project"] = "hydrolase"
        row["pocket_confidence"] = "Unknown Confidence"
        row["enzyme_name"] = hyd_names.get(uid, uid)
        row["uniprot_id"] = uid
        row["organism"] = seq_info["organism"]
        row["sequence_length"] = seq_info["sequence_length"]

        # Native substrates
        ns = native_substrate_data.get(uid, {})
        row["native_reaction"] = ns.get("native_reaction", "")
        row["native_substrates"] = ns.get("native_substrates", "")

        # Pocket analysis (apo)
        pa = hyd_pocket.get(uid, {})
        if pa and pa.get("role") == "candidate":
            bp = pa.get("binding_pocket") or {}
            ev = pa.get("evidence") or {}
            if bp:
                row["apo_pocket_volume_A3"] = fmt_float(bp.get("volume_A3"), 1)
                row["apo_pocket_area_A2"] = fmt_float(bp.get("area_A2"), 1)
                row["apo_pocket_n_residues"] = str(bp.get("n_residues", ""))

            all_pockets = pa.get("all_pockets") or []
            if all_pockets:
                largest = max(p.get("volume_A3", 0) for p in all_pockets)
                row["apo_largest_cavity_A3"] = fmt_float(largest, 1)
            else:
                row["apo_largest_cavity_A3"] = row["apo_pocket_volume_A3"]

            etype = ev.get("type", "none")
            row["evidence_source"] = etype
            row["evidence_homolog_pdb"] = ev.get("homolog_pdb", "")
            row["evidence_tmscore"] = fmt_float(ev.get("tmscore"), 3)

            total = ev.get("active_site_total", "")
            overlap = ev.get("active_site_overlap", "")
            if total != "" and overlap != "":
                overlap_str = f"{overlap}/{total}"
                pct = int(overlap) / int(total) * 100 if int(total) > 0 else 0
                row["active_site_overlap"] = f"{overlap_str} ({pct:.0f}%)"
            else:
                # For hydrolases with foldseek evidence, try n_mapped/n_homolog_sites
                n_mapped = ev.get("n_mapped", "")
                n_sites = ev.get("n_homolog_sites", "")
                if n_mapped != "" and n_sites != "":
                    pct = int(n_mapped) / int(n_sites) * 100 if int(n_sites) > 0 else 0
                    row["active_site_overlap"] = f"{n_mapped}/{n_sites} ({pct:.0f}%)"

            row["pocket_confidence"] = pocket_confidence(ev)

            lining = bp.get("lining_residues", []) if bp else []
            chem = compute_pocket_chemistry(lining)
            row["apo_frac_charged"] = chem["frac_charged"]
            row["apo_frac_hydrophobic"] = chem["frac_hydrophobic"]
            row["apo_frac_polar"] = chem["frac_polar"]
            row["binding_pocket_residues"] = fmt_residues(lining, "compact")

        # Holo pocket data
        hp = holo_pocket_data.get(uid, {})
        if hp:
            row["holo_pocket_volume_A3"] = hp.get("holo_volume_A3", "")
            row["holo_pocket_area_A2"] = hp.get("holo_area_A2", "")
            row["holo_pocket_n_residues"] = hp.get("holo_n_residues", "")
            row["holo_pocket_shape"] = hp.get("holo_shape", "")
            row["holo_ligand_contacts"] = hp.get("holo_ligand_contacts", "")
            row["holo_contact_fraction"] = hp.get("holo_contact_fraction", "")

        hc = holo_chem_data.get(uid, {})
        if hc:
            row["holo_frac_charged"] = fmt_pct(hc.get("holo_frac_charged"))
            row["holo_frac_hydrophobic"] = fmt_pct(hc.get("holo_frac_hydrophobic"))
            row["holo_frac_polar"] = fmt_pct(hc.get("holo_frac_polar"))

        # Substrate contact chemistry (residues within 5A of substrate)
        cofold_pdb = find_best_cofold_pdb(HYD_DIR / "cofold_output", uid)
        cs = compute_contact_shell_chemistry(cofold_pdb)
        row.update(cs)

        hcomp = holo_comparison_data.get(uid, {})
        if hcomp:
            row["volume_change_pct"] = hcomp.get("volume_change_pct", "")
            row["residue_jaccard"] = hcomp.get("residue_jaccard", "")

        # Catalytic site validation
        cat = hyd_cat.get(uid, {})
        if cat:
            row["catalytic_evidence_tier"] = cat.get("evidence_tier", "")
            row["n_catalytic_core"] = str(cat.get("n_core", ""))
            row["n_catalytic_shell"] = str(cat.get("n_shell", ""))
            row["catalytic_center_score"] = fmt_float(
                cat.get("catalytic_center_score"), 3
            )
            row["core_catalytic_residues"] = fmt_residues(
                cat.get("core_residues", [])
            )
            row["shell_catalytic_residues"] = fmt_residues(
                cat.get("shell_residues", [])
            )

        # Cofolding metrics
        cf = cofold_data.get(uid, {})
        if cf:
            row["cofold_strategy"] = cf.get("strategy", "")
            row["cofold_confidence"] = fmt_float(cf.get("confidence_score"), 3)
            row["cofold_iptm"] = fmt_float(cf.get("iptm"), 3)
            row["cofold_ptm"] = fmt_float(cf.get("ptm"), 3)
            row["substrate_sasa_burial"] = fmt_pct(cf.get("substrate_sasa_burial"))
            row["substrate_n_contacts"] = cf.get("substrate_n_contact_residues", "")
            row["substrate_min_distance_A"] = fmt_float(
                cf.get("substrate_min_distance"), 2
            )
            row["substrate_iptm_to_protein"] = fmt_float(
                cf.get("substrate_iptm_to_protein"), 3
            )
            row["mean_protein_plddt"] = fmt_float(cf.get("mean_protein_plddt"), 1)
            row["complex_plddt"] = fmt_float(cf.get("complex_plddt"), 3)
            # Hydrolases don't use cofactors
            row["cofactor"] = cf.get("cofactor_ccd", "")
            row["cofactor_sasa_burial"] = fmt_pct(cf.get("cofactor_sasa_burial"))
            row["cofactor_iptm_to_protein"] = fmt_float(
                cf.get("cofactor_iptm_to_protein"), 3
            )

        # Volume analysis
        va = vol_data.get(uid, {})
        if va:
            row["substrate_volume_A3"] = fmt_float(
                va.get("total_substrate_volume_A3"), 1
            )
            row["apo_pocket_to_substrate_ratio"] = fmt_float(
                va.get("volume_ratio"), 2
            )
            row["apo_pocket_shape"] = va.get("pocket_shape", "")
            row["flags"] = va.get("flags", "")

        # Compute holo pocket-to-substrate ratio
        try:
            holo_vol = float(row["holo_pocket_volume_A3"]) if row["holo_pocket_volume_A3"] else 0
            sub_vol = float(row["substrate_volume_A3"]) if row["substrate_volume_A3"] else 0
            if holo_vol > 0 and sub_vol > 0:
                row["holo_pocket_to_substrate_ratio"] = f"{holo_vol / sub_vol:.2f}"
        except (ValueError, ZeroDivisionError):
            pass

        rows.append(row)

    # ── Sort: synthetases first (by enzyme name), then hydrolases ────
    rows.sort(key=lambda r: (0 if r["project"] == "synthetase" else 1, r["enzyme_name"].lower()))

    # ── Write output ─────────────────────────────────────────────────
    out_path = ANALYSIS / "comprehensive_enzyme_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # ── Print summary stats ──────────────────────────────────────────
    syn_rows = [r for r in rows if r["project"] == "synthetase"]
    hyd_rows = [r for r in rows if r["project"] == "hydrolase"]

    print(f"Comprehensive table written to: {out_path}")
    print(f"  Synthetases: {len(syn_rows)} enzymes")
    print(f"  Hydrolases:  {len(hyd_rows)} enzymes")
    print(f"  Total:       {len(rows)} enzymes")
    print(f"  Columns:     {len(columns)}")

    # Show column listing
    print("\nColumns included:")
    for i, col in enumerate(columns, 1):
        print(f"  {i:2d}. {col}")

    # Quick data completeness check
    print("\nData completeness:")
    for project_name, project_rows in [("synthetase", syn_rows), ("hydrolase", hyd_rows)]:
        n = len(project_rows)
        has_apo = sum(1 for r in project_rows if r["apo_pocket_volume_A3"])
        has_holo = sum(1 for r in project_rows if r["holo_pocket_volume_A3"])
        has_cofold = sum(1 for r in project_rows if r["cofold_iptm"])
        has_cat = sum(1 for r in project_rows if r["n_catalytic_core"] and r["n_catalytic_core"] != "0")
        has_residues = sum(1 for r in project_rows if r["binding_pocket_residues"])
        print(f"  {project_name}: apo_pocket={has_apo}/{n}, holo_pocket={has_holo}/{n}, "
              f"cofold={has_cofold}/{n}, catalytic_residues={has_cat}/{n}, "
              f"pocket_residues={has_residues}/{n}")


if __name__ == "__main__":
    main()
