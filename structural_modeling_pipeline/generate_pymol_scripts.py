#!/usr/bin/env python3
"""
Generate per-enzyme PyMOL (.pml) scripts for pocket visualization.

Uses active site residues (from AlphaFill/FoldSeek/UniProt evidence) as the
pocket core, then expands by 5A to show the immediate binding environment.
This gives a focused, meaningful pocket view instead of the full KVFinder cavity.

Two script types per enzyme:
  1. Apo scripts: pocket surface + optional cofolded overlay
  2. Cofold scripts: substrate bound in protein with pocket surface

Features:
  - YRB surface coloring (per-atom hydrophobic/basic/acidic)
  - Per-chain ligand coloring with distinct carbon colors
  - Polar contacts between ligands and pocket
  - Evidence source and confidence in header comments
  - Ray-tracing ready with PNG export command
"""

import csv
import json
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
ANALYSIS = BASE / "analysis_results"

PROJECTS = [
    {
        "name": "synthetase",
        "pa_path": BASE / "amide_synthetase" / "pocket_analysis.json",
        "apo_dir": BASE / "all_structures" / "amide_synthetases_candidates" / "alphafold_apo_structure",
        "cofold_dir": BASE / "all_structures" / "amide_synthetases_candidates" / "cofolded_structure",
        "cofold_input_dir": BASE / "amide_synthetase" / "cofold_input",
    },
    {
        "name": "hydrolase",
        "pa_path": BASE / "amide_hydrolase" / "pocket_analysis.json",
        "apo_dir": BASE / "all_structures" / "amide_hydrolase_candidates" / "alphafold_apo_structure",
        "cofold_dir": BASE / "all_structures" / "amide_hydrolase_candidates" / "cofolded_structure",
        "cofold_input_dir": BASE / "amide_hydrolase" / "cofold_input",
    },
]
CSV_PATH = BASE / "analysis_results" / "comprehensive_enzyme_table.csv"

# Distinct carbon colors for up to 6 ligand chains (B through G)
# Each is visually distinguishable on a white background with YRB surface
CHAIN_CARBON_COLORS = {
    "B": "cyan",         # cofactor (ATP) or primary substrate
    "C": "palegreen",    # metal or second ligand
    "D": "lightorange",  # substrate 1
    "E": "lightpink",    # substrate 2
    "F": "paleblue",     # additional
    "G": "paleyellow",   # additional
}


def load_csv_data() -> dict:
    data = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            uid = r["uniprot_id"].strip()
            data[uid] = r
    return data


def extract_uid(filename: str) -> str:
    return filename.replace(".pdb", "").rsplit("_", 1)[-1]


def get_active_site_residues(entry: dict) -> list[int]:
    """Extract all functionally annotated residue positions from pocket evidence."""
    ev = entry.get("evidence") or {}
    positions = set()
    for r in ev.get("active_site_residues", []):
        positions.add(int(r))
    for ligand, contacts in (ev.get("ligand_contacts") or {}).items():
        for r in contacts:
            positions.add(int(r))
    for metal, contacts in (ev.get("metal_contacts") or {}).items():
        for r in contacts:
            positions.add(int(r))
    return sorted(positions)


def get_catalytic_residues(entry: dict) -> list[int]:
    """Return active site residues only when evidence is high-confidence.

    High-confidence criteria:
      - uniprot_annotation: always (experimentally curated)
      - alphafill: always (direct ligand transplant)
      - foldseek+alphafill or foldseek+uniprot: only if TM-score >= 0.7
      - none: skip (no data)
    """
    ev = entry.get("evidence") or {}
    etype = ev.get("type", "none")

    if etype == "none":
        return []
    if etype in ("uniprot_annotation", "alphafill"):
        pass  # always high confidence
    elif etype.startswith("foldseek"):
        tm = ev.get("tmscore")
        if tm is None or tm < 0.7:
            return []
    else:
        return []

    return [int(r) for r in ev.get("active_site_residues", [])]


def get_chain_names(cofold_input_dir: Path, uid: str) -> dict:
    """Read cofold input YAML to build chain_id -> human-readable name mapping."""
    yaml_path = cofold_input_dir / f"{uid}.yaml"
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {}

    config_path = cofold_input_dir.parent / "config.yaml"
    smiles_to_name = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                proj_cfg = yaml.safe_load(f)
            for sub in proj_cfg.get("target_reaction", {}).get("substrates", []):
                smi = sub.get("smiles", "")
                name = sub.get("name", "")
                if smi and name:
                    smiles_to_name[smi] = name.replace("_", " ")
        except Exception:
            pass

    chain_names = {}
    for seq in cfg.get("sequences", []):
        if "ligand" in seq:
            lig = seq["ligand"]
            chain_id = lig.get("id", "")
            ccd = lig.get("ccd", "")
            smiles = lig.get("smiles", "")
            if ccd:
                name = ccd
            elif smiles:
                name = smiles_to_name.get(smiles, smiles[:30])
            else:
                name = f"chain {chain_id}"
            chain_names[chain_id] = name
    return chain_names


def get_evidence_summary(entry: dict) -> list[str]:
    """Build comment lines describing evidence source and confidence."""
    ev = entry.get("evidence") or {}
    etype = ev.get("type", "none")
    lines = []

    if etype == "alphafill":
        ligands = list((ev.get("ligand_contacts") or {}).keys())
        metals = list((ev.get("metal_contacts") or {}).keys())
        all_lig = ligands + metals
        lines.append(f"# Evidence: AlphaFill (ligand contacts: {', '.join(all_lig)})")
    elif etype == "uniprot_annotation":
        ec = ", ".join(ev.get("ec_numbers", []))
        ec_str = f", EC {ec}" if ec else ""
        lines.append(f"# Evidence: UniProt curated active/binding sites{ec_str}")
    elif etype == "foldseek+alphafill":
        pdb = ev.get("homolog_pdb", "?")
        uid = ev.get("homolog_uniprot", "")
        tm = ev.get("tmscore")
        tm_str = f", TM={tm:.2f}" if tm is not None else ""
        lines.append(f"# Evidence: FoldSeek -> AlphaFill (homolog {pdb} [{uid}]{tm_str})")
    elif etype == "foldseek+uniprot":
        pdb = ev.get("homolog_pdb", "?")
        uid = ev.get("homolog_uniprot", "")
        tm = ev.get("tmscore")
        tm_str = f", TM={tm:.2f}" if tm is not None else ""
        lines.append(f"# Evidence: FoldSeek -> UniProt (homolog {pdb} [{uid}]{tm_str})")
    elif etype == "none":
        note = ev.get("note", "No catalytic site evidence")
        lines.append(f"# Evidence: None ({note})")
    else:
        lines.append(f"# Evidence: {etype}")

    # Active site residues
    resi = ev.get("active_site_residues", [])
    if resi:
        total = ev.get("active_site_total", len(resi))
        overlap = ev.get("active_site_overlap", 0)
        lines.append(f"# Active site residues ({len(resi)}): {', '.join(str(r) for r in resi[:20])}"
                     + ("..." if len(resi) > 20 else ""))
        lines.append(f"# Pocket overlap: {overlap}/{total}")

    return lines


def yrb_coloring_lines(sel: str) -> list[str]:
    return [
        f"color white, {sel}",
        f"color tv_yellow, {sel} and elem C and resn ALA+VAL+LEU+ILE+PRO+PHE+TRP+MET",
        f"color tv_yellow, {sel} and resn CYS and name CB+SG",
        f"color tv_yellow, {sel} and resn LYS and name CB+CG+CD+CE",
        f"color tv_yellow, {sel} and resn ARG and name CB+CG+CD",
        f"color tv_yellow, {sel} and resn GLU and name CB+CG",
        f"color tv_yellow, {sel} and resn GLN and name CB+CG",
        f"color tv_yellow, {sel} and resn ASP and name CB",
        f"color tv_yellow, {sel} and resn ASN and name CB",
        f"color tv_yellow, {sel} and resn THR and name CG2",
        f"color tv_yellow, {sel} and resn SER and name CB",
        f"color tv_yellow, {sel} and resn TYR and name CB+CG+CD1+CD2+CE1+CE2+CZ",
        f"color tv_yellow, {sel} and resn HIS and name CB+CG",
        f"color marine, {sel} and resn LYS and name NZ",
        f"color marine, {sel} and resn ARG and name NH1+NH2+NE+CZ",
        f"color marine, {sel} and resn HIS and name ND1+NE2",
        f"color tv_red, {sel} and resn ASP and name OD1+OD2",
        f"color tv_red, {sel} and resn GLU and name OE1+OE2",
        "",
    ]


def pml_header(enzyme_name: str, uid: str, meta: dict = None,
               view: str = "apo", evidence_lines: list[str] = None,
               chain_names: dict = None) -> list[str]:
    lines = [f"# PyMOL pocket visualization: {enzyme_name} ({uid})", ""]
    if meta:
        if view == "apo":
            vol = meta.get("apo_pocket_volume_A3", "N/A")
            hydro = meta.get("apo_frac_hydrophobic", "N/A")
            lines.append(f"# Apo pocket: volume={vol} A3, hydrophobic={hydro}")
        elif view == "cofold":
            holo_vol = meta.get("holo_pocket_volume_A3", "N/A")
            holo_hydro = meta.get("holo_frac_hydrophobic", "N/A")
            apo_vol = meta.get("apo_pocket_volume_A3", "N/A")
            vol_change = meta.get("volume_change_pct", "N/A")
            holo_shape = meta.get("holo_pocket_shape", "N/A")
            iptm = meta.get("cofold_iptm", "N/A")
            burial = meta.get("substrate_sasa_burial", "N/A")
            lines.append(f"# Holo pocket: volume={holo_vol} A3, hydrophobic={holo_hydro}, shape={holo_shape}")
            lines.append(f"# Apo pocket:  volume={apo_vol} A3 (volume change: {vol_change}%)")
            lines.append(f"# Cofolding: ipTM={iptm}, substrate burial={burial}")
        lines.append(f"# Organism: {meta.get('organism', '')}")

    if evidence_lines:
        for el in evidence_lines:
            lines.append(el)

    if chain_names:
        lines.append("#")
        lines.append("# Ligand chains:")
        for chain_id, name in sorted(chain_names.items()):
            color = CHAIN_CARBON_COLORS.get(chain_id, "gray")
            lines.append(f"#   Chain {chain_id}: {name} (carbon color: {color})")

    lines.append("")
    lines.extend([
        "bg_color white",
        "set ray_shadow, 0",
        "set spec_reflect, 0.3",
        "set spec_power, 80",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "set surface_quality, 1",
        "set surface_mode, 3",
        "set two_sided_lighting, on",
        "set surface_color_smoothing, 1",
        "set antialias, 2",
        "set ray_opaque_background, 1",
        "",
        "# Define translucent gray to match HTML viewer (opacity 0.7)",
        "set_color prot_gray, [0.8, 0.8, 0.8]",
        "",
    ])
    return lines


def ligand_coloring_lines(chain_names: dict) -> list[str]:
    """Generate per-chain ligand coloring with distinct carbon colors."""
    L = []
    if not chain_names:
        # Fallback: color all non-A chains uniformly
        L.append("select ligands, complex and not chain A")
        L.append("show sticks, ligands")
        L.append("set stick_radius, 0.25, ligands")
        L.append("color cyan, ligands and elem C")
        L.append("color red, ligands and elem O")
        L.append("color blue, ligands and elem N")
        L.append("color orange, ligands and elem P")
        L.append("color yellow, ligands and elem S")
        L.append("")
        return L

    L.append("# Per-chain ligand display with distinct colors")
    for chain_id, name in sorted(chain_names.items()):
        carbon_color = CHAIN_CARBON_COLORS.get(chain_id, "gray")
        sel_name = f"lig_{chain_id}"
        L.append(f"select {sel_name}, complex and chain {chain_id}  # {name}")

        # Metals: show as spheres
        is_metal = name.upper() in ("MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU")
        if is_metal:
            L.append(f"show spheres, {sel_name}")
            L.append(f"set sphere_scale, 0.8, {sel_name}")
            L.append(f"color palegreen, {sel_name}")
        else:
            L.append(f"show sticks, {sel_name}")
            L.append(f"set stick_radius, 0.25, {sel_name}")
            L.append(f"color {carbon_color}, {sel_name} and elem C")
            L.append(f"color red, {sel_name} and elem O")
            L.append(f"color blue, {sel_name} and elem N")
            L.append(f"color orange, {sel_name} and elem P")
            L.append(f"color yellow, {sel_name} and elem S")
        L.append("")

    # Group all ligands for later use
    chain_list = " or ".join(f"chain {c}" for c in sorted(chain_names.keys()))
    L.append(f"select ligands, complex and ({chain_list})")
    L.append("")
    return L


def catalytic_residue_lines(catalytic_resi: list[int], protein_obj: str,
                            descriptions: list[str] = None) -> list[str]:
    """Show UniProt-curated catalytic residues as sticks.

    Only called when UniProt 'Active site' annotations exist for this enzyme.
    These are experimentally verified catalytic residues (typically 1-5).
    """
    if not catalytic_resi:
        return []
    resi_list = "+".join(str(r) for r in catalytic_resi)
    L = ["# UniProt-curated catalytic residues"]
    if descriptions:
        for pos, desc in zip(catalytic_resi, descriptions):
            if desc:
                L.append(f"#   Residue {pos}: {desc}")
    L.extend([
        f"select catalytic, {protein_obj} and chain A and resi {resi_list}",
        "show sticks, catalytic",
        "set stick_radius, 0.15, catalytic",
        "color firebrick, catalytic and elem C",
        "color red, catalytic and elem O",
        "color tv_blue, catalytic and elem N",
        "color yellow, catalytic and elem S",
        "deselect",
        "",
    ])
    return L


def polar_contacts_block(ligand_sel: str = "ligands", pocket_sel: str = "pocket") -> list[str]:
    """Generate polar contact (H-bond) visualization between ligands and pocket."""
    L = [
        "# Polar contacts between ligands and pocket residues",
        f"dist hbonds, {ligand_sel}, {pocket_sel}, 3.5, mode=2",
        "set dash_color, gray50, hbonds",
        "set dash_width, 1.5, hbonds",
        "set dash_gap, 0.3, hbonds",
        "hide labels, hbonds",
        "",
    ]
    return L


def generate_apo_pml(
    apo_filename: str,
    cofold_relative: str | None,
    enzyme_name: str,
    uid: str,
    active_resi: list[int],
    meta: dict = None,
    evidence_lines: list[str] = None,
    chain_names: dict = None,
    apo_lining_resi: list[int] = None,
    catalytic_resi: list[int] = None,
    catalytic_descs: list[str] = None,
) -> str:
    L = pml_header(enzyme_name, uid, meta=meta, view="apo",
                   evidence_lines=evidence_lines, chain_names=chain_names)

    # Use apo lining residues when available (consistent with HTML viewer);
    # fall back to active site residues
    pocket_resi = apo_lining_resi if apo_lining_resi else active_resi
    has_pocket = len(pocket_resi) > 0

    L.append("# ====== SCENE 1: Apo Pocket Analysis ======")
    L.append("")
    L.append(f"load {apo_filename}, apo_protein")
    L.append("hide everything, apo_protein")
    L.append("show cartoon, apo_protein")
    # Translucent gray cartoon matching HTML viewer
    L.append("color prot_gray, apo_protein")
    L.append("set cartoon_color, prot_gray, apo_protein")
    L.append("set cartoon_transparency, 0.3, apo_protein")
    L.append("")

    if has_pocket:
        resi_list = "+".join(str(r) for r in pocket_resi)
        # Create a separate object for the pocket surface so YRB coloring
        # does not bleed into the protein cartoon
        L.append(f"create pocket_obj, apo_protein and chain A and resi {resi_list}")
        L.append("hide everything, pocket_obj")
        L.append("")
        # YRB atom coloring on the pocket object only
        L.extend(yrb_coloring_lines("pocket_obj"))
        L.append("show surface, pocket_obj")
        L.append("set transparency, 0.5, pocket_obj")
        L.append("")
        # No catalytic sticks in apo scene (no ligands to anchor selection);
        # sticks are shown in the overlay scene (scene 2) using distance to substrates
        L.append(f"select pocket_sel, apo_protein and chain A and resi {resi_list}")
        L.append("orient pocket_sel")
        L.append("zoom pocket_sel, 5")
    else:
        L.append("# No active site data available")
        L.append("orient apo_protein")

    L.append("")
    L.append("scene 001_apo_pocket, store")
    L.append("")

    if cofold_relative:
        L.append("# ====== SCENE 2: Apo + Cofolded Overlay ======")
        L.append("")
        L.append(f"load {cofold_relative}, cofold_complex")
        L.append("align cofold_complex and chain A, apo_protein")
        L.append("hide everything, cofold_complex and chain A")
        L.append("")

        if chain_names:
            L.append("# Per-chain ligand display with distinct colors")
            for chain_id, name in sorted(chain_names.items()):
                carbon_color = CHAIN_CARBON_COLORS.get(chain_id, "gray")
                sel_name = f"overlay_{chain_id}"
                is_metal = name.upper() in ("MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU")
                L.append(f"select {sel_name}, cofold_complex and chain {chain_id}  # {name}")
                if is_metal:
                    L.append(f"show spheres, {sel_name}")
                    L.append(f"set sphere_scale, 0.8, {sel_name}")
                    L.append(f"color palegreen, {sel_name}")
                else:
                    L.append(f"show sticks, {sel_name}")
                    L.append(f"set stick_radius, 0.25, {sel_name}")
                    L.append(f"color {carbon_color}, {sel_name} and elem C")
                    L.append(f"color red, {sel_name} and elem O")
                    L.append(f"color blue, {sel_name} and elem N")
                    L.append(f"color orange, {sel_name} and elem P")
                    L.append(f"color yellow, {sel_name} and elem S")
                L.append("")
            L.append("select substrates, cofold_complex and not chain A")
        else:
            L.append("select substrates, cofold_complex and not chain A")
            L.append("show sticks, substrates")
            L.append("set stick_radius, 0.25, substrates")
            L.append("color cyan, substrates and elem C")
            L.append("color red, substrates and elem O")
            L.append("color blue, substrates and elem N")
            L.append("color orange, substrates and elem P")
            L.append("color yellow, substrates and elem S")

        L.append("")
        L.append("set cartoon_transparency, 0.8, apo_protein")
        L.append("")

        if has_pocket:
            L.append(f"select pocket, apo_protein and chain A and resi {resi_list}")
            L.extend(polar_contacts_block("substrates", "pocket"))
            if catalytic_resi:
                L.extend(catalytic_residue_lines(catalytic_resi, "apo_protein",
                                                 catalytic_descs))
            L.append("orient pocket")
            L.append("zoom pocket, 8")
        else:
            L.append("orient apo_protein")

        L.append("")
        L.append("scene 002_overlay, store")
        L.append("")

    # Ray-tracing export hints
    L.append("deselect")
    L.append("scene 001_apo_pocket, recall")
    L.append("")
    L.append(f"# To render: ray 2400, 1800; png {apo_filename.replace('.pdb', '_apo.png')}, dpi=300")
    L.append("")
    return "\n".join(L)


def generate_cofold_pml(
    cofold_filename: str,
    enzyme_name: str,
    uid: str,
    active_resi: list[int],
    meta: dict = None,
    evidence_lines: list[str] = None,
    chain_names: dict = None,
    holo_lining_resi: list[int] = None,
    catalytic_resi: list[int] = None,
    catalytic_descs: list[str] = None,
) -> str:
    L = pml_header(enzyme_name, uid, meta=meta, view="cofold",
                   evidence_lines=evidence_lines, chain_names=chain_names)

    # For holo pocket, prefer holo lining residues (from holo_pocket_analysis)
    # to match the HTML viewer; fall back to active site residues
    pocket_resi = holo_lining_resi if holo_lining_resi else active_resi
    has_pocket = len(pocket_resi) > 0

    L.append("# ====== Cofolded Structure: Substrate in Pocket ======")
    L.append("")
    L.append(f"load {cofold_filename}, complex")
    L.append("")
    # Protein cartoon: gray and semi-transparent
    # NOTE: set cartoon_color / cartoon_transparency require OBJECT name, not selection
    L.append("hide everything, complex")
    L.append("show cartoon, complex and chain A")
    L.append("color prot_gray, complex")
    L.append("set cartoon_color, prot_gray, complex")
    L.append("set cartoon_transparency, 0.3, complex")
    L.append("")

    # Per-chain ligand coloring
    L.extend(ligand_coloring_lines(chain_names))

    if has_pocket:
        resi_list = "+".join(str(r) for r in pocket_resi)
        # Create a separate object for the pocket surface so YRB coloring
        # does not bleed into the protein cartoon
        L.append(f"create pocket_obj, complex and chain A and resi {resi_list}")
        L.append("hide everything, pocket_obj")
        L.append("")
        # YRB atom coloring on the pocket object only
        L.extend(yrb_coloring_lines("pocket_obj"))
        L.append("show surface, pocket_obj")
        L.append("set transparency, 0.5, pocket_obj")
        L.append("")
        if catalytic_resi:
            L.extend(catalytic_residue_lines(catalytic_resi, "complex",
                                             catalytic_descs))
        L.append(f"select pocket, complex and chain A and resi {resi_list}")
        L.extend(polar_contacts_block("ligands", "pocket"))
        L.append("orient pocket")
        L.append("zoom pocket, 8")
    else:
        L.append("orient complex")

    L.append("")
    L.append("scene 001_cofold, store")
    L.append("")
    L.append("deselect")
    L.append("")
    L.append(f"# To render: ray 2400, 1800; png {cofold_filename.replace('.pdb', '_cofold.png')}, dpi=300")
    L.append("")
    return "\n".join(L)


def main():
    csv_data = load_csv_data()

    # Load holo pocket residues (matches HTML pipeline)
    holo_pocket_path = ANALYSIS / "holo_pocket_residues.json"
    holo_pocket_data = {}
    if holo_pocket_path.exists():
        with open(holo_pocket_path) as f:
            holo_pocket_data = json.load(f)
        print(f"Loaded holo pocket residues for {len(holo_pocket_data)} enzymes")

    # Load catalytic site data from catalytic_site_validation.json
    # (produced by validate_catalytic_sites.py + enhance_catalytic_sites.py).
    # Uses only core_residues (UniProt "Active site" features) — NOT shell
    # residues (Binding/Metal site) which are substrate-specific.
    #
    # FILTER: validate_catalytic_sites.py promotes Binding site features to
    # core_residues when no Active site features exist. These promoted entries
    # have core > 5 residues and shell = 0, which is not true catalytic data.
    # We only trust entries where:
    #   - foldseek_homology (our enhancement — Active site only, never promoted)
    #   - uniprot_direct with shell_residues > 0 (real Active site + separate Binding)
    #   - uniprot_direct with core <= 5 (small enough to be real Active site)
    #   - *_homology_* with core <= 5 (InterPro-transferred, small = real)
    catalytic_data = {}
    n_skipped = 0
    for proj in PROJECTS:
        cat_path = proj["pa_path"].parent / "catalytic_site_validation.json"
        if cat_path.exists():
            with open(cat_path) as f:
                cat_json = json.load(f)
            for uid, entry in cat_json.items():
                core = entry.get("core_residues", [])
                shell = entry.get("shell_residues", [])
                tier = entry.get("evidence_tier", "")
                if not core:
                    continue
                # foldseek_homology: always trust (Active site only)
                if tier == "foldseek_homology":
                    pass
                # uniprot_direct: trust if shell exists (real Active site)
                # or if core is small (≤ 5, likely real)
                elif tier == "uniprot_direct":
                    if not shell and len(core) > 5:
                        n_skipped += 1
                        continue  # Promoted Binding sites, skip
                # InterPro homology: trust if small
                elif "homology" in tier:
                    if len(core) > 5:
                        n_skipped += 1
                        continue
                else:
                    continue

                catalytic_data[uid] = {
                    "positions": core,
                    "descriptions": entry.get("core_descriptions", []),
                    "evidence_tier": tier,
                }
    print(f"Loaded catalytic site data: {len(catalytic_data)} enzymes with core residues"
          + (f" (skipped {n_skipped} with promoted Binding sites)" if n_skipped else ""))

    apo_count = 0
    cofold_count = 0

    for proj in PROJECTS:
        with open(proj["pa_path"]) as f:
            pocket_data = json.load(f)

        apo_dir = proj["apo_dir"]
        cofold_dir = proj["cofold_dir"]
        cofold_input_dir = proj.get("cofold_input_dir")

        for pdb_path in sorted(apo_dir.glob("*.pdb")):
            fname = pdb_path.name
            uid = extract_uid(fname)
            pa = pocket_data.get(uid, {})
            active_resi = get_active_site_residues(pa) if isinstance(pa, dict) and pa.get("role") == "candidate" else []
            evidence_lines = get_evidence_summary(pa) if isinstance(pa, dict) and pa.get("role") == "candidate" else []
            meta = csv_data.get(uid, {})
            enzyme_name = meta.get("enzyme_name", uid)
            cofold_path = cofold_dir / fname
            cofold_rel = f"../cofolded_structure/{fname}" if cofold_path.exists() else None
            chain_names = get_chain_names(cofold_input_dir, uid) if cofold_input_dir else {}

            # Use apo lining residues when available (consistent with HTML viewer)
            apo_lining = []
            if isinstance(pa, dict) and pa.get("role") == "candidate":
                bp = pa.get("binding_pocket") or {}
                apo_lining = sorted(set(int(r[1]) for r in bp.get("lining_residues", [])))

            # Catalytic residues: core_residues from catalytic_site_validation.json
            cat_entry = catalytic_data.get(uid, {})
            catalytic = cat_entry.get("positions", [])
            cat_descs = cat_entry.get("descriptions", [])

            pml = generate_apo_pml(fname, cofold_rel, enzyme_name, uid, active_resi,
                                   meta=meta, evidence_lines=evidence_lines,
                                   chain_names=chain_names,
                                   apo_lining_resi=apo_lining,
                                   catalytic_resi=catalytic,
                                   catalytic_descs=cat_descs)
            (apo_dir / fname.replace(".pdb", "_pocket_analysis.pml")).write_text(pml)
            apo_count += 1

        for pdb_path in sorted(cofold_dir.glob("*.pdb")):
            fname = pdb_path.name
            uid = extract_uid(fname)
            pa = pocket_data.get(uid, {})
            active_resi = get_active_site_residues(pa) if isinstance(pa, dict) and pa.get("role") == "candidate" else []
            evidence_lines = get_evidence_summary(pa) if isinstance(pa, dict) and pa.get("role") == "candidate" else []
            meta = csv_data.get(uid, {})
            enzyme_name = meta.get("enzyme_name", uid)
            chain_names = get_chain_names(cofold_input_dir, uid) if cofold_input_dir else {}

            # Use holo lining residues when available (consistent with HTML viewer)
            holo_lining = []
            hp = holo_pocket_data.get(uid)
            if hp:
                holo_lining = sorted(set(int(r[1]) for r in hp.get("lining_residues", [])))

            # Catalytic residues: core_residues from catalytic_site_validation.json
            cat_entry = catalytic_data.get(uid, {})
            catalytic = cat_entry.get("positions", [])
            cat_descs = cat_entry.get("descriptions", [])

            pml = generate_cofold_pml(fname, enzyme_name, uid, active_resi,
                                      meta=meta, evidence_lines=evidence_lines,
                                      chain_names=chain_names,
                                      holo_lining_resi=holo_lining,
                                      catalytic_resi=catalytic,
                                      catalytic_descs=cat_descs)
            (cofold_dir / fname.replace(".pdb", "_pocket_analysis.pml")).write_text(pml)
            cofold_count += 1

        print(f"  {proj['name']}: apo={len(list(apo_dir.glob('*.pdb')))} cofold={len(list(cofold_dir.glob('*.pdb')))}")

    print(f"\nTotal scripts: {apo_count} apo + {cofold_count} cofold = {apo_count + cofold_count}")


if __name__ == "__main__":
    main()
