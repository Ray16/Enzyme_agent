#!/usr/bin/env python3
"""
Generate interactive HTML pocket visualization for all enzymes.

Produces a single self-contained HTML file with side-by-side apo/holo
3D viewers for every enzyme, using 3Dmol.js (loaded from CDN).

Each enzyme card shows:
  - Apo structure (AlphaFold) with pocket surface colored by residue type
  - Holo structure (cofolded) with pocket surface + substrate/cofactor sticks
  - Key metrics: volumes, hydrophobicity, volume change, Jaccard overlap

Surface coloring (YRB-like):
  Gold    = hydrophobic (Ala, Val, Leu, Ile, Phe, Trp, Met, Pro)
  Blue    = basic (Arg, Lys, His)
  Red     = acidic (Asp, Glu)
  White   = polar (Ser, Thr, Asn, Gln, Tyr, Cys, Gly)

Output: all_structures/pocket_visualization.html

Usage:
    python generate_pocket_html.py
"""

import csv
import json
import os
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
ANALYSIS = BASE / "analysis_results"

PROJECTS = [
    {
        "name": "synthetase",
        "label": "Amide Synthetases",
        "pa_path": BASE / "amide_synthetase" / "pocket_analysis.json",
        "apo_dir": BASE / "all_structures" / "amide_synthetases_candidates" / "alphafold_apo_structure",
        "cofold_dir": BASE / "all_structures" / "amide_synthetases_candidates" / "cofolded_structure",
        "cofold_input_dir": BASE / "amide_synthetase" / "cofold_input",
        "cofold_output_dir": BASE / "amide_synthetase" / "cofold_output",
    },
    {
        "name": "hydrolase",
        "label": "Amide Hydrolases",
        "pa_path": BASE / "amide_hydrolase" / "pocket_analysis.json",
        "apo_dir": BASE / "all_structures" / "amide_hydrolase_candidates" / "alphafold_apo_structure",
        "cofold_dir": BASE / "all_structures" / "amide_hydrolase_candidates" / "cofolded_structure",
        "cofold_input_dir": BASE / "amide_hydrolase" / "cofold_input",
        "cofold_output_dir": BASE / "amide_hydrolase" / "cofold_output",
    },
]

CSV_PATH = ANALYSIS / "comprehensive_enzyme_table.csv"


def load_csv_data() -> dict:
    data = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            uid = r["uniprot_id"].strip()
            data[uid] = r
    return data


def extract_uid(filename: str) -> str:
    return filename.replace(".pdb", "").rsplit("_", 1)[-1]


def get_active_site_and_lining(entry: dict) -> tuple[list[int], list[int]]:
    """Extract active site residues and lining residues from pocket_analysis.json entry."""
    ev = entry.get("evidence") or {}
    active = set()
    for r in ev.get("active_site_residues", []):
        active.add(int(r))
    for lig, contacts in (ev.get("ligand_contacts") or {}).items():
        for r in contacts:
            active.add(int(r))
    for metal, contacts in (ev.get("metal_contacts") or {}).items():
        for r in contacts:
            active.add(int(r))

    bp = entry.get("binding_pocket") or {}
    lining = set()
    for r in bp.get("lining_residues", []):
        lining.add(int(r[1]))

    return sorted(active), sorted(lining)


def get_evidence_source(entry: dict) -> dict:
    """Extract a human-readable pocket evidence summary from pocket_analysis.json entry.

    Returns dict with keys:
        source_label: short label for display (e.g. "AlphaFill", "FoldSeek → AlphaFill")
        ligands: list of ligand names involved (e.g. ["ATP", "MG"])
        homolog_pdb: PDB ID of homolog if foldseek was used
        homolog_uniprot: UniProt ID of homolog if foldseek was used
        tmscore: TM-score of structural alignment (foldseek cases)
        detail: one-line description for tooltip/display
    """
    ev = entry.get("evidence") or {}
    etype = ev.get("type", "none")
    info = {"source_label": "None", "ligands": [], "homolog_pdb": None,
            "homolog_uniprot": None, "tmscore": None, "detail": "No evidence"}

    # Collect ligand names from all possible fields
    ligands = []
    for lig_dict_key in ("ligand_contacts", "ligands", "metal_contacts"):
        for lig_name in (ev.get(lig_dict_key) or {}):
            if lig_name not in ligands:
                ligands.append(lig_name)
    # Also from homolog_ligands (foldseek cases)
    for lig_name in (ev.get("homolog_ligands") or {}):
        if lig_name not in ligands:
            ligands.append(lig_name)
    info["ligands"] = ligands
    lig_str = ", ".join(ligands) if ligands else ""

    if etype == "alphafill":
        info["source_label"] = "AlphaFill"
        info["detail"] = f"AlphaFill ligand contacts ({lig_str})" if lig_str else "AlphaFill ligand contacts"

    elif etype == "uniprot_annotation":
        info["source_label"] = "UniProt"
        pname = ev.get("protein_name", "")
        ec = ", ".join(ev.get("ec_numbers", []))
        parts = ["UniProt curated active/binding sites"]
        if ec:
            parts.append(f"EC {ec}")
        info["detail"] = " — ".join(parts)

    elif etype == "foldseek+alphafill":
        homolog_pdb = ev.get("homolog_pdb", "?")
        homolog_uid = ev.get("homolog_uniprot", "")
        tm = ev.get("tmscore")
        info["source_label"] = "FoldSeek → AlphaFill"
        info["homolog_pdb"] = homolog_pdb
        info["homolog_uniprot"] = homolog_uid
        info["tmscore"] = tm
        detail = f"FoldSeek homolog {homolog_pdb}"
        if homolog_uid:
            detail += f" ({homolog_uid})"
        if tm is not None:
            detail += f" TM={tm:.2f}"
        detail += " → AlphaFill contacts"
        if lig_str:
            detail += f" ({lig_str})"
        info["detail"] = detail

    elif etype == "foldseek+uniprot":
        homolog_pdb = ev.get("homolog_pdb", "?")
        homolog_uid = ev.get("homolog_uniprot", "")
        tm = ev.get("tmscore")
        info["source_label"] = "FoldSeek → UniProt"
        info["homolog_pdb"] = homolog_pdb
        info["homolog_uniprot"] = homolog_uid
        info["tmscore"] = tm
        detail = f"FoldSeek homolog {homolog_pdb}"
        if homolog_uid:
            detail += f" ({homolog_uid})"
        if tm is not None:
            detail += f" TM={tm:.2f}"
        detail += " → UniProt annotations"
        if lig_str:
            detail += f" ({lig_str})"
        info["detail"] = detail

    elif etype == "none":
        note = ev.get("note", "")
        info["detail"] = note if note else "No catalytic site evidence found"

    else:
        info["source_label"] = etype
        info["detail"] = etype

    return info


def get_pocket_confidence(entry: dict, evidence_info: dict) -> dict:
    """Classify pocket confidence based on evidence quality and pocket overlap.

    Two axes combined:
      - Evidence quality: how reliable is the active site identification?
        (direct source vs foldseek-mediated, TM-score)
      - Pocket match: does the detected pocket capture the active site residues?
        (overlap fraction, number of residues)

    Returns dict with keys:
        level: "High" | "Medium" | "Low" | "Unknown"
        reason: short explanation of why this level was assigned
    """
    ev = entry.get("evidence") or {}
    etype = ev.get("type", "none")
    tm = evidence_info.get("tmscore")
    total = ev.get("active_site_total", 0)
    overlap = ev.get("active_site_overlap", 0)
    n_res = len(ev.get("active_site_residues", []))
    overlap_frac = overlap / total if total else 0.0

    if etype == "none":
        return {"level": "Unknown Confidence", "reason": "No catalytic site evidence available"}

    is_direct = etype in ("alphafill", "uniprot_annotation")
    good_source = is_direct or (tm is not None and tm >= 0.8)
    moderate_source = not is_direct and tm is not None and 0.6 <= tm < 0.8

    if good_source and overlap_frac >= 0.5 and n_res >= 2:
        src = "direct evidence" if is_direct else f"high TM ({tm:.2f})"
        return {"level": "High Confidence",
                "reason": f"Good source ({src}), {overlap}/{total} residues in pocket ({overlap_frac:.0%})"}

    if (good_source and overlap_frac > 0 and n_res >= 2) or \
       (moderate_source and overlap_frac >= 0.5):
        if good_source:
            src = "direct evidence" if is_direct else f"high TM ({tm:.2f})"
            return {"level": "Medium Confidence",
                    "reason": f"Good source ({src}), but partial overlap {overlap}/{total} ({overlap_frac:.0%})"}
        else:
            return {"level": "Medium Confidence",
                    "reason": f"Moderate TM ({tm:.2f}), but good overlap {overlap}/{total} ({overlap_frac:.0%})"}

    # Low: evidence exists but fails above criteria
    reasons = []
    if not good_source and not moderate_source and tm is not None:
        reasons.append(f"low TM ({tm:.2f})")
    if overlap_frac == 0 and total > 0:
        reasons.append(f"pocket captures 0/{total} active site residues")
    if n_res < 2:
        reasons.append(f"only {n_res} residue(s) identified")
    if not reasons:
        reasons.append(f"partial overlap {overlap}/{total} ({overlap_frac:.0%})")
    return {"level": "Low Confidence", "reason": "; ".join(reasons)}


def get_chain_names(cofold_input_dir: Path, uid: str) -> dict:
    """Read cofold input YAML to build chain_id -> human-readable name mapping.

    Returns e.g. {"B": "ATP", "C": "Mg²⁺", "D": "hexamethylenediamine", "E": "adipic_acid"}
    """
    yaml_path = cofold_input_dir / f"{uid}.yaml"
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return {}

    chain_names = {}
    # Also read substrate names from project config for SMILES->name lookup
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

    for seq in cfg.get("sequences", []):
        if "ligand" in seq:
            lig = seq["ligand"]
            chain_id = lig.get("id", "")
            ccd = lig.get("ccd", "")
            smiles = lig.get("smiles", "")
            if ccd:
                # Use CCD code directly (ATP, MG, etc.)
                name = ccd
                if ccd in ("MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU"):
                    name = {"MG": "Mg\u00b2\u207a", "MN": "Mn\u00b2\u207a", "ZN": "Zn\u00b2\u207a",
                            "FE": "Fe\u00b2\u207a", "CA": "Ca\u00b2\u207a", "CO": "Co\u00b2\u207a",
                            "NI": "Ni\u00b2\u207a", "CU": "Cu\u00b2\u207a"}.get(ccd, ccd)
            elif smiles:
                name = smiles_to_name.get(smiles, smiles[:20])
            else:
                name = f"chain {chain_id}"
            chain_names[chain_id] = name
    return chain_names


METAL_CCDS = {"MG", "MN", "ZN", "FE", "CA", "CO", "NI", "CU"}
COFACTOR_CCDS = {"ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "NADP", "FAD", "FMN",
                 "COA", "SAM", "SAH", "PLP", "TPP", "HEM"} | METAL_CCDS


def load_substrate_iptm(cofold_output_dir: Path, uid: str,
                        chain_names: dict) -> dict[str, float]:
    """Load per-substrate protein→chain ipTM from Boltz-2 confidence JSON.

    Returns dict mapping substrate display name to pairwise ipTM value,
    e.g. {"hexamethylenediamine": 0.323, "adipic acid": 0.388}.
    Skips cofactor/metal chains (ATP, MG, etc.).
    """
    if not cofold_output_dir or not chain_names:
        return {}

    uid_dir = cofold_output_dir / uid
    if not uid_dir.is_dir():
        return {}

    conf_files = sorted(uid_dir.glob("**/confidence_*_model_*.json"))
    if not conf_files:
        return {}

    try:
        with open(conf_files[0]) as f:
            data = json.load(f)
    except Exception:
        return {}

    pair_iptm = data.get("pair_chains_iptm", {})
    if not pair_iptm:
        return {}

    # Map PDB chain letters (B,C,D,...) to Boltz-2 chain indices (1,2,3,...)
    # Non-protein chains are ordered alphabetically starting from B
    sorted_chains = sorted(chain_names.keys())
    result = {}
    for i, chain_letter in enumerate(sorted_chains):
        boltz_idx = str(i + 1)
        name = chain_names[chain_letter]

        # Skip cofactors and metals — only show substrates
        raw_name = name.upper().replace("²⁺", "").replace(" ", "")
        if raw_name in COFACTOR_CCDS:
            continue

        prot_to_chain = pair_iptm.get("0", {}).get(boltz_idx)
        if prot_to_chain is not None:
            result[name] = round(prot_to_chain, 3)

    return result


def read_pdb(path: Path) -> str | None:
    if not path.exists():
        return None
    import gzip
    try:
        with open(path) as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with gzip.open(path, "rt") as f:
                return f.read()
        except Exception:
            return None


def strip_pdb(pdb_text: str) -> str:
    """Strip PDB to essential columns for 3Dmol.js rendering.

    Keeps ATOM/HETATM (cols 0-54 + element cols 76-78).
    Drops occupancy (54-60) and B-factor (60-66) to save space,
    but preserves the element column so 3Dmol.js correctly identifies
    non-standard elements (e.g., MG, ZN, FE).
    Also keeps TER, END, CONECT for chain boundaries and bond connectivity.
    """
    out = []
    for line in pdb_text.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            # Keep coords (0-54), pad to col 76, append element (76-78)
            base = line[:54]
            if len(line) >= 78:
                elem = line[76:78]
                out.append(f"{base:<76s}{elem}")
            else:
                out.append(base)
        elif line.startswith(("TER", "END", "CONECT")):
            out.append(line.rstrip())
    return "\n".join(out)


def escape_for_js(pdb_text: str) -> str:
    """Escape PDB text for embedding in JS template literal."""
    return pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def html_header(title_suffix: str = "") -> str:
    title = f"Pocket Analysis: Apo vs Holo Structures{title_suffix}"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
       margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
h1 {{ text-align: center; margin-bottom: 5px; }}
.page-sub {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }}
.toc {{ background: white; border-radius: 8px; padding: 15px 20px; margin-bottom: 25px;
       box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.toc h2 {{ margin: 0 0 10px 0; font-size: 16px; }}
.toc-list {{ column-count: 4; column-gap: 20px; list-style: none; padding: 0; margin: 0; font-size: 13px; }}
.toc-list li {{ margin-bottom: 3px; }}
.toc-list a {{ color: #2563eb; text-decoration: none; }}
.toc-list a:hover {{ text-decoration: underline; }}
.section-header {{ font-size: 20px; font-weight: bold; margin: 30px 0 15px 0;
                   padding-bottom: 8px; border-bottom: 2px solid #2563eb; }}
.card {{ background: white; border-radius: 8px; padding: 18px; margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card-title {{ font-size: 18px; font-weight: bold; margin-bottom: 3px; }}
.card-sub {{ color: #666; font-size: 13px; margin-bottom: 10px; }}
.metrics {{ display: flex; gap: 10px; margin: 8px 0; flex-wrap: wrap; }}
.m {{ padding: 4px 10px; background: #f0f4ff; border-radius: 4px; font-size: 12px; }}
.ml {{ color: #888; font-size: 10px; }}
.mv {{ font-weight: bold; font-size: 14px; }}
.legend {{ font-size: 11px; color: #555; margin: 5px 0 10px 0; }}
.legend .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px;
                  vertical-align: middle; margin-right: 2px; margin-left: 8px; }}
.viewers {{ display: flex; gap: 15px; }}
.vp {{ flex: 1; min-width: 0; }}
.vl {{ font-weight: bold; text-align: center; font-size: 13px; margin-bottom: 4px; color: #555; }}
.viewer-box {{ width: 100%; height: 450px; position: relative; border: 1px solid #ddd; border-radius: 4px; }}
.no-structure {{ display: flex; align-items: center; justify-content: center;
                height: 450px; border: 1px solid #ddd; border-radius: 4px;
                background: #fafafa; color: #999; font-size: 14px; }}
.evidence-row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 12px; flex-wrap: wrap; }}
.ev-label {{ color: #888; }}
.ev-badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: 600; font-size: 11px; }}
.ev-alphafill {{ background: #e8f5e9; color: #2e7d32; }}
.ev-uniprot {{ background: #e3f2fd; color: #1565c0; }}
.ev-foldseek-alphafill {{ background: #fff3e0; color: #e65100; }}
.ev-foldseek-uniprot {{ background: #f3e5f5; color: #7b1fa2; }}
.ev-none {{ background: #f5f5f5; color: #999; }}
.ev-detail {{ color: #666; font-size: 11px; }}
.ev-ligands {{ display: inline-block; padding: 1px 5px; background: #f0f0f0; border-radius: 2px;
              font-family: monospace; font-size: 11px; margin-left: 2px; }}
.conf-badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: 600;
              font-size: 11px; cursor: help; }}
.conf-high {{ background: #c8e6c9; color: #1b5e20; }}
.conf-medium {{ background: #fff9c4; color: #f57f17; }}
.conf-low {{ background: #ffcdd2; color: #b71c1c; }}
.conf-unknown {{ background: #f5f5f5; color: #999; }}
.footer-note {{ max-width: 900px; margin: 30px auto; padding: 15px 20px; background: white;
               border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 12px; color: #555; }}
.footer-note h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #333; }}
.footer-note table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
.footer-note th, .footer-note td {{ text-align: left; padding: 4px 10px; border-bottom: 1px solid #eee; font-size: 12px; }}
.footer-note th {{ color: #888; font-weight: 600; }}
.cofold-only-note {{ font-size: 12px; color: #e67e22; margin-top: 5px; text-align: center; }}
.loading-placeholder {{ display: flex; align-items: center; justify-content: center;
                        height: 450px; border: 1px solid #ddd; border-radius: 4px;
                        background: #fafafa; color: #aaa; font-size: 13px; }}
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<h1>{title}</h1>
<div class="page-sub">
  Interactive 3D pocket visualization with YRB surface coloring.
  Pocket defined by active site residues (AlphaFill / UniProt / FoldSeek evidence).<br>
  Rotate: left-click drag | Zoom: scroll | Translate: right-click drag<br>
  Viewers load on scroll for performance.
</div>
"""


def html_footer(n_total: int) -> str:
    return f"""
<div class="footer-note">
<h3>Pocket Confidence Levels</h3>
<p>Confidence combines two axes: <b>evidence quality</b> (how reliable is the active site identification)
and <b>pocket match</b> (does the detected cavity capture the identified residues).
Hover over a confidence badge for a per-enzyme explanation.</p>
<table>
<tr><th>Level</th><th>Criteria</th></tr>
<tr><td><span class="conf-badge conf-high">High Confidence</span></td>
    <td>Good source (direct AlphaFill/UniProt, or FoldSeek with TM &ge; 0.8)
        AND &ge; 50% of active site residues fall within the detected pocket
        AND &ge; 2 residues identified</td></tr>
<tr><td><span class="conf-badge conf-medium">Medium Confidence</span></td>
    <td>Good source with partial overlap (&gt; 0% but &lt; 50%) and &ge; 2 residues,
        OR moderate source (FoldSeek TM 0.6&ndash;0.8) with &ge; 50% overlap</td></tr>
<tr><td><span class="conf-badge conf-low">Low Confidence</span></td>
    <td>Evidence exists but: pocket captures 0% of active site residues,
        or structural homology is weak (TM &lt; 0.6),
        or only 1 residue was identified</td></tr>
<tr><td><span class="conf-badge conf-unknown">Unknown Confidence</span></td>
    <td>No catalytic site evidence found from any source</td></tr>
</table>
</div>
<div style="text-align:center;color:#999;font-size:12px;margin-top:30px;">
  {n_total} enzymes visualized. Generated by generate_pocket_html.py
</div>
<script>
// Lazy-load 3Dmol viewers when cards scroll into view
var yrb = {{prop:"resn",map:{{ALA:"#FFD700",VAL:"#FFD700",LEU:"#FFD700",ILE:"#FFD700",
  PHE:"#FFD700",TRP:"#FFD700",MET:"#FFD700",PRO:"#FFD700",
  ARG:"#4169E1",LYS:"#4169E1",HIS:"#4169E1",
  ASP:"#DC143C",GLU:"#DC143C",
  SER:"#FFFFFF",THR:"#FFFFFF",ASN:"#FFFFFF",GLN:"#FFFFFF",
  TYR:"#FFFFFF",CYS:"#FFFFFF",GLY:"#FFFFFF"}}}};

function initViewers(card) {{
  var cfgEl = card.querySelector('.viewer-config');
  if (!cfgEl) return;
  var cfg = JSON.parse(cfgEl.textContent);
  cfgEl.remove();  // prevent re-init

  if (cfg.apo_id) {{
    var apoDiv = document.getElementById(cfg.apo_id);
    var pdbEl = document.getElementById('pdb_' + cfg.apo_id);
    if (apoDiv && pdbEl) {{
      var a = $3Dmol.createViewer(apoDiv, {{backgroundColor: "white"}});
      a.addModel(pdbEl.textContent, "pdb");
      a.setStyle({{}}, {{cartoon: {{color: "gray", opacity: 0.7}}}});
      var apoPocketSel = {{resi: cfg.apo_pocket_resi}};
      if (cfg.apo_pocket_resi.length > 0) {{
        a.addSurface($3Dmol.SurfaceType.VDW,
          {{opacity: 0.6, colorscheme: yrb}}, apoPocketSel, apoPocketSel);
        a.zoomTo(apoPocketSel);
      }} else {{
        a.zoomTo();
      }}
      a.render();
      pdbEl.remove();
    }}
  }}

  if (cfg.holo_id) {{
    var holoDiv = document.getElementById(cfg.holo_id);
    var pdbEl = document.getElementById('pdb_' + cfg.holo_id);
    if (holoDiv && pdbEl) {{
      var h = $3Dmol.createViewer(holoDiv, {{backgroundColor: "white", hoverDuration: 100}});
      h.addModel(pdbEl.textContent, "pdb");
      // Protein backbone as gray cartoon
      h.setStyle({{chain: "A"}}, {{cartoon: {{color: "gray", opacity: 0.7}}}});
      var holoPocketSel = {{resi: cfg.holo_pocket_resi, chain: "A"}};
      if (cfg.holo_pocket_resi.length > 0) {{
        h.addSurface($3Dmol.SurfaceType.VDW,
          {{opacity: 0.6, colorscheme: yrb}}, holoPocketSel, holoPocketSel);
      }}
      // Substrates/cofactors (non-protein chains) as sticks
      h.setStyle({{not: {{chain: "A"}}}},
        {{stick: {{radius: 0.3}}}});
      // Override Mg2+ ions as larger green spheres (replaces ball-and-stick)
      h.setStyle({{elem: "MG"}},
        {{sphere: {{radius: 1.0, color: "green"}}}});
      // Hover labels for ligands — use chain_names for readable names
      var chainNames = cfg.chain_names || {{}};
      h.setHoverable({{not: {{chain: "A"}}}}, true,
        function(atom, viewer, event, container) {{
          if (!atom.hovLabel) {{
            var lbl = chainNames[atom.chain] || atom.resn || atom.elem;
            atom.hovLabel = viewer.addLabel(lbl, {{
              position: atom, backgroundColor: "black",
              fontColor: "white", fontSize: 12,
              borderThickness: 1, borderColor: "gray",
              showBackground: true
            }});
          }}
        }},
        function(atom, viewer) {{
          if (atom.hovLabel) {{
            viewer.removeLabel(atom.hovLabel);
            delete atom.hovLabel;
          }}
        }}
      );
      if (cfg.holo_pocket_resi.length > 0)
        h.zoomTo(holoPocketSel);
      else
        h.zoomTo();
      h.render();
      pdbEl.remove();
    }}
  }}
}}

// Use IntersectionObserver to lazy-load
var observer = new IntersectionObserver(function(entries) {{
  entries.forEach(function(entry) {{
    if (entry.isIntersecting) {{
      initViewers(entry.target);
      observer.unobserve(entry.target);
    }}
  }});
}}, {{rootMargin: "200px"}});

document.querySelectorAll('.card').forEach(function(card) {{
  observer.observe(card);
}});
</script>
</body></html>"""


def enzyme_card_html(
    idx: int,
    uid: str,
    enzyme_name: str,
    meta: dict,
    apo_pdb: str | None,
    cofold_pdb: str | None,
    active_resi: list[int],
    apo_lining_resi: list[int],
    holo_lining_resi: list[int],
    evidence_info: dict | None = None,
    confidence: dict | None = None,
    chain_names: dict | None = None,
    substrate_iptm: dict[str, float] | None = None,
    enzyme_family: str = "",
) -> str:
    """Generate HTML for one enzyme card with embedded 3Dmol viewers."""

    organism = meta.get("organism", "")
    apo_vol = meta.get("apo_pocket_volume_A3", "")
    holo_vol = meta.get("holo_pocket_volume_A3", "")
    apo_hydro = meta.get("apo_frac_hydrophobic", "")
    holo_hydro = meta.get("holo_frac_hydrophobic", "")
    vol_change = meta.get("volume_change_pct", "")
    jaccard = meta.get("residue_jaccard", "")
    overlap = meta.get("active_site_overlap", "")
    cofold_iptm = meta.get("cofold_iptm", "")
    contact_frac = meta.get("holo_contact_fraction", "")
    holo_vol_ratio = meta.get("holo_pocket_to_substrate_ratio", "")
    substrate_burial = meta.get("substrate_sasa_burial", "")
    cofactor_burial = meta.get("cofactor_sasa_burial", "")
    mean_plddt = meta.get("mean_protein_plddt", "")
    holo_shape = meta.get("holo_pocket_shape", "")
    cofold_strategy = meta.get("cofold_strategy", "")

    apo_id = f"apo_{idx}"
    holo_id = f"holo_{idx}"

    parts = []
    parts.append(f'<div class="card" id="enzyme_{uid}">')
    parts.append(f'<div class="card-title">{enzyme_name} ({uid})</div>')
    sub_parts = [organism]
    if enzyme_family:
        sub_parts.append(f'<b>{enzyme_family}</b>')
    parts.append(f'<div class="card-sub">{" | ".join(p for p in sub_parts if p)}</div>')

    # Metrics row
    metrics = []
    if apo_vol:
        metrics.append(f'<div class="m"><div class="ml">Apo Vol</div><div class="mv">{float(apo_vol):.0f} \u00c5\u00b3</div></div>')
    if holo_vol:
        metrics.append(f'<div class="m"><div class="ml">Holo Vol</div><div class="mv">{float(holo_vol):.0f} \u00c5\u00b3</div></div>')
    if holo_vol_ratio:
        ratio_val = float(holo_vol_ratio)
        metrics.append(f'<div class="m"><div class="ml">Vol Ratio (Pocket/Substrate)</div><div class="mv">{ratio_val:.1f}x</div></div>')
    if vol_change:
        sign = "+" if float(vol_change) > 0 else ""
        metrics.append(f'<div class="m"><div class="ml">\u0394 Vol</div><div class="mv">{sign}{float(vol_change):.1f}%</div></div>')
    if apo_hydro:
        metrics.append(f'<div class="m"><div class="ml">Apo Hydro</div><div class="mv">{apo_hydro}</div></div>')
    if holo_hydro:
        metrics.append(f'<div class="m"><div class="ml">Holo Hydro</div><div class="mv">{holo_hydro}</div></div>')
    if jaccard:
        metrics.append(f'<div class="m"><div class="ml">Jaccard</div><div class="mv">{jaccard}</div></div>')
    if overlap:
        metrics.append(f'<div class="m"><div class="ml">Binding Site Overlap</div><div class="mv">{overlap}</div></div>')
    if substrate_iptm:
        for sub_name, sub_val in substrate_iptm.items():
            metrics.append(f'<div class="m"><div class="ml">ipTM ({sub_name})</div><div class="mv">{sub_val:.3f}</div></div>')
    if cofold_iptm:
        metrics.append(f'<div class="m"><div class="ml">ipTM (overall)</div><div class="mv">{cofold_iptm}</div></div>')
    if contact_frac:
        metrics.append(f'<div class="m"><div class="ml">Contact Frac</div><div class="mv">{contact_frac}</div></div>')
    if holo_shape:
        shape_label = holo_shape.replace("_", " ").title()
        metrics.append(f'<div class="m"><div class="ml">Holo Shape</div><div class="mv">{shape_label}</div></div>')
    if substrate_burial:
        metrics.append(f'<div class="m"><div class="ml">Substrate Burial</div><div class="mv">{substrate_burial}</div></div>')
    if cofactor_burial:
        metrics.append(f'<div class="m"><div class="ml">Cofactor Burial</div><div class="mv">{cofactor_burial}</div></div>')
    if cofold_strategy:
        strat_label = cofold_strategy.replace("_", " ").title()
        metrics.append(f'<div class="m"><div class="ml">Cofold Strategy</div><div class="mv">{strat_label}</div></div>')
    if mean_plddt:
        metrics.append(f'<div class="m"><div class="ml">pLDDT</div><div class="mv">{mean_plddt}</div></div>')

    parts.append(f'<div class="metrics">{"".join(metrics)}</div>')

    # Evidence source and confidence row
    conf_html = ""
    if confidence:
        level = confidence["level"]
        reason = confidence["reason"]
        conf_css = {"High Confidence": "conf-high", "Medium Confidence": "conf-medium",
                    "Low Confidence": "conf-low", "Unknown Confidence": "conf-unknown"}.get(level, "conf-unknown")
        conf_html = f'<span class="conf-badge {conf_css}" title="{reason}">{level}</span>'

    if evidence_info and evidence_info.get("source_label", "None") != "None":
        ev = evidence_info
        label = ev["source_label"]
        # Map source label to CSS class
        css_map = {
            "AlphaFill": "ev-alphafill",
            "UniProt": "ev-uniprot",
            "FoldSeek → AlphaFill": "ev-foldseek-alphafill",
            "FoldSeek → UniProt": "ev-foldseek-uniprot",
        }
        badge_css = css_map.get(label, "ev-none")
        ev_parts = [f'<span class="ev-label">Pocket Source:</span>']
        ev_parts.append(f'<span class="ev-badge {badge_css}">{label}</span>')
        if ev.get("ligands"):
            for lig in ev["ligands"]:
                ev_parts.append(f'<span class="ev-ligands">{lig}</span>')
        if ev.get("homolog_pdb"):
            tm_str = f' TM={ev["tmscore"]:.2f}' if ev.get("tmscore") is not None else ""
            homolog_label = f'via {ev["homolog_pdb"]}'
            if ev.get("homolog_uniprot"):
                homolog_label += f' ({ev["homolog_uniprot"]})'
            homolog_label += tm_str
            ev_parts.append(f'<span class="ev-detail">{homolog_label}</span>')
        if conf_html:
            ev_parts.append(conf_html)
        parts.append(f'<div class="evidence-row">{"".join(ev_parts)}</div>')
    elif confidence:
        ev_parts = ['<span class="ev-label">Pocket Source:</span>',
                     '<span class="ev-badge ev-none">No evidence</span>']
        if conf_html:
            ev_parts.append(conf_html)
        parts.append(f'<div class="evidence-row">{"".join(ev_parts)}</div>')

    # Legend
    parts.append('''<div class="legend">
Surface: <span class="swatch" style="background:#FFD700"></span>Hydrophobic
<span class="swatch" style="background:#4169E1"></span>Basic
<span class="swatch" style="background:#DC143C"></span>Acidic
<span class="swatch" style="background:#FFFFFF;border:1px solid #ccc"></span>Polar
</div>''')

    # Viewers
    parts.append('<div class="viewers">')

    # Apo panel
    parts.append('<div class="vp">')
    parts.append('<div class="vl">Apo (AlphaFold)</div>')
    if apo_pdb and apo_lining_resi:
        parts.append(f'<div id="{apo_id}" class="viewer-box"></div>')
    elif apo_pdb:
        parts.append(f'<div id="{apo_id}" class="viewer-box"></div>')
    else:
        parts.append('<div class="no-structure">No AlphaFold structure available</div>')
    parts.append('</div>')

    # Holo panel
    parts.append('<div class="vp">')
    parts.append('<div class="vl">Holo (Cofolded)</div>')
    if cofold_pdb:
        parts.append(f'<div id="{holo_id}" class="viewer-box"></div>')
    else:
        parts.append('<div class="no-structure">No cofolded structure available</div>')
    parts.append('</div>')

    parts.append('</div>')  # viewers

    if not apo_pdb and cofold_pdb:
        parts.append('<div class="cofold-only-note">No AlphaFold structure; showing cofolded model only</div>')

    # Store PDB data in hidden script tags INSIDE the card for lazy loading
    if apo_pdb:
        parts.append(f'<script type="text/plain" id="pdb_{apo_id}">{escape_for_js(strip_pdb(apo_pdb))}</script>')
    if cofold_pdb:
        parts.append(f'<script type="text/plain" id="pdb_{holo_id}">{escape_for_js(strip_pdb(cofold_pdb))}</script>')

    # Store viewer config as JSON INSIDE the card for the lazy-loading observer
    init_config = {
        "apo_id": apo_id if apo_pdb else None,
        "holo_id": holo_id if cofold_pdb else None,
        "apo_pocket_resi": list(apo_lining_resi),
        "holo_pocket_resi": list(holo_lining_resi) if holo_lining_resi else list(apo_lining_resi),
        "active_resi": list(active_resi),
        "chain_names": chain_names or {},
    }
    parts.append(f'<script type="application/json" class="viewer-config">{json.dumps(init_config)}</script>')

    parts.append('</div>')  # card

    return "\n".join(parts)


def main():
    csv_data = load_csv_data()

    # Load enzyme family classification (Pfam-based)
    family_data = {}
    family_csv = ANALYSIS / "enzyme_family_classification.csv"
    if family_csv.exists():
        with open(family_csv) as f:
            for r in csv.DictReader(f):
                uid = r.get("uniprot_id", "").strip()
                fam = r.get("assigned_family", "")
                pfam = r.get("pfam_ids", "").strip()
                # Map to human-readable names
                if fam == "Gamma_PGA_hydro":
                    family_data[uid] = "Polyglutamate hydrolase (PF05908)"
                elif fam == "Peptidase_S51":
                    family_data[uid] = "Cyanophycinase, S51 (PF03575)"
                elif fam in ("Transglut_core", "Peptidase_C71", "Pseudomurein endoisopeptidase"):
                    family_data[uid] = "Pseudomurein hydrolase"
                elif fam == "no_domain_annotation":
                    family_data[uid] = ""
                elif fam:
                    family_data[uid] = fam + (f" ({pfam})" if pfam else "")
        print(f"Loaded family classification for {len(family_data)} enzymes")

    # Load holo pocket residues (from holo_pocket_analysis.py)
    holo_pocket_path = ANALYSIS / "holo_pocket_residues.json"
    holo_pocket_data = {}
    if holo_pocket_path.exists():
        with open(holo_pocket_path) as f:
            holo_pocket_data = json.load(f)
        print(f"Loaded holo pocket residues for {len(holo_pocket_data)} enzymes")

    html_parts = [html_header()]

    # Table of contents
    toc_items = {"synthetase": [], "hydrolase": []}

    total = 0
    all_cards = {"synthetase": [], "hydrolase": []}

    for proj in PROJECTS:
        pa_data = {}
        if proj["pa_path"].exists():
            with open(proj["pa_path"]) as f:
                pa_data = json.load(f)

        apo_dir = proj["apo_dir"]
        cofold_dir = proj["cofold_dir"]

        # Collect all UIDs from both apo and cofold directories
        uids = set()
        for pdb in apo_dir.glob("*.pdb"):
            uids.add(extract_uid(pdb.name))
        for pdb in cofold_dir.glob("*.pdb"):
            uids.add(extract_uid(pdb.name))

        for uid in sorted(uids):
            meta = csv_data.get(uid, {})
            enzyme_name = meta.get("enzyme_name", uid)

            # Find PDB files
            apo_pdbs = list(apo_dir.glob(f"*_{uid}.pdb")) + list(apo_dir.glob(f"*{uid}*.pdb"))
            cofold_pdbs = list(cofold_dir.glob(f"*_{uid}.pdb")) + list(cofold_dir.glob(f"*{uid}*.pdb"))

            apo_pdb = read_pdb(apo_pdbs[0]) if apo_pdbs else None
            cofold_pdb = read_pdb(cofold_pdbs[0]) if cofold_pdbs else None

            if not apo_pdb and not cofold_pdb:
                continue

            # Get pocket residues — separate apo and holo
            pa = pa_data.get(uid, {})
            if isinstance(pa, dict) and pa.get("role") == "candidate":
                active_resi, apo_lining_resi = get_active_site_and_lining(pa)
                evidence_info = get_evidence_source(pa)
                confidence = get_pocket_confidence(pa, evidence_info)
            else:
                active_resi, apo_lining_resi = [], []
                evidence_info = None
                confidence = None

            # Holo pocket residues from holo_pocket_analysis
            holo_pa = holo_pocket_data.get(uid, {})
            if holo_pa and holo_pa.get("lining_residues"):
                holo_lining_resi = sorted(set(
                    int(r[1]) for r in holo_pa["lining_residues"]
                ))
            else:
                holo_lining_resi = []

            # Chain name mapping for ligand hover labels
            cofold_input_dir = proj.get("cofold_input_dir")
            chain_names = get_chain_names(cofold_input_dir, uid) if cofold_input_dir else {}

            # Per-substrate ipTM from Boltz-2 confidence files
            cofold_output_dir = proj.get("cofold_output_dir")
            sub_iptm = load_substrate_iptm(cofold_output_dir, uid, chain_names) if cofold_output_dir else {}

            total += 1
            card = enzyme_card_html(
                idx=total,
                uid=uid,
                enzyme_name=enzyme_name,
                meta=meta,
                apo_pdb=apo_pdb,
                cofold_pdb=cofold_pdb,
                active_resi=active_resi,
                apo_lining_resi=apo_lining_resi,
                holo_lining_resi=holo_lining_resi,
                evidence_info=evidence_info,
                confidence=confidence,
                chain_names=chain_names,
                substrate_iptm=sub_iptm,
                enzyme_family=family_data.get(uid, ""),
            )
            all_cards[proj["name"]].append(card)
            toc_items[proj["name"]].append((uid, enzyme_name))

    # Write separate HTML files per project
    out_dir = BASE / "all_structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for proj in PROJECTS:
        pname = proj["name"]
        cards = all_cards[pname]
        items = toc_items[pname]
        if not cards:
            continue

        file_parts = [html_header(f" — {proj['label']}")]

        # TOC
        file_parts.append('<div class="toc">')
        file_parts.append(f'<h2>{proj["label"]} ({len(items)} enzymes)</h2>')
        file_parts.append('<ul class="toc-list">')
        for uid, ename in items:
            file_parts.append(f'<li><a href="#enzyme_{uid}">{ename} ({uid})</a></li>')
        file_parts.append('</ul>')
        file_parts.append('</div>')

        file_parts.extend(cards)
        file_parts.append(html_footer(len(cards)))

        out_path = out_dir / f"pocket_visualization_{pname}s.html"
        with open(out_path, "w") as f:
            f.write("\n".join(file_parts))

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"Written: {out_path}")
        print(f"  Size: {size_mb:.1f} MB, {len(cards)} enzymes")

    print(f"\nTotal: {total} enzymes ({len(all_cards['synthetase'])} synthetases, {len(all_cards['hydrolase'])} hydrolases)")


if __name__ == "__main__":
    main()
