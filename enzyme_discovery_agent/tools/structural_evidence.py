"""Structural evidence integration tool.

Reads co-folding pipeline results (CSV/JSON) and returns a structured
structural report for each candidate enzyme, allowing the agent to
incorporate binding quality, pocket chemistry, and apo-holo stability
into its evaluation and ranking.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from enzyme_discovery_agent.tool_registry import tool

logger = logging.getLogger("enzyme_agent")


def _read_csv_for_uid(csv_path: str, uid: str, uid_col: str = "uid") -> dict | None:
    """Read a CSV and return the row matching the given UID as a dict."""
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(uid_col, "").strip() == uid:
                return row
    return None


def _safe_float(val: str | None, default: float | None = None) -> float | None:
    """Convert a string to float, returning default on failure."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _find_analysis_results(cofold_results_dir: str) -> str:
    """Resolve the analysis_results directory from the given path."""
    p = Path(cofold_results_dir)
    if p.name == "analysis_results":
        return str(p)
    candidate = p / "analysis_results"
    if candidate.is_dir():
        return str(candidate)
    return str(p)


@tool(
    description=(
        "Retrieve structural evaluation data for a candidate enzyme from the "
        "co-folding pipeline results. Returns binding quality (iPTM, SASA "
        "burial), pocket chemistry (charged/hydrophobic/polar fractions), "
        "apo-holo comparison (residue Jaccard, volume change), and pocket "
        "geometry. Use this after co-folding data is available to add "
        "structural evidence to your candidate evaluation."
    ),
    category="reasoning",
    parallel_safe=True,
    params={
        "accession": {
            "type": "string",
            "description": (
                "UniProt accession of the candidate enzyme (e.g. "
                "'A0A0F7RJ52' for AsbA)."
            ),
        },
        "cofold_results_dir": {
            "type": "string",
            "description": (
                "Path to the directory containing co-folding analysis "
                "results. Should contain CSV files like "
                "cofolding_metrics.csv, holo_pocket_chemistry.csv, etc. "
                "Can be the analysis_results/ directory itself or its "
                "parent directory."
            ),
        },
    },
)
def get_structural_evaluation(
    accession: str,
    cofold_results_dir: str,
) -> dict:
    """Retrieve structural evaluation data for a candidate enzyme."""
    accession = accession.strip()
    if not accession:
        return {
            "status": "error",
            "summary": "accession is required.",
            "data": None,
            "count": 0,
            "suggestions": [],
        }

    results_dir = _find_analysis_results(cofold_results_dir)
    if not os.path.isdir(results_dir):
        return {
            "status": "error",
            "summary": f"Analysis results directory not found: {results_dir}",
            "data": None,
            "count": 0,
            "suggestions": [
                "Verify the cofold_results_dir path points to a directory "
                "containing cofolding_metrics.csv."
            ],
        }

    # Read co-folding metrics
    cofold = _read_csv_for_uid(
        os.path.join(results_dir, "cofolding_metrics.csv"), accession
    )
    if cofold is None:
        return {
            "status": "error",
            "summary": f"No co-folding data found for {accession} in {results_dir}.",
            "data": None,
            "count": 0,
            "suggestions": [
                f"Check that {accession} was included in the co-folding run.",
                "List available UIDs with: head -1 cofolding_metrics.csv",
            ],
        }

    # Read holo pocket chemistry
    holo_chem = _read_csv_for_uid(
        os.path.join(results_dir, "holo_pocket_chemistry.csv"), accession
    )

    # Read apo-holo comparison
    apo_holo = _read_csv_for_uid(
        os.path.join(results_dir, "holo_vs_apo_comparison.csv"), accession
    )

    # Read sub-region descriptors (if available from B1)
    subregions_path = os.path.join(results_dir, "substrate_pocket_subregions.json")
    sub_regions = None
    if os.path.isfile(subregions_path):
        try:
            with open(subregions_path) as f:
                all_subregions = json.load(f)
            sub_regions = all_subregions.get(accession)
        except Exception:
            pass

    # Read clash data (if available)
    clash_path = os.path.join(results_dir, "clash_analysis.json")
    clash_data = None
    if os.path.isfile(clash_path):
        try:
            with open(clash_path) as f:
                all_clashes = json.load(f)
            clash_data = all_clashes.get(accession)
        except Exception:
            pass

    # Build structured report
    substrate_iptm = _safe_float(cofold.get("substrate_iptm_to_protein"))
    if substrate_iptm is None:
        substrate_iptm = _safe_float(cofold.get("iptm"))
    sasa_burial = _safe_float(cofold.get("substrate_sasa_burial"))

    binding_quality = {
        "substrate_iptm": substrate_iptm,
        "sasa_burial_pct": round(sasa_burial * 100, 1) if sasa_burial else None,
        "confidence_score": _safe_float(cofold.get("confidence_score")),
        "complex_plddt": _safe_float(cofold.get("complex_plddt")),
        "cofactor_sasa_burial": _safe_float(cofold.get("cofactor_sasa_burial")),
        "substrate_min_distance": _safe_float(cofold.get("substrate_min_distance")),
        "n_contact_residues": _safe_float(cofold.get("substrate_n_contact_residues")),
    }

    pocket_chemistry = {}
    if holo_chem:
        pocket_chemistry = {
            "frac_charged": _safe_float(holo_chem.get("holo_frac_charged")),
            "frac_hydrophobic": _safe_float(holo_chem.get("holo_frac_hydrophobic")),
            "frac_polar": _safe_float(holo_chem.get("holo_frac_polar")),
            "avg_hydropathy": _safe_float(holo_chem.get("holo_avg_hydropathy")),
            "n_residues": _safe_float(holo_chem.get("holo_n_residues")),
        }
    else:
        pocket_chemistry = {
            "frac_charged": _safe_float(cofold.get("pocket_frac_charged")),
            "frac_hydrophobic": _safe_float(cofold.get("pocket_frac_hydrophobic")),
            "frac_polar": _safe_float(cofold.get("pocket_frac_polar")),
        }

    apo_holo_comparison = {}
    if apo_holo:
        apo_holo_comparison = {
            "residue_jaccard": _safe_float(apo_holo.get("residue_jaccard")),
            "volume_change_pct": _safe_float(apo_holo.get("volume_change_pct")),
            "apo_volume_A3": _safe_float(apo_holo.get("apo_volume_A3")),
            "holo_volume_A3": _safe_float(apo_holo.get("holo_volume_A3")),
            "apo_shape": apo_holo.get("apo_shape", ""),
            "holo_shape": apo_holo.get("holo_shape", ""),
        }

    pocket_geometry = {}
    if holo_chem:
        pocket_geometry = {
            "volume_A3": _safe_float(holo_chem.get("holo_volume_A3")),
            "area_A2": _safe_float(holo_chem.get("holo_area_A2")),
            "sa_v_ratio": _safe_float(holo_chem.get("holo_sa_v_ratio")),
        }

    # Build interpretive assessment
    assessment_parts = []
    if substrate_iptm is not None:
        if substrate_iptm >= 0.95:
            assessment_parts.append(f"High binding confidence (iPTM={substrate_iptm:.3f})")
        elif substrate_iptm >= 0.80:
            assessment_parts.append(f"Moderate binding confidence (iPTM={substrate_iptm:.3f})")
        else:
            assessment_parts.append(f"Low binding confidence (iPTM={substrate_iptm:.3f})")

    if sasa_burial is not None:
        burial_pct = sasa_burial * 100
        if burial_pct >= 85:
            assessment_parts.append(f"substrate well-enclosed ({burial_pct:.0f}% buried)")
        elif burial_pct >= 60:
            assessment_parts.append(f"substrate partially enclosed ({burial_pct:.0f}% buried)")
        else:
            assessment_parts.append(f"substrate poorly enclosed ({burial_pct:.0f}% buried)")

    frac_hydro = pocket_chemistry.get("frac_hydrophobic")
    frac_charged = pocket_chemistry.get("frac_charged")
    if frac_hydro is not None and frac_charged is not None:
        if frac_hydro > 0.40:
            assessment_parts.append(
                f"hydrophobic pocket ({frac_hydro:.0%}) favorable for uncharged substrates"
            )
        elif frac_charged > 0.35:
            assessment_parts.append(
                f"charged pocket ({frac_charged:.0%}) may cause electrostatic issues "
                "with charged substrate termini"
            )

    if apo_holo_comparison.get("residue_jaccard") is not None:
        jac = apo_holo_comparison["residue_jaccard"]
        if jac >= 0.8:
            assessment_parts.append("substrate placed in predicted catalytic site (high apo-holo overlap)")
        elif jac >= 0.5:
            assessment_parts.append("moderate apo-holo pocket overlap")
        else:
            assessment_parts.append("substrate placed outside predicted pocket (low apo-holo overlap)")

    assessment = "; ".join(assessment_parts) if assessment_parts else "Insufficient data for assessment."

    report = {
        "accession": accession,
        "enzyme_name": cofold.get("enzyme_name", ""),
        "project": cofold.get("project", ""),
        "cofold_strategy": cofold.get("strategy", ""),
        "binding_quality": binding_quality,
        "pocket_chemistry": pocket_chemistry,
        "apo_holo_comparison": apo_holo_comparison,
        "pocket_geometry": pocket_geometry,
        "assessment": assessment,
    }

    if sub_regions:
        report["sub_regions"] = sub_regions
    if clash_data:
        report["clash_data"] = clash_data

    summary_parts = [f"Structural evaluation for {accession}"]
    if substrate_iptm is not None:
        summary_parts.append(f"iPTM={substrate_iptm:.3f}")
    if sasa_burial is not None:
        summary_parts.append(f"burial={sasa_burial*100:.1f}%")
    if frac_hydro is not None:
        summary_parts.append(f"hydrophobic={frac_hydro:.0%}")

    return {
        "status": "success",
        "summary": ", ".join(summary_parts),
        "data": report,
        "count": 1,
        "suggestions": [],
    }
