"""Per-candidate provenance file: the single source of truth for its evidence.

One JSON file per candidate under ``<run>/provenance/<accession>.json``, recording
WHAT we know about the candidate and WHERE each piece came from:

  * discovery — how the candidate was found (search / structural homolog /
    full-text mining of a deposited accession);
  * native substrates — each with its SOURCE (curated UniProt/Rhea/ChEBI vs
    extracted from a specific paper), so a substrate the enzyme was ranked on is
    always traceable;
  * mechanism — bond formed, direction, cofactor, reactive groups, and source
    (feeds ranking, the structural-modelling handoff, and paper-relevance);
  * substrate-accommodation facts vs the target;
  * structure / expression feasibility and the linked literature.

Evidence-gathering WRITES this file; ranking READS it (a compact projection).
Keeping the two decoupled makes every ranking decision auditable against a cited
source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("enzyme_agent")


def _provenance_dir(run_dir: Path) -> Path:
    d = Path(run_dir) / "provenance"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _discovery_from_source(source_tool: str) -> dict:
    st = source_tool or ""
    if st.startswith("fulltext_genbank:"):
        return {"route": "full_text_mining", "source_tool": st,
                "deposited_accession": st.split(":", 1)[1],
                "note": "recovered via a deposited GenBank accession in a paper"}
    if st.startswith("fulltext_uniprot"):
        return {"route": "full_text_mining", "source_tool": st,
                "note": "UniProt accession mined from paper full text"}
    if "structural" in st:
        return {"route": "structural_homology", "source_tool": st}
    return {"route": "sequence_search", "source_tool": st or "search_uniprot"}


def build_candidate_provenance(
    eval_data: dict,
    source_tool: str = "",
    literature_extraction: dict | None = None,
    accommodation_facts: list[dict] | None = None,
) -> dict:
    """Assemble the provenance record for one candidate from its evaluation data
    plus any literature extraction, tagging every substrate/mechanism with its
    source."""
    acc = eval_data.get("accession", "")
    reviewed = bool(eval_data.get("is_swissprot"))

    # Native substrates: curated (from evaluation) carry source 'curated';
    # literature-extracted carry source 'literature' + the paper.
    native = []
    for ns in eval_data.get("native_substrates", []) or []:
        if ns.get("name"):
            native.append({
                "name": ns.get("name"),
                "smiles": ns.get("smiles"),
                "source": "curated:UniProt/Rhea/ChEBI",
            })
    mechanism = None
    chem = eval_data.get("reaction_chemistry") or {}
    if chem.get("bond_types") or chem.get("reaction_direction"):
        mechanism = {
            "bond_types": chem.get("bond_types"),
            "direction": chem.get("reaction_direction"),
            "source": "curated:EC/catalytic-activity",
        }

    paper_ref = None
    if literature_extraction:
        paper = literature_extraction.get("paper") or {}
        paper_ref = {"doi": paper.get("doi"), "title": paper.get("title"),
                     "deposited_accession": paper.get("deposited_accession")}
        for s in literature_extraction.get("native_substrates", []) or []:
            native.append({
                "name": s.get("name"), "smiles": s.get("smiles"),
                "source": "literature", "paper_doi": paper.get("doi"),
                "paper_title": paper.get("title"),
            })
        lm = literature_extraction.get("mechanism")
        if lm:
            mechanism = {
                "bond_formed": lm.get("bond_formed"),
                "direction": lm.get("direction"),
                "cofactor": lm.get("cofactor"),
                "reactive_groups": lm.get("reactive_groups"),
                "summary": lm.get("summary"),
                "source": "literature",
                "paper_doi": paper.get("doi"),
                "paper_title": paper.get("title"),
            }

    struct = eval_data.get("structural_data") or {}
    lit = eval_data.get("literature") or {}
    return {
        "accession": acc,
        "protein_name": eval_data.get("protein_name", ""),
        "gene_names": eval_data.get("gene_names", []),
        "organism": eval_data.get("organism", ""),
        "ec_numbers": eval_data.get("ec_numbers", []),
        "interpro_family": eval_data.get("interpro_family", ""),
        "annotation_source": "SwissProt (reviewed)" if reviewed else "TrEMBL (unreviewed)",
        "discovery": _discovery_from_source(source_tool),
        "native_substrates": native,
        "mechanism": mechanism,
        "paper": paper_ref,
        "substrate_accommodation_vs_target": accommodation_facts or [],
        "provenance_note": (
            "native substrate recovered from literature; not present in UniProt"
            if (literature_extraction and literature_extraction.get("native_substrates"))
            else "annotation from curated databases"
        ),
        "structure": {
            "has_pdb": bool(struct.get("pdb_ids")),
            "pdb_ids": struct.get("pdb_ids", []),
            "has_alphafold": bool(struct.get("has_alphafold")),
        },
        "characterization": {
            "characterization_papers": lit.get("characterization_papers", 0),
            "engineering_papers": lit.get("engineering_papers", 0),
        },
        "critical_warnings": eval_data.get("critical_warnings", []),
    }


def write_candidate_provenance(run_dir: Path, provenance: dict) -> Path:
    acc = provenance.get("accession", "unknown")
    path = _provenance_dir(run_dir) / f"{acc}.json"
    path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    return path


def load_candidate_provenance(run_dir: Path, accession: str) -> dict | None:
    path = _provenance_dir(run_dir) / f"{accession}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def compact_provenance_for_ranking(provenance: dict) -> dict:
    """The decision-relevant projection the LLM ranks over — small enough that the
    model does not exhaust its budget (the full file stays on disk for audit and
    the structural-modelling handoff)."""
    mech = provenance.get("mechanism") or {}
    subs = []
    for s in provenance.get("native_substrates", []):
        subs.append({"name": s.get("name"), "source": s.get("source", "").split(":")[0]})
    return {
        "accession": provenance.get("accession"),
        "name": provenance.get("protein_name") or (provenance.get("gene_names") or [""])[0],
        "annotation_source": provenance.get("annotation_source"),
        "native_substrates": subs[:6],
        "mechanism": {
            "bond": mech.get("bond_formed") or mech.get("bond_types"),
            "direction": mech.get("direction"),
            "cofactor": mech.get("cofactor"),
        },
        "substrate_accommodation_vs_target": provenance.get("substrate_accommodation_vs_target", []),
        "characterization": provenance.get("characterization"),
        "has_structure": provenance.get("structure", {}).get("has_pdb")
        or provenance.get("structure", {}).get("has_alphafold"),
        "critical_warnings": provenance.get("critical_warnings", []),
    }
