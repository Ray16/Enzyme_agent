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
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("enzyme_agent")


def _provenance_dir(run_dir: Path) -> Path:
    d = Path(run_dir) / "provenance"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _name_slug(provenance: dict) -> str:
    """A filesystem-safe enzyme-name slug for the provenance filename.

    Prefers the protein name, then the first gene name; empty when neither is
    known (some TrEMBL leads have no name), so the caller falls back to the
    accession alone.  Non-alphanumerics collapse to single hyphens; the
    accession is never part of the slug so it stays an unambiguous suffix.
    """
    name = (provenance.get("protein_name") or "").strip()
    if not name:
        genes = provenance.get("gene_names") or []
        name = (genes[0] if genes else "") or ""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return slug[:60].strip("-")


def _provenance_filename(provenance: dict) -> str:
    """``<enzyme-name-slug>_<accession>.json`` — human-readable, and the
    accession suffix keeps it unique when names are blank or duplicated."""
    acc = provenance.get("accession") or "unknown"
    slug = _name_slug(provenance)
    return f"{slug}_{acc}.json" if slug else f"{acc}.json"


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
    catalytic_residues: list[dict] = []
    characterization: list[dict] = []
    homologs: list[dict] = []
    is_characterization_paper = False
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
        catalytic_residues = literature_extraction.get("catalytic_residues") or []
        # Every measured result keeps the conditions it was obtained under and the
        # paper it came from — a number without its conditions/source is not usable.
        for m in literature_extraction.get("characterization", []) or []:
            characterization.append({
                "metric": m.get("metric"),
                "value": m.get("value"),
                "units": m.get("units"),
                "substrate": m.get("substrate"),
                "conditions": m.get("conditions") or {},
                "paper_doi": paper.get("doi"),
                "paper_title": paper.get("title"),
            })
        homologs = literature_extraction.get("homologs_or_related_enzymes") or []
        is_characterization_paper = bool(literature_extraction.get("is_characterization_paper"))

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
        "catalytic_residues": catalytic_residues,
        "is_characterization_paper": is_characterization_paper,
        "characterization_results": characterization,
        "homologs_or_related_enzymes": homologs,
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
    path = _provenance_dir(run_dir) / _provenance_filename(provenance)
    path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    return path


def load_candidate_provenance(run_dir: Path, accession: str) -> dict | None:
    """Look up a candidate's provenance by accession.

    Files are named ``<name-slug>_<accession>.json`` (accession suffix), so match
    on the suffix; fall back to the legacy ``<accession>.json`` layout.
    """
    d = _provenance_dir(run_dir)
    candidates = sorted(d.glob(f"*_{accession}.json")) + [d / f"{accession}.json"]
    for path in candidates:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _accommodation_facts(target_smiles: str, native_smiles: list[str]) -> list[dict]:
    """Compute per-target-substrate accommodation facts from stored/extracted
    native SMILES (RDKit only, no network)."""
    if not target_smiles or not native_smiles:
        return []
    try:
        from enzyme_discovery_agent.chemistry_logic import compare_substrate_properties
        from enzyme_discovery_agent.tools.chemistry import compute_fingerprint_similarity
        from enzyme_discovery_agent.tools.substrate import compute_substrate_properties
    except Exception:
        return []
    facts = []
    for t in [s for s in target_smiles.split(".") if s.strip()]:
        best, bn = 0.0, ""
        for n in native_smiles:
            sim = compute_fingerprint_similarity(t, n).get("data", {}).get("similarity", 0.0)
            if sim > best:
                best, bn = sim, n
        if bn:
            p = compare_substrate_properties(t, bn, compute_substrate_properties)
            if p:
                facts.append(p)
    return facts


def enrich_leads_and_write_provenance(run_dir, evaluations, query, llm_client, log_fn=None):
    """Write a provenance file per evaluated candidate, recovering native
    substrates + mechanism from the paper for fact-poor leads that carry a paper
    link (curation misses them).

    Extracts ONCE PER PAPER (cached) — a paper characterises several enzymes, so
    one call serves every candidate deposited in it — and matches each candidate
    to its own enzyme by organism/name.  Cost is naturally bounded by the number
    of DISTINCT papers that deposited a mined accession (a small set), so there
    is no arbitrary paper cap — every fact-poor lead with a paper gets enriched.
    Fully defensive: any per-item failure is logged and skipped; never blocks
    finalization.  Mutates ``evaluations`` in place so ranking sees the recovery.
    """
    import json as _json
    from pathlib import Path as _Path
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry

    run = _Path(run_dir)
    reg = load_candidate_registry(run).get("candidates", {})
    try:
        tc = _json.loads((run / "target_context.json").read_text())
        target_smiles = tc.get("substrate_smiles") or ""
    except Exception:
        target_smiles = ""

    # Group fact-poor, paper-linked candidates by their paper so each paper is
    # read at most once.
    from enzyme_discovery_agent.tools.literature_extraction import (
        find_candidate_paper, extract_from_paper, _match_enzyme_record,
        select_relevant_sections,
    )
    from enzyme_discovery_agent.chemistry_logic import resolve_substrate_names_to_smiles

    by_paper: dict[str, list] = {}
    src_by_acc: dict[str, str] = {}
    for ev in evaluations:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if not acc:
            continue
        src = reg.get(acc, {}).get("metadata", {}).get("source_tool", "")
        src_by_acc[acc] = src
        if not d.get("native_substrates") and "fulltext_genbank" in src and llm_client is not None:
            paper = find_candidate_paper(run, src)
            if paper:
                by_paper.setdefault(paper["text_path"], []).append((d, acc, src, paper))

    extraction_by_acc: dict[str, dict] = {}
    n_enriched = 0
    for path, group in by_paper.items():
        try:
            text = _Path(path).read_text(errors="ignore")
        except OSError:
            continue
        organisms = [g[0].get("organism", "") for g in group]
        names = [g[0].get("protein_name", "") for g in group]
        anchors = [group[0][3]["deposited_accession"]]
        for o in organisms:
            anchors += o.split()[:2]
        for n in names:
            anchors += [w for w in n.split() if len(w) >= 3]
        try:
            obj = extract_from_paper(
                select_relevant_sections(text, identifiers=anchors),
                group[0][3]["deposited_accession"], names[0], llm_client,
                query=query, organism=organisms[0],
            )
        except Exception as exc:
            logger.warning("literature extraction failed for paper %s: %s", path, exc)
            continue
        paper = group[0][3]
        for d, acc, src, _ in group:
            rec = _match_enzyme_record(obj.get("enzymes") or [], paper["deposited_accession"],
                                       organism=d.get("organism", ""),
                                       protein_name=d.get("protein_name", ""))
            if not rec:
                continue
            subs = [{"name": n, "smiles": resolve_substrate_names_to_smiles(n)}
                    for n in (rec.get("native_substrates") or [])]
            lit_natives = [{"name": s["name"], "smiles": s["smiles"], "chebi_id": ""}
                           for s in subs if s.get("smiles")]
            extraction_by_acc[acc] = {
                "paper": paper, "native_substrates": subs,
                "mechanism": rec.get("reaction"),
                "catalytic_residues": rec.get("catalytic_residues") or [],
                "characterization": rec.get("characterization") or [],
                "homologs_or_related_enzymes": obj.get("homologs_or_related_enzymes") or [],
                "is_characterization_paper": obj.get("is_characterization_paper", False),
                "paper_relevant": obj.get("paper_relevant", True),
                "relevance_reason": obj.get("relevance_reason", ""),
            }
            if lit_natives:
                d["native_substrates"] = lit_natives
                facts = _accommodation_facts(target_smiles, [s["smiles"] for s in lit_natives])
                if facts:
                    d.setdefault("substrate_similarity", {})["property_comparisons"] = facts
                n_enriched += 1

    # Write a provenance file for every evaluated candidate.
    n_prov = 0
    for ev in evaluations:
        d = ev.get("data", ev)
        acc = d.get("accession", "")
        if not acc:
            continue
        accom = (d.get("substrate_similarity") or {}).get("property_comparisons", [])
        try:
            prov = build_candidate_provenance(
                d, source_tool=src_by_acc.get(acc, ""),
                literature_extraction=extraction_by_acc.get(acc),
                accommodation_facts=accom,
            )
            write_candidate_provenance(run, prov)
            n_prov += 1
        except Exception as exc:
            logger.warning("provenance write failed for %s: %s", acc, exc)
    if log_fn:
        log_fn(f"  Provenance: wrote {n_prov} files; recovered substrates for "
               f"{n_enriched} fact-poor lead(s) from {len(by_paper)} paper(s)")
    return n_enriched


def compact_provenance_for_ranking(provenance: dict) -> dict:
    """The decision-relevant projection the LLM ranks over — small enough that the
    model does not exhaust its budget (the full file stays on disk for audit and
    the structural-modelling handoff)."""
    mech = provenance.get("mechanism") or {}
    subs = []
    for s in provenance.get("native_substrates", []):
        subs.append({"name": s.get("name"), "source": s.get("source", "").split(":")[0]})
    # Compact measured results: metric=value(units) on substrate — the strongest
    # activity signal for ranking; full conditions stay in the on-disk file.
    measured = [
        f"{m.get('metric')}={m.get('value')}{m.get('units') or ''}"
        + (f" on {m.get('substrate')}" if m.get("substrate") else "")
        for m in (provenance.get("characterization_results") or [])[:6]
    ]
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
        "measured_results": measured,
        "characterization_paper_count": provenance.get("characterization", {}).get("characterization_papers", 0),
        "has_structure": provenance.get("structure", {}).get("has_pdb")
        or provenance.get("structure", {}).get("has_alphafold"),
        "critical_warnings": provenance.get("critical_warnings", []),
    }
