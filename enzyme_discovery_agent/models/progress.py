"""Measure changes in research evidence, excluding conversation and bookkeeping."""

import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import discovery_status, load_candidate_registry
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.models.modeling_feedback import modeling_feedback_basis
from enzyme_discovery_agent.run_layout import run_file

_TRANSIENT = {"timestamp", "ts", "updated_at", "last_updated", "first_seen", "derived_at"}
_EVIDENCE_FIELDS = (
    "protein_name", "gene_names", "ec_numbers", "catalytic_activities",
    "interpro_family_id", "claim_validation", "literature_evidence", "literature",
    "native_substrates", "substrate_similarity", "mechanism_evidence",
    "structural_evidence", "modeling_result",
    "expression_data",
    "literature_lead", "identity_resolution_status",
)


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in _TRANSIENT}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def research_signature(run_dir: Path, papers: list[dict]) -> str:
    """Notebook reads/writes and repeated identical evidence are not progress."""
    from enzyme_discovery_agent.tools.literature import _paper_key
    registry = load_candidate_registry(run_dir, strict=True).get("candidates", {})
    evidence = {}
    for acc, entry in registry.items():
        meta = entry.get("metadata") or {}
        evidence[acc] = {"status": discovery_status(entry),
                         **{field: meta.get(field) for field in _EVIDENCE_FIELDS}}
        evidence[acc]["modeling_feedback"] = modeling_feedback_basis(meta)
        mechanism = evidence[acc].get("mechanism_evidence")
        if isinstance(mechanism, dict):
            # Rewording an extraction is not newly retrieved evidence. Track
            # source identity rather than arbitrary changes in generated prose.
            source = mechanism.get("paper") or (mechanism.get("facts") or {}).get("provenance") or {}
            evidence[acc]["mechanism_evidence"] = (
                {"source": source, "text_sha256": mechanism.get("text_sha256")}
                if mechanism.get("status") == "ok" and isinstance(source, dict) and
                any(source.get(k) for k in ("doi", "pmid", "pmcid")) else None)
    records = {}
    for name in ("mechanism_families.json", "family_landscape.json",
                 "artifacts/fulltext/manifest.json"):
        try:
            records[name] = json.loads((run_file(run_dir, name)).read_text())
        except FileNotFoundError:
            pass
    payload = {"candidates": evidence, "records": records,
               "papers": sorted({str(key) for p in papers if (key := _paper_key(p)) is not None})}
    # A novel negative search still closes a research question. Cached repeats,
    # errors, notebook activity and completion notices do not create new work.
    payload["evidence_queries"] = sorted({
        json.dumps({"tool": event.get("tool"), "args": event.get("args") or {}}, sort_keys=True)
        for event in read_events(run_dir)
        if event.get("type") == "tool_call"
        and (str(event.get("tool", "")).startswith(("search_", "fetch_"))
             or event.get("tool") in {"deep_literature_search", "follow_paper_citations", "get_protein_domains",
                                      "get_candidate_evidence",
                                      "get_modelseed_compound", "lookup_genbank_to_uniprot",
                                      "survey_family_reactions",
                                      "extract_reaction_mechanism", "extract_experimental_measurements"})
        and event.get("status") not in {"error", "skipped"}
    })
    return hashlib.sha256(json.dumps(_stable(payload), sort_keys=True, default=str).encode()).hexdigest()
