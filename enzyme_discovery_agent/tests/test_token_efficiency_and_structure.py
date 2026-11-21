"""Regression tests for token-efficiency compaction, tool filtering, and structural helpers."""
from __future__ import annotations

import json
import pytest

from enzyme_discovery_agent import config
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.context import stable_tool_definitions
from enzyme_discovery_agent.models import research_decisions as decisions
from enzyme_discovery_agent.models import target_review as review
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json, file_sha256, json_sha256, read_json
from enzyme_discovery_agent.tests.test_review_protocol_recovery import pending


def test_compact_record_removes_boilerplate_while_preserving_scientific_facts():
    rxn = "synthetic_donor + synthetic_acceptor + ATP = synthetic_product + ADP + phosphate"
    sub = {
        "chebi_id": "CHEBI:999001",
        "name": "synthetic_donor",
        "smiles": "CCO",
        "occurrence_id": "native:0:0",
        "stoichiometry": 1,
        "stoichiometry_status": "source_recorded",
        "stoichiometry_source": rxn,
    }
    raw = {
        "accession": "SYNTH1",
        "protein_name": "Synthetic catalyst",
        "catalytic_activities": [rxn],
        "native_substrates": [dict(sub)],
        "native_substrate_reaction_scope": {
            "reaction_index": 1,
            "reaction": rxn,
            "ec_number": "6.9.9.9",
            "native_substrates": [dict(sub)],
        },
        "claim_validation": {
            "schema_version": "1.0",
            "subject": "SYNTH1",
            "verdict": "partially_supported",
            "promotion_eligible": True,
            "blocking_claims": [],
            "claims": [
                {
                    "claim_type": "accession_identity",
                    "subject": "SYNTH1",
                    "asserted_value": "SYNTH1",
                    "verdict": "supported",
                    "evidence": {"source": "UniProtKB", "locator": "primaryAccession", "observed_value": "SYNTH1"},
                    "reason": "retrieved primary accession matches exactly",
                },
                {
                    "claim_type": "reviewed_status",
                    "subject": "SYNTH1",
                    "asserted_value": True,
                    "verdict": "contradicted",
                    "evidence": {"source": "UniProtKB", "locator": "entryType", "observed_value": "unreviewed"},
                    "reason": "record is not reviewed",
                },
            ],
        },
        "domain_architecture": [
            {
                "accession": "IPR999999",
                "name": "Synthetic catalytic domain",
                "type": "domain",
                "coordinates": [{"start": 10, "end": 220}],
            }
        ],
        "mechanism_evidence": {
            "activation": {
                "activation_mode": "synthetic_activation",
                "basis": {"source_type": "uniprot_catalytic_activity", "source_value": rxn},
            },
            "modeled_reaction_scope": {"reaction_index": 1, "reaction": rxn, "ec_number": "6.9.9.9"},
        },
        "identity_resolution": {
            "status": "resolved",
            "resolved_accession": "SYNTH1",
            "database_record_path": "/tmp/local/record.json",
            "database_record_sha256": "a" * 64,
        },
        "reaction_roles": None,
        "critical_warnings": [],
    }

    sci = review.scientific_record(raw)
    before_digest = review.digest(sci)
    compacted = review.compact_record(sci)

    # Scientific digest is unaffected by compact_record projection.
    assert review.digest(review.scientific_record(raw)) == before_digest

    # Routine supported claims collapse; contradicted/flagged claims stay intact.
    cv = compacted["claim_validation"]
    assert cv["supported_claims"] == ["accession_identity"]
    assert len(cv["flagged_claims"]) == 1
    assert cv["flagged_claims"][0]["claim_type"] == "reviewed_status"
    assert cv["flagged_claims"][0]["verdict"] == "contradicted"

    # Identical native_substrates is not duplicated outside native_substrate_reaction_scope,
    # and repeated stoichiometry_source reaction text is stripped.
    assert "native_substrates" not in compacted
    scoped_sub = compacted["native_substrate_reaction_scope"]["native_substrates"][0]
    assert scoped_sub["smiles"] == "CCO"
    assert "stoichiometry_source" not in scoped_sub

    # Domain architecture keeps ID, name, type, and residue span concisely.
    assert compacted["domain_architecture"] == ["IPR999999: Synthetic catalytic domain (domain, 10-220)"]

    # Local paths/hashes stripped from identity_resolution; empty/None fields omitted.
    assert compacted["identity_resolution"] == {"status": "resolved", "resolved_accession": "SYNTH1"}
    assert "reaction_roles" not in compacted
    assert "critical_warnings" not in compacted

    # Substantially smaller than raw JSON.
    assert len(json.dumps(compacted)) < len(json.dumps(sci)) * 0.65


def test_agent_filters_disabled_and_design_tools_and_exposes_named_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_CANDIDATE_MAP", False)
    agent = EnzymeAgent(output_dir=tmp_path, token_budget=1000)
    names = {tool["function"]["name"] for tool in agent.tools}
    assert "design_de_novo_enzyme" not in names
    assert "design_enzyme_with_disco" not in names
    assert "map_candidate_neighbors" not in names
    assert "expand_candidates_esm" not in names
    assert "search_uniprot" in names
    assert "plan_research_step" in names

    # Stable tool definitions remain identical regardless of include_inspections.
    assert stable_tool_definitions(agent.tools, include_inspections=True) == stable_tool_definitions(
        agent.tools, include_inspections=False
    )

    # Named gate attributes alias the exact pipeline instances for backward compatibility.
    assert agent._entity_gate is agent._gate_pipeline.gates[0]
    assert agent._coverage_gate is agent._gate_pipeline.gates[1]
    assert agent._eval_gate is agent._gate_pipeline.gates[2]
    assert agent._expansion_gate is agent._gate_pipeline.gates[3]


def test_plan_research_step_accepts_inline_outcome_interpretation(tmp_path):
    identifier, outcome = pending(tmp_path)
    proposal = {
        "hypothesis": "A complementary source may resolve the missing catalytic evidence.",
        "alternative": "No complementary source resolves it.",
        "scope": "retrieval",
        "prediction": "The complementary lookup returns a primary record.",
        "contrasting_prediction": "The lookup returns no primary record.",
        "why_this_test": "Test whether another database carries the missing record.",
        "revision_trigger": "Revisit if the complementary lookup is empty.",
        "evidence_refs": [outcome],
        "revises": identifier,
        "belief_update": "suspend",
        "outcome_interpretation": "The prior lookup was deduplicated and added no new evidence; switching route.",
        "tool_name": "search_uniprot",
        "arguments": {"query": "synthetic catalyst"},
    }
    validated = decisions.validate_plan(tmp_path, proposal, {"search_uniprot"})
    assert validated["outcome_interpretation"] == proposal["outcome_interpretation"]
    new_id = decisions.start_decision(tmp_path, proposal, {"search_uniprot"})
    saved = json.loads((decisions._root(tmp_path) / new_id / "plan.json").read_text())
    assert saved["outcome_interpretation"] == proposal["outcome_interpretation"]

    bad = {**proposal, "outcome_interpretation": "x" * (decisions.BRIEF_FIELD_LIMIT + 1)}
    with pytest.raises(ValueError, match="outcome_interpretation"):
        decisions.validate_plan(tmp_path, bad, {"search_uniprot"})


def test_state_store_shared_json_and_hash_helpers(tmp_path):
    missing = tmp_path / "missing.json"
    assert read_json(missing, {"fallback": True}) == {"fallback": True}
    target = tmp_path / "data.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert read_json(target) == {"a": 1, "b": 2}
    assert json_sha256({"a": 1, "b": 2}) == json_sha256({"b": 2, "a": 1})
    assert len(file_sha256(target)) == 64
