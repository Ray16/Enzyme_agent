"""Unit tests for chemical-reasoning upgrades: reaction-center diff, entailment guardrails,
targeted scope follow-up, comparative synthesis, and expanded counterfactual evals."""
from __future__ import annotations

import json
import pytest

from enzyme_discovery_agent.eval import chemical_reasoning
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models.comparative_synthesis import build_comparative_brief, write_comparative_synthesis
from enzyme_discovery_agent.models.evidence_followup import run_scope_followup, select_scope_followup_candidates
from enzyme_discovery_agent.models.reaction_center import analyze_reaction_center
from enzyme_discovery_agent.models import target_review as review
from enzyme_discovery_agent.tests.test_target_review_gate import assessment, setup
from enzyme_discovery_agent.tools.family_reactions import _product_side_chebi_ids
from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences


def test_reaction_center_distinguishes_condensation_spectators_cyclization_and_cleavage():
    # 1. Intermolecular condensation: acid + amine -> amide
    cond = analyze_reaction_center(
        reactants=[
            {"occurrence_id": "native:0:0", "name": "propionic acid", "smiles": "CCC(=O)O"},
            {"occurrence_id": "native:1:0", "name": "propylamine", "smiles": "CCCN"},
        ],
        products=[
            {"occurrence_id": "product:0:0", "name": "N-propylpropanamide", "smiles": "CCC(=O)NCCC"},
        ],
    )
    assert cond is not None
    assert cond["status"] == "complete"
    assert cond["molecularity"] == "intermolecular_condensation"
    assert cond["consumed_groups"].get("carboxyl") == 1
    assert cond["consumed_groups"].get("primary_amine") == 1
    assert cond["formed_groups"].get("amide") == 1

    # 2. Spectator functional group trap: hydroxy-acid esterifies at OH while carboxyl stays unchanged
    spec = analyze_reaction_center(
        reactants=[
            {"occurrence_id": "native:0:0", "name": "3-hydroxypropanoic acid", "smiles": "OCCC(=O)O"},
            {"occurrence_id": "native:1:0", "name": "acetic acid", "smiles": "CC(=O)O"},
        ],
        products=[
            {"occurrence_id": "product:0:0", "name": "3-acetoxypropanoic acid", "smiles": "CC(=O)OCCC(=O)O"},
        ],
    )
    assert spec is not None
    assert spec["formed_groups"].get("ester") == 1
    assert "hydroxyl" in spec["consumed_groups"]
    # One carboxyl was consumed (esterified) and one remained spectator; ester formed.
    assert spec["consumed_groups"].get("carboxyl") == 1

    # 3. Intramolecular cyclization: 4-aminobutanoic acid -> 2-pyrrolidone (ring_delta = +1)
    cyc = analyze_reaction_center(
        reactants=[{"occurrence_id": "native:0:0", "name": "4-aminobutanoic acid", "smiles": "NCCCC(=O)O"}],
        products=[{"occurrence_id": "product:0:0", "name": "2-pyrrolidone", "smiles": "O=C1CCCN1"}],
    )
    assert cyc is not None
    assert cyc["molecularity"] == "intramolecular_cyclization"
    assert cyc["ring_delta"] == 1

    # 4. Cleavage: ester -> acid + alcohol
    clv = analyze_reaction_center(
        reactants=[{"occurrence_id": "native:0:0", "name": "ethyl acetate", "smiles": "CC(=O)OCC"}],
        products=[
            {"occurrence_id": "product:0:0", "name": "acetic acid", "smiles": "CC(=O)O"},
            {"occurrence_id": "product:1:0", "name": "ethanol", "smiles": "CCO"},
        ],
    )
    assert clv is not None
    assert clv["molecularity"] == "cleavage"
    assert clv["consumed_groups"].get("ester") == 1


def test_product_side_chebi_extraction_and_right_side_occurrences():
    entry = {
        "comments": [{
            "commentType": "CATALYTIC ACTIVITY",
            "reaction": {
                "name": "compound_a + compound_b = 2 compound_c",
                "reactionCrossReferences": [
                    {"database": "ChEBI", "id": "CHEBI:1"},
                    {"database": "ChEBI", "id": "CHEBI:2"},
                    {"database": "ChEBI", "id": "CHEBI:3"},
                ],
            },
        }]
    }
    names = {"CHEBI:1": ["compound_a"], "CHEBI:2": ["compound_b"], "CHEBI:3": ["compound_c"]}
    assert _product_side_chebi_ids(entry, lambda cid: names[cid]) == ["CHEBI:3"]

    prods = substrate_occurrences(
        "compound_a + compound_b = 2 compound_c",
        [{"chebi_id": "CHEBI:3", "name": "compound_c", "smiles": "CCO"}],
        side="right",
    )
    assert len(prods) == 2
    assert [p["occurrence_id"] for p in prods] == ["product:0:0", "product:0:1"]


def test_entailment_guardrails_cap_overclaimed_direct_observation_and_reject_bad_occurrence(tmp_path):
    packet = setup(tmp_path)
    # 1. Overclaiming direct_target_observation when target_roles rely on transfer/hypothesis
    overclaimed = {
        **assessment("exclude"),
        "evidence_level": "direct_target_observation",
        "target_roles": [
            {"participant": "T1", "role": "Assumed donor analog", "basis": "hypothesis"},
            {"participant": "T2", "role": "Analog class reported inactive", "basis": "incompatible"},
        ],
        "constraints": [{"finding": "Analog class inactive", "basis": "observed_mismatch", "severity": "major"}],
    }
    saved = review.save_assessment(tmp_path, packet, overclaimed)
    assert saved["evidence_level"] == "mechanistic_transfer"
    assert saved["requested_evidence_level"] == "direct_target_observation"
    assert "capped at mechanistic_transfer" in saved["admission_note"]

    # 2. Hallucinated native occurrence ID is rejected when scope defines occurrences
    update_candidate_registry(tmp_path, "SYNTH1", None, {
        "native_substrate_reaction_scope": {
            "reaction_index": 0,
            "reaction": "A + B = C",
            "native_substrates": [{"occurrence_id": "native:0:0", "name": "A", "smiles": "CCO"}],
        }
    })
    row = review.review_queue(tmp_path, force=True)[0]
    row["source_id"] = "R1"
    packet2 = {"target": review.target_contract(tmp_path), "target_inputs": packet["target_inputs"], "candidates": [row]}
    bad_occ = {
        **assessment("investigate"),
        "target_roles": [
            {"participant": "T1", "role": "Maps to native:9:0", "basis": "transfer"},
            {"participant": "T2", "role": "Maps to native:0:0", "basis": "transfer"},
        ],
    }
    with pytest.raises(ValueError, match="native:9:0"):
        review.save_assessment(tmp_path, packet2, bad_occ)


def test_scope_followup_enriches_diverse_investigate_candidates_and_reopens_review(tmp_path):
    packet = setup(tmp_path)
    update_candidate_registry(tmp_path, "SYNTH1", None, {
        "gene_names": ["synA"], "organism": "Syntheticus modelus", "interpro_family_id": "IPR_FAM1",
    })
    row = review.review_queue(tmp_path, force=True)[0]
    row["source_id"] = "R1"
    packet = {"target": review.target_contract(tmp_path), "target_inputs": packet["target_inputs"], "candidates": [row]}
    review.save_assessment(tmp_path, packet, assessment("investigate"))

    # Before follow-up, candidate is already reviewed and not in review_queue.
    assert review.review_queue(tmp_path) == []
    selected = select_scope_followup_candidates(tmp_path, limit=2)
    assert [s["accession"] for s in selected] == ["SYNTH1"]

    fake_papers = [{
        "title": "Biochemical characterization and substrate specificity of SynA",
        "doi": "10.1000/syna",
        "pmid": "99999999",
        "year": 2026,
        "abstract": "Purified SynA exhibits broad substrate specificity and measurable kcat on aliphatic analogs.",
    }]
    out = run_scope_followup(tmp_path, limit=2, search_fn=lambda q: fake_papers)
    assert out["enriched"] == ["SYNTH1"]

    # Enriched scope_followup_evidence changes scientific_record so candidate re-enters review_queue.
    meta = load_candidate_registry(tmp_path)["candidates"]["SYNTH1"]["metadata"]
    assert meta["scope_followup_evidence"]["passages"][0]["doi"] == "10.1000/syna"
    requeued = review.review_queue(tmp_path)
    assert [r["accession"] for r in requeued] == ["SYNTH1"]
    assert "scope_followup_evidence" in requeued[0]["record"]

    # Running again without a new assessment does not repeat the same lookup.
    assert select_scope_followup_candidates(tmp_path, limit=2) == []


def test_comparative_synthesis_groups_families_and_collapses_orthologs(tmp_path):
    packet = setup(tmp_path)
    base_meta = load_candidate_registry(tmp_path)["candidates"]["SYNTH1"]["metadata"]
    for acc, fam, gene, seq, dec in [
        ("SYNTH1", "IPR_F1", "synA", "MAAAA", "prioritize"),
        ("SYNTH2", "IPR_F1", "synA", "MAAAA", "prioritize"),  # exact-sequence ortholog
        ("SYNTH3", "IPR_F2", "synB", "MCCCC", "investigate"),
    ]:
        claims = [{"claim_type": k, "subject": acc, "verdict": "supported"} for k in ("accession_identity", "record_active")]
        update_candidate_registry(tmp_path, acc, "evaluated", {
            **base_meta,
            "protein_name": f"Enzyme {gene}",
            "gene_names": [gene],
            "sequence": seq,
            "interpro_family_id": fam,
            "claim_validation": {"promotion_eligible": True, "claims": claims},
        })
        row = review.review_queue(tmp_path, accessions=[acc], force=True)[0]
        row["source_id"] = "R1"
        p = {"target": review.target_contract(tmp_path), "target_inputs": packet["target_inputs"], "candidates": [row]}
        review.save_assessment(tmp_path, p, {**assessment(dec), "accession": acc})

    brief = write_comparative_synthesis(tmp_path)
    assert brief["total_admitted_candidates"] == 3
    assert brief["total_families"] == 2
    fam1 = next(f for f in brief["families"] if "IPR_F1" in f["family"])
    assert fam1["n_prioritized"] == 2
    assert fam1["n_distinct_representatives"] == 1
    assert len(fam1["collapsed_orthologs"]) == 1
    assert (tmp_path / "artifacts" / "comparative_synthesis.json").is_file()
    assert "Cross-Family Comparative Chemical Synthesis" in brief["markdown"]


def test_expanded_chemical_reasoning_eval_cases_grade_cleanly():
    suite = chemical_reasoning.cases()
    assert len(suite) >= 16
    ids = {c["id"] for c in suite}
    assert {"c12", "c13", "c14", "c15", "c16"}.issubset(ids)
    for case in suite:
        perfect = {
            "claims": [
                {"id": cid, "verdict": verdict, "evidence_refs": ["S01"], "reason": "Grounded in S01."}
                for cid, verdict in case["expected"].items()
            ],
            "next_action": case["acceptable_actions"][0],
        }
        graded = chemical_reasoning.grade(case, perfect)
        assert graded["passed"] is True, f"{case['id']} failed: {graded}"
        assert graded["action_matches_reference"] is True
