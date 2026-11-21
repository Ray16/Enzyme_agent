"""Unit tests for holistic non-native discovery upgrades:
directional pocket envelope, UniProt architecture/promiscuity extraction,
two-axis target review, sequence-identity ortholog filtering, and structural feedback."""
from __future__ import annotations

import csv
import json
import pytest

from enzyme_discovery_agent.chemistry.accommodation import accommodation_facts
from enzyme_discovery_agent.chemistry.protein_context import (
    extract_architecture_and_tractability,
    extract_promiscuity_indicators,
)
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.family_learning import build_family_learning
from enzyme_discovery_agent.models.modeling_portfolio import ranked_portfolio
from enzyme_discovery_agent.models.ortholog import filter_highly_similar_orthologs, sequence_identity
from enzyme_discovery_agent.models import target_review as review
from enzyme_discovery_agent.tests.test_target_review_gate import assessment, setup


def test_directional_pocket_envelope_distinguishes_larger_vs_smaller_native_substrates():
    # Target: small 4-carbon acid (butyric acid, MW ~88)
    # Native large: 12-carbon acid (dodecanoic acid, MW ~200 -> ratio ~2.27 -> larger_native_envelope)
    large = accommodation_facts(["CCCC(=O)O"], ["CCCCCCCCCCCC(=O)O"])
    assert large is not None
    p_large = large["per_target"][0]
    assert p_large["pocket_size_relation"] == "larger_native_envelope"
    assert p_large["native_to_target_mw_ratio"] > 1.35

    # Target: 12-carbon acid (MW ~200) vs Native small: 2-carbon acetic acid (MW ~60 -> ratio ~0.30 -> smaller_native_envelope)
    small = accommodation_facts(["CCCCCCCCCCCC(=O)O"], ["CC(=O)O"])
    assert small is not None
    p_small = small["per_target"][0]
    assert p_small["pocket_size_relation"] == "smaller_native_envelope"
    assert p_small["native_to_target_mw_ratio"] < 0.70

    # Comparable: butyric acid vs propionic acid (MW ~74 vs ~88 -> ratio ~0.84 -> comparable_envelope)
    comp = accommodation_facts(["CCCC(=O)O"], ["CCC(=O)O"])
    assert comp is not None
    assert comp["per_target"][0]["pocket_size_relation"] == "comparable_envelope"


def test_protein_context_extracts_architecture_tractability_and_promiscuity():
    entry = {
        "features": [
            {"type": "Active site", "location": {"start": {"value": 120}, "end": {"value": 120}}, "description": "Nucleophile"},
            {"type": "Binding site", "location": {"start": {"value": 45}, "end": {"value": 45}}, "ligand": {"name": "ATP"}},
            {"type": "Binding site", "location": {"start": {"value": 48}, "end": {"value": 48}}, "ligand": {"name": "ATP"}},
        ],
        "comments": [
            {"commentType": "SUBUNIT", "texts": [{"value": "Homodimer."}]},
            {"commentType": "FUNCTION", "texts": [{"value": "Exhibits broad substrate specificity across various aliphatic substrates in secondary metabolite biosynthesis."}]},
            {"commentType": "CATALYTIC ACTIVITY", "reaction": {"name": "A + B = C"}},
            {"commentType": "CATALYTIC ACTIVITY", "reaction": {"name": "A + D = E"}},
        ],
    }
    arch = extract_architecture_and_tractability(entry, {"pdb_ids": ["1ABC", "2DEF"]})
    assert arch["subunit_class"] == "homooligomer"
    assert arch["is_membrane_protein"] is False
    assert arch["has_experimental_structure"] is True
    assert arch["pdb_count"] == 2
    assert arch["catalytic_residues"][0]["position"] == "120"
    assert arch["binding_sites"][0]["ligand"] == "ATP"
    assert arch["binding_sites"][0]["positions"] == ["45", "48"]

    prom = extract_promiscuity_indicators(entry, None, {"engineering_papers": 4})
    assert prom["distinct_annotated_reactions"] == 2
    assert prom["metabolic_context"] == "secondary_or_specialized"
    assert prom["engineering_precedent_papers"] == 4
    assert prom["broad_specificity_cues"]


def test_two_axis_target_review_accepts_and_validates_mechanism_and_pocket_outlook(tmp_path):
    packet = setup(tmp_path)
    two_axis = {
        **assessment("prioritize"),
        "mechanism_fit": "strong",
        "pocket_engineering_outlook": "larger_or_promiscuous_pocket",
    }
    saved = review.save_assessment(tmp_path, packet, two_axis)
    assert saved["mechanism_fit"] == "strong"
    assert saved["pocket_engineering_outlook"] == "larger_or_promiscuous_pocket"

    bad = {**assessment("investigate"), "mechanism_fit": "invalid_value"}
    with pytest.raises(ValueError, match="mechanism_fit"):
        review.save_assessment(tmp_path, packet, bad)


def test_sequence_identity_ortholog_filter_collapses_near_identical_but_keeps_divergent(tmp_path):
    seq_base = "M" + "ACDEFGHIKLMNPQRSTVWY" * 5  # 101 aa
    seq_near = seq_base[:-3] + "AAA"           # ~97% identical -> should collapse
    seq_div  = "M" + "WYVTSRQPNMLKIHGFEDCA" * 5  # divergent ortholog -> should KEEP!

    assert sequence_identity(seq_base, seq_base) == 1.0
    assert sequence_identity(seq_base, seq_near) >= 0.90
    assert sequence_identity(seq_base, seq_div) < 0.50

    rows = [
        {"accession": "ORTH_1", "family": "IPR_X", "data": {"sequence": seq_base, "protein_name": "SynA"}},
        {"accession": "ORTH_NEAR", "family": "IPR_X", "data": {"sequence": seq_near, "protein_name": "SynA"}},
        {"accession": "ORTH_DIV", "family": "IPR_X", "data": {"sequence": seq_div, "protein_name": "SynA"}},
        {"accession": "ORTH_NOSEQ", "family": "IPR_X", "data": {"protein_name": "SynA"}},
    ]
    kept, collapsed = filter_highly_similar_orthologs(rows, identity_threshold=0.85)
    assert [r["accession"] for r in kept] == ["ORTH_1", "ORTH_DIV", "ORTH_NOSEQ"]
    assert len(collapsed) == 1
    assert collapsed[0]["accession"] == "ORTH_NEAR"
    assert collapsed[0]["representative_accession"] == "ORTH_1"

    # Also verify ranked_portfolio writes collapsed_near_identical_orthologs cleanly
    for r in rows[:3]:
        update_candidate_registry(tmp_path, r["accession"], "evaluated", {
            "protein_name": "SynA", "interpro_family_id": "IPR_X", "sequence": r["data"]["sequence"],
        })
    with (tmp_path / "ranked_candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["accession", "protein_name", "rank"])
        w.writeheader()
        for i, r in enumerate(rows[:3], 1):
            w.writerow({"accession": r["accession"], "protein_name": "SynA", "rank": i})
    selected = ranked_portfolio(tmp_path)
    assert [r["accession"] for r in selected] == ["ORTH_1", "ORTH_DIV"]
    portfolio_doc = json.loads((tmp_path / "modeling_portfolio.json").read_text())
    assert len(portfolio_doc["collapsed_near_identical_orthologs"]) == 1


def test_family_learning_incorporates_structural_modeling_feedback():
    base_meta = {
        "catalytic_activities": ["A + B = C"],
        "reaction_chemistry": {"reaction_direction": "formation", "bond_types": ["ester"]},
        "claim_validation": {"promotion_eligible": True, "claims": []},
    }
    candidates = {
        "PASS_1": {
            "statuses": [{"status": "evaluated"}],
            "metadata": {
                **base_meta,
                "interpro_family_id": "IPR_PASS",
                "modeling_validation_evidence": {
                    "execution_status": "complete",
                    "scientific_status": "ok",
                    "pose_validation": {"n_predictions": 50, "n_passing": 25, "passing_fraction": 0.5},
                },
            },
        },
        "UNMOD_1": {
            "statuses": [{"status": "evaluated"}],
            "metadata": {**base_meta, "interpro_family_id": "IPR_UNMOD"},
        },
        "FAIL_1": {
            "statuses": [{"status": "evaluated"}],
            "metadata": {
                **base_meta,
                "interpro_family_id": "IPR_FAIL",
                "gene_names": ["f1"],
                "modeling_validation_evidence": {
                    "execution_status": "complete",
                    "scientific_status": "failed",
                    "pose_validation": {"n_predictions": 50, "n_passing": 0, "passing_fraction": 0.0},
                },
            },
        },
        "FAIL_2": {
            "statuses": [{"status": "evaluated"}],
            "metadata": {
                **base_meta,
                "interpro_family_id": "IPR_FAIL",
                "gene_names": ["f2"],
                "modeling_validation_evidence": {
                    "execution_status": "complete",
                    "scientific_status": "failed",
                    "pose_validation": {"n_predictions": 50, "n_passing": 0, "passing_fraction": 0.0},
                },
            },
        },
    }
    policy = build_family_learning(candidates, [], {}, ("formation", ["ester"], None))
    order = policy["priority_order"]
    # Passing structural family ranks ahead of unmodeled, which ranks ahead of consistently failing family.
    assert order.index("ipr:IPR_PASS") < order.index("ipr:IPR_UNMOD") < order.index("ipr:IPR_FAIL")
    assert policy["families"]["ipr:IPR_PASS"]["structural_feedback"]["n_passing_geometry"] == 1
    assert policy["families"]["ipr:IPR_FAIL"]["structural_feedback"]["structural_warning"] is not None
