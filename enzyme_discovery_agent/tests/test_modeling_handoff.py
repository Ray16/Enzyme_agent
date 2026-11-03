"""Safety tests for discovery-to-structural-modeling projection."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.chemistry_logic import (
    derive_activation_mechanism,
    derive_modeling_cofactors,
)
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.modeling_handoff import (
    build_modeling_input,
    build_target_substrates,
    characterize_smiles,
    is_execution_ready,
)
from enzyme_discovery_agent.output.structural_modeling_handoff import (
    write_structural_modeling_handoff,
)


TARGET = {
    "substrate_smiles": "O=C(O)CCCCC(=O)O.NCCCCCCN",
    "target_reasoning": {
        "bond_types": ["amide"],
        "reaction_direction": "formation",
        "modeling_specification": {
            "state": "carboxyl_activation_complex",
            "components": [
                {
                    "source_component_index": 1,
                    "name": "adipic acid",
                    "role": "acyl_donor",
                    "reactive": True,
                    "reactive_atom": "either terminal carboxylate oxygen",
                    "modeling_smiles": "[O-]C(=O)CCCCC(=O)[O-]",
                    "state_rationale": "carboxylate attacks the activating nucleotide",
                    "include_in_model": True,
                    "affinity_target": True,
                },
                {
                    "source_component_index": 2,
                    "name": "hexamethylenediamine",
                    "role": "amine_nucleophile_for_later_step",
                    "reactive": True,
                    "reactive_atom": "either terminal amine nitrogen",
                    "modeling_smiles": "NCCCCCCN",
                    "state_rationale": "neutral nitrogen represents the nucleophilic form",
                    "include_in_model": False,
                    "affinity_target": False,
                },
            ],
            "geometry": {
                "nucleophile": {
                    "component_index": 1,
                    "atom": "either terminal carboxylate oxygen",
                },
            },
            "uncertainties": [],
        },
    },
}


class TestActivationProjection(unittest.TestCase):
    def test_amp_ppi_maps_to_alpha(self):
        result = derive_activation_mechanism([
            "acid + ATP + amine = amide + AMP + diphosphate",
        ])
        self.assertEqual(result["activation_mode"], "acyl_adenylate")
        self.assertEqual(result["reactive_atom"], "PA")
        self.assertEqual(result["leaving_group_atom"], "O3A")
        self.assertEqual(result["status"], "assigned")

    def test_adp_pi_maps_to_gamma(self):
        result = derive_activation_mechanism([
            "acid + ATP + amine = amide + ADP + phosphate",
        ])
        self.assertEqual(result["activation_mode"], "acyl_phosphate")
        self.assertEqual(result["reactive_atom"], "PG")
        self.assertEqual(result["leaving_group_atom"], "O3B")

    def test_products_on_reactant_side_do_not_trigger(self):
        self.assertIsNone(derive_activation_mechanism([
            "AMP + diphosphate = acid + ATP",
        ]))

    def test_conflicting_equations_are_not_resolved_by_vote(self):
        result = derive_activation_mechanism([
            "acid + ATP = product + AMP + diphosphate",
            "acid + ATP = product + AMP + diphosphate",
            "acid + ATP = product + ADP + phosphate",
        ])
        self.assertEqual(result["status"], "conflicting")
        self.assertNotIn("activation_mode", result)

    def test_non_ntp_reaction_is_unresolved(self):
        self.assertIsNone(derive_activation_mechanism(["amide + water = acid + amine"]))

    def test_metal_requires_explicit_identity_and_count(self):
        projection = derive_modeling_cofactors([
            "acid + ATP = product + AMP + diphosphate",
        ])
        self.assertEqual([c["ccd"] for c in projection["cofactors"]], ["ATP"])
        self.assertTrue(any("metal" in item for item in projection["unresolved"]))

        projection = derive_modeling_cofactors(
            ["acid + ATP = product + AMP + diphosphate"],
            ["Mg(2+)"],
            ["Binds 2 Mg(2+) ions per subunit."],
        )
        self.assertEqual(projection["cofactors"][1]["ccd"], "MG")
        self.assertEqual(projection["cofactors"][1]["count"], 2)

    def test_metal_count_must_match_annotated_metal(self):
        projection = derive_modeling_cofactors(
            ["acid + ATP = product + AMP + diphosphate"],
            ["Mg(2+)"],
            ["Binds 2 Mn(2+) ions per subunit."],
        )
        self.assertEqual([c["ccd"] for c in projection["cofactors"]], ["ATP"])
        self.assertTrue(any("stoichiometry" in item for item in projection["unresolved"]))


class TestTargetProjection(unittest.TestCase):
    def test_charge_and_roles_are_explicit(self):
        substrates, unresolved = build_target_substrates(TARGET)
        donor, acceptor = substrates
        self.assertEqual(donor["role"], "acyl_donor")
        self.assertEqual(donor["source_representation"]["formal_charge"], 0)
        self.assertEqual(donor["modeling_representation"]["formal_charge"], -2)
        self.assertEqual(donor["review_status"], "required")
        self.assertEqual(acceptor["role"], "amine_nucleophile_for_later_step")
        self.assertEqual(acceptor["modeling_representation"]["formal_charge"], 0)
        self.assertTrue(any("requires review" in item for item in unresolved))

    def test_specified_stereochemistry_is_preserved(self):
        facts = characterize_smiles("C[C@@H](O)CC(=O)O")
        self.assertEqual(facts["stereochemistry"]["status"], "specified")
        self.assertEqual(facts["stereochemistry"]["assigned_centers"][0]["cip"], "R")
        self.assertIn("@@", facts["canonical_isomeric_smiles"])

    def test_unspecified_stereocenter_blocks_readiness(self):
        target = {
            "substrate_smiles": "CC(O)CC(=O)O.N",
            "target_reasoning": {
                "bond_types": ["amide"],
                "reaction_direction": "formation",
                "modeling_specification": {
                    "state": "reactant_complex",
                    "components": [
                        {
                            "source_component_index": 1,
                            "role": "substrate",
                            "reactive": True,
                            "reactive_atom": "carbon",
                            "modeling_smiles": "CC(O)CC(=O)O",
                        },
                        {
                            "source_component_index": 2,
                            "role": "cosubstrate",
                            "reactive": True,
                            "reactive_atom": "nitrogen",
                            "modeling_smiles": "N",
                        },
                    ],
                    "geometry": {"nucleophile": {"component_index": 1, "atom": "oxygen"}},
                    "uncertainties": [],
                },
            },
        }
        modeling = build_modeling_input({
            "catalytic_activities": [
                "acid + ATP + amine = amide + AMP + diphosphate",
            ],
            "required_cofactors": ["Mg(2+)"],
            "cofactor_notes": ["Binds 1 Mg(2+) ion per subunit."],
            "modeling_specification": {
                "enzyme_chains": [{"source": "full_sequence"}],
            },
        }, target)
        ready, errors = is_execution_ready(modeling)
        self.assertFalse(ready)
        self.assertTrue(any("stereochemistry" in item for item in errors))

    def test_grounded_non_ntp_mechanism_can_be_ready(self):
        target = {
            "substrate_smiles": "CC(=O)NCC",
            "target_reasoning": {
                "modeling_specification": {
                    "state": "reactant_complex",
                    "components": [{
                        "source_component_index": 1,
                        "role": "electrophilic_substrate",
                        "reactive": True,
                        "reactive_atom": "amide carbonyl carbon",
                        "modeling_smiles": "CC(=O)NCC",
                        "include_in_model": True,
                        "affinity_target": True,
                    }],
                    "geometry": {"kind": "candidate_specific"},
                    "uncertainties": [],
                },
            },
        }
        candidate = {
            "catalytic_activities": ["an amide + H2O = acid + amine"],
            "mechanism_evidence": {
                "status": "ok",
                "paper": {"doi": "10.example/mechanism"},
                "facts": {
                    "confidence": "high",
                    "mechanism_class": "hydrolysis",
                    "is_clean_inline_substitution": True,
                    "nucleophile": {
                        "source": "protein_residue",
                        "identity": "catalytic residue",
                        "atom": "side-chain oxygen",
                    },
                    "electrophile": {
                        "identity": "substrate",
                        "atom": "amide carbonyl carbon",
                    },
                    "leaving_group": {
                        "identity": "amine",
                        "atom": "amide nitrogen",
                    },
                    "cofactors": [],
                    "metals": [],
                    "enzyme_state": {
                        "state": "processed",
                        "mature_chains": [
                            {"start": 1, "end": 266},
                            {"start": 267, "end": 400},
                        ],
                        "evidence": "Cleavage exposes the catalytic residue.",
                    },
                },
            },
        }
        modeling = build_modeling_input(candidate, target)
        self.assertEqual(modeling["status"], "ready")
        self.assertIsNone(modeling["activation"])
        self.assertEqual(modeling["cofactors"], [])
        self.assertEqual(modeling["geometry"]["kind"], "inline_substitution")
        self.assertEqual(modeling["enzyme_state"]["chains"][1]["start"], 267)

    def test_cofold_can_be_ready_without_a_geometry_observable(self):
        target = {
            "substrate_smiles": "CCO",
            "target_reasoning": {
                "modeling_specification": {
                    "state": "binding_complex",
                    "components": [{
                        "source_component_index": 1,
                        "role": "substrate",
                        "reactive": False,
                        "reactive_atom": None,
                        "modeling_smiles": "CCO",
                        "include_in_model": True,
                        "affinity_target": True,
                    }],
                    "geometry": {"kind": "none"},
                    "uncertainties": [],
                },
            },
        }
        candidate = {
            "catalytic_activities": ["substrate + water = products"],
            "mechanism_evidence": {
                "status": "ok",
                "paper": {"doi": "10.example.binding"},
                "facts": {
                    "confidence": "high",
                    "mechanism_class": "other",
                    "cofactors": [],
                    "metals": [],
                    "enzyme_state": {"state": "intact"},
                },
            },
        }
        modeling = build_modeling_input(candidate, target)
        ready, errors = is_execution_ready(modeling)
        self.assertTrue(ready, errors)
        self.assertIsNone(modeling["geometry"])
        self.assertEqual(modeling["geometry_status"], "unresolved")
        self.assertTrue(any("geometry" in item for item in modeling["warnings"]))

    def test_mechanism_metal_does_not_displace_reaction_nucleotide(self):
        candidate = {
            "catalytic_activities": [
                "acid + ATP = product + AMP + diphosphate",
            ],
            "mechanism_evidence": {
                "status": "ok",
                "facts": {
                    "confidence": "high",
                    "metals": [{"ccd": "MG", "count": 1}],
                    "enzyme_state": {"state": "intact"},
                },
            },
        }
        modeling = build_modeling_input(candidate, TARGET)
        self.assertEqual(
            [cofactor["ccd"] for cofactor in modeling["cofactors"]],
            ["ATP", "MG"],
        )


class TestHandoffWriter(unittest.TestCase):
    def setUp(self):
        # Isolate target/structure serialization from the separate admission tests.
        from unittest.mock import patch
        guard=patch('enzyme_discovery_agent.models.target_review.eligible',return_value=True)
        guard.start();self.addCleanup(guard.stop)
    def test_registry_mechanism_wins_and_neighbor_is_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "evaluated", metadata={
                "claim_validation": {"promotion_eligible": True, "claims": [{"claim_type": "accession_identity", "subject": "P12345", "verdict": "supported", "evidence": {"source": "UniProtKB"}}]},
                "catalytic_activities": [
                    "acid + ATP + amine = amide + AMP + diphosphate",
                ],
                "mechanism_evidence": {"status": "ok", "facts": {"confidence": "high"}},
            })
            path = write_structural_modeling_handoff(
                run_dir,
                [{"status": "success", "data": {
                    "accession": "P12345",
                    "mechanism_evidence": {"status": "stale"},
                }}],
                TARGET,
                neighbor_map={"promising": [{
                    "accession": "P12345",
                    "neighbors": [{"accession": "Q99999", "distance": 0.2}],
                }]},
            )
            document = json.loads(path.read_text())

        self.assertEqual(document["schema_version"], "3.0")
        self.assertEqual(
            document["candidates"][0]["evidence"]["mechanism"]["status"], "ok",
        )
        self.assertEqual(
            document["unresolved_leads"][0]["neighbor_of"], "P12345",
        )
        self.assertEqual(document["unresolved_leads"][0]["accession"], "Q99999")

    def test_handoff_preserves_target_context_for_execution_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "evaluated", metadata={
                "claim_validation": {"promotion_eligible": True, "claims": [{"claim_type": "accession_identity", "subject": "P12345", "verdict": "supported", "evidence": {"source": "UniProtKB"}}]},
                "catalytic_activities": [
                    "acid + ATP + amine = amide + AMP + diphosphate",
                ],
                "required_cofactors": ["Mg(2+)"],
                "cofactor_notes": ["Binds 1 Mg(2+) ion per subunit."],
                "modeling_specification": {
                    "enzyme_chains": [{"source": "full_sequence"}],
                },
            })
            path = write_structural_modeling_handoff(
                run_dir,
                [{"status": "success", "data": {"accession": "P12345"}}],
                TARGET,
            )
            document = json.loads(path.read_text())

        self.assertEqual(document["target"]["substrate_smiles"], TARGET["substrate_smiles"])
        candidate = document["candidates"][0]
        self.assertEqual(
            set(candidate),
            {"accession", "identity", "sequence", "sequence_identity", "input_availability", "evidence", "modeling_input", "selection"},
        )
        self.assertEqual(
            candidate["evidence"]["identity"]["accession"],
            candidate["accession"],
        )
        self.assertEqual(
            candidate["evidence"]["native_reactions"],
            ["acid + ATP + amine = amide + AMP + diphosphate"],
        )
        self.assertEqual(
            candidate["evidence"]["cofactor_annotations"]["names"],
            ["Mg(2+)"],
        )
        self.assertEqual(candidate["selection"]["source"], "evaluated_discovery_candidate")
        modeling = candidate["modeling_input"]
        self.assertEqual(len(modeling["target_substrates"]), 2)
        self.assertEqual(
            [item["ccd"] for item in modeling["cofactors"]],
            ["ATP", "MG"],
        )
        self.assertEqual(
            modeling["enzyme_state"]["chains"],
            [{"source": "full_sequence"}],
        )


if __name__ == "__main__":
    unittest.main()
