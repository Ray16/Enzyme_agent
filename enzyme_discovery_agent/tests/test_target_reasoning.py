"""Tests for the shared derived chemistry contract (target_reasoning.json)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.models.target_reasoning import (
    SCHEMA_VERSION,
    STATUSES,
    UNCERTAINTIES,
    derive_target_reasoning,
    infer_substrate_topology,
    load_target_reasoning,
    normalize_target_reasoning,
    target_claims,
)

# A generic formation reaction between two small molecules.  No answer
# substrates, products, or enzyme names appear anywhere in this file.
_FORMATION_QUERY = "enzymes that catalyze the formation of an ester bond from an alcohol and an acid"
_TWO_INPUTS = "CCO.CC(=O)O"  # a generic alcohol + a generic acid


class TestDeterministicDerivation(unittest.TestCase):
    def test_direction_bonds_topology_reactive_groups(self):
        contract = derive_target_reasoning(
            _FORMATION_QUERY, substrate_smiles=_TWO_INPUTS, llm_client=None,
        )
        self.assertEqual(contract["reaction_direction"], "formation")
        self.assertIn("ester", contract["bond_types"])
        self.assertEqual(contract["substrate_topology"], "free_small_molecule")
        # RDKit should detect at least the carboxyl/hydroxyl groups present.
        self.assertTrue(contract["reactive_groups"])

    def test_cleavage_direction_is_derived(self):
        contract = derive_target_reasoning(
            "enzymes that catalyze the hydrolysis of an ester bond",
            substrate_smiles=None, llm_client=None,
        )
        self.assertEqual(contract["reaction_direction"], "cleavage")

    def test_every_claim_carries_basis_status_uncertainty(self):
        contract = derive_target_reasoning(
            _FORMATION_QUERY, substrate_smiles=_TWO_INPUTS, llm_client=None,
        )
        self.assertTrue(contract["claims"])
        for claim in contract["claims"]:
            self.assertTrue(claim["claim"])
            self.assertTrue(claim["basis"])
            self.assertTrue(claim["implication"])
            self.assertIn(claim["status"], STATUSES)
            self.assertIn(claim["uncertainty"], UNCERTAINTIES)

    def test_no_llm_degrades_to_deterministic_only(self):
        contract = derive_target_reasoning(
            _FORMATION_QUERY, substrate_smiles=_TWO_INPUTS, llm_client=None,
        )
        # No inferred claims without a client; deterministic fields still present.
        self.assertEqual(contract["activation_categories"], [])
        self.assertEqual(contract["disqualifiers"], [])
        self.assertTrue(all(c["status"] == "derived_from_target"
                            for c in contract["claims"]))

    def test_explicit_profile_direction_lowers_uncertainty(self):
        contract = derive_target_reasoning(
            "a reaction", substrate_smiles=None, llm_client=None,
            target_profile={"reaction": {"direction": "formation"}},
        )
        self.assertEqual(contract["reaction_direction"], "formation")
        dir_claim = next(c for c in contract["claims"]
                         if "direction is formation" in c["claim"])
        self.assertEqual(dir_claim["uncertainty"], "low")


class TestTopologyInference(unittest.TestCase):
    def test_explicit_topology_wins(self):
        self.assertEqual(
            infer_substrate_topology("CCO", {"topology": "polymer"}), "polymer")

    def test_dot_separated_inputs_are_free(self):
        self.assertEqual(
            infer_substrate_topology("CCO.CC(=O)O", {}), "free_small_molecule")

    def test_carrier_description(self):
        self.assertEqual(
            infer_substrate_topology(None, {"physical_state": "carrier tethered"}),
            "carrier_protein_tethered")

    def test_no_signal_is_none(self):
        self.assertIsNone(infer_substrate_topology(None, {}))


class TestNormalizeAndLoad(unittest.TestCase):
    def test_roundtrip(self):
        contract = derive_target_reasoning(
            _FORMATION_QUERY, substrate_smiles=_TWO_INPUTS, llm_client=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target_reasoning.json").write_text(json.dumps(contract))
            loaded = load_target_reasoning(root)
        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded["reaction_direction"], "formation")

    def test_missing_contract_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_target_reasoning(Path(tmp)))

    def test_invalid_claim_status_rejected(self):
        with self.assertRaises(ValueError):
            normalize_target_reasoning({
                "claims": [{"claim": "x", "basis": "y", "implication": "z",
                            "status": "bogus", "uncertainty": "low"}],
            })

    def test_invalid_topology_rejected(self):
        with self.assertRaises(ValueError):
            normalize_target_reasoning({"claims": [], "substrate_topology": "gas"})


class TestConsumersReadContract(unittest.TestCase):
    def test_target_claims_and_readers_agree(self):
        contract = derive_target_reasoning(
            _FORMATION_QUERY, substrate_smiles=_TWO_INPUTS, llm_client=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target_reasoning.json").write_text(json.dumps(contract))
            (root / "target_context.json").write_text(json.dumps({
                "query": _FORMATION_QUERY, "substrate_smiles": _TWO_INPUTS,
                "target_profile": {},
            }))
            direction, bonds, topology = target_claims(root)
            self.assertEqual(direction, "formation")
            self.assertIn("ester", bonds)
            self.assertEqual(topology, "free_small_molecule")

            # Downstream readers must return the SAME facts from the contract.
            from enzyme_discovery_agent.tools.family_reactions import (
                _target_profile_for_run,
                _target_topology_for_run,
            )
            from enzyme_discovery_agent.models.candidate_registry import _target_direction
            self.assertEqual(_target_profile_for_run(root), (direction, bonds))
            self.assertEqual(_target_topology_for_run(root), topology)
            self.assertEqual(_target_direction(root), direction)

    def test_legacy_run_without_contract_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target_context.json").write_text(json.dumps({
                "query": _FORMATION_QUERY, "substrate_smiles": _TWO_INPUTS,
                "target_profile": {},
            }))
            from enzyme_discovery_agent.tools.family_reactions import (
                _target_profile_for_run,
                _target_topology_for_run,
            )
            direction, bonds = _target_profile_for_run(root)
            self.assertEqual(direction, "formation")
            self.assertIn("ester", bonds)
            self.assertEqual(_target_topology_for_run(root), "free_small_molecule")


if __name__ == "__main__":
    unittest.main()
