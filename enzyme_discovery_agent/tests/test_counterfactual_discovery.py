"""Counterfactual discovery tests: decisions must track chemistry, not recall.

For each paired target set (one changed chemical dimension), assert that the
shared chemistry contract derives the changed constraint AND that the relevant
deterministic guard flips its verdict.  This measures constraint-sensitive
behavior — the operational distinction between reasoning and answer recall —
without an LLM in the loop, so it is fast and reproducible.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.eval.counterfactuals import (
    COUNTERFACTUAL_PAIRS,
    SYNTHETIC_ANALOG_CASE,
)
from enzyme_discovery_agent.models.claim_validation import validate_uniprot_claims
from enzyme_discovery_agent.models.target_reasoning import derive_target_reasoning
from enzyme_discovery_agent.tools.mechanism_families import family_passes_constraints


def _contract(side: dict) -> dict:
    return derive_target_reasoning(
        side["query"], substrate_smiles=side.get("substrate_smiles"),
        llm_client=None, target_profile=side.get("profile"),
    )


def _entry(*, accession="P00001", reaction, ec, fold="IPR000001",
           entry_type="UniProtKB reviewed (Swiss-Prot)"):
    return {
        "entryType": entry_type,
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {
            "fullName": {"value": "Example"},
            "ecNumbers": ([{"value": ec}] if ec else []),
        }},
        "comments": [{"commentType": "CATALYTIC ACTIVITY",
                      "reaction": {"name": reaction}}],
        "uniProtKBCrossReferences": [{"database": "InterPro", "id": fold}],
    }


def _claim(result, kind):
    return next(c for c in result["claims"] if c["claim_type"] == kind)


class TestContractTracksChemistry(unittest.TestCase):
    """Each pair's contract must flip exactly the changed constraint."""

    def test_pairs_flip_expected_field(self):
        for pair in COUNTERFACTUAL_PAIRS:
            with self.subTest(pair=pair["id"]):
                base = _contract(pair["base"])
                variant = _contract(pair["variant"])
                field = pair["expect"]["field"]
                base_val, var_val = base[field], variant[field]
                if field == "bond_types":
                    self.assertIn(pair["expect"]["base"], base_val)
                    self.assertIn(pair["expect"]["variant"], var_val)
                    self.assertNotIn(pair["expect"]["variant"], base_val)
                else:
                    self.assertEqual(base_val, pair["expect"]["base"])
                    self.assertEqual(var_val, pair["expect"]["variant"])


class TestGuardsFlipOnDirection(unittest.TestCase):
    def test_formation_only_family_dropped_for_cleavage_target(self):
        fam = {"substrates_free": True, "native_direction": "formation"}
        ok_formation, _ = family_passes_constraints(fam, "formation")
        ok_cleavage, reason = family_passes_constraints(fam, "cleavage")
        self.assertTrue(ok_formation)
        self.assertFalse(ok_cleavage)
        self.assertIn("opposite", reason)

    def test_member_direction_contradicted_when_target_flips(self):
        formation_member = _entry(
            reaction="acid + amine + ATP = amide + ADP + phosphate", ec="6.3.2.1")
        supported = validate_uniprot_claims(
            formation_member, expected_accession="P00001",
            target_direction="formation")
        contradicted = validate_uniprot_claims(
            formation_member, expected_accession="P00001",
            target_direction="cleavage")
        self.assertEqual(_claim(supported, "reaction_direction")["verdict"], "supported")
        self.assertEqual(
            _claim(contradicted, "reaction_direction")["verdict"], "contradicted")


class TestGuardsFlipOnTopology(unittest.TestCase):
    def test_carrier_member_matches_carrier_target_but_not_free(self):
        carrier = _entry(
            reaction="acyl-[carrier protein] + acceptor = product + carrier protein",
            ec="2.3.1.1")
        free_target = validate_uniprot_claims(
            carrier, expected_accession="P00001",
            target_topology="free_small_molecule")
        carrier_target = validate_uniprot_claims(
            carrier, expected_accession="P00001",
            target_topology="carrier_protein_tethered")
        self.assertEqual(
            _claim(free_target, "substrate_topology")["verdict"], "contradicted")
        self.assertEqual(
            _claim(carrier_target, "substrate_topology")["verdict"], "supported")

    def test_monomer_only_member_not_equivalent_for_polymer_target(self):
        free_member = _entry(reaction="acid + amine = amide + H2O", ec="6.3.2.1")
        result = validate_uniprot_claims(
            free_member, expected_accession="P00001", target_topology="polymer")
        self.assertEqual(
            _claim(result, "substrate_topology")["verdict"], "contradicted")
        self.assertFalse(result["promotion_eligible"])


class TestGuardsFlipOnBond(unittest.TestCase):
    def test_bond_type_contradicted_on_scaffold_change(self):
        amide_member = _entry(
            reaction="acid + amine + ATP = amide + ADP + phosphate", ec="6.3.2.1")
        as_amide = validate_uniprot_claims(
            amide_member, expected_accession="P00001", target_bond_types={"amide"})
        as_ester = validate_uniprot_claims(
            amide_member, expected_accession="P00001", target_bond_types={"ester"})
        self.assertEqual(_claim(as_amide, "bond_type")["verdict"], "supported")
        self.assertEqual(_claim(as_ester, "bond_type")["verdict"], "contradicted")


class TestSyntheticAnalogRetainsMechanism(unittest.TestCase):
    def test_mechanism_retained_without_native_substrate_evidence(self):
        case = SYNTHETIC_ANALOG_CASE
        member = _entry(
            reaction=case["member_native_reaction"], ec=case["member_ec"])
        # Correct native mechanism (direction + bond); substrate identity is NOT
        # a gate, so mechanism-level possibility is retained for the analog target.
        result = validate_uniprot_claims(
            member, expected_accession="P00001",
            target_direction=case["target_direction"],
            target_bond_types=set(case["target_bond_types"]))
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(_claim(result, "reaction_direction")["verdict"], "supported")
        self.assertEqual(_claim(result, "bond_type")["verdict"], "supported")
        # No substrate-identity claim is asserted — native scope stays unproven.
        self.assertNotIn(
            "substrate_identity", {c["claim_type"] for c in result["claims"]})


if __name__ == "__main__":
    unittest.main()
