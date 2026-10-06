"""Tests for falsifiable family-hypothesis records and retrieval verdicts."""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.models.family_hypothesis import (
    FAMILY_VERDICTS,
    family_verdict,
    normalize_family_hypothesis,
)


def _validation(*, eligible, contradicted=False):
    """A minimal member claim-validation as produced by claim_validation."""
    claims = []
    if contradicted:
        claims = [{"claim_type": "reaction_direction", "verdict": "contradicted"}]
    return {"promotion_eligible": eligible, "claims": claims}


class TestNormalizeFamilyHypothesis(unittest.TestCase):
    def test_full_record_shape(self):
        rec = normalize_family_hypothesis({
            "name": "generic ligase fold",
            "descriptors": ["adenylation domain", "adenylation domain"],
            "native_reaction": "acid + amine + ATP -> product + AMP",
            "native_direction": "formation",
            "substrates_free": True,
            "required_catalytic_capability": "carboxylate activation",
            "proposed_activation_strategy": "acyl-adenylate intermediate",
            "disqualifying_observations": ["acts only on a carrier-bound donor"],
            "role": "direct_catalysis",
        })
        self.assertEqual(rec["name"], "generic ligase fold")
        self.assertEqual(rec["search_descriptors"], ["adenylation domain"])
        self.assertEqual(rec["predicted_native_direction"], "formation")
        self.assertEqual(rec["predicted_substrate_topology"], "free_small_molecule")
        self.assertEqual(rec["role"], "direct_catalysis")
        self.assertEqual(rec["validation_status"], "hypothesis")
        self.assertTrue(rec["disqualifying_observations"])

    def test_partial_record_is_not_dropped(self):
        rec = normalize_family_hypothesis({"name": "x", "native_reaction": "a -> b"})
        self.assertEqual(rec["name"], "x")
        self.assertEqual(rec["predicted_native_reaction"], "a -> b")
        self.assertIsNone(rec["predicted_substrate_topology"])
        self.assertEqual(rec["disqualifying_observations"], [])

    def test_unknown_role_defaults_to_direct_catalysis(self):
        self.assertEqual(
            normalize_family_hypothesis({"role": "bogus"})["role"], "direct_catalysis")

    def test_non_free_substrate_leaves_topology_unknown(self):
        rec = normalize_family_hypothesis({"substrates_free": False})
        self.assertIsNone(rec["predicted_substrate_topology"])


class TestFamilyVerdict(unittest.TestCase):
    def test_supported_when_a_member_is_eligible(self):
        v = family_verdict([_validation(eligible=False), _validation(eligible=True)])
        self.assertEqual(v, "supported_family_hypothesis")
        self.assertIn(v, FAMILY_VERDICTS)

    def test_contradicted_when_no_eligible_and_a_member_contradicts(self):
        v = family_verdict([_validation(eligible=False, contradicted=True)])
        self.assertEqual(v, "contradicted_family_hypothesis")

    def test_no_evidence_is_insufficient_not_feasible(self):
        # Empty, and reviewed-but-only-insufficient, must both be insufficient.
        self.assertEqual(family_verdict([]), "insufficient_family_evidence")
        self.assertEqual(
            family_verdict([_validation(eligible=False)]),
            "insufficient_family_evidence")

    def test_support_wins_over_a_contradicting_sibling(self):
        # One eligible member establishes the pattern even if another contradicts.
        v = family_verdict([
            _validation(eligible=True),
            _validation(eligible=False, contradicted=True),
        ])
        self.assertEqual(v, "supported_family_hypothesis")

    def test_none_input_is_insufficient(self):
        self.assertEqual(family_verdict(None), "insufficient_family_evidence")


if __name__ == "__main__":
    unittest.main()
