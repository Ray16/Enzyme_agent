"""Offline tests for the family native-reaction survey classification logic.

These exercise the pure WARM/COLD/neutral scoring with fixture dicts only — no
network access — so the family-navigation decision logic is verified in CI.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.tools.family_reactions import (
    classify_member_reaction,
    summarise_family_profile,
)


class TestMemberClassification(unittest.TestCase):
    def test_same_direction_ligase_is_warm(self):
        # An EC 6.3.* ligase catalyses bond FORMATION; against a formation target
        # it is a WARM member the agent should expand toward.
        result = classify_member_reaction(
            ["6.3.2.1"], "A + B = product", target_direction="formation",
            target_bond_types=set(),
        )
        self.assertEqual(result["direction"], "formation")
        self.assertEqual(result["tag"], "warm")
        self.assertGreater(result["score"], 0)

    def test_opposite_direction_hydrolase_is_cold(self):
        # An EC 3.* hydrolase runs CLEAVAGE; against a formation target it is COLD.
        result = classify_member_reaction(
            ["3.5.1.4"], "amide + H2O = acid + amine",
            target_direction="formation", target_bond_types=set(),
        )
        self.assertEqual(result["direction"], "cleavage")
        self.assertEqual(result["tag"], "cold")
        self.assertLess(result["score"], 0)

    def test_shared_bond_type_raises_warm_score(self):
        base = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types=set(),
        )
        with_bond = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types={"amide", "C-N"},
        )
        self.assertEqual(with_bond["tag"], "warm")
        self.assertGreater(with_bond["score"], base["score"])

    def test_unknown_direction_is_neutral(self):
        # No EC and an unclassifiable reaction -> no direction -> neutral.
        result = classify_member_reaction(
            [], "", target_direction="formation", target_bond_types=set(),
        )
        self.assertIsNone(result["direction"])
        self.assertEqual(result["tag"], "neutral")
        self.assertEqual(result["score"], 0)

    def test_no_target_direction_never_warms(self):
        # If the run declares no direction, nothing can be scored WARM/COLD.
        result = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction=None, target_bond_types=set(),
        )
        self.assertEqual(result["tag"], "neutral")

    def test_right_direction_similar_substrate_is_warm(self):
        # Same direction AND a high substrate_fit -> WARM.
        result = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types={"amide"}, substrate_fit=0.8,
        )
        self.assertEqual(result["tag"], "warm")

    def test_right_direction_dissimilar_substrate_is_tepid(self):
        # Same direction but a LOW substrate_fit -> tepid, not warm: the fold
        # makes the target bond, but on a substrate unlike the target.
        result = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types={"amide"}, substrate_fit=0.1,
        )
        self.assertEqual(result["tag"], "tepid")

    def test_substrate_fit_orders_within_warm(self):
        hi = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types=set(), substrate_fit=0.9,
        )
        lo = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types=set(), substrate_fit=0.6,
        )
        self.assertGreater(hi["score"], lo["score"])

    def test_opposite_direction_stays_cold_regardless_of_substrate(self):
        result = classify_member_reaction(
            ["3.5.1.4"], "amide + H2O = acid + amine", target_direction="formation",
            target_bond_types=set(), substrate_fit=0.9,
        )
        self.assertEqual(result["tag"], "cold")


class TestFamilyProfile(unittest.TestCase):
    def test_profile_counts_and_distinct_reactions(self):
        members = [
            {"tag": "warm", "direction": "formation", "native_reaction": "A = B"},
            {"tag": "warm", "direction": "formation", "native_reaction": "A = B"},
            {"tag": "cold", "direction": "cleavage", "native_reaction": "C + H2O = D"},
            {"tag": "neutral", "direction": None, "native_reaction": ""},
        ]
        profile = summarise_family_profile(members)
        self.assertEqual(profile["warm_members"], 2)
        self.assertEqual(profile["cold_members"], 1)
        self.assertEqual(profile["neutral_members"], 1)
        self.assertEqual(profile["directions"], {"formation": 2, "cleavage": 1})
        # Distinct reaction names only (the duplicate "A = B" collapses).
        self.assertEqual(profile["distinct_reactions"], ["A = B", "C + H2O = D"])


if __name__ == "__main__":
    unittest.main()
