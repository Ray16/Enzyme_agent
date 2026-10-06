"""Tests for deterministic mechanism-family assignment of candidates.

The families here are hand-written stand-ins with the same *shape* as a real
``mechanism_families.json`` (name + descriptors + promise); the point is the
matching logic, not any particular reaction.
"""

import unittest

from enzyme_discovery_agent.tools.mechanism_family_assignment import (
    assign_mechanism_family,
    mechanism_family_level,
    summarize_assignments,
)

FAMILIES = [
    {
        "name": "NIS (NRPS-independent siderophore) synthetase fold",
        "descriptors": ["IucA/IucC-like acyltransferase fold", "siderophore synthetase A/B domain"],
        "promise": "high",
        "proposed_activation_strategy": "acyl-AMP intermediate",
    },
    {
        "name": "ATP-grasp fold amide ligase",
        "descriptors": ["ATP-grasp domain", "glutathione-synthetase-like fold"],
        "promise": "medium",
        "proposed_activation_strategy": "phosphoanhydride intermediate",
    },
    {
        "name": "ANL superfamily acyl-adenylate synthetase",
        "descriptors": ["AMP-binding domain", "acyl-CoA synthetase-like fold", "GH3 auxin-amido synthetase"],
        "promise": "medium",
        "proposed_activation_strategy": "adenylation intermediate",
    },
]


class TestAssignment(unittest.TestCase):
    def test_interpro_name_matches_family(self):
        meta = {
            "interpro_family": "Aerobactin siderophore biosynthesis, IucA/IucC-like",
            "protein_name": "Aerobactin synthase",
        }
        got = assign_mechanism_family(meta, FAMILIES)
        self.assertIsNotNone(got)
        self.assertTrue(got["name"].startswith("NIS"))
        self.assertIn("iuca", got["matched_tokens"])
        self.assertGreaterEqual(got["score"], 1)

    def test_distinct_family_matches_atp_grasp(self):
        meta = {"interpro_family": "Glutathione synthetase, ATP-grasp domain"}
        got = assign_mechanism_family(meta, FAMILIES)
        self.assertIsNotNone(got)
        self.assertTrue(got["name"].startswith("ATP-grasp"))

    def test_no_overlap_returns_none(self):
        meta = {"interpro_family": "Peptidase S58, DmpA/ArgJ", "protein_name": "6-aminohexanoate hydrolase"}
        self.assertIsNone(assign_mechanism_family(meta, FAMILIES))

    def test_no_families_returns_none(self):
        self.assertIsNone(assign_mechanism_family({"interpro_family": "x IucA/IucC"}, []))

    def test_generic_tokens_do_not_match(self):
        # "synthetase"/"domain"/"fold" are generic stopwords; a candidate whose
        # only overlap is generic vocabulary must NOT be assigned.
        meta = {"interpro_family": "Some ATP-dependent synthetase domain fold"}
        self.assertIsNone(assign_mechanism_family(meta, FAMILIES))

    def test_tie_breaks_toward_higher_promise(self):
        # Craft a candidate that shares exactly one distinctive token with a
        # high-promise and a medium-promise family; high promise should win.
        fams = [
            {"name": "Low family", "descriptors": ["widget alpha"], "promise": "low"},
            {"name": "High family", "descriptors": ["widget beta"], "promise": "high"},
        ]
        meta = {"interpro_family": "widget alpha widget beta"}
        got = assign_mechanism_family(meta, fams)
        self.assertEqual(got["name"], "High family")

    def test_level_from_name(self):
        self.assertEqual(mechanism_family_level("ANL superfamily acyl-adenylate synthetase"), "superfamily")
        self.assertEqual(mechanism_family_level("ATP-grasp fold amide ligase"), "fold")
        self.assertEqual(mechanism_family_level("Mur ligase superfamily"), "superfamily")
        self.assertIsNone(mechanism_family_level("Some unnamed class"))


class TestRollup(unittest.TestCase):
    def test_rollup_counts_and_best_rank(self):
        entries = [
            {"accession": "A1", "mechanism_family": "NIS (NRPS-independent siderophore) synthetase fold", "rank": 5},
            {"accession": "A2", "mechanism_family": "NIS (NRPS-independent siderophore) synthetase fold", "rank": 2},
            {"accession": "B1", "mechanism_family": "ATP-grasp fold amide ligase", "rank": 9},
            {"accession": "C1", "mechanism_family": None, "rank": 12},
        ]
        rows = summarize_assignments(entries, FAMILIES)
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(by_name["NIS (NRPS-independent siderophore) synthetase fold"]["n_candidates"], 2)
        self.assertEqual(by_name["NIS (NRPS-independent siderophore) synthetase fold"]["best_rank"], 2)
        self.assertEqual(by_name["ATP-grasp fold amide ligase"]["best_rank"], 9)
        self.assertIn("Unassigned", by_name)
        self.assertEqual(by_name["Unassigned"]["n_candidates"], 1)
        # A family with zero assigned candidates is omitted from the rollup.
        self.assertNotIn("ANL superfamily acyl-adenylate synthetase", by_name)

    def test_rollup_preserves_derived_order(self):
        entries = [
            {"accession": "B1", "mechanism_family": "ATP-grasp fold amide ligase", "rank": 1},
            {"accession": "A1", "mechanism_family": "NIS (NRPS-independent siderophore) synthetase fold", "rank": 3},
        ]
        rows = summarize_assignments(entries, FAMILIES)
        # NIS is derived first, so it precedes ATP-grasp regardless of insertion order.
        self.assertTrue(rows[0]["name"].startswith("NIS"))


if __name__ == "__main__":
    unittest.main()
