"""Offline tests for the family native-reaction survey.

These exercise the GROUNDED-fact reporting (direction vs target, substrate
resemblance numbers, family distributions) with fixtures only — no network — so
the survey's fact reporting is verified in CI.  The survey reports numbers; it
does not emit warm/tepid verdicts or apply any threshold.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.tools.family_reactions import (
    classify_member_reaction,
    summarise_family_profile,
    select_relevant_members,
    _substrate_facts,
    _trembl_members_query,
)


def _member(acc, direction, fg, mw=None, reviewed=True, ec=None):
    """Build a survey-member dict with a resolved substrate fit for selection tests."""
    dims = {"functional_group_similarity": fg}
    if mw is not None:
        dims["mw_similarity"] = mw
    return {
        "accession": acc, "direction_vs_target": direction, "reviewed": reviewed,
        "ec": ec or [], "name": acc.lower(), "native_reaction": "A + B = C",
        "substrate": {"dimensions": dims} if fg is not None else None,
    }


class TestMemberDirectionFacts(unittest.TestCase):
    def test_same_direction_ligase(self):
        r = classify_member_reaction(
            ["6.3.2.1"], "A + B = product", target_direction="formation",
            target_bond_types=set(),
        )
        self.assertEqual(r["direction"], "formation")
        self.assertEqual(r["direction_vs_target"], "same")

    def test_opposite_direction_hydrolase(self):
        r = classify_member_reaction(
            ["3.5.1.4"], "amide + H2O = acid + amine",
            target_direction="formation", target_bond_types=set(),
        )
        self.assertEqual(r["direction"], "cleavage")
        self.assertEqual(r["direction_vs_target"], "opposite")

    def test_shared_bond_type_is_reported_not_scored(self):
        r = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction="formation",
            target_bond_types={"amide", "C-N"},
        )
        self.assertTrue(r["shared_bond_type"])
        # No verdict fields: the survey reports facts, it does not decide.
        self.assertNotIn("tag", r)
        self.assertNotIn("score", r)

    def test_unknown_direction(self):
        r = classify_member_reaction(
            [], "", target_direction="formation", target_bond_types=set(),
        )
        self.assertIsNone(r["direction"])
        self.assertEqual(r["direction_vs_target"], "unknown")

    def test_no_target_direction(self):
        r = classify_member_reaction(
            ["6.3.2.1"], "x = y", target_direction=None, target_bond_types=set(),
        )
        self.assertEqual(r["direction_vs_target"], "unknown")


class TestSubstrateFacts(unittest.TestCase):
    def test_identical_substrate_scores_high(self):
        # Native == target substrate -> functional_group_similarity == 1.0.
        facts = _substrate_facts(["CC(=O)O"], ["CC(=O)O"])
        self.assertIsNotNone(facts)
        self.assertEqual(facts["functional_group_similarity"], 1.0)

    def test_dissimilar_substrate_scores_lower(self):
        # A large aromatic vs a small acid -> lower functional-group overlap.
        near = _substrate_facts(["OC(=O)CCCCC(=O)O"], ["OC(=O)CCCC(=O)O"])  # diacids
        far = _substrate_facts(["c1ccccc1"], ["OC(=O)CCCC(=O)O"])          # benzene vs diacid
        self.assertIsNotNone(near)
        self.assertIsNotNone(far)
        self.assertGreater(near["functional_group_similarity"],
                            far["functional_group_similarity"])

    def test_none_when_unresolvable(self):
        self.assertIsNone(_substrate_facts([], ["CC(=O)O"]))
        self.assertIsNone(_substrate_facts(["CC(=O)O"], []))


class TestFamilyProfile(unittest.TestCase):
    def test_profile_reports_direction_counts_and_distributions(self):
        members = [
            {"direction_vs_target": "same", "native_reaction": "A = B",
             "substrate": {"functional_group_similarity": 0.8, "size_ratio": 0.9}},
            {"direction_vs_target": "same", "native_reaction": "A = B",
             "substrate": {"functional_group_similarity": 0.4, "size_ratio": 0.3}},
            {"direction_vs_target": "opposite", "native_reaction": "C + H2O = D",
             "substrate": None},
        ]
        p = summarise_family_profile(members)
        self.assertEqual(p["n_members"], 3)
        self.assertEqual(p["direction_vs_target_counts"], {"same": 2, "opposite": 1})
        self.assertEqual(p["functional_group_similarity"]["max"], 0.8)
        self.assertEqual(p["functional_group_similarity"]["min"], 0.4)
        self.assertEqual(p["functional_group_similarity"]["median"], 0.6)
        # Distinct reaction names only (the duplicate "A = B" collapses).
        self.assertEqual(p["distinct_reactions"], ["A = B", "C + H2O = D"])


class TestSelectRelevantMembers(unittest.TestCase):
    """The substrate-fit member selection that feeds promotion — pure, relative,
    no tuned relevance cutoff."""

    def test_ranks_by_fit_and_caps(self):
        members = [
            _member("A", "same", 0.5, 0.5),
            _member("B", "same", 0.9, 0.9),
            _member("C", "same", 0.7, 0.7),
        ]
        sel = select_relevant_members(members, max_select=2)
        self.assertEqual([s["accession"] for s in sel], ["B", "C"])  # top-2 by fit
        self.assertAlmostEqual(sel[0]["substrate_fit"], 0.9)

    def test_default_promotes_EVERY_relevant_member_no_cap(self):
        # Discovery must not drop relevant members: with no max_select, ALL
        # same-direction resolved-fit members are returned (best-fit first), not a
        # top-N slice — so a large fold's long tail of genuine members is kept.
        members = [_member(chr(65 + i), "same", 0.9 - i * 0.05, 0.9 - i * 0.05)
                   for i in range(12)]
        sel = select_relevant_members(members)          # default: uncapped
        self.assertEqual(len(sel), 12)                  # every relevant member kept
        self.assertEqual([s["accession"] for s in sel],
                         [chr(65 + i) for i in range(12)])  # ordered best-fit first

    def test_default_still_excludes_irrelevant(self):
        # Uncapped does NOT mean unfiltered: the relevance gate (same-direction +
        # resolved fit) still drops opposite/unknown/unresolved members.
        members = [
            _member("SAME", "same", 0.5, 0.5),
            _member("OPP", "opposite", 1.0, 1.0),
            _member("UNK", "unknown", 0.9, 0.9),
            _member("NOFIT", "same", None),
        ]
        sel = select_relevant_members(members)
        self.assertEqual([s["accession"] for s in sel], ["SAME"])

    def test_excludes_opposite_and_unrelated(self):
        members = [
            _member("A", "opposite", 1.0, 1.0),   # perfect fit but wrong direction
            _member("B", "unrelated", 1.0, 1.0),  # different bond
            _member("C", "same", 0.3, 0.3),
        ]
        sel = select_relevant_members(members, max_select=5)
        self.assertEqual([s["accession"] for s in sel], ["C"])

    def test_excludes_unknown_direction(self):
        # Promotion (which spends eval budget) requires a GROUNDED same-direction
        # reaction. 'unknown' (no EC/reaction) is the leak channel for off-target
        # folds (hydrolases/esterases w/o clean EC), so it is NOT promoted.
        sel = select_relevant_members([_member("A", "unknown", 0.8, 0.8)], max_select=5)
        self.assertEqual(sel, [])

    def test_promotes_only_same_direction(self):
        members = [
            _member("SAME", "same", 0.5, 0.5),
            _member("UNK", "unknown", 0.9, 0.9),   # higher fit but not grounded
        ]
        sel = select_relevant_members(members, max_select=5)
        self.assertEqual([s["accession"] for s in sel], ["SAME"])

    def test_drops_members_without_resolved_fit(self):
        # No substrate resolved -> nothing to compare -> excluded (not promoted blind).
        sel = select_relevant_members([_member("A", "same", None)], max_select=5)
        self.assertEqual(sel, [])

    def test_no_absolute_floor_best_available_is_selected(self):
        # Even a low-but-best fit is selected: selection is RELATIVE, not a cutoff.
        sel = select_relevant_members([_member("A", "same", 0.1, 0.1)], max_select=5)
        self.assertEqual([s["accession"] for s in sel], ["A"])

    def test_reviewed_wins_tie_over_trembl(self):
        members = [
            _member("T", "same", 0.8, 0.8, reviewed=False),
            _member("R", "same", 0.8, 0.8, reviewed=True),
        ]
        sel = select_relevant_members(members, max_select=2)
        self.assertEqual([s["accession"] for s in sel], ["R", "T"])


class TestTremblQuery(unittest.TestCase):
    def test_trembl_query_targets_unreviewed_specific_ec(self):
        q = _trembl_members_query("IPR000873", "6.3.2.1")
        self.assertIn("xref:interpro-IPR000873", q)
        self.assertIn("reviewed:false", q)
        self.assertIn("ec:6.3.2.1", q)
        # Never widens to the whole (reviewed) fold or a partial EC.
        self.assertNotIn("reviewed:true", q)


if __name__ == "__main__":
    unittest.main()
