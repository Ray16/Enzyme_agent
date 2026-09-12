"""Substrate fit is DECOMPOSED, never a single collapsed score.

Ranking reasons over independent accommodation axes: (1) how many of the target's
substrates the enzyme can host (a hard physical constraint, kept as a COUNT) and
(2) the weakest per-substrate functional-group overlap (the chemistry axis).  This
guards the two behaviours that matter: the axes stay separate, and a candidate that
cannot host every target substrate ranks BELOW one that can — even if the
non-accommodating one has a higher functional-group number.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent_improved.output.report_generator import (
    _accommodation_axes,
    _accommodation_sort_key,
    _deterministic_mechanistic_ranking,
    _mechanism_evidence_level,
)


def _accom(per, unaccommodated, weak, n=2, shape_mismatch=0):
    # shape_mismatch property comparisons carry an aromaticity/ring mismatch flag
    comps = [{"aromaticity_match": False, "cyclicity_match": False}
             for _ in range(shape_mismatch)]
    return {"substrate_similarity": {
        "accommodation": {
            "per_target": per,
            "n_target_substrates": n,
            "unaccommodated_targets": unaccommodated,
            "weakest_functional_group_similarity": weak,
        },
        "property_comparisons": comps,
    }}


class TestAccommodationAxes(unittest.TestCase):
    def test_absent_when_no_comparison(self):
        self.assertFalse(_accommodation_axes({})["present"])
        self.assertFalse(_accommodation_axes({"substrate_similarity": {}})["present"])

    def test_axes_kept_separate(self):
        ax = _accommodation_axes(_accom([{}, {}], [], 0.5))
        self.assertTrue(ax["present"])
        self.assertEqual(ax["n_target_substrates"], 2)
        self.assertEqual(ax["n_accommodated"], 2)
        self.assertEqual(ax["n_unaccommodated"], 0)
        self.assertTrue(ax["all_accommodated"])
        self.assertEqual(ax["weakest_fg_similarity"], 0.5)

    def test_unaccommodated_counted_not_blended(self):
        ax = _accommodation_axes(_accom([{}, {}], ["NCCCCCCN"], 0.0))
        self.assertEqual(ax["n_unaccommodated"], 1)
        self.assertEqual(ax["n_accommodated"], 1)
        self.assertFalse(ax["all_accommodated"])


class TestAccommodationSortKey(unittest.TestCase):
    def test_hosting_every_substrate_is_the_first_axis(self):
        # Fewer unaccommodated substrates ranks first (lower key) EVEN when the
        # unaccommodated candidate has a higher weakest-FG number — the host-every-
        # substrate count is the hard first axis, not blended with chemistry.
        accommodated = _accommodation_sort_key(_accom([{}, {}], [], 0.46))
        unaccommodated = _accommodation_sort_key(_accom([{}, {}], ["NCCCCCCN"], 0.9))
        self.assertLess(accommodated, unaccommodated)

    def test_missing_sorts_neutral_between(self):
        accommodated = _accommodation_sort_key(_accom([{}], [], 0.9, n=1))
        neutral = _accommodation_sort_key({})
        unaccommodated = _accommodation_sort_key(_accom([{}, {}], ["x"], 0.9))
        self.assertLess(accommodated, neutral)
        self.assertLess(neutral, unaccommodated)

    def test_higher_weakest_fg_breaks_tie_when_both_hosted(self):
        strong = _accommodation_sort_key(_accom([{}, {}], [], 0.8))
        weak = _accommodation_sort_key(_accom([{}, {}], [], 0.3))
        self.assertLess(strong, weak)


class TestRankingUsesDecomposedAxes(unittest.TestCase):
    def test_accommodated_ranks_above_unaccommodated(self):
        base = {"literature": {}, "structural_data": {}, "is_swissprot": False,
                "gene_names": [], "critical_warnings": []}
        accommodated = {"data": {**base, "accession": "ACC001",
                                 **_accom([{}, {}], [], 0.46)}}
        unaccommodated = {"data": {**base, "accession": "UNA001",
                                   **_accom([{}, {}], ["NCCCCCCN"], 0.9)}}
        ranked = _deterministic_mechanistic_ranking(
            [unaccommodated, accommodated], {1: 8.0, 2: 5.0, 3: 2.0})
        order = [r["accession"] for r in ranked]
        self.assertLess(order.index("ACC001"), order.index("UNA001"))
        # rationale surfaces the decomposed facts, not a single score
        acc_row = next(r for r in ranked if r["accession"] == "ACC001")
        self.assertIn("substrates 2/2 hosted", acc_row["rationale"])


class TestMechanismEvidenceRanking(unittest.TestCase):
    def test_evidence_levels_distinguish_grounding(self):
        experimental = {"mechanism_evidence": {
            "status": "ok", "facts": {"confidence": "high"},
            "paper": {"doi": "10.1/example"},
        }}
        tentative = {"mechanism_evidence": {
            "status": "low_confidence", "facts": {"confidence": "low"},
        }}
        self.assertEqual(_mechanism_evidence_level(experimental), "experimental")
        self.assertEqual(_mechanism_evidence_level(tentative), "tentative")
        self.assertEqual(_mechanism_evidence_level({}), "unknown")

    def test_grounded_mechanism_breaks_otherwise_equal_tie(self):
        base = {
            "substrate_similarity": {}, "literature": {}, "structural_data": {},
            "is_swissprot": False, "gene_names": [], "critical_warnings": [],
        }
        unknown = {"data": {**base, "accession": "A00001"}}
        grounded = {"data": {**base, "accession": "Z00001", "mechanism_evidence": {
            "status": "ok", "facts": {"confidence": "high"},
            "paper": {"pmcid": "PMC1"},
        }}}
        ranked = _deterministic_mechanistic_ranking(
            [unknown, grounded], {1: 8.0, 2: 5.0, 3: 2.0}
        )
        self.assertEqual(ranked[0]["accession"], "Z00001")
        self.assertIn("mechanism=experimental", ranked[0]["rationale"])


if __name__ == "__main__":
    unittest.main()
