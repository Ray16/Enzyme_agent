"""The ranking substrate signal must DEMOTE a candidate that cannot host every
target substrate, not score its unresolvable partner as neutral.

Regression guard for the subtle bug: reactant-side resolution makes a
cofactor-scale partner (e.g. a one-atom amine donor) unresolvable, so the raw
per-pair fit is dropped and the score would be None -> treated as a neutral
midpoint in ranking, letting a non-accommodating candidate outrank a real,
moderate both-substrate fit.  An unaccommodated target must score 0.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.output.report_generator import (
    _accommodation_score,
    _deterministic_mechanistic_ranking,
    _mechanism_evidence_level,
)


class TestAccommodationScore(unittest.TestCase):
    def test_unaccommodated_target_scores_zero_not_neutral(self):
        data = {"substrate_similarity": {
            "property_comparisons": [],
            "accommodation": {"unaccommodated_targets": ["NCCCCCCN"],
                              "all_target_substrates_accommodated": False},
        }}
        self.assertEqual(_accommodation_score(data), 0.0)

    def test_no_substrate_data_is_unknown_none(self):
        self.assertIsNone(_accommodation_score({"substrate_similarity": {}}))
        self.assertIsNone(_accommodation_score({}))

    def test_accommodated_candidate_uses_physics_fit(self):
        # A fully accommodated candidate with a real property comparison keeps a
        # positive physics-based score (not zeroed).
        comp = {
            "functional_group_similarity": 0.8, "mw_similarity": 0.7,
            "volume_similarity": 0.7, "chain_length_similarity": 0.7,
            "mw_target": 146.0, "mw_native": 150.0,
        }
        data = {"substrate_similarity": {
            "property_comparisons": [comp],
            "accommodation": {"unaccommodated_targets": [],
                              "all_target_substrates_accommodated": True},
        }}
        score = _accommodation_score(data)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)

    def test_ranking_orders_accommodated_above_unaccommodated(self):
        # The concrete relevance win: a real 0.46 fit ranks ABOVE an unaccommodated
        # 0.0 under the ranking's ascending fit_key (lower key = higher rank).
        accommodated = {"substrate_similarity": {
            "property_comparisons": [{
                "functional_group_similarity": 0.6, "mw_similarity": 0.7,
                "volume_similarity": 0.6, "chain_length_similarity": 0.6,
                "mw_target": 146.0, "mw_native": 160.0}],
            "accommodation": {"unaccommodated_targets": []},
        }}
        unaccommodated = {"substrate_similarity": {
            "property_comparisons": [],
            "accommodation": {"unaccommodated_targets": ["NCCCCCCN"]},
        }}
        s_acc = _accommodation_score(accommodated)
        s_un = _accommodation_score(unaccommodated)

        def fit_key(s):
            return -(s if s is not None else 0.5)
        self.assertLess(fit_key(s_acc), fit_key(s_un))  # accommodated ranks first


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
