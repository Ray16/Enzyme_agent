"""Unit tests for the developability layer (predictor wrapper, evidence, assessment).

These are offline: the real NetSolP model is exercised by a separate integration check,
not here. Assessment logic is tested with an injected fake predictor.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictors as P  # noqa: E402
from evidence import LiabilityKnowledge  # noqa: E402
import assess as A  # noqa: E402


class FakePredictor(P.SolubilityPredictor):
    name = "fake"

    def __init__(self, score, threshold=0.5, higher_is_better=True):
        self._score = score
        self._threshold = threshold
        self._higher_is_better = higher_is_better

    def predict(self, sequence):
        return P.SolubilityPrediction(score=self._score, method="fake",
                                      citation="test", note="fake",
                                      threshold=self._threshold,
                                      higher_is_better=self._higher_is_better)


class TestNetSolPParser(unittest.TestCase):
    def test_parse_csv(self):
        text = ("sid,fasta,predicted_solubility,predicted_usability\n"
                "query,MKV,0.71,0.39\n")
        recs = P.parse_netsolp_csv(text)
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0]["predicted_solubility"], 0.71)
        self.assertAlmostEqual(recs[0]["predicted_usability"], 0.39)

    def test_parse_csv_solubility_only(self):
        recs = P.parse_netsolp_csv("sid,fasta,predicted_solubility\nq,MKV,0.5\n")
        self.assertNotIn("predicted_usability", recs[0])


class TestNetSolPAvailability(unittest.TestCase):
    def test_unavailable_when_models_missing(self):
        pred = P.NetSolPPredictor(netsolp_dir="/nonexistent/dir")
        self.assertFalse(pred.available())

    def test_predict_abstains_when_unavailable(self):
        pred = P.NetSolPPredictor(netsolp_dir="/nonexistent/dir")
        out = pred.predict("MKVLAAALLLLLAAA")
        self.assertIsNone(out.score)
        self.assertIn("not installed", out.note)


class TestDescriptors(unittest.TestCase):
    def test_grounded_descriptors_present(self):
        d = P.compute_descriptors("MKVLAAALLLLLAAAGGGDDDEEEKKKRRR")
        if not d or "gravy_kyte_doolittle" not in d:
            self.skipTest("Biopython not available")
        self.assertIn("instability_index_guruprasad", d)
        self.assertEqual(d["length"], 30)

    def test_short_sequence_empty(self):
        self.assertEqual(P.compute_descriptors("MK"), {})


class TestEvidence(unittest.TestCase):
    def _kb(self):
        doc = {"records": [
            {"id": "u", "uniprot": "P00001", "liability": "solubility",
             "severity": "veto", "claim": "insoluble (cited)", "families": [],
             "exploration_hint": "find homolog",
             "evidence": [{"pmid": "123", "quote": "..."}]},
            {"id": "f", "uniprot": None, "families": ["IPR011761"],
             "liability": "stability", "severity": "flag", "claim": "labile", "evidence": []},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(doc, fh)
            return fh.name

    def test_lookup_by_uniprot_and_family(self):
        kb = LiabilityKnowledge.load(self._kb())
        self.assertEqual(len(kb.lookup(uniprot="P00001")), 1)
        self.assertEqual(len(kb.lookup(family_ids=["IPR011761"])), 1)
        self.assertEqual(len(kb.lookup(uniprot="P99999")), 0)

    def test_shipped_store_has_no_unsupported_records(self):
        # The shipped store must not assert liabilities without citations.
        kb = LiabilityKnowledge.load()
        self.assertEqual(kb.records, [])


class TestAssessment(unittest.TestCase):
    def _kb(self, records):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"records": records}, fh)
            path = fh.name
        return LiabilityKnowledge.load(path)

    def test_curated_veto_dominates_predictor(self):
        kb = self._kb([{"id": "v", "uniprot": "P1", "liability": "solubility",
                        "severity": "veto", "claim": "insoluble (cited)", "families": [],
                        "evidence": [{"pmid": "1"}]}])
        a = A.assess_developability(sequence="DEKR" * 20, uniprot="P1",
                                    knowledge=kb, predictor=FakePredictor(0.99))
        self.assertEqual(a.verdict, "veto")
        self.assertEqual(a.score, 0.0)
        self.assertTrue(a.exploration_trigger["triggered"])

    def test_predicted_soluble_ok(self):
        kb = self._kb([])
        a = A.assess_developability(sequence="DEKR" * 20, knowledge=kb,
                                    predictor=FakePredictor(0.8))
        self.assertEqual(a.verdict, "ok")
        self.assertIsNone(a.exploration_trigger)

    def test_predicted_low_flags_and_triggers(self):
        kb = self._kb([])
        a = A.assess_developability(sequence="DEKR" * 20, knowledge=kb,
                                    predictor=FakePredictor(0.2))
        self.assertEqual(a.verdict, "flag")
        self.assertTrue(a.exploration_trigger["triggered"])

    def test_no_prediction_no_evidence_abstains(self):
        kb = self._kb([])
        a = A.assess_developability(sequence="DEKR" * 20, knowledge=kb,
                                    predictor=FakePredictor(None))
        self.assertEqual(a.verdict, "abstain")
        self.assertIsNone(a.score)

    def test_no_sequence_abstains(self):
        kb = self._kb([])
        a = A.assess_developability(knowledge=kb, predictor=FakePredictor(0.9))
        self.assertEqual(a.verdict, "abstain")


if __name__ == "__main__":
    unittest.main()
