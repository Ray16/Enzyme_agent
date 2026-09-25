"""Offline tests for the family-survey gate PROMOTING substrate-fit members.

The survey identifies the relevant enzymes in a family (best native-substrate
fit, same/unknown direction).  Those members must become CANDIDATES so they enter
evaluation — the earlier gate only showed them to the model.  These tests exercise
the promotion helper directly (no network, no LLM) against a temp registry.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry


class TestPromoteRelevant(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        self.gate = FamilySurveyGate()

    def _relevant(self, acc, fit, name="", reviewed=True):
        return {"accession": acc, "substrate_fit": fit, "name": name,
                "reviewed": reviewed, "direction_vs_target": "same"}

    def test_promotes_and_tags_provenance(self):
        n = self.gate._promote_relevant(
            self.run_dir, "IPR000873",
            [self._relevant("P00001", 0.9, name="alpha"),
             self._relevant("Q00002", 0.6)],
        )
        self.assertEqual(n, 2)
        reg = load_candidate_registry(self.run_dir)["candidates"]
        self.assertIn("P00001", reg)
        self.assertIn("Q00002", reg)
        meta = reg["P00001"]["metadata"]
        self.assertEqual(meta["source_tool"], "family_substrate_fit:IPR000873")
        self.assertEqual(meta["substrate_fit"], 0.9)
        # The paper/UniProt name rides along as the candidate label.
        self.assertEqual(reg["P00001"]["enzyme_name"], "alpha")
        # Registered as a discovered candidate (enters the evaluation pipeline).
        self.assertEqual(reg["P00001"]["statuses"][-1]["status"], "discovered")

    def test_dedup_across_folds(self):
        n1 = self.gate._promote_relevant(self.run_dir, "IPR0001",
                                         [self._relevant("P00001", 0.9)])
        # Same accession surfaces again in another fold — promoted only once.
        n2 = self.gate._promote_relevant(self.run_dir, "IPR0002",
                                         [self._relevant("P00001", 0.7)])
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 0)

    def test_empty_relevant_is_noop(self):
        self.assertEqual(self.gate._promote_relevant(self.run_dir, "IPR0001", []), 0)
        self.assertEqual(load_candidate_registry(self.run_dir)["candidates"], {})

    def test_skips_entries_without_accession(self):
        n = self.gate._promote_relevant(
            self.run_dir, "IPR0001",
            [{"accession": "", "substrate_fit": 0.9}, self._relevant("P00001", 0.9)],
        )
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
