"""Tests for the literature-exploration run scorer."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.eval import score_run
from enzyme_discovery_agent.eval.benchmarks import BENCHMARKS


def _write_run(root: Path):
    (root / "artifacts" / "literature").mkdir(parents=True)
    (root / "artifacts" / "database").mkdir(parents=True)
    (root / "artifacts" / "fulltext").mkdir(parents=True)

    events = [
        {"type": "run_start"},
        {"type": "tool_call", "tool": "search_semantic_scholar", "iteration": 2},
        {"type": "tool_call", "tool": "search_europepmc", "iteration": 2},
        {"type": "tool_call", "tool": "search_pubmed", "iteration": 2},  # fan-out 3
        {"type": "tool_call", "tool": "consolidate_literature", "iteration": 5},
        {"type": "tool_call", "tool": "fetch_full_text", "iteration": 6},
    ]
    (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))

    papers = [
        {"title": "The ATP-dependent amide ligases DdaG and DdaF",
         "abstract": "dapdiamide biosynthesis", "doi": "10.1/a"},
        {"title": "duplicate record", "doi": "10.1/a"},          # dup by DOI
        {"title": "unrelated review", "doi": "10.1/b", "abstract": "x"},
    ]
    (root / "artifacts" / "literature" / "s1.json").write_text(
        json.dumps({"result": {"data": papers}}))
    # accession surfaces via a UniProt (database) artifact, not a paper
    (root / "artifacts" / "database" / "u1.json").write_text(
        json.dumps({"result": {"data": [{"accession": "E2JA32"}]}}))
    (root / "artifacts" / "fulltext" / "manifest.json").write_text(
        json.dumps([{"source": "europepmc", "format": "xml"}]))
    (root / "candidate_registry.json").write_text(json.dumps({"candidates": {
        "answer": {"accession": "E2JA32", "metadata": {"claim_validation": {"promotion_eligible": True}}, "statuses": [
            {"status": "discovered"}, {"status": "screened"},
            {"status": "evaluated"}, {"status": "ranked"},
        ]},
        "other": {"accession": "P12345", "statuses": [
            {"status": "discovered"}, {"status": "screened"},
        ]},
    }}))
    (root / "pass1_ranking.json").write_text(json.dumps([
        {"rank": 1, "accession": "P12345"},
        {"rank": 2, "accession": "E2JA32"},
    ]))


class TestScoreRun(unittest.TestCase):
    def test_scores_a_synthetic_run(self):
        bench = BENCHMARKS["amide_synthetase"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root)
            m = score_run(root, bench)

        self.assertEqual(m["max_fanout"], 3)                 # 3 searches in iter 2
        self.assertEqual(m["n_sources_used"], 3)
        self.assertEqual(m["consolidate_calls"], 1)
        self.assertEqual(m["fetch_full_text_calls"], 1)
        self.assertEqual(m["papers_raw"], 3)
        self.assertEqual(m["papers_unique"], 2)              # DOI dup merged
        self.assertGreater(m["redundancy"], 0)
        self.assertEqual(m["fulltext_downloaded"], 1)
        # adaptive recall: specific member from paper text, accession from the
        # database artifact, mechanism-class as a diagnostic.
        self.assertIn("ddag", m["recall"]["specific_member"]["found"])
        self.assertIn("dapdiamide", m["recall"]["specific_member"]["found"])
        self.assertIn("e2ja32", m["recall"]["specific_accession"]["found"])
        self.assertTrue(m["recall"]["mechanism_class"]["hit"])  # "amide ligase"
        self.assertTrue(m["found_answer"])                      # a primary target hit
        self.assertEqual(m["candidate_stages"]["retention"]["discovered"]["count"], 2)
        self.assertEqual(m["candidate_stages"]["retention"]["evaluated"]["count"], 1)
        self.assertEqual(m["candidate_stages"]["best_known_positive_rank"], 2)

    def test_empty_run_scores_zero_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "artifacts" / "literature").mkdir(parents=True)
            (Path(tmp) / "events.jsonl").write_text("")
            m = score_run(tmp, BENCHMARKS["amide_synthetase"])
        self.assertFalse(m["found_answer"])
        self.assertEqual(m["papers_unique"], 0)
        self.assertEqual(m["max_fanout"], 0)


if __name__ == "__main__":
    unittest.main()
