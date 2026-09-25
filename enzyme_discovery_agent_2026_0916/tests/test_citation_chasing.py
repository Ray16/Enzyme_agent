"""Tests for forward/backward citation chasing in deep_literature_search."""

import json
import unittest
from unittest import mock

from enzyme_discovery_agent.tools import literature as L

_HTTP = "enzyme_discovery_agent.tools.literature.request_with_retry"


def _work(i, cites=0):
    return {
        "id": f"https://openalex.org/W{i}",
        "title": f"Work {i}",
        "publication_year": 2020,
        "doi": f"https://doi.org/10.1/{i}",
        "cited_by_count": cites,
        "abstract_inverted_index": {"amide": [0], "ligase": [1]},
        "authorships": [{"author": {"display_name": "A B"}}],
    }


class TestForwardCitations(unittest.TestCase):
    def test_fetches_top_cited_with_correct_params(self):
        captured = {}

        def fake(req, timeout=30):
            captured["url"] = req.full_url
            return json.dumps({"results": [_work(i) for i in range(3)]}).encode()

        with mock.patch(_HTTP, side_effect=fake):
            out = L._fetch_forward_citations(
                ["https://api.openalex.org/works?filter=cites:W1"],
                per_seed=3, total_cap=5)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["source"], "openalex")
        self.assertIn("per-page=3", captured["url"])
        self.assertIn("sort=cited_by_count:desc", captured["url"])  # top-cited first

    def test_total_cap_across_seeds(self):
        def fake(req, timeout=30):
            return json.dumps({"results": [_work(i) for i in range(5)]}).encode()

        with mock.patch(_HTTP, side_effect=fake):
            out = L._fetch_forward_citations(["u1", "u2", "u3"], per_seed=5, total_cap=7)
        self.assertEqual(len(out), 7)  # stops at the global cap

    def test_parses_work_fields(self):
        p = L._openalex_work_to_paper(_work(9, cites=42))
        self.assertEqual(p["doi"], "10.1/9")
        self.assertEqual(p["cited_by_count"], 42)
        self.assertIn("amide", p["abstract"])
        self.assertEqual(p["openalex_id"], "https://openalex.org/W9")


class TestDedupPapers(unittest.TestCase):
    def test_dedup_by_id_then_doi(self):
        papers = [
            {"openalex_id": "W1", "doi": "10.1/a"},
            {"openalex_id": "W1", "doi": "10.1/b"},   # dup by id
            {"doi": "10.1/c"},
            {"doi": "10.1/C"},                          # dup by DOI (case-insensitive)
        ]
        self.assertEqual(len(L._dedup_papers(papers)), 2)


class TestRoundRobin(unittest.TestCase):
    def test_interleaves_multiple_sources(self):
        self.assertEqual(list(L._round_robin([[1, 2, 3], [4, 5], [6]])),
                         [1, 4, 6, 2, 5, 3])


if __name__ == "__main__":
    unittest.main()
