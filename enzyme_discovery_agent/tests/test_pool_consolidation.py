"""Tests for cross-source pool consolidation: dedup, record-merge, funnel.

Rerank is off by default (no model), so ``consolidate_pool`` exercises the
dedup/merge path and returns native order -- exactly what we want to assert
here (the rerank fusion itself is covered by test_rerank_fusion).
"""

import unittest

from enzyme_discovery_agent.tools import literature as L


class TestDedupeAndMerge(unittest.TestCase):
    def test_dedup_by_doi_and_merge_fills_fields(self):
        papers = [
            {"title": "Amide synthetase X", "doi": "10.1/x", "pmid": "111",
             "abstract": "short", "source": "pubmed"},
            {"title": "Amide synthetase X", "doi": "10.1/X",  # case-different DOI
             "pmcid": "PMC9", "abstract": "a much longer abstract with detail",
             "source": "europepmc"},
        ]
        unique, stats = L.dedupe_and_merge(papers)
        self.assertEqual(len(unique), 1)
        self.assertEqual(stats, {"raw": 2, "unique": 1, "duplicates_merged": 1})
        rec = unique[0]
        self.assertEqual(rec["pmid"], "111")          # kept from record A
        self.assertEqual(rec["pmcid"], "PMC9")        # filled from record B
        self.assertIn("longer abstract", rec["abstract"])   # kept the longer one
        self.assertEqual(rec["sources"], ["europepmc", "pubmed"])  # provenance union

    def test_dedup_by_pmid_when_no_doi(self):
        papers = [
            {"title": "A", "pmid": "222", "source": "pubmed"},
            {"title": "A (different formatting)", "pmid": "222", "source": "semantic_scholar"},
        ]
        unique, stats = L.dedupe_and_merge(papers)
        self.assertEqual(len(unique), 1)
        self.assertEqual(sorted(unique[0]["sources"]), ["pubmed", "semantic_scholar"])

    def test_dedup_by_normalized_title_when_no_ids(self):
        papers = [
            {"title": "Directed Evolution of an Amide Synthetase!"},
            {"title": "directed evolution of an amide synthetase"},  # punctuation/case
        ]
        unique, stats = L.dedupe_and_merge(papers)
        self.assertEqual(len(unique), 1)

    def test_distinct_papers_are_kept(self):
        papers = [
            {"title": "A", "doi": "10.1/a"},
            {"title": "B", "doi": "10.1/b"},
        ]
        unique, _ = L.dedupe_and_merge(papers)
        self.assertEqual(len(unique), 2)

    def test_unkeyable_records_pass_through(self):
        papers = [{"abstract": "no title, no ids"}]
        unique, stats = L.dedupe_and_merge(papers)
        self.assertEqual(len(unique), 1)


class TestConsolidatePool(unittest.TestCase):
    def test_funnel_counts(self):
        papers = [
            {"title": "X", "doi": "10.1/x", "source": "pubmed"},
            {"title": "X", "doi": "10.1/x", "source": "openalex"},   # dup
            {"title": "Y", "doi": "10.1/y", "source": "europepmc"},
        ]
        out = L.consolidate_pool("amide synthetase", papers, max_results=25)
        self.assertEqual(out["funnel"],
                         {"raw": 3, "unique": 2, "duplicates_merged": 1, "kept": 2})
        titles = {p["title"] for p in out["ranked"]}
        self.assertEqual(titles, {"X", "Y"})

    def test_respects_max_results(self):
        papers = [{"title": t, "doi": f"10.1/{t}"} for t in "ABCDE"]
        out = L.consolidate_pool("q", papers, max_results=2)
        self.assertEqual(out["funnel"]["unique"], 5)
        self.assertEqual(out["funnel"]["kept"], 2)


if __name__ == "__main__":
    unittest.main()
