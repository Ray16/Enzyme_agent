"""Tests for RRF rank-fusion in semantic re-ranking.

`_rrf_fuse` fuses each source's NATIVE retrieval order (list position) with the
semantic cosine order, so a paper strong on *either* axis surfaces. These tests
exercise the pure fusion logic directly (no embedding model required): the key
property is that native retrieval order is fused in, not discarded.
"""

import unittest

from enzyme_discovery_agent.tools import literature


class TestRRFFuse(unittest.TestCase):
    def _titles(self, papers):
        return [p["title"] for p in papers]

    def test_native_order_is_fused_not_discarded(self):
        # Native order A,B,C; cosine says B best, C mid, A worst.
        papers = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        sims = [0.1, 0.9, 0.2]
        out = literature._rrf_fuse(papers, sims, max_results=10, min_score=0.0)
        # B wins (top on both-ish); A (native #1) still beats C despite the
        # lowest cosine -- proving native retrieval rank is not discarded.
        self.assertEqual(self._titles(out), ["B", "A", "C"])

    def test_scores_are_attached_for_audit(self):
        papers = [{"title": "A"}, {"title": "B"}]
        out = literature._rrf_fuse(papers, [0.3, 0.7], max_results=10, min_score=0.0)
        for p in out:
            self.assertIn("semantic_score", p)
            self.assertIn("rrf_score", p)

    def test_cosine_floor_gates_inclusion(self):
        papers = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        sims = [0.1, 0.9, 0.2]
        out = literature._rrf_fuse(papers, sims, max_results=10, min_score=0.5)
        self.assertEqual(self._titles(out), ["B"])  # only B clears 0.5 cosine

    def test_floor_never_starves_to_empty(self):
        papers = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        sims = [0.1, 0.2, 0.15]
        # An over-tight floor removes everything -> degrade to fused order,
        # not an empty result.
        out = literature._rrf_fuse(papers, sims, max_results=10, min_score=0.99)
        self.assertEqual(len(out), 3)

    def test_respects_max_results(self):
        papers = [{"title": t} for t in "ABCDE"]
        sims = [0.5, 0.4, 0.3, 0.2, 0.1]
        out = literature._rrf_fuse(papers, sims, max_results=2, min_score=0.0)
        self.assertEqual(len(out), 2)

    def test_empty_input(self):
        self.assertEqual(literature._semantic_rerank("q", [], max_results=5), [])


if __name__ == "__main__":
    unittest.main()
