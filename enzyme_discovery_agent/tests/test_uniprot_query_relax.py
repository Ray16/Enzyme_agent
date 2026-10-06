"""Offline tests for UniProt free-text query relaxation.

UniProt ANDs free-text terms, so an over-specified query zeros out the moment one
term misses (observed: an over-specified name+organism+keyword query for a real,
structurally-characterised enzyme returned 0). The relaxer drops trailing
qualifier terms so a retry can find it.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.tools.database import _relax_uniprot_query


class TestRelaxUniprotQuery(unittest.TestCase):
    def test_full_query_first_then_progressively_shorter(self):
        v = _relax_uniprot_query("A B C D E")
        self.assertEqual(v[0], "A B C D E")           # full query tried first
        self.assertEqual(v[-1], "A B")                # relaxes down to first two
        # strictly decreasing length, core-first
        lengths = [len(x.split()) for x in v]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_stops_at_two_terms(self):
        v = _relax_uniprot_query("one two three")
        self.assertIn("one two", v)
        self.assertNotIn("one", v)                    # never relaxes to a single term

    def test_single_and_two_term_queries_unchanged(self):
        self.assertEqual(_relax_uniprot_query("solo"), ["solo"])
        self.assertEqual(_relax_uniprot_query("two terms"), ["two terms"])

    def test_dedup_preserves_order(self):
        v = _relax_uniprot_query("a a b")
        self.assertEqual(v, ["a a b", "a a"])


if __name__ == "__main__":
    unittest.main()
