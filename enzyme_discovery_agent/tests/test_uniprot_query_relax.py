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


def test_structured_queries_never_drop_a_boolean_operand():
    from unittest.mock import patch
    from enzyme_discovery_agent.tools.database import search_uniprot
    query = "gene:ExampleQ AND taxonomy_name:ExampleGenus"
    assert _relax_uniprot_query(query) == [query]
    with patch("enzyme_discovery_agent.tools.database.request_with_retry", return_value=b'{"results": []}') as request:
        result = search_uniprot(query=query, function_filter="synthetic enzyme", max_results=5)
    assert result["status"] == "partial"
    assert result["count"] == 0
    assert request.call_count == 1


def test_completed_empty_gene_search_suppresses_only_equal_or_narrower_queries():
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.tool_registry import check_search_duplicate, record_search
    register_builtin_tools()
    history = []
    prior = {"gene_name": "exampleQ", "function_filter": "specific enzyme", "max_results": 5}
    record_search('search_uniprot', prior, {"status": "partial", "count": 0, "search_completed": True}, history=history)
    narrower = {**prior, "function_filter": "specific enzyme bacterial", "max_results": 20}
    assert check_search_duplicate('search_uniprot', narrower, history=history)['deduplicated']
    assert check_search_duplicate('search_uniprot', {**prior, "function_filter": "specific"}, history=history) is None
    assert check_search_duplicate('search_uniprot', {**prior, "gene_name": "other"}, history=history) is None
    assert check_search_duplicate('search_uniprot', {**prior, "function_filter": "specific OR enzyme"}, history=history) is None
    assert check_search_duplicate('search_uniprot', {**prior, "function_filter": '"specific enzyme"'}, history=history) is None


def test_incomplete_or_failed_search_is_never_cached_as_zero():
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.tool_registry import check_search_duplicate, record_search
    register_builtin_tools()
    args = {"gene_name": "exampleQ", "function_filter": "specific enzyme"}
    for result in ({"status": "error", "count": 0, "search_completed": True},
                   {"status": "partial", "count": 0},
                   {"status": "partial", "count": 1, "search_completed": True}):
        history = []
        record_search('search_uniprot', args, result, history=history)
        assert not history
        assert check_search_duplicate('search_uniprot', args, history=history) is None
