"""Retroactive stop-fold exclusion at finalization.

A fold the agent STOPS late (after its members were already screened/evaluated)
leaves those members in the ranked pool, where they otherwise rank on their own
facts (observed: an evaluated member whose fold was stopped minutes later still
landed in the ranked CSV).  At finalization we drop such members below the
reporting line — Tier 3, flagged ``excluded_stopped_fold``, omitted from the CSV
decision list but kept in ``pass1_ranking.json`` for provenance — with a carve-out
that NEVER excludes a literature-named enzyme (a stop is a coarse fold-level
breadth call and can be wrong).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.output.report_generator import (
    _annotate_reasoned_tiers,
)
from enzyme_discovery_agent.output.csv_generator import _load_ranking


def _eval(acc, *, swissprot=True, char_papers=1):
    return {"data": {
        "accession": acc,
        "is_swissprot": swissprot,
        "critical_warnings": [],
        "literature": {"characterization_papers": char_papers},
    }}


class TestAnnotateSink(unittest.TestCase):
    def _annotate(self, ranking, acc_to_eval, ontarget, stopped, sink):
        # Base signature keeps ``fold_fit`` as the 5th positional arg; pass an
        # empty map so these unit tests exercise only the sink behaviour.
        _annotate_reasoned_tiers(
            ranking, acc_to_eval, ontarget, stopped, {}, sink=sink
        )

    def test_sunk_member_is_tier3_and_flagged(self):
        ranking = [{"accession": "ON1", "rank": 1},
                   {"accession": "STOPLATE", "rank": 2}]
        acc_to_eval = {"ON1": _eval("ON1"),
                       "STOPLATE": _eval("STOPLATE")}
        acc_to_eval['ON1']['data']['priority_assessment']={'decision':'prioritize',
            'reason':'Source-grounded role correspondence supports this investigation.',
            'strengths':['The native reacting roles correspond to the target hypothesis.'],
            'weaknesses':[],'unknowns':['Target activity is untested.']}
        self._annotate(ranking, acc_to_eval, ontarget={"ON1"},
                       stopped={"STOPLATE"}, sink={"STOPLATE"})
        by = {e["accession"]: e for e in ranking}
        self.assertEqual(by["STOPLATE"]["tier"], 3)
        self.assertTrue(by["STOPLATE"].get("excluded_stopped_fold"))
        # The on-target lead is untouched.
        self.assertEqual(by["ON1"]["tier"], 1)
        self.assertFalse(by["ON1"].get("excluded_stopped_fold"))

    def test_literature_named_stopped_member_is_not_sunk(self):
        # A stopped-fold member that is literature-named is kept OUT of ``sink`` by
        # the caller; here we prove that a stopped member absent from ``sink`` is
        # neither Tier-3-gated nor flagged (annotation-only, ranks on its facts).
        ranking = [{"accession": "NAMED", "rank": 1}]
        acc_to_eval = {"NAMED": _eval("NAMED")}
        self._annotate(ranking, acc_to_eval, ontarget=set(),
                       stopped={"NAMED"}, sink=set())
        e = ranking[0]
        self.assertNotEqual(e["tier"], 3)
        self.assertFalse(e.get("excluded_stopped_fold"))
        self.assertEqual(e["family_relevance"], "off-target(stopped)")

    def test_no_sink_preserves_prior_behaviour(self):
        # With an empty sink nothing is flagged; a plain stopped member stays a
        # ranked, annotation-only candidate (back-compat with the default call).
        ranking = [{"accession": "S", "rank": 1}]
        _annotate_reasoned_tiers(ranking, {"S": _eval("S")}, set(), {"S"}, {})
        self.assertFalse(ranking[0].get("excluded_stopped_fold"))


class TestCsvExcludesFlagged(unittest.TestCase):
    def test_load_ranking_omits_excluded_stopped_fold(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "pass1_ranking.json").write_text(json.dumps([
                {"accession": "KEEP", "rank": 1, "tier": 2},
                {"accession": "DROP", "rank": 2, "tier": 3,
                 "excluded_stopped_fold": True},
            ]))
            lookup = _load_ranking(run)
            self.assertIn("KEEP", lookup)
            self.assertNotIn("DROP", lookup)


if __name__ == "__main__":
    unittest.main()
