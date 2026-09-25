"""A fold-STOP is an informational annotation, never a ranking penalty.

Regression guard for the observed failure: the agent stopped the fold that
actually answered the query, which then hard-gated its members to the bottom
tier and sank them below unrelated analogues.  A stop must not gate, must not
sink ``fam_rank``, and must not remove a candidate from consideration — only a
genuine per-candidate mechanistic contradiction (``critical_warnings``) does.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.output.report_generator import (
    _deterministic_mechanistic_ranking,
    _family_rank_map,
)


class TestFamilyRankMap(unittest.TestCase):
    def test_stopped_is_neutral_not_sunk(self):
        fr = _family_rank_map(stopped_accs={"STOP1"}, ontarget_accs={"ON1"})
        self.assertEqual(fr.get("ON1"), 0)          # on-target still boosted
        self.assertNotIn("STOP1", fr)               # stopped -> default neutral (1)


class TestStopDoesNotSinkRanking(unittest.TestCase):
    def _eval(self, acc, fit_smiles=None, warnings=None):
        # A minimal eval record; substrate fit is driven by a native-vs-target
        # accommodation score, here supplied directly via the cached field.
        d = {"accession": acc, "critical_warnings": warnings or []}
        if fit_smiles is not None:
            d["substrate_accommodation"] = fit_smiles
        return {"data": d}

    def test_stopped_member_outranks_neutral_when_no_contradiction(self):
        # STOP1 sits in a stopped fold but has no mechanistic contradiction; it
        # must be free to rank by its own facts, above a plain neutral candidate.
        evals = [self._eval("NEU1"), self._eval("STOP1")]
        fam = _family_rank_map(stopped_accs={"STOP1"}, ontarget_accs={"STOP1"})
        ranked = _deterministic_mechanistic_ranking(
            evals, tier_score={1: 8.0, 2: 5.0, 3: 2.0}, family_rank_by_acc=fam)
        order = [r["accession"] for r in ranked]
        # On-target STOP1 (fam_rank 0) ranks ahead of neutral NEU1 (fam_rank 1);
        # the stop did not gate it to the bottom.
        self.assertLess(order.index("STOP1"), order.index("NEU1"))
        self.assertNotEqual(ranked[order.index("STOP1")]["tier"], 3)

    def test_critical_warning_still_gates(self):
        evals = [self._eval("GOOD"), self._eval("BAD", warnings=["direction mismatch"])]
        ranked = _deterministic_mechanistic_ranking(
            evals, tier_score={1: 8.0, 2: 5.0, 3: 2.0})
        by_acc = {r["accession"]: r for r in ranked}
        self.assertEqual(by_acc["BAD"]["tier"], 3)          # hard-gated
        self.assertLess([r["accession"] for r in ranked].index("GOOD"),
                        [r["accession"] for r in ranked].index("BAD"))


if __name__ == "__main__":
    unittest.main()
