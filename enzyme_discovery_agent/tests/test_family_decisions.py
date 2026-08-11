"""Tests for agent expand/stop decisions and their effect on the eval budget."""

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.gates.family_relevance import allocate_evaluation_budget
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.models.family_decisions import (
    load_family_decisions,
    record_family_decisions,
    stopped_folds,
)


class TestStopAwareAllocation(unittest.TestCase):
    def _folds(self):
        return [
            {"fold_id": "IPR_NIS", "n_members": 5, "dir_rank": 0, "fg_max": 0.8},
            {"fold_id": "IPR_tRNA", "n_members": 3, "dir_rank": 1, "fg_max": None},
        ]

    def test_floor_covers_offtarget_without_a_stop(self):
        # Default behaviour: the off-target fold still gets its floor (the bug).
        t = allocate_evaluation_budget(self._folds(), budget=6, floor=1, cap=5)
        self.assertEqual(t["IPR_tRNA"], 1)
        self.assertEqual(t["IPR_NIS"], 5)

    def test_stop_waives_the_floor_and_reallocates(self):
        # Agent stops the off-target fold: it gets zero, its floor is waived,
        # and the freed budget stays on the on-target fold (never exceeds cap).
        t = allocate_evaluation_budget(
            self._folds(), budget=6, floor=1, cap=5, stopped={"IPR_tRNA"})
        self.assertEqual(t["IPR_tRNA"], 0)
        self.assertEqual(t["IPR_NIS"], 5)

    def test_stopped_fold_never_gets_extra_depth(self):
        # Even a same-direction, high-resemblance fold gets zero once stopped.
        folds = [{"fold_id": "A", "n_members": 10, "dir_rank": 0, "fg_max": 0.9}]
        t = allocate_evaluation_budget(folds, budget=5, floor=1, cap=10, stopped={"A"})
        self.assertEqual(t["A"], 0)


class TestFamilyDecisions(unittest.TestCase):
    def test_record_load_and_stopped_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_family_decisions(
                root, stop=["IPR_tRNA"], expand=["IPR_NIS"],
                reasons={"IPR_tRNA": "aminoacyl-tRNA, unlike aliphatic diacid",
                         "IPR_NIS": "condenses diamine+diacid"})
            self.assertEqual(stopped_folds(root), {"IPR_tRNA"})
            d = load_family_decisions(root)
            self.assertEqual(d["IPR_NIS"]["decision"], "expand")
            self.assertIn("diamine", d["IPR_NIS"]["reason"])

    def test_decision_is_revisitable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_family_decisions(root, stop=["IPR_x"])
            self.assertEqual(stopped_folds(root), {"IPR_x"})
            # Resume it later — latest call wins, so it is no longer stopped.
            record_family_decisions(root, expand=["IPR_x"])
            self.assertEqual(stopped_folds(root), set())

    def test_emits_family_decision_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_family_decisions(root, stop=["IPR_a"], expand=["IPR_b"])
            evs = [e for e in read_events(root) if e["type"] == "family_decision"]
            self.assertEqual({e["fold"] for e in evs}, {"IPR_a", "IPR_b"})
            self.assertEqual({e["decision"] for e in evs}, {"stop", "expand"})


if __name__ == "__main__":
    unittest.main()
