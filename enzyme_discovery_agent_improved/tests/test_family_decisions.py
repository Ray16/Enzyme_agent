"""Tests for agent expand/stop decisions and their effect on the eval budget."""

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent_improved.gates.family_relevance import allocate_evaluation_budget
from enzyme_discovery_agent_improved.models.event_log import read_events
from enzyme_discovery_agent_improved.models.family_decisions import (
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


class TestProtectedMembers(unittest.TestCase):
    """A literature-named candidate is evaluated on its own merits even when its
    fold is stopped — the stop pruned breadth, not that specific enzyme."""

    def _run(self, reps, stopped, protected):
        # build_relevance_and_allocate needs a run_dir for its side files and a
        # fold resolver.  Exercise the pure selection instead by patching the
        # network-touching helpers with a temp dir.
        import tempfile
        from unittest import mock

        from enzyme_discovery_agent_improved.gates import family_relevance as fr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Each rep resolves to its own singleton fold named after its acc, and
            # every fold is "same" direction with no fg survey (offline).
            with mock.patch.object(fr, "_most_relevant_fold",
                                   side_effect=lambda ids, cache: ids[0]), \
                 mock.patch.object(fr, "_candidate_direction", return_value="same"), \
                 mock.patch.object(fr, "_fold_fg_max", return_value=None), \
                 mock.patch("enzyme_discovery_agent_improved.models.family_decisions."
                            "stopped_folds", return_value=stopped):
                sel, land = fr.build_relevance_and_allocate(
                    root, reps, protected=protected)
        return sel, land

    def test_named_member_survives_a_stopped_singleton_fold(self):
        # DdaG-shaped case: the sole member of a stopped fold, surfaced by name.
        reps = [("E2JA32", {"interpro_ids": ["IPR053158"], "seq_length": 300})]
        sel, _ = self._run(reps, stopped={"IPR053158"}, protected={"E2JA32"})
        self.assertIn("E2JA32", sel)

    def test_unprotected_member_of_a_stopped_fold_is_still_dropped(self):
        reps = [("X99999", {"interpro_ids": ["IPR053158"], "seq_length": 300})]
        sel, _ = self._run(reps, stopped={"IPR053158"}, protected=set())
        self.assertNotIn("X99999", sel)


class TestReviewedTremblSplit(unittest.TestCase):
    """A relevant fold evaluates EVERY reviewed member (comprehensive annotated
    coverage) but only a bounded slice of its unreviewed (TrEMBL) tail."""

    def _run(self, reps):
        import tempfile
        from unittest import mock
        from enzyme_discovery_agent_improved.gates import family_relevance as fr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(fr, "_most_relevant_fold",
                                   side_effect=lambda ids, cache: ids[0]), \
                 mock.patch.object(fr, "_candidate_direction", return_value="same"), \
                 mock.patch.object(fr, "_fold_fg_max", return_value=None), \
                 mock.patch.object(fr.config, "FAMILY_TREMBL_EVAL_CAP_PER_FOLD", 2), \
                 mock.patch("enzyme_discovery_agent_improved.models.family_decisions."
                            "stopped_folds", return_value=set()):
                sel, _ = fr.build_relevance_and_allocate(root, reps)
        return sel

    def test_all_reviewed_kept_trembl_capped(self):
        fold = ["IPRSHARED"]
        reviewed = [(f"R{i}", {"interpro_ids": fold, "seq_length": 300 + i,
                               "reviewed": True}) for i in range(4)]
        trembl = [(f"U{i}", {"interpro_ids": fold, "seq_length": 200 + i,
                             "reviewed": False}) for i in range(5)]
        sel = self._run(reviewed + trembl)
        # Every reviewed member is evaluated; the TrEMBL tail is capped at 2.
        self.assertTrue(all(f"R{i}" in sel for i in range(4)))
        self.assertEqual(sum(1 for i in range(5) if f"U{i}" in sel), 2)


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
