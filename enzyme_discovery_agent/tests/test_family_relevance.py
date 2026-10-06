"""Offline tests for fold-family relevance budgeting.

These exercise the PURE budget allocation (no network): every fold keeps its
floor (breadth is never dropped), extra depth flows to same-direction /
substrate-resembling folds first, opposite-direction folds never rise above the
floor, and per-fold and total caps are respected.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.gates.family_relevance import allocate_evaluation_budget

# dir_rank: same=0, unknown=1, unrelated=2, opposite=3
SAME, UNKNOWN, UNRELATED, OPPOSITE = 0, 1, 2, 3


def _fold(fid, n, dir_rank, fg=None):
    return {"fold_id": fid, "n_members": n, "dir_rank": dir_rank, "fg_max": fg}


class TestBudgetFloor(unittest.TestCase):
    def test_every_fold_gets_at_least_floor(self):
        folds = [_fold("A", 5, SAME, 0.6), _fold("B", 3, OPPOSITE, None),
                 _fold("C", 1, UNKNOWN, 0.2)]
        alloc = allocate_evaluation_budget(folds, budget=10, floor=1, cap=6)
        self.assertGreaterEqual(alloc["A"], 1)
        self.assertGreaterEqual(alloc["B"], 1)  # opposite still keeps its floor
        self.assertEqual(alloc["C"], 1)         # singleton fold

    def test_floor_never_dropped_even_when_folds_exceed_budget(self):
        # More folds than the budget: every fold must still get its floor
        # (breadth is guaranteed; the budget only bounds EXTRA depth).
        folds = [_fold(chr(65 + i), 4, SAME, 0.5) for i in range(8)]
        alloc = allocate_evaluation_budget(folds, budget=3, floor=1, cap=6)
        self.assertEqual(len(alloc), 8)
        self.assertTrue(all(v >= 1 for v in alloc.values()))


class TestBudgetRelevance(unittest.TestCase):
    def test_extra_depth_prefers_same_direction(self):
        folds = [_fold("same", 6, SAME, 0.5), _fold("unknown", 6, UNKNOWN, 0.9)]
        alloc = allocate_evaluation_budget(folds, budget=8, floor=1, cap=6)
        # 2 floors + 6 extra.  Same-direction is ranked ahead of unknown even
        # though unknown has higher fg, so it fills first.
        self.assertEqual(alloc["same"], 6)
        self.assertEqual(alloc["unknown"], 2)

    def test_extra_depth_prefers_higher_resemblance_within_direction(self):
        folds = [_fold("hi", 6, SAME, 0.8), _fold("lo", 6, SAME, 0.2)]
        alloc = allocate_evaluation_budget(folds, budget=8, floor=1, cap=6)
        self.assertEqual(alloc["hi"], 6)   # higher fg fills first
        self.assertEqual(alloc["lo"], 2)

    def test_opposite_direction_never_exceeds_floor(self):
        folds = [_fold("opp", 20, OPPOSITE, None), _fold("good", 6, SAME, 0.5)]
        alloc = allocate_evaluation_budget(folds, budget=40, floor=1, cap=6)
        self.assertEqual(alloc["opp"], 1)          # floored despite 20 members
        self.assertEqual(alloc["good"], 6)         # on-target fold gets depth

    def test_inversion_is_fixed(self):
        # The observed failure: one big off-target fold vs a small on-target one.
        folds = [_fold("acyltransferase", 16, OPPOSITE, None),
                 _fold("dapdiamide", 2, SAME, 0.6)]
        alloc = allocate_evaluation_budget(folds, budget=40, floor=1, cap=6)
        self.assertEqual(alloc["acyltransferase"], 1)   # 16 -> 1 (floored)
        self.assertEqual(alloc["dapdiamide"], 2)        # gets both members


class TestBudgetCaps(unittest.TestCase):
    def test_per_fold_cap_respected(self):
        folds = [_fold("A", 20, SAME, 0.9)]
        alloc = allocate_evaluation_budget(folds, budget=40, floor=1, cap=6)
        self.assertEqual(alloc["A"], 6)

    def test_cannot_exceed_member_count(self):
        folds = [_fold("A", 3, SAME, 0.9)]
        alloc = allocate_evaluation_budget(folds, budget=40, floor=1, cap=6)
        self.assertEqual(alloc["A"], 3)

    def test_total_extra_bounded_by_budget(self):
        folds = [_fold("A", 6, SAME, 0.9), _fold("B", 6, SAME, 0.8),
                 _fold("C", 6, SAME, 0.7)]
        alloc = allocate_evaluation_budget(folds, budget=10, floor=1, cap=6)
        # 3 floors (3) + 7 extra = 10 total.
        self.assertEqual(sum(alloc.values()), 10)


if __name__ == "__main__":
    unittest.main()
