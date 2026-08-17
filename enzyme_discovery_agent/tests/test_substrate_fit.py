"""Tests for the shared one-to-one substrate assignment."""

import unittest

from enzyme_discovery_agent.substrate_fit import assign_one_to_one


class TestAssignOneToOne(unittest.TestCase):
    def test_each_native_serves_one_target(self):
        # Two targets, two natives. Native "acid" fits target "A" best AND target
        # "B" second-best; native "amine" fits "B" poorly. Independent-best would
        # map BOTH targets onto "acid"; one-to-one must give each a DISTINCT native.
        score = lambda t, n: {("A", "acid"): 0.9, ("A", "amine"): 0.1,
                              ("B", "acid"): 0.6, ("B", "amine"): 0.0}[(t, n)]
        res = assign_one_to_one(["A", "B"], ["acid", "amine"], score)
        natives = {t: n for t, n, _ in res}
        self.assertEqual(len(set(natives.values())), 2)          # distinct natives
        self.assertEqual(natives["A"], "acid")                   # A keeps its best
        self.assertEqual(natives["B"], "amine")                  # B forced onto amine

    def test_ammonia_case_is_demoted(self):
        # Target = {diacid, diamine}; native = {aspartate-like, ammonia-like}.
        # The diamine matches aspartate (size/some FG) 0.48 but ammonia 0.0.
        # Independent-best: both targets grab aspartate -> worst 0.48 (too high).
        # One-to-one: diamine forced onto ammonia -> worst ~0.0 (correctly low).
        score = lambda t, n: {("diacid", "asp"): 0.55, ("diacid", "nh3"): 0.0,
                              ("diamine", "asp"): 0.48, ("diamine", "nh3"): 0.0}[(t, n)]
        res = assign_one_to_one(["diacid", "diamine"], ["asp", "nh3"], score)
        worst = min(s for _, _, s in res)
        self.assertLess(worst, 0.1)                              # bottleneck is ammonia

    def test_maximin_prefers_balanced_assignment(self):
        # Assignment 1: (0.9, 0.2) worst=0.2 ; Assignment 2: (0.5, 0.5) worst=0.5.
        # Maximin must pick the balanced one.
        score = lambda t, n: {("A", "x"): 0.9, ("A", "y"): 0.5,
                              ("B", "x"): 0.5, ("B", "y"): 0.2}[(t, n)]
        res = assign_one_to_one(["A", "B"], ["x", "y"], score)
        self.assertEqual(min(s for _, _, s in res), 0.5)

    def test_more_targets_than_natives_leaves_one_unmatched(self):
        score = lambda t, n: 0.7
        res = assign_one_to_one(["A", "B"], ["only"], score)
        matched = [n for _, n, _ in res if n is not None]
        self.assertEqual(matched, ["only"])                      # one native used once
        self.assertTrue(any(n is None for _, n, _ in res))       # a target unmatched
        self.assertEqual(min(s for _, _, s in res), 0.0)         # unmatched -> 0

    def test_single_substrate_each(self):
        res = assign_one_to_one(["A"], ["x"], lambda t, n: 0.42)
        self.assertEqual(res, [("A", "x", 0.42)])

    def test_empty_inputs(self):
        self.assertEqual(assign_one_to_one([], ["x"], lambda t, n: 1.0), [])
        self.assertEqual(assign_one_to_one(["A"], [], lambda t, n: 1.0),
                         [("A", None, 0.0)])

    def test_greedy_fallback_matches_exact_on_small(self):
        # Force the greedy branch by exceeding the exact-enumeration bounds and
        # check it still gives distinct natives, best-first.
        targets = [f"t{i}" for i in range(8)]
        natives = [f"n{i}" for i in range(8)]
        score = lambda t, n: 1.0 if t[1:] == n[1:] else 0.0     # identity is best
        res = assign_one_to_one(targets, natives, score)
        self.assertTrue(all(t[1:] == n[1:] for t, n, _ in res)) # each matched to its own
        self.assertEqual(len({n for _, n, _ in res}), 8)         # all distinct


if __name__ == "__main__":
    unittest.main()
