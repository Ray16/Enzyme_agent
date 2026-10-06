"""Plateau signal: exploration_saturation reads recent search novelty and flags
when outward search has stopped surfacing new families (a cue for the agent to
broaden, not a hard stop)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.models.event_log import append_event, exploration_saturation


class TestExplorationSaturation(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())

    def _reg(self, new_families, new_accessions):
        append_event(self.run_dir, "candidates_registered",
                     new_families=new_families, new_accessions=new_accessions)

    def test_empty_is_not_plateaued(self):
        sat = exploration_saturation(self.run_dir, window=4)
        self.assertFalse(sat["plateaued"])
        self.assertEqual(sat["events_in_window"], 0)

    def test_window_not_full_is_not_plateaued(self):
        self._reg(0, 0)
        self._reg(0, 0)
        sat = exploration_saturation(self.run_dir, window=4)
        self.assertFalse(sat["plateaued"])        # only 2 of 4 -> undecided

    def test_full_window_no_new_families_is_plateaued(self):
        for _ in range(4):
            self._reg(0, 1)                        # churning orthologs, no new families
        sat = exploration_saturation(self.run_dir, window=4)
        self.assertTrue(sat["plateaued"])
        self.assertEqual(sat["recent_new_families"], 0)
        self.assertEqual(sat["recent_new_accessions"], 4)

    def test_recent_new_family_clears_plateau(self):
        for _ in range(3):
            self._reg(0, 1)
        self._reg(2, 3)                            # a fresh family in the window
        sat = exploration_saturation(self.run_dir, window=4)
        self.assertFalse(sat["plateaued"])
        self.assertEqual(sat["recent_new_families"], 2)

    def test_only_recent_window_counts(self):
        self._reg(5, 5)                            # old productive searches...
        for _ in range(4):
            self._reg(0, 0)                        # ...but the last 4 added nothing
        sat = exploration_saturation(self.run_dir, window=4)
        self.assertTrue(sat["plateaued"])


if __name__ == "__main__":
    unittest.main()
