#!/usr/bin/env python3
"""Tests for vendoring the discovery-agent cache into the coordinate store (generic)."""
import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)

import refresh_coordinates as rc  # noqa: E402

# A store already holding a STRUCTURED coordinate (compiler schema: attack roles/atoms).
STRUCTURED = {
    "mechanisms": {
        "mech-A": {
            "keys": {"activation_mode": ["mode-a"], "families": ["IPR0001"]},
            "steps": [{"index": 1, "name": "s1", "attack": {"nucleophile": {"role": "r", "atom_selector": "sel"}}}],
            "provenance": {"status": "literature_extracted"},
        }
    }
}


class RefreshTest(unittest.TestCase):
    def test_cache_to_mechanisms_groups_by_class(self):
        cache = {"families": {
            "IPR0001": {"mechanism_class": "mech-A", "steps": [{"index": 1}]},
            "IPR0002": {"mechanism_class": "mech-A", "steps": [{"index": 1}]},
            "IPR0003": {"mechanism_class": "mech-B", "steps": [{"index": 1}]},
        }}
        ex = rc.cache_to_mechanisms(cache)
        self.assertCountEqual(ex["mech-A"]["keys"]["families"], ["IPR0001", "IPR0002"])
        self.assertIn("mech-B", ex)

    def test_empty_cache_is_noop(self):
        current = copy.deepcopy(STRUCTURED)
        merged, report = rc.merge(current, {})
        self.assertEqual(report["added_mechanisms"], [])
        self.assertEqual(merged, STRUCTURED)

    def test_new_mechanism_added_flagged_needs_structuring(self):
        current = copy.deepcopy(STRUCTURED)
        exported = rc.cache_to_mechanisms({"families": {
            "IPR9999": {"mechanism_class": "mech-NEW", "steps": [{"index": 1, "name": "x"}],
                        "provenance": {"source_enzyme": "E"}},
        }})
        merged, report = rc.merge(current, exported)
        self.assertIn("mech-NEW", merged["mechanisms"])
        self.assertTrue(merged["mechanisms"]["mech-NEW"]["provenance"]["needs_structuring"])
        self.assertEqual(report["added_mechanisms"], ["mech-NEW"])

    def test_existing_mechanism_unions_families_but_keeps_structured_steps(self):
        current = copy.deepcopy(STRUCTURED)
        exported = rc.cache_to_mechanisms({"families": {
            "IPR0002": {"mechanism_class": "mech-A", "steps": [{"index": 1, "name": "prose-step"}]},
        }})
        merged, report = rc.merge(current, exported)
        fams = merged["mechanisms"]["mech-A"]["keys"]["families"]
        self.assertCountEqual(fams, ["IPR0001", "IPR0002"])          # widened transfer
        # structured steps preserved (not overwritten by the prose extraction)
        steps = merged["mechanisms"]["mech-A"]["steps"]
        self.assertIn("attack", steps[0])
        self.assertEqual(report["families_added"], {"mech-A": ["IPR0002"]})


if __name__ == "__main__":
    unittest.main()
