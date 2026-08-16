"""Offline tests for the deterministic mechanism-family constraint filter.

The derivation prompt STATES a direction + free-substrate constraint, but the
model does not reliably self-apply it (observed on the 20260816 run: it listed
serine-hydrolase and cysteine-protease families — native CLEAVAGE — for a bond-
FORMATION target). These tests pin the CODE enforcement that drops such families
regardless of what the model labelled them.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type
from enzyme_discovery_agent.tools.mechanism_families import (
    family_passes_constraints,
    _family_native_direction,
)


def _fam(native_reaction, native_direction="", substrates_free=None):
    return {"name": "x", "native_reaction": native_reaction,
            "native_direction": native_direction, "substrates_free": substrates_free}


class TestNativeDirection(unittest.TestCase):
    def test_water_reactant_is_cleavage_even_if_mislabelled(self):
        # H2O on the LEFT of the arrow => hydrolysis => cleavage, overriding a
        # wrong 'formation' label the model may have written.
        f = _fam("peptide + H2O -> two peptides", native_direction="formation")
        self.assertEqual(_family_native_direction(f), "cleavage")

    def test_uses_stated_direction_when_no_water(self):
        f = _fam("acid + amine + ATP -> amide + AMP + PPi", native_direction="formation")
        self.assertEqual(_family_native_direction(f), "formation")

    def test_water_on_product_side_is_not_cleavage(self):
        # Condensation releases water on the RIGHT — that is still formation.
        f = _fam("acid + amine -> amide + H2O", native_direction="formation")
        self.assertEqual(_family_native_direction(f), "formation")


class TestFamilyPassesConstraints(unittest.TestCase):
    def test_keeps_formation_free_substrate_family(self):
        ok, reason = family_passes_constraints(
            _fam("free acid + free amine + ATP -> amide + AMP + PPi",
                 "formation", True), "formation")
        self.assertTrue(ok, reason)

    def test_drops_inverse_direction(self):
        ok, reason = family_passes_constraints(
            _fam("amide + H2O -> acid + amine", "cleavage", True), "formation")
        self.assertFalse(ok)
        self.assertIn("opposite", reason)

    def test_drops_inverse_direction_despite_wrong_label(self):
        # Model labelled a hydrolase 'formation'; the H2O-reactant signal wins.
        ok, _ = family_passes_constraints(
            _fam("peptide + H2O -> two peptides", "formation", True), "formation")
        self.assertFalse(ok)

    def test_drops_non_free_substrate(self):
        ok, reason = family_passes_constraints(
            _fam("cofactor-acid + ATP + protein-lysine -> amide", "formation", False),
            "formation")
        self.assertFalse(ok)
        self.assertIn("free", reason)

    def test_unknown_target_direction_skips_direction_check(self):
        # If the target direction can't be determined, do NOT empty the list on
        # the direction axis (free-substrate still applies).
        ok, _ = family_passes_constraints(
            _fam("amide + H2O -> acid + amine", "cleavage", True), None)
        self.assertTrue(ok)

    def test_missing_substrates_free_is_not_dropped(self):
        # Absent field (older/partial responses) => unknown, keep (don't over-drop).
        ok, _ = family_passes_constraints(
            _fam("free acid + free amine + ATP -> amide", "formation", None), "formation")
        self.assertTrue(ok)


class TestTargetDirectionFromQuery(unittest.TestCase):
    def test_synthesis_query_is_formation(self):
        d = extract_target_bond_type(
            "find amide synthetases for nylon-6,6 synthesis").get("direction")
        self.assertEqual(d, "formation")


if __name__ == "__main__":
    unittest.main()
