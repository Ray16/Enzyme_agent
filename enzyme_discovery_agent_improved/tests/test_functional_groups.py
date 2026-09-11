"""Wide coverage tests for functional-group detection.

The substrate-fit matcher is only as good as its functional-group recognition:
it must (1) name as many groups as possible across chemistries, (2) NOT confuse
a functional-group-less donor (ammonia/water) with a real amine, and (3) detect
the ACTIVATED acyl derivatives (thioester, anhydride, acyl-phosphate) and the
hydroxamate/omega chemistry that enzyme substrates actually use.
"""

from __future__ import annotations

import unittest

from rdkit import Chem

from enzyme_discovery_agent_improved.tools.substrate import (
    _detect_functional_groups,
    compute_substrate_properties,
)


def _groups(smiles: str) -> set[str]:
    return set(_detect_functional_groups(Chem.MolFromSmiles(smiles))["groups"])


def _counts(smiles: str) -> dict:
    return _detect_functional_groups(Chem.MolFromSmiles(smiles))["counts"]


class TestAmineDiscrimination(unittest.TestCase):
    """A functional-group-less nitrogen donor is not a primary amine — the
    distinction the one-to-one accommodation guard relies on."""

    def test_ammonia_and_water_carry_no_reactive_counts(self):
        # Ammonia / ammonium / water contribute NOTHING to the count vector that
        # drives similarity, so they can never flatter a target amine/acid role.
        self.assertEqual(_counts("N"), {})          # NH3
        self.assertEqual(_counts("[NH4+]"), {})      # NH4+
        self.assertEqual(_counts("O"), {})           # water

    def test_substituted_amines_are_graded_by_substitution(self):
        self.assertEqual(_counts("CN").get("primary_amine"), 1)
        self.assertEqual(_counts("CNC").get("secondary_amine"), 1)
        self.assertEqual(_counts("CN(C)C").get("tertiary_amine"), 1)
        self.assertNotIn("primary_amine", _counts("CN(C)C"))


class TestTargetAndOmegaSubstrates(unittest.TestCase):
    def test_diacid_and_diamine_counts(self):
        self.assertEqual(_counts("O=C(O)CCCCC(=O)O").get("carboxyl"), 2)
        self.assertEqual(_counts("NCCCCCCN").get("primary_amine"), 2)

    def test_omega_amino_acid_presents_both_reactive_roles(self):
        # A single omega-amino-acid molecule carries BOTH a carboxyl and an amine
        # -- the bifunctional/self-condensation case that must not be scored as
        # missing a reactive role.
        c = _counts("NCCCCCC(=O)O")
        self.assertEqual(c.get("carboxyl"), 1)
        self.assertEqual(c.get("primary_amine"), 1)


class TestActivatedAcylAndEnzymeGroups(unittest.TestCase):
    """The activated acyl intermediates that drive ligase/synthetase chemistry,
    which the original hand list missed and RDKit Fragments also lacks."""

    def test_thioester(self):
        self.assertIn("thioester", _groups("CC(=O)SCCN"))
        self.assertEqual(_counts("CC(=O)SCCN").get("thioester"), 1)

    def test_acyl_phosphate(self):
        self.assertIn("acyl_phosphate", _groups("CC(=O)OP(=O)(O)O"))

    def test_anhydride_not_double_counted_as_ester(self):
        c = _counts("CC(=O)OC(C)=O")
        self.assertEqual(c.get("anhydride"), 1)
        self.assertNotIn("ester", c)  # ester SMARTS excludes anhydride

    def test_hydroxamate(self):
        self.assertIn("hydroxamate", _groups("CC(=O)NO"))


class TestBreadthAcrossChemistries(unittest.TestCase):
    """Recognition must span many chemistries, not only amide-relevant ones."""

    def test_diverse_groups_are_named(self):
        expect = {
            "Clc1ccccc1": "organohalogen",
            "Oc1ccccc1": "phenol",
            "SCC(N)C(=O)O": "thiol",
            "CS(=O)(=O)N": "sulfonamide",
            "OP(=O)(O)OCC": "phosphoester",
            "[C-]#[N+]C": "isonitrile",
            "NC(=O)N": "urea",
            "C=CC(=O)O": "alkene",
        }
        for smiles, group in expect.items():
            self.assertIn(group, _groups(smiles), f"{group} missing for {smiles}")

    def test_recognition_is_broader_than_similarity_core(self):
        # The Fragments-augmented presence list names more than the clean
        # similarity-count core (e.g. it adds heterocycle/decoration labels),
        # while counts stay restricted to the canonical core.
        props = compute_substrate_properties(
            "OCC1OC(n2cnc3c2ncnc3N)C(O)C1OP(=O)(O)O")  # AMP-like
        self.assertTrue(set(props.functional_groups) >= set(props.functional_group_counts))
        self.assertIn("phosphoester", props.functional_group_counts)


if __name__ == "__main__":
    unittest.main()
