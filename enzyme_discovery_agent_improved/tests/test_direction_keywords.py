"""Reaction-direction derivation must recognise verb/plural surface forms.

Regression: a hydrolysis query phrased with the verb ("hydrolyzes") or the
plural enzyme class ("hydrolases") derived direction=None (the keyword map only
had "hydrolysis"/"hydrolase" and matching is whole-word), which silently
mis-directs a cleavage campaign as formation.
"""
import unittest

from enzyme_discovery_agent_improved.chemistry_logic import extract_target_bond_type


class TestDirectionKeywords(unittest.TestCase):
    def test_hydrolysis_verb_and_plural(self):
        for q in (
            "find amide hydrolases that hydrolyzes the amide bond in nylon-6,6",
            "an enzyme that hydrolyses peptide bonds",
            "amide hydrolase for nylon degradation",
            "enzymes that cleave amide bonds",
            "depolymerize the polyamide",
        ):
            self.assertEqual(extract_target_bond_type(q)["direction"], "cleavage", q)

    def test_formation_verbs_unchanged(self):
        for q in (
            "find amide synthetases that ligate adipic acid and HMDA for nylon-6,6 synthesis",
            "an enzyme that synthesizes the amide bond",
            "a ligase that condenses two substrates",
        ):
            self.assertEqual(extract_target_bond_type(q)["direction"], "formation", q)

    def test_cleavage_wins_when_both_present(self):
        # "hydrolysis (NOT synthesis)" style — cleavage terms are checked first.
        self.assertEqual(
            extract_target_bond_type("amide hydrolysis, not synthesis")["direction"],
            "cleavage")


if __name__ == "__main__":
    unittest.main()
