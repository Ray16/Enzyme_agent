"""Tests for deterministic NTP-activation-chemistry derivation.

``derive_activation_mechanism`` reads only the products of an enzyme's own native
reaction equation(s) and classifies which phosphorus of the NTP the acyl group is
joined at:
  * mono-phosphate (+ di-/pyrophosphate) released -> alpha phosphorus
  * di-phosphate (+ inorganic phosphate) released -> gamma phosphorus
It is family-agnostic post-hoc candidate annotation, and returns None when the
reaction carries no interpretable NTP-condensation signature.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from enzyme_discovery_agent_improved.chemistry_logic import (
    derive_activation_mechanism,
    derive_modeling_cofactors,
)


class TestActivationDerivation(unittest.TestCase):

    def test_diphosphate_route_gamma(self):
        # products: ADP + phosphate  -> gamma phosphorus
        acts = ["putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine "
                "+ ADP + phosphate + H(+)"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["activation_mode"], "acyl_phosphate")
        self.assertEqual(r["reactive_phosphate"], "Pgamma")
        self.assertEqual(r["confidence"], "high")
        self.assertEqual(r["source"], "catalytic_activity_reaction_products")

    def test_monophosphate_route_alpha(self):
        # products: AMP + diphosphate -> alpha phosphorus
        acts = ["(S)-2,3-diaminopropanoate + fumarate + ATP = "
                "N(3)-fumaroyl-(S)-2,3-diaminopropanoate + AMP + diphosphate"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["activation_mode"], "acyl_adenylate")
        self.assertEqual(r["reactive_phosphate"], "Palpha")
        self.assertEqual(r["confidence"], "high")

    def test_pyrophosphate_synonym(self):
        acts = ["benzoate + ATP + glycine = benzoylglycine + AMP + pyrophosphate"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["reactive_phosphate"], "Palpha")

    def test_no_corroboration_is_medium(self):
        # AMP present but no explicit di-/pyrophosphate token -> medium confidence
        acts = ["acid + ATP + amine = amide + AMP"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["activation_mode"], "acyl_adenylate")
        self.assertEqual(r["confidence"], "medium")

    def test_diphosphate_does_not_satisfy_pi_corroboration(self):
        # ADP route with only "diphosphate" (no standalone phosphate) -> medium,
        # confirming the Pi test does not fire on the substring inside diphosphate.
        acts = ["x + ATP = y + ADP + diphosphate"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["reactive_phosphate"], "Pgamma")
        self.assertEqual(r["confidence"], "medium")

    def test_non_ntp_reaction_returns_none(self):
        acts = ["an amide + H2O = an acid + an amine"]  # hydrolase, no NTP
        self.assertIsNone(derive_activation_mechanism(acts))

    def test_empty_returns_none(self):
        self.assertIsNone(derive_activation_mechanism([]))
        self.assertIsNone(derive_activation_mechanism(None))

    def test_gtp_monophosphate_route(self):
        # generality across bases: GTP -> GMP is still alpha-phosphorus activation
        acts = ["acid + GTP + amine = amide + GMP + diphosphate"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["reactive_phosphate"], "Palpha")

    def test_conflicting_activities_low_confidence(self):
        acts = [
            "acid + ATP + amine = amide + AMP + diphosphate",
            "x + ATP = y + ADP + phosphate",
        ]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["confidence"], "low")
        self.assertIn("conflicting", r["basis"])

    def test_word_boundary_no_false_amp(self):
        # "example"/"clamp" must not register as AMP; no NTP anyway -> None
        self.assertIsNone(derive_activation_mechanism(["for example a clamp forms"]))

    def test_provenance_records_equation(self):
        acts = ["putrescine + L-glutamate + ATP = product + ADP + phosphate"]
        r = derive_activation_mechanism(acts)
        self.assertEqual(r["derived_from"], acts[0])
        self.assertIn("reaction products", r["basis"])


class TestModelingCofactors(unittest.TestCase):

    def test_atp_and_metal_count_from_note(self):
        # glnA2-style: reaction gives ATP; note states the Mg count.
        r = derive_modeling_cofactors(
            ["putrescine + L-glutamate + ATP = product + ADP + phosphate"],
            cofactor_names=["Mg(2+)"],
            cofactor_notes=["Binds 2 Mg(2+) ions per subunit."],
        )
        cof = {c["role"]: c for c in r["cofactors"]}
        self.assertEqual(cof["nucleotide_substrate"]["ccd"], "ATP")
        self.assertEqual(cof["nucleotide_substrate"]["count"], 1)
        self.assertEqual(cof["catalytic_metal"]["ccd"], "MG")
        self.assertEqual(cof["catalytic_metal"]["count"], 2)
        self.assertEqual(cof["catalytic_metal"]["basis"], "uniprot_cofactor_note")

    def test_metal_count_word_form(self):
        r = derive_modeling_cofactors(
            ["acid + ATP + amine = amide + AMP + diphosphate"],
            cofactor_names=["Magnesium"],
            cofactor_notes=["Requires two magnesium ions."],
        )
        metal = [c for c in r["cofactors"] if c["role"] == "catalytic_metal"][0]
        self.assertEqual(metal["count"], 2)
        self.assertEqual(metal["ccd"], "MG")

    def test_metal_alternative_recorded(self):
        # puuA-style: Mg with Mn alternative, no explicit count -> default 2.
        r = derive_modeling_cofactors(
            ["putrescine + L-glutamate + ATP = product + ADP + phosphate"],
            cofactor_names=["Mg(2+)", "Mn(2+)"],
            cofactor_notes=["Requires Mg(2+), that can be substituted by Mn(2+)."],
        )
        metal = [c for c in r["cofactors"] if c["role"] == "catalytic_metal"][0]
        self.assertEqual(metal["ccd"], "MG")
        self.assertEqual(metal["alternatives"], ["MN"])
        self.assertEqual(metal["count"], 2)
        self.assertTrue(metal["basis"].startswith("mechanistic_default"))

    def test_no_cofactor_annotation_defaults_flagged(self):
        # ddaG/iucA-style: NTP in reaction but no cofactor comment at all.
        r = derive_modeling_cofactors(
            ["acid + ATP + amine = amide + AMP + diphosphate"],
            cofactor_names=[], cofactor_notes=[],
        )
        metal = [c for c in r["cofactors"] if c["role"] == "catalytic_metal"][0]
        self.assertEqual(metal["ccd"], "MG")
        self.assertTrue(metal["identity_basis"].startswith("mechanistic_default"))
        self.assertTrue(metal["basis"].startswith("mechanistic_default"))

    def test_gtp_detected(self):
        r = derive_modeling_cofactors(["x + GTP = y + GMP + diphosphate"])
        ntp = [c for c in r["cofactors"] if c["role"] == "nucleotide_substrate"][0]
        self.assertEqual(ntp["ccd"], "GTP")

    def test_non_ntp_returns_none(self):
        self.assertIsNone(derive_modeling_cofactors(["an amide + H2O = acid + amine"]))
        self.assertIsNone(derive_modeling_cofactors([]))

    def test_diphosphate_note_not_miscounted(self):
        # "diphosphate" in a note must not be read as a metal count of any kind;
        # with no real count stated, the default applies.
        r = derive_modeling_cofactors(
            ["acid + ATP + amine = amide + AMP + diphosphate"],
            cofactor_names=["Mg(2+)"], cofactor_notes=["Releases diphosphate."],
        )
        metal = [c for c in r["cofactors"] if c["role"] == "catalytic_metal"][0]
        self.assertEqual(metal["count"], 2)
        self.assertTrue(metal["basis"].startswith("mechanistic_default"))


class TestMergeActivationEvidence(unittest.TestCase):

    def test_merge_into_none(self):
        from enzyme_discovery_agent_improved.tools.reasoning import _merge_activation_evidence
        act = {"activation_mode": "acyl_phosphate", "reactive_phosphate": "Pgamma"}
        merged = _merge_activation_evidence(None, act)
        self.assertEqual(merged["activation"], act)

    def test_merge_preserves_literature_facts(self):
        from enzyme_discovery_agent_improved.tools.reasoning import _merge_activation_evidence
        lit = {"status": "ok", "facts": {"mechanism_class": "acyl_transfer"}}
        act = {"activation_mode": "acyl_adenylate", "reactive_phosphate": "Palpha"}
        merged = _merge_activation_evidence(lit, act)
        self.assertEqual(merged["facts"], {"mechanism_class": "acyl_transfer"})
        self.assertEqual(merged["activation"], act)

    def test_no_activation_returns_existing_unchanged(self):
        from enzyme_discovery_agent_improved.tools.reasoning import _merge_activation_evidence
        lit = {"status": "ok"}
        self.assertIs(_merge_activation_evidence(lit, None), lit)


if __name__ == "__main__":
    unittest.main()
