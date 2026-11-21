"""Nucleophile-class gate.

An enzyme whose RECORDED reaction/EC uses a nucleophile class (glutamine amido donor,
free CoA/ACP thiol) that the run's own target partners do not carry cannot form the
target bond and must leave the final ranking (kept in excluded_candidates.csv).  Free
ammonia is only flagged and demoted: that annotation is often transferred by
similarity.  The gate is keyed on recorded reactions/EC and on the target's own partner
structures, never on family names or a hard-coded target; no recorded reaction, or an
unresolved target, means no action.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from enzyme_discovery_agent.chemistry.reactions import (
    classify_nucleophile_partner,
    evaluate_partner_gate,
    partner_gate_reason,
    target_nucleophile_classes,
)

# This run's target partners (an acid and an amine); other targets are exercised below.
TARGET = ["O=C(O)CCCC(=O)O", "NCCCCCN"]

ASN = "L-aspartate + L-glutamine + ATP + H2O = L-asparagine + L-glutamate + AMP + diphosphate + H(+)"
GS = "L-glutamate + NH4(+) + ATP = L-glutamine + ADP + phosphate + H(+)"

# Recorded reactions copied from UniProt-derived rows of run 20261106_000850.
COMPATIBLE = {
    "GlnA3 Q9KZC7": (["spermine + L-glutamate + ATP = gamma-L-glutamylspermine + ADP + phosphate + H(+)",
                      "spermidine + L-glutamate + ATP = gamma-L-glutamylspermidine + ADP + phosphate + H(+)",
                      "putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+)",
                      "cadaverine + L-glutamate + ATP = gamma-L-glutamylcadaverine + ADP + phosphate + H(+)"],
                     ["6.3.1.-"]),
    "PuuA P78061": (["putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+)"],
                    ["6.3.1.11"]),
    "BacD P39641": (["L-anticapsin + L-alanine + ATP = bacilysin + ADP + phosphate + H(+)"], ["6.3.2.49"]),
    "PubC P0DXD6": (["2 N-(3-carboxypropanoyl)-N-hydroxyputrescine + 2 ATP = putrebactin + 2 AMP + 2 diphosphate + 2 H(+)",
                     "2 N-(3-carboxypropanoyl)-N-hydroxyputrescine + ATP = pre-putrebactin + AMP + diphosphate"],
                    ["6.3.2.63"]),
    "Gss P0AES1": (["spermidine + glutathione + ATP = glutathionylspermidine + ADP + phosphate + H(+)",
                    "glutathionylspermidine + H2O = spermidine + glutathione"], []),
    "iucA Q76BS7": (["N(6)-acetyl-N(6)-hydroxy-L-lysine + citrate + ATP = "
                     "N(2)-citryl-N(6)-acetyl-N(6)-hydroxy-L-lysine + AMP + diphosphate + H(+)"], ["6.3.2.38"]),
    "AsbB Q81RQ8": (["N(8)-citryl-spermidine + spermidine + ATP = N(8),N'(8)-citryl-bis(spermidine) + AMP + diphosphate + H(+)"],
                    ["6.3.2.-"]),
    "GCL2 A0A1H4U7K7": (["L-cysteine + L-glutamate + ATP = gamma-L-glutamyl-L-cysteine + ADP + phosphate + H(+)"],
                        ["6.3.2.2"]),
    "StaphyloferrinA Q2FW72": (["N(5)-[(S)-citryl]-D-ornithine + citrate + ATP = staphyloferrin A + AMP + diphosphate + H(+)"],
                               ["6.3.2.57"]),
    "StaphyloferrinB Q2G1N1": (["2-[(L-alanin-3-ylcarbamoyl)methyl]-3-(2-aminoethylcarbamoyl)-2-hydroxypropanoate + "
                                "2-oxoglutarate + ATP = staphyloferrin B + AMP + diphosphate + H(+)"], ["6.3.2.56"]),
    "Trypanothione O60993": (["spermidine + glutathione + ATP = glutathionylspermidine + ADP + phosphate + H(+)"],
                             ["6.3.1.9"]),
    "acyl-CoA:amine N-acyltransferase": (["acyl-CoA + cadaverine = N-acylcadaverine + CoA + H(+)"], ["2.3.1.-"]),
}


class TestPartnerClassification(unittest.TestCase):
    def test_known_compatible_enzymes_are_never_flagged(self):
        for name, (reactions, ec) in COMPATIBLE.items():
            with self.subTest(name):
                gate = evaluate_partner_gate(reactions, ec, TARGET)
                self.assertEqual(gate["action"], "none", f"{name} wrongly gated: {gate}")
                self.assertEqual(gate["partner_class"], "no_conflict_recorded")

    def test_no_recorded_reaction_is_unknown_and_unpenalised(self):
        # DesD entries (and YjfC) carry no recorded reaction in UniProt.
        for ec in ([], ["6.3.1.-"], ["6.-.-.-"]):
            gate = evaluate_partner_gate([], ec, TARGET)
            self.assertEqual((gate["partner_class"], gate["action"]), ("unknown", "none"))

    def test_asparagine_synthetases_are_excluded_by_recorded_reaction(self):
        for ec in (["6.3.5.4"], []):
            gate = evaluate_partner_gate([ASN], ec, TARGET)
            self.assertEqual(gate["partner_class"], "amido_donor_glutamine")
            self.assertEqual(gate["action"], "exclude")
            self.assertEqual(gate["basis"], "recorded_reaction")
        self.assertIn("ammonia", partner_gate_reason(gate))

    def test_amido_donor_ec_alone_is_enough_when_no_reaction_is_recorded(self):
        gate = evaluate_partner_gate([], ["6.3.5.4"], TARGET)
        self.assertEqual((gate["partner_class"], gate["basis"], gate["action"]),
                         ("amido_donor_glutamine", "EC", "exclude"))

    def test_free_coa_ligase_is_excluded(self):
        gate = evaluate_partner_gate(
            ["glutarate + ATP + CoA = glutaryl-CoA + AMP + diphosphate"], ["6.2.1.-"], TARGET)
        self.assertEqual((gate["partner_class"], gate["action"]), ("thiol_or_CoA", "exclude"))
        self.assertEqual(evaluate_partner_gate([], ["6.2.1.1"], TARGET)["action"], "exclude")

    def test_canonical_glutamine_synthetase_is_reviewed_and_demoted_not_excluded(self):
        gate = evaluate_partner_gate([GS], ["6.3.1.2"], TARGET)
        self.assertEqual((gate["partner_class"], gate["action"]), ("ammonia", "review_demote"))
        self.assertEqual(evaluate_partner_gate([], ["6.3.1.2"], TARGET)["action"], "review_demote")

    def test_recorded_alkylamine_reaction_overrides_a_second_ammonia_reaction(self):
        # A promiscuous entry with BOTH a polyamine and an ammonia reaction is compatible.
        gate = evaluate_partner_gate([GS, COMPATIBLE["PuuA P78061"][0][0]], ["6.3.1.2"], TARGET)
        self.assertEqual((gate["partner_class"], gate["action"]), ("no_conflict_recorded", "none"))

    def test_gate_does_not_key_on_family_names(self):
        # Same fold label, different recorded chemistry -> different outcome.
        self.assertEqual(evaluate_partner_gate(COMPATIBLE["GlnA3 Q9KZC7"][0], ["6.3.1.-"], TARGET)["action"], "none")
        self.assertEqual(evaluate_partner_gate([GS], ["6.3.1.2"], TARGET)["action"], "review_demote")

    def test_reactant_side_only(self):
        # Ammonia RELEASED by a hydrolysis must not be read as the nucleophile.
        gate = classify_nucleophile_partner(["L-asparagine + H2O = L-aspartate + NH4(+)"], [])
        self.assertNotEqual(gate["partner_class"], "ammonia")


class TestTargetDrivenNotHardCoded(unittest.TestCase):
    """The same recorded reaction is a conflict or not depending on the run's target."""

    def test_target_nucleophile_classes_are_read_from_structures(self):
        self.assertEqual(target_nucleophile_classes(TARGET), {"alkylamine"})
        self.assertEqual(target_nucleophile_classes(["N"]), {"ammonia"})            # ammonia itself
        self.assertEqual(target_nucleophile_classes(["CCS"]), {"thiol"})            # a free thiol
        self.assertEqual(target_nucleophile_classes(["CC(=O)NC"]), set())          # acylated N is not a nucleophile
        self.assertEqual(target_nucleophile_classes(["not a smiles", None, ""]), set())

    def test_thiol_ligase_is_right_for_a_thiol_target_and_wrong_for_an_amine_target(self):
        coa = ["glutarate + ATP + CoA = glutaryl-CoA + AMP + diphosphate"]
        self.assertEqual(evaluate_partner_gate(coa, ["6.2.1.-"], TARGET)["action"], "exclude")
        self.assertEqual(evaluate_partner_gate(coa, ["6.2.1.-"], ["O=C(O)CCCC(=O)O", "CCS"])["action"], "none")

    def test_glutamine_amido_donor_is_right_for_an_ammonia_target(self):
        self.assertEqual(evaluate_partner_gate([ASN], ["6.3.5.4"], TARGET)["action"], "exclude")
        self.assertEqual(evaluate_partner_gate([ASN], ["6.3.5.4"], ["N", "OC(=O)CC(N)C(=O)O"])["action"], "none")

    def test_unresolved_target_means_no_action(self):
        for target in ([], None, ["not a smiles"]):
            gate = evaluate_partner_gate([ASN], ["6.3.5.4"], target)
            self.assertEqual(gate["action"], "none")
            self.assertEqual(gate["partner_class"], "amido_donor_glutamine")  # still reported
            self.assertIn("unresolved", gate["note"])


def _evaluation(acc, name, reactions, ec):
    return {"status": "success", "data": {
        "accession": acc, "protein_name": name, "gene_names": [name.split()[0].lower()],
        "organism": "Testus exemplaris", "ec_numbers": list(ec), "interpro_family": "Some family",
        "catalytic_activities": list(reactions),
        "substrate_similarity": {"property_comparisons": [{"target_smiles": smi} for smi in TARGET]}}}


class TestRankingIntegration(unittest.TestCase):
    def test_incompatible_is_excluded_and_recorded_ammonia_is_demoted(self):
        from enzyme_discovery_agent.output.report_generator import _deterministic_mechanistic_ranking
        evals = [
            _evaluation("P78061", "PuuA synthetase", *COMPATIBLE["PuuA P78061"]),
            _evaluation("A0A1I2XEU9", "Asparagine synthase", [ASN], ["6.3.5.4"]),
            _evaluation("P9WN36", "Glutamine synthetase", [GS], ["6.3.1.2"]),
        ]
        excluded = []
        ranking = _deterministic_mechanistic_ranking(evals, {1: 8.0, 2: 5.0, 3: 2.0}, excluded_out=excluded)
        accs = [r["accession"] for r in ranking]
        self.assertNotIn("A0A1I2XEU9", accs)
        self.assertEqual([e["accession"] for e in excluded], ["A0A1I2XEU9"])
        self.assertEqual(excluded[0]["partner_class"], "amido_donor_glutamine")
        by_acc = {r["accession"]: r for r in ranking}
        self.assertEqual(by_acc["P9WN36"]["tier"], 3)
        self.assertIn("Nucleophile review", by_acc["P9WN36"]["rationale"])
        self.assertEqual(by_acc["P9WN36"]["partner_class"], "ammonia")
        self.assertEqual(by_acc["P78061"]["partner_class"], "no_conflict_recorded")
        self.assertLess(by_acc["P78061"]["tier"], 3)
        self.assertLess(accs.index("P78061"), accs.index("P9WN36"))

    def test_csv_writes_partner_columns_and_a_separate_exclusion_file(self):
        from enzyme_discovery_agent.output.csv_generator import generate_candidates_csv
        evals = [_evaluation("P78061", "PuuA synthetase", *COMPATIBLE["PuuA P78061"]),
                 _evaluation("A0A1I2XEU9", "Asparagine synthase", [ASN], ["6.3.5.4"])]
        ranking = [{"rank": 1, "accession": "P78061", "tier": 2, "tier_label": "candidate",
                    "substrate_fit": 0.4, "evidence_level": "low", "family_relevance": "on-target",
                    "direction": "formation"},
                   # a stale persisted ranking that still lists the asparagine synthetase
                   {"rank": 2, "accession": "A0A1I2XEU9", "tier": 2, "tier_label": "candidate",
                    "substrate_fit": 0.3, "evidence_level": "none", "family_relevance": "on-target",
                    "direction": "formation"}]
        guard = patch("enzyme_discovery_agent.models.target_review.admitted_evaluations",
                      side_effect=lambda root, rows: rows)
        guard.start()
        self.addCleanup(guard.stop)
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "pass1_ranking.json").write_text(json.dumps(ranking))
            generate_candidates_csv(run_dir, evals)
            rows = list(csv.DictReader(open(run_dir / "ranked_candidates.csv")))
            self.assertEqual([r["accession"] for r in rows], ["P78061"])
            self.assertEqual(rows[0]["partner_class"], "no_conflict_recorded")
            self.assertEqual(rows[0]["partner_class_basis"], "recorded_reaction")
            excluded = list(csv.DictReader(open(run_dir / "excluded_candidates.csv")))
            self.assertEqual([r["accession"] for r in excluded], ["A0A1I2XEU9"])
            self.assertIn("ammonia", excluded[0]["reason"])


class TestMechanismFamilyGrounding(unittest.TestCase):
    FAMILIES = [
        {"name": "Glutamine synthetase-fold gamma-glutamyl-polyamine ligase (GlnA-like)",
         "descriptors": ["glutamine synthetase catalytic domain", "GlnA-like",
                         "gamma-glutamylpolyamine synthetase", "glutamate-ammonia ligase fold"]},
        {"name": "Asparagine synthetase B-like amidotransferase",
         "descriptors": ["asparagine synthetase B", "class II glutamine amidotransferase",
                         "N-type ATP pyrophosphatase"]},
    ]

    def test_substrate_vocabulary_cannot_pull_asparagine_synthetase_into_the_glna_family(self):
        from enzyme_discovery_agent.tools.mechanism_family_assignment import assign_mechanism_family
        meta = {"interpro_family": "Asparagine synthase, glutamine-hydrolyzing",
                "protein_name": "asparagine synthase (glutamine-hydrolyzing)", "ec_numbers": ["6.3.5.4"],
                "gene_names": [], "catalytic_activities": [ASN]}
        self.assertTrue(assign_mechanism_family(meta, self.FAMILIES)["name"].startswith("Asparagine synthetase"))

    def test_a_real_glna_member_keeps_its_label(self):
        from enzyme_discovery_agent.tools.mechanism_family_assignment import assign_mechanism_family
        meta = {"interpro_family": "Glutamine synthetase, catalytic domain",
                "protein_name": "Gamma-glutamylpolyamine synthetase GlnA3", "ec_numbers": ["6.3.1.-"],
                "gene_names": ["glnA3"], "catalytic_activities": COMPATIBLE["GlnA3 Q9KZC7"][0]}
        self.assertTrue(assign_mechanism_family(meta, self.FAMILIES)["name"].startswith("Glutamine synthetase-fold"))


if __name__ == "__main__":
    unittest.main()
