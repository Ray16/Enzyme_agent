"""Tests for enzymology-profile paper ranking (full-text harvest selection).

The harvest can deep-read only a few papers; these tests pin the behaviour that
it ranks the most search-guiding papers first, answer-agnostically and
direction-aware, and degrades safely when the contract is absent.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from enzyme_discovery_agent.tools.paper_relevance import (
    assess_primary_evidence,
    build_enzymology_profile,
    looks_like_review,
    mine_deposited_accessions,
    rank_papers_by_enzymology,
    score_paper,
)

# A formation/amide contract like the nylon target's target_reasoning.json.
_FORMATION_CONTRACT = {
    "reaction_direction": "formation",
    "bond_types": ["amide"],
    "substrate_topology": "free_small_molecule",
    "reactive_groups": ["carbonyl", "carboxyl", "primary_amine"],
    "activation_categories": [
        "acyl-adenylate (acyl-AMP) intermediate", "acyl-phosphate intermediate",
    ],
    "cofactor_requirements": ["ATP or other nucleoside triphosphate"],
}
_MECH_FAMILIES = [
    {"name": "ANL superfamily adenylate-forming amido synthetase"},
    {"name": "ATP-grasp fold amide/peptide ligase"},
]

_ON_POINT = {"title": "The Amide Bond Synthetase McbA: an ATP-dependent adenylate-forming ligase",
             "abstract": "An enzyme that catalyzes amide bond formation from a carboxylic acid."}
_MATERIALS = {"title": "Novel poly(amide-imide)s from dicarboxylic acids for dielectric membranes",
              "abstract": "Thermal properties of copolyamide films."}
_HYDROLASE = {"title": "A nylon hydrolase for enzymatic amide bond hydrolysis and depolymerization",
              "abstract": "Enzymatic degradation of polyamide by hydrolysis."}
_OFFTOPIC = {"title": "Genome sequence of a hypervirulent Klebsiella pneumoniae isolate",
             "abstract": "Clinical isolate genome."}
_INHIBITOR = {
    "title": "Targeting adenylate-forming enzymes with designed inhibitors",
    "abstract": "Potent inhibition of ATP-dependent enzymes.",
}
_TETHERED = {
    "title": "Thioester-mediated RNA aminoacylation and peptidyl-RNA synthesis",
    "abstract": "A ribosomal precursor is aminoacylated.",
}
_COA = {
    "title": "A pimeloyl-CoA synthetase mechanism",
    "abstract": "The enzyme forms a CoA thioester through an acyl-AMP.",
}


class TestEnzymologyProfileRanking(unittest.TestCase):
    def setUp(self):
        self.profile = build_enzymology_profile(_FORMATION_CONTRACT, _MECH_FAMILIES)

    def test_profile_is_direction_aware(self):
        # Formation target -> hydrolysis terms are disqualifiers, not positives.
        self.assertIn("hydrolase", self.profile["negative_dir"])
        self.assertNotIn("hydrolase", self.profile["weak"])
        self.assertTrue(any("synthetase" in s for s in self.profile["strong"]))

    def test_cleavage_target_inverts_direction(self):
        prof = build_enzymology_profile(
            {"reaction_direction": "cleavage", "bond_types": ["ester"]}, None)
        self.assertIn("synthetase", prof["negative_dir"])
        self.assertTrue(any("hydrolase" in s for s in prof["strong"]))

    def test_on_point_scores_above_materials_and_offtopic(self):
        s_on = score_paper(_ON_POINT, self.profile)
        s_mat = score_paper(_MATERIALS, self.profile)
        s_hyd = score_paper(_HYDROLASE, self.profile)
        self.assertGreater(s_on, s_mat)
        self.assertGreater(s_on, s_hyd)  # wrong direction penalized

    def test_non_catalysis_and_tethered_contexts_are_penalized(self):
        self.assertGreater(
            score_paper(_ON_POINT, self.profile),
            score_paper(_INHIBITOR, self.profile),
        )
        self.assertGreater(
            score_paper(_ON_POINT, self.profile),
            score_paper(_TETHERED, self.profile),
        )
        self.assertGreater(
            score_paper(_ON_POINT, self.profile),
            score_paper(_COA, self.profile),
        )
        self.assertEqual(score_paper(_COA, self.profile), float("-inf"))

    def test_primary_evidence_requires_original_enzymology_content(self):
        mechanistic = {
            "title": "Kinetic mechanism of an ATP-dependent amide ligase",
            "abstract": (
                "We purified the enzyme, measured kcat, determined substrate "
                "specificity, and identified an acyl-phosphate intermediate."
            ),
        }
        review = {
            "title": "ATP-dependent amide ligases: a review",
            "abstract": "An overview of enzyme mechanisms.",
        }
        inhibitor = {
            "title": "Targeting adenylate-forming enzymes with inhibitors",
            "abstract": "Designed inhibitors bind the enzyme active site.",
        }
        assessed = assess_primary_evidence(mechanistic)
        self.assertTrue(assessed["eligible"])
        self.assertIn("mechanism", assessed["evidence_types"])
        self.assertIn("kinetics", assessed["evidence_types"])
        self.assertFalse(assess_primary_evidence(review)["eligible"])
        self.assertFalse(assess_primary_evidence(inhibitor)["eligible"])

    def test_offtopic_without_enzyme_signal_is_gated_out(self):
        self.assertEqual(score_paper(_OFFTOPIC, self.profile), float("-inf"))

    def test_ranking_puts_on_point_first(self):
        pool = [_OFFTOPIC, _MATERIALS, _HYDROLASE, _ON_POINT]

        class _RunDir:
            # rank_papers_by_enzymology reads two JSON files from run_dir; simulate
            # a run dir with the contract present via a tmp path is covered below.
            pass

        # No contract on disk -> must fall back to the provided order untouched.
        fallback_order = list(pool)
        got = rank_papers_by_enzymology("/nonexistent", pool, fallback=lambda: fallback_order)
        self.assertEqual(got, fallback_order)

    def test_ranking_with_contract_on_disk(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            (rd / "target_reasoning.json").write_text(json.dumps(_FORMATION_CONTRACT))
            (rd / "mechanism_families.json").write_text(json.dumps(_MECH_FAMILIES))
            pool = [_OFFTOPIC, _MATERIALS, _HYDROLASE, _ON_POINT]
            ranked = rank_papers_by_enzymology(str(rd), pool, fallback=lambda: pool)
            self.assertEqual(ranked[0], _ON_POINT)
            # Both no-enzyme-signal papers sink to the bottom, but are retained.
            self.assertEqual({id(p) for p in ranked[-2:]},
                             {id(_OFFTOPIC), id(_MATERIALS)})
            # Wrong-direction hydrolase has enzyme signal but ranks below on-point.
            self.assertLess(ranked.index(_HYDROLASE), ranked.index(_MATERIALS))
            self.assertGreater(ranked.index(_HYDROLASE), ranked.index(_ON_POINT))
            self.assertEqual(len(ranked), len(pool))

    def test_empty_pool(self):
        self.assertEqual(rank_papers_by_enzymology("/x", [], fallback=lambda: []), [])


class TestDepositedAccessionMining(unittest.TestCase):
    def test_extracts_refseq_and_uniprot(self):
        text = ("The gene was deposited as WP_073481158.1 and its ortholog "
                "WP_051713392. The UniProt entry A0A1M5ABR5 was annotated. "
                "See also GenBank AAB12345.")
        accs = mine_deposited_accessions(text)
        self.assertIn("WP_073481158.1", accs)
        self.assertIn("WP_051713392", accs)
        self.assertIn("A0A1M5ABR5", accs)
        self.assertIn("AAB12345", accs)

    def test_pmc_and_common_abbrevs_are_not_mistaken_for_accessions(self):
        text = "Available in PMC89847; funded by NIH12345. ATP was the cofactor."
        accs = mine_deposited_accessions(text)
        self.assertFalse(any(a.startswith(("PMC", "NIH", "ATP")) for a in accs))

    def test_mining_is_deduped_and_bounded(self):
        text = " ".join(["WP_000000001"] * 50) + " WP_000000002"
        accs = mine_deposited_accessions(text, cap=5)
        self.assertEqual(accs.count("WP_000000001"), 1)
        self.assertLessEqual(len(accs), 5)

    def test_empty_text(self):
        self.assertEqual(mine_deposited_accessions(""), [])

    def test_review_vs_primary_detection(self):
        self.assertTrue(looks_like_review({"title": "Amide bond formation: a review"}))
        self.assertTrue(looks_like_review(
            {"title": "X", "publication_types": ["Review"]}))
        self.assertFalse(looks_like_review(
            {"title": "Broad-spectrum amide bond synthetase from a bacterium"}))

    def test_mining_uses_no_answer_knowledge(self):
        # Leakage-safety intent: the miner is a pure regex over the given text and
        # does not read benchmarks / filter by any answer. A string containing no
        # accessions yields nothing, regardless of enzyme names present.
        self.assertEqual(mine_deposited_accessions("SfaB DdaG McbA amide ligase"), [])


if __name__ == "__main__":
    unittest.main()
