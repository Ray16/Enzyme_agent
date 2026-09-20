"""Regression tests for the direction-extraction + topology-gating fixes.

Covers two intrinsic failures found on the nylon-6,6 hydrolase run:
  1. hydrolysis/cleavage verbs+plurals were not recognized -> target direction
     None -> the direction gate never engaged -> bond-formation ligases ranked.
  2. substrate_topology was a HARD promotion gate -> a polymer/oligomer-native
     true positive (NylC) was rejected against a small-molecule proxy target.
All offline; no network/LLM.
"""

import unittest

from enzyme_discovery_agent.chemistry_logic import extract_target_bond_type
from enzyme_discovery_agent.models.claim_validation import validate_uniprot_claims
from enzyme_discovery_agent.models.target_reasoning import derive_target_reasoning


def _entry(reaction, ec, accession="P00001"):
    return {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {
            "fullName": {"value": "Example enzyme"},
            "ecNumbers": [{"value": ec}],
        }},
        "comments": [{"commentType": "CATALYTIC ACTIVITY", "reaction": {"name": reaction}}],
        "uniProtKBCrossReferences": [{"database": "InterPro", "id": "IPR000001"}],
        "sequence": {"length": 300, "value": "A" * 300},
    }


class _StubResp:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        self.choices = [choice]


class _StubLLM:
    def __init__(self, content):
        self._content = content

    def call(self, **_):
        return _StubResp(self._content)


class TestDirectionExtraction(unittest.TestCase):
    def test_hydrolysis_query_is_cleavage(self):
        q = "find amide hydrolases that hydrolyze the amide bond in nylon-6,6"
        self.assertEqual(extract_target_bond_type(q)["direction"], "cleavage")

    def test_synthesis_query_is_formation(self):
        q = "find amide synthetases that ligates adipic acid and hexamethylenediamine"
        self.assertEqual(extract_target_bond_type(q)["direction"], "formation")

    def test_depolymerize_is_cleavage(self):
        self.assertEqual(
            extract_target_bond_type("enzymes that depolymerize the polyester")["direction"],
            "cleavage")

    def test_no_false_positive_on_lysine(self):
        # A broad "lys" stem would wrongly hit "lysine"; whole-word matching must not.
        self.assertIsNone(extract_target_bond_type("a protein with a lysine residue")["direction"])


class TestTargetReasoningLLMAuthoritative(unittest.TestCase):
    Q = "find amide hydrolases that hydrolyze the amide bond in nylon-6,6"
    TET = "NCCCCCCNC(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)O"

    def test_deterministic_fallback_when_no_llm(self):
        c = derive_target_reasoning(self.Q, self.TET, llm_client=None)
        self.assertEqual(c["reaction_direction"], "cleavage")

    def test_llm_direction_is_authoritative(self):
        # LLM says formation even for a hydrolysis-worded query -> LLM wins (it is
        # the authoritative semantic source; here we only assert it is preferred).
        llm = _StubLLM('{"reaction_direction":"formation","bond_types":["amide"],'
                       '"substrate_topology":"polymer","activation_problem":"",'
                       '"activation_categories":[],"cofactor_requirements":[],'
                       '"disqualifiers":[],"uncertainty":"low"}')
        c = derive_target_reasoning(self.Q, self.TET, llm_client=llm)
        self.assertEqual(c["reaction_direction"], "formation")
        self.assertEqual(c["substrate_topology"], "polymer")

    def test_unrecognized_llm_value_falls_back(self):
        llm = _StubLLM('{"reaction_direction":"nonsense","substrate_topology":"bogus",'
                       '"activation_problem":"","activation_categories":[],'
                       '"cofactor_requirements":[],"disqualifiers":[],"uncertainty":"low"}')
        c = derive_target_reasoning(self.Q, self.TET, llm_client=llm)
        self.assertEqual(c["reaction_direction"], "cleavage")  # deterministic fallback


class TestTopologyIsAdvisory(unittest.TestCase):
    def test_topology_mismatch_does_not_hard_block(self):
        # A polymer-native hydrolase validated against a small-molecule PROXY target:
        # topology contradicts, but must NOT block promotion (recall for the enzymes
        # we actually want). The claim is still recorded as advisory evidence.
        entry = _entry("a peptidoglycan polymer + H2O = shorter polymer + amino acid",
                       ec="3.5.1.28")
        v = validate_uniprot_claims(
            entry, expected_accession="P00001",
            target_topology="free_small_molecule",
        )
        self.assertNotIn("substrate_topology", v["required_claims"])
        self.assertNotIn("substrate_topology", v["blocking_claims"])
        claim_types = {c["claim_type"] for c in v["claims"]}
        self.assertIn("substrate_topology", claim_types)  # still recorded


if __name__ == "__main__":
    unittest.main()
