"""Tests for polymer-aware oligomer derivation in target reasoning.

When the target reaction builds or breaks a repeating chain (even from monomeric
inputs), the agent should reason out a representative oligomer (>= a tetramer of
residues) for structural modeling, and that proposal should be RDKit-validated so
a monomer/dimer or an invalid string is rejected. All offline (LLM stubbed).
"""

import json
import unittest

from enzyme_discovery_agent.models.target_reasoning import (
    derive_target_reasoning,
    validate_polymer_oligomer,
)

# Generic repeating chains (not the benchmark target): a tetra-/di-glycine peptide.
TETRA_PEPTIDE = "NCC(=O)NCC(=O)NCC(=O)NCC(=O)O"   # 3 peptide (amide) bonds
DI_PEPTIDE = "NCC(=O)NCC(=O)O"                     # 1 peptide (amide) bond


class _StubResp:
    def __init__(self, content):
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        self.choices = [choice]


class _StubLLM:
    def __init__(self, payload):
        self._content = json.dumps(payload)

    def call(self, **_):
        return _StubResp(self._content)


class TestValidateOligomer(unittest.TestCase):
    def test_accepts_tetramer(self):
        out = validate_polymer_oligomer(TETRA_PEPTIDE, ["amide"], "NCC(=O)O")
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out["n_backbone_bonds"], 3)

    def test_rejects_dimer(self):
        # A single amide bond is not a >= tetramer chain.
        self.assertIsNone(validate_polymer_oligomer(DI_PEPTIDE, ["amide"], "NCC(=O)O"))

    def test_rejects_invalid_smiles(self):
        self.assertIsNone(validate_polymer_oligomer("not_a_smiles", ["amide"], None))

    def test_rejects_empty(self):
        self.assertIsNone(validate_polymer_oligomer(None, ["amide"], None))

    def test_unrecognized_bond_uses_size_floor(self):
        # Bond type not SMARTS-mappable -> must be clearly larger than the monomer.
        big = TETRA_PEPTIDE
        self.assertIsNotNone(validate_polymer_oligomer(big, ["glycosidic"], "NCC(=O)O"))
        self.assertIsNone(validate_polymer_oligomer("NCC(=O)O", ["glycosidic"], "NCC(=O)O"))


class TestDeriveWithPolymerReasoning(unittest.TestCase):
    Q = "find ligases that build a repeating polyamide from a diacid and a diamine"

    def _payload(self, **over):
        base = {
            "reaction_direction": "formation",
            "bond_types": ["amide"],
            "substrate_topology": "free_small_molecule",
            "activation_problem": "condensation is unfavorable in water",
            "activation_categories": ["carboxylate activation"],
            "cofactor_requirements": [],
            "disqualifiers": [],
            "polymer_target": True,
            "repeat_unit": "diacid-diamine amide",
            "oligomer_smiles": TETRA_PEPTIDE,
            "uncertainty": "medium",
        }
        base.update(over)
        return base

    def test_polymer_target_yields_validated_oligomer(self):
        c = derive_target_reasoning(self.Q, "NCC(=O)O", llm_client=_StubLLM(self._payload()))
        self.assertTrue(c["polymer_target"])
        self.assertIsNotNone(c["modeling_oligomer"])
        self.assertGreaterEqual(c["modeling_oligomer"]["n_backbone_bonds"], 3)
        self.assertEqual(c["modeling_oligomer"]["repeat_unit"], "diacid-diamine amide")
        self.assertTrue(any("oligomer" in cl["claim"] for cl in c["claims"]))

    def test_non_polymer_target_has_no_oligomer(self):
        c = derive_target_reasoning(
            self.Q, "NCC(=O)O",
            llm_client=_StubLLM(self._payload(polymer_target=False, oligomer_smiles=None)))
        self.assertFalse(c["polymer_target"])
        self.assertIsNone(c["modeling_oligomer"])

    def test_monomer_proposal_is_rejected(self):
        # Agent claims a polymer target but proposes only a single amide (dimer):
        # validation rejects it rather than passing a non-oligomer downstream.
        c = derive_target_reasoning(
            self.Q, "NCC(=O)O",
            llm_client=_StubLLM(self._payload(oligomer_smiles=DI_PEPTIDE)))
        self.assertTrue(c["polymer_target"])
        self.assertIsNone(c["modeling_oligomer"])

    def test_contract_normalizes_without_llm(self):
        c = derive_target_reasoning(self.Q, "NCC(=O)O", llm_client=None)
        self.assertFalse(c["polymer_target"])
        self.assertIsNone(c["modeling_oligomer"])


if __name__ == "__main__":
    unittest.main()
