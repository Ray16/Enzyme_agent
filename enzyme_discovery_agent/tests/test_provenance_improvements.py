"""Tests for the three provenance improvements:

1. the chemistry contract is referenced in the run manifest;
2. a dirty run captures an exact, reconstructible package patch;
3. per-candidate provenance records reasoning alignment vs the target contract.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent import config, provenance
from enzyme_discovery_agent.models.candidate_provenance import (
    _reasoning_alignment,
    build_candidate_provenance,
)


def _write_contract(run_dir: Path) -> None:
    (run_dir / "target_reasoning.json").write_text(json.dumps({
        "schema_version": "1.0",
        "reaction_direction": "formation",
        "bond_types": ["amide"],
        "substrate_topology": "free_small_molecule",
        "reactive_groups": ["carboxyl", "primary_amine"],
        "activation_categories": ["acyl-adenylate intermediate"],
        "claims": [
            {"claim": "a", "basis": "b", "implication": "c",
             "status": "derived_from_target", "uncertainty": "low"},
            {"claim": "d", "basis": "e", "implication": "f",
             "status": "inferred", "uncertainty": "medium"},
        ],
    }))


class TestContractInManifest(unittest.TestCase):
    def test_manifest_references_the_contract_with_a_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_contract(run_dir)
            manifest = provenance.write_manifest_start(
                run_dir, query="q", model="m", config_module=config)
            tr = manifest["target_reasoning"]
            self.assertEqual(tr["reaction_direction"], "formation")
            self.assertEqual(tr["bond_types"], ["amide"])
            self.assertEqual(tr["n_claims"], 2)
            self.assertEqual(tr["n_inferred"], 1)
            self.assertEqual(len(tr["sha256"]), 64)

    def test_legacy_run_without_contract_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = provenance.write_manifest_start(
                Path(tmp), query="q", model="m", config_module=config)
            self.assertIsNone(manifest["target_reasoning"])


class TestPackageDiffCapture(unittest.TestCase):
    def test_code_provenance_carries_package_dirty_flag(self):
        # In this repo the tree may be clean or dirty; either way the new fields
        # must be present and internally consistent, and a patch file is written
        # only when the package actually has tracked changes.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            code = provenance.capture_code_provenance(run_dir)
            if code.get("vcs") != "git":
                self.skipTest("not a git repo")
            self.assertIn("package_dirty", code)
            if code["package_dirty"] and code.get("package_diff_bytes"):
                self.assertEqual(code.get("patch_file"), "code_changes.patch")
                patch = (run_dir / "code_changes.patch").read_text()
                self.assertEqual(len(patch.encode()), code["package_diff_bytes"])
                self.assertEqual(len(code["package_diff_sha256"]), 64)


class TestReasoningAlignment(unittest.TestCase):
    def _cv(self, claims, eligible):
        return {"promotion_eligible": eligible, "claims": claims}

    def test_supported_claims_retain_for_evaluation(self):
        cv = self._cv([
            {"claim_type": "reaction_direction", "verdict": "supported"},
            {"claim_type": "bond_type", "verdict": "supported"},
            {"claim_type": "substrate_topology", "verdict": "insufficient_evidence"},
        ], eligible=True)
        ra = _reasoning_alignment(cv)
        self.assertEqual(ra["decision"], "retain_for_evaluation")
        self.assertIn("reaction_direction", ra["target_constraints_addressed"])
        self.assertIn("substrate_topology", ra["unsupported_or_unknown"])
        self.assertEqual(ra["contradictions"], [])

    def test_contradiction_drives_decision(self):
        cv = self._cv([
            {"claim_type": "reaction_direction", "verdict": "contradicted"},
        ], eligible=False)
        ra = _reasoning_alignment(cv)
        self.assertEqual(ra["decision"], "contradicted")
        self.assertIn("reaction_direction", ra["contradictions"])

    def test_no_contradiction_no_eligibility_is_insufficient(self):
        cv = self._cv([
            {"claim_type": "native_reaction", "verdict": "insufficient_evidence"},
        ], eligible=False)
        self.assertEqual(_reasoning_alignment(cv)["decision"], "insufficient_evidence")

    def test_none_validation_yields_none(self):
        self.assertIsNone(_reasoning_alignment(None))

    def test_build_candidate_provenance_embeds_alignment(self):
        cv = self._cv([
            {"claim_type": "reaction_direction", "verdict": "supported"},
        ], eligible=True)
        prov = build_candidate_provenance(
            {"accession": "P00001", "protein_name": "x"},
            claim_validation=cv)
        self.assertEqual(prov["reasoning_alignment"]["decision"], "retain_for_evaluation")

    def test_build_candidate_provenance_without_validation(self):
        prov = build_candidate_provenance({"accession": "P00001"})
        self.assertIsNone(prov["reasoning_alignment"])


if __name__ == "__main__":
    unittest.main()
