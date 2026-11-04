"""Offline regression tests for exact-record, claim-level validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.claim_validation import (
    reaction_topology,
    validate_uniprot_claims,
)
from enzyme_discovery_agent.tools.database import fetch_uniprot_sequence
from enzyme_discovery_agent.tools.family_reactions import select_relevant_members
from enzyme_discovery_agent.tools.reasoning import screen_candidates


def _entry(
    accession="P00001", *, entry_type="UniProtKB reviewed (Swiss-Prot)",
    fold="IPR000001", reaction="acid + amine + ATP = product + ADP + phosphate",
    ec="6.3.2.1",
):
    comments = [] if reaction is None else [{
        "commentType": "CATALYTIC ACTIVITY",
        "reaction": {"name": reaction},
    }]
    xrefs = [] if fold is None else [{"database": "InterPro", "id": fold}]
    return {
        "entryType": entry_type,
        "primaryAccession": accession,
        "proteinDescription": {"recommendedName": {
            "fullName": {"value": "Example enzyme"},
            "ecNumbers": ([{"value": ec}] if ec else []),
        }},
        "comments": comments,
        "uniProtKBCrossReferences": xrefs,
        "sequence": {"length": 300, "value": "A" * 300},
    }


def _claim(result, kind):
    return next(c for c in result["claims"] if c["claim_type"] == kind)


class TestAtomicClaimVerdicts(unittest.TestCase):
    def test_reviewed_exact_record_supports_required_claims(self):
        result = validate_uniprot_claims(
            _entry(), expected_accession="P00001", expected_fold_id="IPR000001",
            target_direction="formation", target_bond_types={"amide"},
            target_topology="free_small_molecule", require_reviewed=True,
        )
        self.assertTrue(result["promotion_eligible"])
        self.assertTrue(all(
            _claim(result, kind)["verdict"] == "supported"
            for kind in result["required_claims"]
        ))

    def test_deleted_accession_is_contradicted(self):
        deleted = {
            "entryType": "Inactive", "primaryAccession": "D00000",
            "inactiveReason": {"inactiveReasonType": "DELETED"},
        }
        result = validate_uniprot_claims(
            deleted, expected_accession="D00000", require_native_reaction=False,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(_claim(result, "record_active")["verdict"], "contradicted")

    def test_name_without_reaction_is_insufficient(self):
        result = validate_uniprot_claims(
            _entry(reaction=None, ec=None), expected_accession="P00001",
            target_direction="formation",
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(_claim(result, "native_reaction")["verdict"],
                         "insufficient_evidence")
        self.assertEqual(_claim(result, "reaction_direction")["verdict"],
                         "insufficient_evidence")

    def test_wrong_fold_is_contradicted(self):
        result = validate_uniprot_claims(
            _entry(fold="IPR999999"), expected_accession="P00001",
            expected_fold_id="IPR000001",
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(_claim(result, "family_membership")["verdict"], "contradicted")

    def test_unreviewed_record_cannot_satisfy_reviewed_claim(self):
        result = validate_uniprot_claims(
            _entry(entry_type="UniProtKB unreviewed (TrEMBL)"),
            expected_accession="P00001",
            require_reviewed=True,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(_claim(result, "reviewed_status")["verdict"], "contradicted")

    def test_opposite_direction_is_contradicted(self):
        result = validate_uniprot_claims(
            _entry(reaction="product + H2O = acid + amine", ec="3.5.1.1"),
            expected_accession="P00001", target_direction="formation",
        )
        self.assertEqual(_claim(result, "reaction_direction")["verdict"], "contradicted")
        self.assertFalse(result["promotion_eligible"])

    def test_verbose_target_bond_phrase_matches_amide_native_reaction(self):
        # Regression: the target bond type is stored as a human-readable phrase
        # ("amide C-N bond formed ...") while the native reaction classifies to the
        # token "amide". These must be compared in one vocabulary; otherwise a
        # clearly amide-forming ligase (e.g. a gamma-glutamyl-polyamine synthetase:
        # glutamate + putrescine -> gamma-glutamyl-putrescine) is wrongly rejected.
        verbose = {"amide C–N bond formed (carboxylic acid + primary amine condensation)"}
        result = validate_uniprot_claims(
            _entry(reaction="putrescine + L-glutamate + ATP = "
                            "gamma-L-glutamylputrescine + ADP + phosphate + H(+)",
                   ec="6.3.1.11"),
            expected_accession="P00001", target_direction="formation",
            target_bond_types=verbose,
        )
        self.assertEqual(_claim(result, "bond_type")["verdict"], "supported")
        self.assertTrue(result["promotion_eligible"])

    def test_topology_compatibility_is_directional(self):
        # Carrier-tethered vs a free target is a real machinery mismatch -> contradicted
        # and blocking (both directions).
        tethered = "acyl-[carrier protein] + acceptor = product + carrier protein"
        self.assertEqual(reaction_topology([tethered]), "carrier_protein_tethered")
        result = validate_uniprot_claims(
            _entry(reaction=tethered), expected_accession="P00001",
            target_topology="free_small_molecule",
        )
        self.assertEqual(_claim(result, "substrate_topology")["verdict"], "contradicted")
        self.assertFalse(result["promotion_eligible"])

        # A different native topology is not proof of accepting the target;
        # retain it as an unblocked possibility rather than claiming support.
        polymer = "polymer(n) + monomer + ATP = polymer(n+1) + ADP + phosphate"
        self.assertEqual(reaction_topology([polymer]), "polymer")
        result_poly = validate_uniprot_claims(
            _entry(reaction=polymer), expected_accession="P00001",
            target_topology="free_small_molecule",
        )
        self.assertEqual(_claim(result_poly, "substrate_topology")["verdict"], "insufficient_evidence")
        self.assertNotIn("substrate_topology", result_poly["blocking_claims"])

        # ...but a monomer-only (free) enzyme CANNOT act on a genuine polymer target.
        monomer = "an amide + H2O = a carboxylate + an amine"  # free small-molecule
        result_rev = validate_uniprot_claims(
            _entry(reaction=monomer), expected_accession="P00001",
            target_topology="polymer",
        )
        self.assertEqual(_claim(result_rev, "substrate_topology")["verdict"], "contradicted")
        self.assertFalse(result_rev["promotion_eligible"])

    def test_unspecific_reaction_topology_is_insufficient_not_assumed_free(self):
        # An unspecific reaction is NOT assumed to be a free small molecule...
        reaction = "substrate + cosubstrate = product"
        self.assertIsNone(reaction_topology([reaction]))
        result = validate_uniprot_claims(
            _entry(reaction=reaction), expected_accession="P00001",
            target_topology="free_small_molecule",
        )
        self.assertEqual(_claim(result, "substrate_topology")["verdict"],
                         "insufficient_evidence")
        # ...and insufficient topology evidence is ADVISORY: it must not hard-block
        # promotion (only a definite contradiction does). A native reaction whose
        # topology cannot be parsed should not be rejected on that basis.
        self.assertNotIn("substrate_topology", result["blocking_claims"])

    def test_literature_topology_does_not_overstate_cross_topology_transfer(self):
        evidence = {"evidence_quote": "Example enzyme forms a polymer(n) product.",
                    "mechanism": {"direction": "formation", "bond_types": ["amide"]},
                    "paper": {"doi": "10.test/example"}, "text_sha256": "test"}
        with mock.patch("enzyme_discovery_agent.models.literature_evidence.verify_literature_evidence", return_value=True), \
             mock.patch("enzyme_discovery_agent.models.claim_validation._quote_reaction_support", return_value=True):
            result = validate_uniprot_claims(
                _entry(reaction=None, ec=None), expected_accession="P00001",
                target_direction="formation", target_bond_types={"amide"},
                target_topology="free_small_molecule", literature_evidence=evidence)
        self.assertEqual(_claim(result, "substrate_topology")["verdict"], "insufficient_evidence")
        self.assertNotIn("substrate_topology", result["blocking_claims"])


class TestPromotionBoundary(unittest.TestCase):
    def _member(self, eligible):
        return {
            "accession": "P00001", "name": "x", "ec": ["6.3.2.1"],
            "reviewed": True, "direction_vs_target": "same",
            "native_reaction": "A + B = C",
            "substrate": {"dimensions": {"functional_group_similarity": 1.0}},
            "claim_validation": {
                "promotion_eligible": eligible,
                "blocking_claims": [] if eligible else ["substrate_topology"],
            },
        }

    def test_selection_requires_supported_claims(self):
        self.assertEqual(select_relevant_members([self._member(False)]), [])
        self.assertEqual([m["accession"] for m in select_relevant_members(
            [self._member(True)])], ["P00001"])

    def test_gate_preserves_validation_in_candidate_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = FamilySurveyGate()
            gate._promote_relevant(root, "IPR000001", [self._member(True)])
            metadata = load_candidate_registry(root)["candidates"]["P00001"]["metadata"]
            self.assertTrue(metadata["claim_validation"]["promotion_eligible"])


class TestIdentityIntegration(unittest.TestCase):
    def test_full_fetch_rejects_deleted_record(self):
        body = json.dumps({
            "entryType": "Inactive", "primaryAccession": "D00000",
            "inactiveReason": {"inactiveReasonType": "DELETED"},
        }).encode()
        with mock.patch("enzyme_discovery_agent.tools.database.request_with_retry",
                        return_value=body):
            result = fetch_uniprot_sequence("D00000")
        self.assertEqual(result["status"], "error")
        self.assertIn("record_active", result["data"]["claim_validation"]["blocking_claims"])

    def test_screening_rejects_deleted_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "target_context.json").write_text(json.dumps({"query": "target"}))
            validation = validate_uniprot_claims(
                {"entryType": "Inactive", "primaryAccession": "D00000",
                 "inactiveReason": {"inactiveReasonType": "DELETED"}},
                expected_accession="D00000", require_native_reaction=False,
            )
            with mock.patch(
                "enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light",
                return_value={"accession": "D00000", "claim_validation": validation},
            ):
                result = screen_candidates(root, ["D00000"])
            self.assertEqual(result["data"][0]["screening_decision"], "rejected")
            statuses = load_candidate_registry(root)["candidates"]["D00000"]["statuses"]
            self.assertEqual(statuses[-1]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
