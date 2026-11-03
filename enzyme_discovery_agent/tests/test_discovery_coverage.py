"""Regression tests for coverage-first candidate evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.gates.discovery_coverage import (
    DiscoveryCoverageGate,
    assess_discovery_coverage,
)
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    load_candidate_registry,
    reopen_candidate,
    update_candidate_registry,
)
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    reopen_candidate,
)
from enzyme_discovery_agent.models.target_profile import normalize_target_profile
from enzyme_discovery_agent.tools.reasoning import evaluate_candidate


class TestDiscoveryCoverage(unittest.TestCase):
    def _write_context(self, run_dir: Path) -> None:
        profile = normalize_target_profile({
            "discovery_coverage": {
                "minimum_screened_candidates": 2,
                "minimum_distinct_families": 2,
                "evidence_buckets": [{
                    "name": "activated-carboxyl evidence",
                    "match_any": ["adenylate", "activated carboxyl"],
                    "minimum_candidates": 1,
                }],
            },
        })
        (run_dir / "target_context.json").write_text(json.dumps({"target_profile": profile}))

    def test_reports_coverage_gaps_without_blocking_rolling_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_context(run_dir)
            update_candidate_registry(run_dir, "P12345", "screened", metadata={
                "gene_names": ["one"], "protein_name": "unrelated protein",
            })
            assessment = assess_discovery_coverage(run_dir)
            self.assertFalse(assessment["covered"])
            self.assertIn("more screened candidates", " ".join(assessment["missing"]))
            result = DiscoveryCoverageGate().execute(AgentState(run_dir=run_dir, candidate_count=1), [])
            self.assertTrue(result.intercept)
            self.assertIn("continue evaluating", result.new_messages[0]["content"])
            self.assertFalse(result.state_updates["discovery_coverage_incomplete"])
            self.assertTrue(result.state_updates["discovery_coverage_redirected"])

            redirected = AgentState(
                run_dir=run_dir, candidate_count=1,
                discovery_coverage_redirected=True,
            )
            follow_up = DiscoveryCoverageGate().execute(redirected, [])
            self.assertFalse(follow_up.intercept)
            self.assertFalse(follow_up.state_updates["discovery_coverage_incomplete"])

    def test_accepts_distinct_families_with_required_annotation_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_context(run_dir)
            update_candidate_registry(run_dir, "P12345", "screened", metadata={
                "gene_names": ["one"], "protein_name": "adenylate ligase",
            })
            update_candidate_registry(run_dir, "Q12345", "screened", metadata={
                "gene_names": ["two"], "protein_name": "different ligase",
            })
            self.assertTrue(assess_discovery_coverage(run_dir)["covered"])
            result = DiscoveryCoverageGate().execute(AgentState(run_dir=run_dir, candidate_count=2), [])
            self.assertFalse(result.intercept)

    def test_canonical_interpro_family_collapses_different_gene_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_context(run_dir)
            for accession, gene in (("P12345", "one"), ("Q12345", "two")):
                update_candidate_registry(run_dir, accession, "screened", metadata={
                    "gene_names": [gene], "protein_name": "adenylate ligase",
                    "screening_family_id": "IPR000001",
                })
            assessment = assess_discovery_coverage(run_dir)
            self.assertEqual(assessment["distinct_families"], 1)
            self.assertEqual(assessment["canonical_families"], 1)
            self.assertFalse(assessment["covered"])

    def test_profile_validation_rejects_malformed_coverage_contract(self):
        with self.assertRaises(ValueError):
            normalize_target_profile({"discovery_coverage": {"minimum_distinct_families": 0}})

    def test_rejected_candidate_cannot_be_evaluated_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "rejected", metadata={
                "rejection_reason": "direction mismatch",
            })
            result = evaluate_candidate(run_dir, "P12345")
            self.assertEqual(result["status"], "error")
            self.assertIn("direction mismatch", result["summary"])

    def test_rejected_candidate_requires_explicit_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "rejected", metadata={
                "rejection_reason": "old evidence",
            })
            update_candidate_registry(run_dir, "P12345", "discovered")
            entry = load_candidate_registry(run_dir)["candidates"]["P12345"]
            self.assertEqual(current_candidate_status(entry), "rejected")
            reopen_candidate(run_dir, "P12345", reason="new primary evidence",
                             evidence={"doi": "10.1/example"})
            entry = load_candidate_registry(run_dir)["candidates"]["P12345"]
            self.assertEqual(current_candidate_status(entry), "reopened")
            self.assertIn("evidence_fingerprint", entry["statuses"][-1])

    def test_rejected_candidate_requires_explicit_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "rejected", metadata={
                "rejection_reason": "old evidence",
            })
            update_candidate_registry(run_dir, "P12345", "discovered")
            entry = __import__(
                "enzyme_discovery_agent.models.candidate_registry", fromlist=["load_candidate_registry"]
            ).load_candidate_registry(run_dir)["candidates"]["P12345"]
            self.assertEqual(current_candidate_status(entry), "rejected")
            reopen_candidate(run_dir, "P12345", reason="new primary evidence",
                             evidence={"doi": "10.1/example"})
            entry = __import__(
                "enzyme_discovery_agent.models.candidate_registry", fromlist=["load_candidate_registry"]
            ).load_candidate_registry(run_dir)["candidates"]["P12345"]
            self.assertEqual(current_candidate_status(entry), "reopened")
            self.assertIn("evidence_fingerprint", entry["statuses"][-1])
