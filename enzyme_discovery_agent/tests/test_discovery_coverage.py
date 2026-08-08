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
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
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

    def test_defers_evaluation_until_profile_coverage_is_represented(self):
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
            self.assertIn("do not evaluate", result.new_messages[0]["content"])
            self.assertTrue(result.state_updates["discovery_coverage_incomplete"])
            self.assertTrue(result.state_updates["discovery_coverage_redirected"])

            redirected = AgentState(
                run_dir=run_dir, candidate_count=1,
                discovery_coverage_redirected=True,
            )
            follow_up = DiscoveryCoverageGate().execute(redirected, [])
            self.assertFalse(follow_up.intercept)
            self.assertTrue(follow_up.state_updates["discovery_coverage_incomplete"])

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
