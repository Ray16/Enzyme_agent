"""Closed-loop modeling lifecycle on the candidate registry.

Covers the loop-ready write surface added for the discovery <-> structural-modeling
connection: ``mark_modeling_queued`` (async hand-off), ``record_modeling_result``
(factor-separated evidence + fail-closed interaction with the sticky ``rejected``
state), and the ``candidates_for_modeling`` / ``modeling_validated`` queries the
loop reads to decide what to model next and what to expand around.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import (
    candidates_for_modeling,
    current_candidate_status,
    load_candidate_registry,
    mark_modeling_queued,
    modeling_validated,
    record_modeling_result,
    update_candidate_registry,
)


def _statuses(run_dir: Path, acc: str) -> list[str]:
    reg = load_candidate_registry(run_dir)["candidates"]
    return [s["status"] for s in reg[acc]["statuses"]]


def _meta(run_dir: Path, acc: str) -> dict:
    return load_candidate_registry(run_dir)["candidates"][acc]["metadata"]


class TestModelingLifecycle(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())

    def _evaluated(self, acc: str) -> None:
        update_candidate_registry(self.run_dir, acc, "discovered", metadata={"gene_names": [acc.lower()]})
        update_candidate_registry(self.run_dir, acc, "screened")
        update_candidate_registry(self.run_dir, acc, "evaluated")

    # ---- record_modeling_result (happy path) -------------------------------
    def test_record_result_advances_and_stores_factors(self):
        self._evaluated("P1")
        status = record_modeling_result(
            self.run_dir, "P1",
            result={"mechanistic_plausibility": "ready",
                    "activity_proxy": {"gated_mean_angle": 158.0, "gated_fraction": 0.4},
                    "developability": {"netsolp": 0.71},
                    "structural_tier": "compatible"},
            verdict="compatible", proxy_version="angle_gate_v1", source="cofold/run42",
        )
        self.assertEqual(status, "structurally_evaluated")
        self.assertEqual(current_candidate_status(
            load_candidate_registry(self.run_dir)["candidates"]["P1"]), "structurally_evaluated")
        block = _meta(self.run_dir, "P1")["modeling_result"]
        self.assertEqual(block["verdict"], "compatible")
        self.assertEqual(block["proxy_version"], "angle_gate_v1")
        self.assertEqual(block["source"], "cofold/run42")
        # Factor-separated, not fused into a single composite number.
        self.assertIn("activity_proxy", block["result"])
        self.assertIn("developability", block["result"])
        self.assertNotIn("score", block)
        self.assertTrue(block["fingerprint"])

    # ---- async hand-off ----------------------------------------------------
    def test_mark_modeling_queued(self):
        self._evaluated("Q1")
        mark_modeling_queued(self.run_dir, "Q1", source="driver")
        self.assertEqual(_statuses(self.run_dir, "Q1")[-1], "modeling_queued")
        self.assertEqual(_meta(self.run_dir, "Q1")["modeling_queued"]["source"], "driver")

    # ---- rejected interaction (fail-closed) --------------------------------
    def test_validated_verdict_reopens_rejected(self):
        self._evaluated("R1")
        update_candidate_registry(self.run_dir, "R1", "rejected", metadata={"rejection_reason": "weak fit"})
        status = record_modeling_result(
            self.run_dir, "R1", result={"structural_tier": "strong"}, verdict="validated",
            source="cofold/run42",
        )
        self.assertEqual(status, "structurally_evaluated")
        hist = _statuses(self.run_dir, "R1")
        # reopened (contradictory evidence) THEN structurally_evaluated.
        self.assertIn("reopened", hist)
        self.assertEqual(hist[-1], "structurally_evaluated")
        self.assertEqual(_meta(self.run_dir, "R1")["modeling_result"]["verdict"], "validated")

    def test_weak_verdict_leaves_rejection_standing(self):
        self._evaluated("R2")
        update_candidate_registry(self.run_dir, "R2", "rejected", metadata={"rejection_reason": "weak fit"})
        status = record_modeling_result(
            self.run_dir, "R2", result={"structural_tier": "weak"}, verdict="weak",
        )
        self.assertEqual(status, "rejected")
        self.assertEqual(current_candidate_status(
            load_candidate_registry(self.run_dir)["candidates"]["R2"]), "rejected")
        # Evidence is still attached for the record even though status is unchanged.
        self.assertEqual(_meta(self.run_dir, "R2")["modeling_result"]["verdict"], "weak")

    # ---- loop queries ------------------------------------------------------
    def test_candidates_for_modeling_selects_and_excludes(self):
        self._evaluated("E1")                 # eligible
        self._evaluated("E2")
        mark_modeling_queued(self.run_dir, "E2")   # already queued -> excluded
        update_candidate_registry(self.run_dir, "D1", "discovered")  # never evaluated -> excluded
        self._evaluated("X1")
        update_candidate_registry(self.run_dir, "X1", "rejected")    # rejected -> excluded
        eligible = set(candidates_for_modeling(self.run_dir))
        self.assertEqual(eligible, {"E1"})

    def test_modeling_validated_returns_confirmed_hits(self):
        self._evaluated("V1")
        self._evaluated("V2")
        record_modeling_result(self.run_dir, "V1", result={"t": "strong"}, verdict="validated")
        record_modeling_result(self.run_dir, "V2", result={"t": "weak"}, verdict="weak")
        hits = modeling_validated(self.run_dir)
        self.assertEqual(set(hits), {"V1"})
        self.assertEqual(hits["V1"]["verdict"], "validated")


if __name__ == "__main__":
    unittest.main()
