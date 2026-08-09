"""Tests that finalization reports honest per-stage and run-level status.

A run whose LLM died before any tool call used to record every finalization
stage as "completed", including reports that were skipped for lack of
evaluations. An empty run must not look like a successful one.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from enzyme_discovery_agent.output.finalizer import RunFinalizer


class _NullLLM:
    """LLM stub that always fails, mimicking an unreachable provider."""

    def call(self, *args, **kwargs):
        return None


class _TokenUsage:
    total_tokens = 0

    def to_dict(self):
        return {"total_tokens": 0}

    def record(self, *args, **kwargs):
        pass


class TestFinalizationStageStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        (self.run_dir / "agent_state").mkdir(parents=True, exist_ok=True)
        self.finalizer = RunFinalizer(self.run_dir, _NullLLM(), lambda msg, level="info": None)

    def tearDown(self):
        self.tmp.cleanup()

    def _status(self) -> dict:
        return json.loads((self.run_dir / "finalization_status.json").read_text())

    def test_missing_artifact_records_incomplete_not_completed(self):
        self.finalizer._run_finalization_stage(
            "confidence_report",
            lambda: None,
            produces=self.run_dir / "report_confidence.md",
        )
        stage = self._status()["stages"]["confidence_report"]
        self.assertEqual(stage["status"], "incomplete")
        self.assertIn("report_confidence.md", stage["detail"])

    def test_written_artifact_records_completed(self):
        target = self.run_dir / "report_confidence.md"
        self.finalizer._run_finalization_stage(
            "confidence_report",
            lambda: target.write_text("# report"),
            produces=target,
        )
        self.assertEqual(self._status()["stages"]["confidence_report"]["status"], "completed")

    def test_raising_stage_records_failed(self):
        def _boom():
            raise RuntimeError("provider down")

        self.finalizer._run_finalization_stage("candidates_csv", _boom)
        stage = self._status()["stages"]["candidates_csv"]
        self.assertEqual(stage["status"], "failed")
        self.assertIn("provider down", stage["error"])

    def test_stage_without_declared_artifact_still_completes(self):
        self.finalizer._run_finalization_stage("exploration_summary", lambda: None)
        self.assertEqual(self._status()["stages"]["exploration_summary"]["status"], "completed")


class TestEmptyRunOutcome(unittest.TestCase):
    """An end-to-end finalize() over a run that never produced a candidate."""

    def test_empty_run_is_reported_as_such(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "agent_state").mkdir(parents=True, exist_ok=True)
            finalizer = RunFinalizer(run_dir, _NullLLM(), lambda msg, level="info": None)

            report, n_evals = finalizer.finalize(
                messages=[],
                token_usage=_TokenUsage(),
                query="find enzymes for a target reaction",
                final_text="no candidates",
            )

            self.assertEqual(n_evals, 0)
            status = json.loads((run_dir / "finalization_status.json").read_text())
            outcome = status["run_outcome"]
            self.assertEqual(outcome["outcome"], "no_candidates_evaluated")
            self.assertEqual(outcome["evaluations"], 0)
            self.assertFalse(outcome["discovery_coverage_met"])
            self.assertIn("confidence_report", outcome["stages_not_completed"])
            self.assertIn("explorative_report", outcome["stages_not_completed"])
            self.assertFalse((run_dir / "report_confidence.md").exists())


if __name__ == "__main__":
    unittest.main()
