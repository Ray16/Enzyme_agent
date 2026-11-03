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


class _RecordingTokenUsage:
    """Token accumulator whose total grows as report stages record LLM calls."""

    def __init__(self):
        self.total_tokens = 100  # main-loop spend before finalization

    def to_dict(self):
        return {"total_tokens": self.total_tokens}

    def record(self, *args, **kwargs):
        self.total_tokens += 1000


class TestTokenUsageWrittenAfterReports(unittest.TestCase):
    """token_usage.json must include the report stages' own LLM spend.

    Regression: the file used to be written before report generation, so it
    undercounted the run and disagreed with the run_complete event / manifest.
    """

    def test_token_usage_json_reflects_report_stage_records(self):
        import enzyme_discovery_agent.output.finalizer as fin

        def _fake_report(run_dir, evals, query, llm, token_usage, log_fn):
            token_usage.record("confidence_report", None)
            (run_dir / "report_confidence.md").write_text("# report\n")

        def _fake_explorative(run_dir, evals, query, llm, token_usage, log_fn):
            token_usage.record("explorative_report", None)
            (run_dir / "report_explorative.md").write_text("# report\n")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "agent_state").mkdir(parents=True, exist_ok=True)
            finalizer = RunFinalizer(run_dir, _NullLLM(), lambda msg, level="info": None)
            token_usage = _RecordingTokenUsage()

            orig_report = fin.generate_two_pass_report
            orig_expl = fin.generate_two_pass_explorative
            fin.generate_two_pass_report = _fake_report
            fin.generate_two_pass_explorative = _fake_explorative
            try:
                finalizer.finalize(
                    messages=[], token_usage=token_usage,
                    query="find enzymes for a target reaction", final_text="",
                )
            finally:
                fin.generate_two_pass_report = orig_report
                fin.generate_two_pass_explorative = orig_expl

            on_disk = json.loads((run_dir / "token_usage.json").read_text())
            # 100 seed + 1000 (confidence) + 1000 (explorative) = 2100.
            self.assertEqual(on_disk["total_tokens"], 2100)
            self.assertEqual(on_disk["total_tokens"], token_usage.total_tokens)


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


def test_checkpoint_report_discloses_unranked_queue(tmp_path, monkeypatch):
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
    from enzyme_discovery_agent.output import finalizer as module
    from enzyme_discovery_agent.output import report_generator
    from enzyme_discovery_agent.run_layout import run_file
    update_candidate_registry(tmp_path, 'P10000', 'evaluated')
    update_candidate_registry(tmp_path, 'Q10000', 'screened')
    update_candidate_registry(tmp_path, 'R10000', 'evidence_pending')
    runner = RunFinalizer(tmp_path, _NullLLM(), lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, '_load_evaluations', lambda: [{'data': {'accession': 'P10000'}}])
    monkeypatch.setattr(report_generator, '_family_relevance_partition', lambda *_: (set(), set(), {}))
    monkeypatch.setattr(report_generator, '_family_rank_map', lambda *_: {})
    monkeypatch.setattr(report_generator, '_deterministic_mechanistic_ranking', lambda *_args, **_kw: [])
    def write_report(root, name, query, evaluations, ranking):
        run_file(root, name).write_text('# Report\n\n## Ranked candidates\n\nExisting content\n')
    monkeypatch.setattr(report_generator, '_write_deterministic_report', write_report)
    monkeypatch.setattr(module, 'generate_candidates_csv', lambda *_args, **_kw: None)
    monkeypatch.setattr(module, 'write_structural_modeling_handoff', lambda *_args, **_kw: None)
    text, count = runner.snapshot(query='synthetic', reason='budget reached')
    assert count == 1
    assert '1 evaluated record(s) available for assessment' in text
    assert '2 discovered, screened, or evidence-pending records' in text
    assert 'do not establish activity on the target' in text
