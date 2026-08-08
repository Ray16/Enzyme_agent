"""Tests for run-level provenance capture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent import config, provenance


class TestConfigSnapshot(unittest.TestCase):
    def test_snapshot_captures_upper_case_constants(self):
        snap = provenance.snapshot_config(config)
        self.assertIn("MAX_ITERATIONS", snap)
        self.assertEqual(snap["MAX_ITERATIONS"], config.MAX_ITERATIONS)

    def test_secrets_are_redacted(self):
        snap = provenance.snapshot_config(config)
        # ARGO_API_KEY resolves to a truthy value (username fallback); it must
        # never appear verbatim in a shareable provenance record.
        self.assertEqual(snap.get("ARGO_API_KEY"), "<redacted>")

    def test_lowercase_and_callables_excluded(self):
        snap = provenance.snapshot_config(config)
        self.assertNotIn("os", snap)


class TestCodeProvenance(unittest.TestCase):
    def test_captures_git_commit(self):
        code = provenance.capture_code_provenance()
        # The workspace is a git repo after initialization; if git is somehow
        # unavailable the function must degrade gracefully rather than raise.
        if code.get("vcs") == "git":
            self.assertTrue(code["commit"])
            self.assertEqual(code["commit_short"], code["commit"][:12])
            self.assertIn("dirty", code)
        else:
            self.assertIsNone(code.get("vcs"))


class TestManifestLifecycle(unittest.TestCase):
    def _run_dir(self, tmp: str) -> Path:
        rd = Path(tmp) / "run"
        rd.mkdir()
        return rd

    def test_start_then_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = self._run_dir(tmp)
            provenance.write_manifest_start(
                rd, query="q", model="m", config_module=config,
                args={"max_iterations": 5}, substrate={"smiles": "CCO", "name": "ethanol"},
                capabilities={"rdkit": True},
            )
            started = json.loads((rd / "manifest.json").read_text())
            self.assertEqual(started["status"], "running")
            self.assertIsNone(started["result"])
            self.assertEqual(started["substrate"]["name"], "ethanol")

            provenance.finalize_manifest(
                rd, status="completed", iterations_used=3,
                candidates_evaluated=2, tokens={"total_tokens": 10}, outcome="ok",
            )
            done = json.loads((rd / "manifest.json").read_text())
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["result"]["candidates_evaluated"], 2)
            self.assertIsNotNone(done["timing"]["finished_at"])
            # Code/environment captured at start must survive finalize.
            self.assertIn("code", done)
            self.assertIn("environment", done)

    def test_finalize_writes_human_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = self._run_dir(tmp)
            provenance.write_manifest_start(
                rd, query="find enzymes", model="argo:claude-5-sonnet",
                config_module=config,
            )
            provenance.finalize_manifest(
                rd, status="completed", iterations_used=1,
                candidates_evaluated=0, tokens={"total_tokens": 42}, outcome="done",
            )
            summary = (rd / "RUN_SUMMARY.md").read_text()
            self.assertIn("Run Summary", summary)
            self.assertIn("find enzymes", summary)
            self.assertIn("42", summary)

    def test_finalize_without_start_does_not_raise(self):
        # A crash before the start manifest is written must not make finalize
        # fail; it should still record what it can.
        with tempfile.TemporaryDirectory() as tmp:
            rd = self._run_dir(tmp)
            provenance.finalize_manifest(rd, status="failed", outcome="crash")
            done = json.loads((rd / "manifest.json").read_text())
            self.assertEqual(done["status"], "failed")


if __name__ == "__main__":
    unittest.main()
