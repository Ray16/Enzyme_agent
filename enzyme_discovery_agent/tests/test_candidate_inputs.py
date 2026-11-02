"""Input retrieval preserves independent identity and sequence checks."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry
from enzyme_discovery_agent.tools.candidate_inputs import inspect_discovery_source, retrieve_candidate_modeling_inputs


class CandidateInputTests(unittest.TestCase):
    def test_source_inspection_is_bounded_and_rejects_external_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifacts/source.txt"
            artifact.parent.mkdir()
            artifact.write_text("first\nidentifier association\nlast\n")
            result = inspect_discovery_source(str(artifact), find="association", max_lines=2, run_dir=root)
            self.assertEqual(result["data"]["matching_lines"], [2])
            self.assertNotIn("last", result["data"]["passage"])
            outside = root / "private.txt"
            outside.write_text("not retrieved evidence")
            self.assertEqual(inspect_discovery_source(str(outside), run_dir=root)["status"], "error")

    def prepare(self, root):
        sequence = "MACK"
        update_candidate_registry(root, "P12345", "discovered", {"identity_resolution": {
            "status": "resolved", "resolved_accession": "P12345",
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}})
        return sequence

    def test_changed_sequence_never_becomes_a_modeling_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            with patch("enzyme_discovery_agent.tools.candidate_inputs._official_record", return_value=("MACD", "UniProtKB", {})):
                result = retrieve_candidate_modeling_inputs("P12345", False, root)
            self.assertEqual(result["status"], "error")
            self.assertFalse((root / "artifacts/modeling_inputs").exists())

    def test_registered_identity_gets_fasta_without_activity_or_compute_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.prepare(root)
            with patch("enzyme_discovery_agent.tools.candidate_inputs._official_record", return_value=(sequence, "UniProtKB", {})):
                result = retrieve_candidate_modeling_inputs("P12345", False, root)
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["data"]["activity_validated"])
            self.assertEqual(result["data"]["new_prediction_jobs"], 0)
            self.assertIn(sequence, Path(result["data"]["fasta_path"]).read_text())
            entry = load_candidate_registry(root)["candidates"]["P12345"]
            self.assertEqual(entry["statuses"][-1]["status"], "discovered")
            self.assertEqual(entry["metadata"]["sequence"], sequence)


if __name__ == "__main__":
    unittest.main()


def test_a_long_source_is_crossed_in_a_few_pages_not_many(tmp_path):
    """Each extra page costs a whole turn, and a turn re-sends the entire context."""
    from enzyme_discovery_agent.tools.candidate_inputs import SOURCE_PAGE_LINES
    (tmp_path / 'artifacts').mkdir()
    source = tmp_path / 'artifacts/paper.txt'
    source.write_text("\n".join(
        f"line {i} carrying a realistic amount of prose about enzymes and their substrates"
        for i in range(800)))
    pages, start = 0, 1
    seen = []
    while start and pages < 20:
        result = inspect_discovery_source(str(source), start_line=start, run_dir=str(tmp_path))
        assert result['status'] == 'success'
        seen.append(result['data']['passage'])
        start = result['data']['next_start_line']
        pages += 1
    assert start is None, 'pagination must terminate'
    assert pages <= 4, f'800 lines should take a few pages, took {pages}'
    assert SOURCE_PAGE_LINES >= 320
    # Every line must still be delivered exactly once: a bigger page must not skip text.
    joined = "\n".join(seen)
    assert "line 0 " in joined and "line 799 " in joined
    assert joined.count("line 400 ") == 1
