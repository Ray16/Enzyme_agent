"""Tests for artifact-backed literature entity extraction."""

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.tools.literature import extract_entities


class TestLiteratureArtifactRecovery(unittest.TestCase):
    def test_omitted_papers_are_recovered_from_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts" / "literature"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "search_pubmed_001.json").write_text(json.dumps({
                "result": {"data": [{
                    "title": "Characterization of XyzA enzyme",
                    "abstract": "The xyzA gene encodes an ATP-dependent ligase.",
                }]}
            }))
            result = extract_entities(run_dir=tmp)
            names = [item["name"].lower() for item in result["data"]["gene_names"]]
            self.assertIn("xyza", names)


class TestExtractEntitiesNonPaperEntries(unittest.TestCase):
    """A model may pass bare strings instead of paper objects.

    The real failure was ``AttributeError: 'str' object has no attribute 'get'``
    from iterating a non-paper entry as if it were a paper dict.
    """

    def test_mixed_list_scans_papers_and_reports_the_drop(self):
        result = extract_entities(papers=[
            {"title": "Characterization of XyzA enzyme",
             "abstract": "The xyzA gene encodes an ATP-dependent ligase."},
            "a bare title string", 42, None,
        ])
        self.assertEqual(result["status"], "success")
        names = [item["name"].lower() for item in result["data"]["gene_names"]]
        self.assertIn("xyza", names)
        self.assertIn("skipped 3", result["summary"])

    def test_all_non_paper_entries_error_rather_than_report_zero(self):
        result = extract_entities(papers=["title one", "title two"])
        self.assertEqual(result["status"], "error")
        self.assertIn("not paper objects", result["summary"])

    def test_non_paper_entries_fall_back_to_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts" / "literature"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "search_pubmed_001.json").write_text(json.dumps({
                "result": {"data": [{
                    "title": "Characterization of XyzA enzyme",
                    "abstract": "The xyzA gene encodes an ATP-dependent ligase.",
                }]}
            }))
            result = extract_entities(papers=["a bare title string"], run_dir=tmp)
            self.assertEqual(result["status"], "success")
            names = [item["name"].lower() for item in result["data"]["gene_names"]]
            self.assertIn("xyza", names)
