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
