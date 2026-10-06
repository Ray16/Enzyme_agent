"""Tests for the candidate-map visualization (best-effort, offline)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from enzyme_discovery_agent.models import embedding_store as es
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.output.candidate_map_plot import render_candidate_map


def _seed(root: Path):
    accs = ["P1", "N1", "N2", "P2", "N3"]
    rng = np.random.default_rng(0)
    rows = [(a, "h", rng.normal(size=8)) for a in accs]
    es.append_vectors(root, "test-model", rows)
    for a in accs:
        update_candidate_registry(root, a, "discovered",
                                  metadata={"interpro_family_id": "IPR000001"})
    (root / "candidate_map.json").write_text(json.dumps({
        "promising": [
            {"accession": "P1", "neighbors": [{"accession": "N1"}, {"accession": "N2"}]},
            {"accession": "P2", "neighbors": [{"accession": "N3"}]},
        ]
    }))


class TestCandidateMapPlot(unittest.TestCase):
    def test_renders_or_skips_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _seed(root)
            # References supplied out-of-band (never hardcoded in package source).
            out = render_candidate_map(root, reference_accessions=["P1"])
            # Best-effort: a PNG when umap/matplotlib present, else None — never raises.
            if out is not None:
                self.assertTrue(out.exists())
                self.assertGreater(out.stat().st_size, 0)

    def test_no_store_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(render_candidate_map(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
