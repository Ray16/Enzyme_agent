"""Tests for the agent-invoked candidate-map tool (offline, no GPU/network).

The GPU embedding subprocess and the UniProt fetch are stubbed, so these exercise
the tool's orchestration/steering logic deterministically.
"""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from enzyme_discovery_agent import config
from enzyme_discovery_agent.models.candidate_registry import (
    current_candidate_status,
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.tools import embedding_map as em

# Deterministic embedding vectors: N1 sits next to P1, N3 next to P2.
_VECS = {
    "P1": [1.0, 0.0, 0.0, 0.0],
    "P2": [0.0, 0.0, 1.0, 0.0],
    "N1": [0.98, 0.2, 0.0, 0.0],
    "N2": [0.0, 1.0, 0.0, 0.0],
    "N3": [0.0, 0.0, 0.98, 0.2],
}


def _seed_registry(root: Path):
    # Two evaluated (promising) candidates with substrate_fit, three discovered.
    update_candidate_registry(root, "P1", "evaluated", metadata={"substrate_fit": 0.9})
    update_candidate_registry(root, "P2", "evaluated", metadata={"substrate_fit": 0.8})
    for acc in ("N1", "N2", "N3"):
        update_candidate_registry(root, acc, "discovered", metadata={"source_tool": "search"})


def _fake_execute_tool(name, args, injected=None):
    return {"status": "success", "data": {"sequence": "M" + "A" * 40}}


def _fake_embed(run_dir, items, model):
    rows = [(it["accession"], it["seq_sha1"], np.array(_VECS[it["accession"]])) for it in items]
    return rows, "stub embedded"


class TestEmbeddingMapTool(unittest.TestCase):
    def test_disabled_returns_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", False):
            out = em.map_candidate_neighbors(run_dir=tmp)
            self.assertEqual(out["status"], "skipped")

    def test_end_to_end_with_stubbed_embedder(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", True), \
                mock.patch.object(em, "execute_tool", _fake_execute_tool), \
                mock.patch.object(em, "_run_embed_subprocess", _fake_embed):
            root = Path(tmp)
            _seed_registry(root)
            out = em.map_candidate_neighbors(run_dir=str(root), k=2, n_promising=2)

            self.assertEqual(out["status"], "success")
            # Map artifact persisted with both promising candidates.
            payload = json.loads((root / "candidate_map.json").read_text())
            promising = {p["accession"]: p for p in payload["promising"]}
            self.assertEqual(set(promising), {"P1", "P2"})
            # Exploit-first: P1's closest neighbor is N1, P2's is N3.
            self.assertEqual(promising["P1"]["neighbors"][0]["accession"], "N1")
            self.assertEqual(promising["P2"]["neighbors"][0]["accession"], "N3")
            # Neighbors are registered as leads with esm_neighbor provenance.
            reg = load_candidate_registry(root).get("candidates", {})
            self.assertTrue(
                reg["N1"]["metadata"]["source_tool"].startswith("esm_neighbor:")
            )

    def test_no_gpu_at_tool_level_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "ENABLE_CANDIDATE_MAP", True), \
                mock.patch.object(em, "execute_tool", _fake_execute_tool), \
                mock.patch.object(em, "_run_embed_subprocess",
                                  lambda run_dir, items, model: (None, "no free GPU")):
            root = Path(tmp)
            _seed_registry(root)
            out = em.map_candidate_neighbors(run_dir=str(root))
            # Empty store -> partial, run not blocked, no neighbors registered.
            self.assertEqual(out["status"], "partial")
            reg = load_candidate_registry(root).get("candidates", {})
            self.assertEqual(current_candidate_status(reg["N1"]), "discovered")
            self.assertNotIn("esm_neighbor", reg["N1"]["metadata"].get("source_tool", ""))

    def test_run_embed_subprocess_no_free_gpu_returns_none(self):
        # gate exists, but the listing has no line for THIS host -> graceful skip.
        listing = types.SimpleNamespace(
            stdout="definitely-not-this-host 0\ndefinitely-not-this-host 1\n",
            returncode=0,
        )
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "CANDIDATE_MAP_GPU_RESERVE", em.__file__), \
                mock.patch.object(em.subprocess, "run", return_value=listing):
            rows, note = em._run_embed_subprocess(
                Path(tmp), [{"accession": "P1", "seq_sha1": "h", "sequence": "MA"}], "m"
            )
            self.assertIsNone(rows)
            self.assertIn("no free GPU", note)


if __name__ == "__main__":
    unittest.main()
