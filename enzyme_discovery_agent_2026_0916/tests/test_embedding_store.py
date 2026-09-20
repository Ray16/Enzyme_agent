"""Tests for the numpy-only embedding store (offline, no GPU)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from enzyme_discovery_agent.models import embedding_store as es


class TestEmbeddingStore(unittest.TestCase):
    def test_empty_store_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = es.load_store(Path(tmp))
            self.assertEqual(len(store), 0)
            self.assertIsNone(store.matrix)

    def test_append_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                ("P1", es.seq_sha1("AAAA"), np.array([2.0, 0.0, 0.0, 0.0])),  # not normalized
                ("P2", es.seq_sha1("BBBB"), np.array([0.0, 1.0, 0.0, 0.0])),
            ]
            added = es.append_vectors(root, "test-model", rows)
            self.assertEqual(added, 2)
            store = es.load_store(root)
            self.assertEqual(len(store), 2)
            self.assertEqual(store.dim, 4)
            self.assertEqual(store.acc_to_row, {"P1": 0, "P2": 1})
            # Rows must be L2-normalized on write.
            self.assertAlmostEqual(float(np.linalg.norm(store.matrix[0])), 1.0, places=5)

    def test_append_skips_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            es.append_vectors(root, "m", [("P1", "h1", np.array([1.0, 0.0]))])
            added = es.append_vectors(root, "m", [
                ("P1", "h1", np.array([0.0, 1.0])),  # duplicate acc -> skipped
                ("P2", "h2", np.array([0.0, 1.0])),
            ])
            self.assertEqual(added, 1)
            self.assertEqual(len(es.load_store(root)), 2)

    def test_pending_accessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            es.append_vectors(root, "m", [("P1", "h1", np.array([1.0, 0.0]))])
            pending = es.pending_accessions(root, ["P1", "P2", "P2", "P3"])
            self.assertEqual(pending, ["P2", "P3"])  # deduped, order-preserving

    def test_knn_finds_closest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            es.append_vectors(root, "m", [
                ("P1", "h", np.array([1.0, 0.0, 0.0])),
                ("N1", "h", np.array([0.98, 0.2, 0.0])),   # near P1
                ("N2", "h", np.array([0.0, 1.0, 0.0])),    # far from P1
            ])
            store = es.load_store(root)
            nbrs = es.knn(store, [store.acc_to_row["P1"]], k=2, exclude={store.acc_to_row["P1"]})
            self.assertIn("P1", nbrs)
            self.assertEqual(nbrs["P1"][0][0], "N1")           # closest first
            self.assertLess(nbrs["P1"][0][1], nbrs["P1"][1][1])  # ascending distance

    def test_sequence_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            es.cache_sequences(root, {"P1": "MAAA", "P2": ""})  # empty dropped
            cached = es.load_sequences(root)
            self.assertEqual(cached.get("P1"), "MAAA")
            self.assertNotIn("P2", cached)


if __name__ == "__main__":
    unittest.main()
