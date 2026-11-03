"""Organized outputs remain resumable; evolving maps preserve their geometry."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from enzyme_discovery_agent.capabilities import create_run_dir
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models import embedding_store as es
from enzyme_discovery_agent.output.evolving_map import stable_projection, update_candidate_map, _color, _retrieved_family_id
from enzyme_discovery_agent.run_layout import organize_run, run_file


class RunLayoutTest(unittest.TestCase):
    def test_new_runs_use_state_subdirectories_and_legacy_runs_remain_readable(self):
        with tempfile.TemporaryDirectory() as d:
            root = create_run_dir(Path(d), "a generic query")
            update_candidate_registry(root, "P12345", "discovered", {"protein_name": "Example enzyme"})
            self.assertTrue((root / "agent_state/candidate_registry.json").exists())
            self.assertFalse((root / "candidate_registry.json").exists())
            self.assertEqual(run_file(root, "report_confidence.md"), root / "report_confidence.md")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "discovered", {})
            self.assertTrue((root / "candidate_registry.json").exists())

    def test_migration_preserves_candidate_data_and_primary_reports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "evaluated", {"protein_name": "Example enzyme"})
            before = load_candidate_registry(root)
            (root / "report_confidence.md").write_text("Saved scientific report")
            (root / "report_interim.md").write_text("Earlier partial report")
            moves = organize_run(root)
            self.assertEqual(load_candidate_registry(root), before)
            self.assertEqual((root / "report_confidence.md").read_text(), "Saved scientific report")
            self.assertEqual(run_file(root, "report_interim.md").read_text(), "Earlier partial report")
            self.assertFalse((root / "candidate_registry.json.lock").exists())
            self.assertTrue((root / "agent_state/candidate_registry.json.lock").exists())
            self.assertGreater(len(moves), 0)


class EvolvingMapTest(unittest.TestCase):
    def seed(self, root):
        rng = np.random.default_rng(4)
        rows = [(acc, "sequence-hash", rng.normal(size=12)) for acc in ["P12345", "Q12345", "Q54321", "P54321"]]
        es.append_vectors(root, "test-model", rows)
        for acc, _, _ in rows:
            update_candidate_registry(root, acc, "discovered", {"interpro_family_id": "IPR000001"})

    def test_more_candidates_keep_existing_coordinates_and_family_colors(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.seed(root)
            first = stable_projection(root, es.load_store(root))
            color = _color("Example family")
            es.append_vectors(root, "test-model", [("Q22222", "other-hash", np.ones(12))])
            second = stable_projection(root, es.load_store(root))
            np.testing.assert_allclose(first, second[:len(first)], atol=1e-7)
            self.assertEqual(_color("Example family"), color)

    def test_family_and_evaluation_updates_create_revisions_but_identical_reads_do_not(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.seed(root)
            update_candidate_map(root, iteration=1)
            self.assertTrue((root / "artifacts/candidate_map/current.json").exists())
            self.assertFalse((root / "candidate_map.html").exists())
            update_candidate_map(root, iteration=2)
            history = root / "artifacts/candidate_map/history"
            self.assertEqual(len(list(history.glob("*.json"))), 1)
            update_candidate_registry(root, "P12345", "evaluated", {"interpro_family_id": "IPR000002"})
            update_candidate_map(root, iteration=3)
            self.assertEqual(len(list(history.glob("*.json"))), 2)
            current = json.loads((root / "artifacts/candidate_map/current.json").read_text())
            self.assertEqual(current["embedded"], 4)

    def test_no_embeddings_never_produce_a_blank_png(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "discovered", {})
            self.assertIsNone(update_candidate_map(root))
            self.assertTrue((root / "artifacts/candidate_map/current.json").exists())
            self.assertFalse((root / "candidate_map.png").exists())
            self.assertFalse((root / "candidate_map.html").exists())

    def test_only_supported_exact_database_membership_resolves_a_family(self):
        claim = {"claim_type": "family_membership", "verdict": "supported", "asserted_value": "IPR000001",
                 "evidence": {"source": "UniProtKB", "observed_value": ["IPR000001"]}}
        metadata = {"claim_validation": {"claims": [claim]}}
        self.assertEqual(_retrieved_family_id(metadata), "IPR000001")
        claim["verdict"] = "unresolved"
        self.assertIsNone(_retrieved_family_id(metadata))
        claim["verdict"] = "supported"
        claim["evidence"]["observed_value"] = ["IPR000002"]
        self.assertIsNone(_retrieved_family_id(metadata))

    def test_different_models_cannot_be_mixed_in_one_projection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.seed(root)
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                es.append_vectors(root, "other-model", [("Q22222", "hash", np.ones(12))])


if __name__ == "__main__":
    unittest.main()
