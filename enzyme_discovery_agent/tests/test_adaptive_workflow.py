"""Evidence feedback, bounded reading and pending evaluations stay resumable."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry
from enzyme_discovery_agent.models.expansion import select_expansion_seeds, record_seed_search, expansion_assessments
from enzyme_discovery_agent.models.family_learning import refresh_family_learning
from enzyme_discovery_agent.models.literature_reading import pending_reading, reading_state
from enzyme_discovery_agent.models.rolling import new_evaluation_work, record_evaluation_batch
from enzyme_discovery_agent.tools import fulltext as F


class AdaptiveWorkflowTest(unittest.TestCase):
    def test_pending_control_stops_before_next_paper_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            manifest = [{"key": f"10.test/{i}", "text_file": f"{i}.txt"} for i in range(2)]
            for row in manifest:
                (dest / row["text_file"]).write_text("Unrelated short passage.")
            with patch("enzyme_discovery_agent.steering.has_pending", side_effect=[False, True]), \
                 patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", return_value={"paper_relevant": False}) as extract:
                F._register_named_enzymes_from_fulltext(root, "target", dest, manifest,
                    SimpleNamespace(budget_exhausted=False), Mock(), set(), 10)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(len(reading_state(root)["extractions"]), 1)

    def test_cached_text_progress_is_reported_without_counting_known_candidates_as_new(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            manifest = [{"key": f"10.test/{i}", "doi": f"10.test/{i}", "text_file": f"{i}.txt"} for i in range(2)]
            for row in manifest:
                (dest / row["text_file"]).write_text("A related catalyst was characterized.")
            (dest / "manifest.json").write_text(json.dumps(manifest))
            update_candidate_registry(root, "P12345", "evaluated", {})
            messages = []
            with patch.object(F, "_recover_literature_papers", return_value=manifest), \
                 patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", return_value={
                     "paper_relevant": True, "enzymes": [{"name": "Synthetic enzyme", "organism": ""}]}), \
                 patch.object(F, "_resolve_named_enzyme", return_value="P12345"), \
                 patch.object(F, "fetch_one") as fetch:
                F.mine_and_register_fulltext_accessions(root, "target", log_fn=messages.append,
                    llm_client=SimpleNamespace(budget_exhausted=False))
            stats = json.loads((root / "agent_state/literature_harvest_step.json").read_text())
            self.assertEqual(stats["new_downloads"], 0)
            self.assertEqual(stats["chunks_processed"], 2)
            self.assertEqual(stats["new_candidates"], 0)
            self.assertEqual(stats["existing_candidates_refreshed"], 1)
            self.assertIn("2 text chunk(s) processed", messages[-1])
            fetch.assert_not_called()

    def test_named_leads_survive_an_overstrict_negative_relevance_label(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            (dest / "paper.txt").write_text("A related catalyst was experimentally characterized.")
            meta = {"key": "10.test/related", "doi": "10.test/related", "text_file": "paper.txt"}
            stats = {}
            with patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", return_value={
                    "paper_relevant": False, "enzymes": [{"name": "Synthetic enzyme", "organism": ""}]}), \
                 patch.object(F, "_resolve_named_enzyme", return_value=None):
                F._register_named_enzymes_from_fulltext(root, "target", dest, [meta],
                    SimpleNamespace(budget_exhausted=False), Mock(), set(), 10, stats=stats)
            self.assertEqual(reading_state(root)["extractions"][meta["key"]]["status"], "done")
            self.assertEqual(stats["chunks_processed"], 1)
            self.assertEqual(stats["unresolved_leads"], 1)
            self.assertEqual(stats["extraction_failures"], 0)

    def make_target(self, root):
        (root / "target_reasoning.json").write_text(json.dumps({
            "reaction_direction": "formation", "bond_types": ["ester"], "substrate_topology": "free_small_molecule"}))

    def seed(self, root, acc, name, fit, status="evaluated"):
        update_candidate_registry(root, acc, status, {
            "protein_name": name, "interpro_family_id": "IPR000001", "substrate_fit": fit,
            "catalytic_activities": ["A + B = C"],
            "reaction_chemistry": {"reaction_direction": "formation", "bond_types": ["ester"]},
            "claim_validation": {"promotion_eligible": True, "claims": [{
                "claim_type": "substrate_topology", "evidence": {"observed_value": "free_small_molecule"}}]}})

    def test_previous_expansion_is_assessed_before_more_searches_in_that_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.make_target(root)
            self.seed(root, "P12345", "Synthetic parent", .8)
            self.seed(root, "Q12345", "Independent returned enzyme", .6, "discovered")
            self.seed(root, "Q54321", "Independent future seed", .7)
            record_seed_search(root, "foldseek", "P12345", {"status": "success", "new_accessions": ["Q12345"]})
            registry = load_candidate_registry(root)["candidates"]
            self.assertEqual(select_expansion_seeds(registry, 2, root, method="foldseek"), [])
            update_candidate_registry(root, "Q12345", "evaluated")
            registry = load_candidate_registry(root)["candidates"]
            self.assertEqual(select_expansion_seeds(registry, 2, root, method="foldseek"), ["Q54321"])

    def test_unproductive_method_is_revisited_when_a_better_supported_seed_arrives(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.make_target(root)
            for acc, name, fit in [("P12345", "Parent one", .4), ("Q12345", "Parent two", .3), ("Q54321", "Parent three", .2)]:
                self.seed(root, acc, name, fit)
            for acc in ("P12345", "Q12345"):
                record_seed_search(root, "foldseek", acc, {"status": "success", "new_accessions": []})
            registry = load_candidate_registry(root)["candidates"]
            policy = refresh_family_learning(root)
            self.assertEqual(expansion_assessments(root, registry, policy, "foldseek")["Q54321"]["decision"], "change_method")
            self.seed(root, "P54321", "Improved supported seed", .9)
            self.assertEqual(select_expansion_seeds(load_candidate_registry(root)["candidates"], 2, root, method="foldseek"), ["P54321"])

    def test_remaining_screened_candidates_are_not_lost_at_a_batch_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "screened", {})
            record_evaluation_batch(root)
            self.assertTrue(new_evaluation_work(root))

    def test_a_negative_excerpt_does_not_hide_a_later_enzyme_and_progress_is_saved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            (dest / "paper.txt").write_text("x" * 12000 + "tail-marker enzyme evidence")
            meta = {"key": "10.test/paper", "doi": "10.test/paper", "text_file": "paper.txt"}
            (dest / "manifest.json").write_text(json.dumps([meta]))
            def extract(text, *args, **kwargs):
                if "tail-marker" in text:
                    return {"paper_relevant": True, "enzymes": [{"name": "Synthetic enzyme", "organism": "Synthetic organism"}]}
                return {"paper_relevant": False, "relevance_reason": "no relevant reaction in this passage"}
            with patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", side_effect=extract), \
                 patch.object(F, "_resolve_named_enzyme", return_value="P12345"):
                register = Mock()
                F._register_named_enzymes_from_fulltext(root, "target", dest, [meta], SimpleNamespace(budget_exhausted=False), register, set(), 10)
                self.assertEqual(reading_state(root)["extractions"][meta["key"]]["cursor"], 12000)
                self.assertTrue(pending_reading(root, []))
                F._register_named_enzymes_from_fulltext(root, "target", dest, [meta], SimpleNamespace(budget_exhausted=False), register, set(), 10)
                register.assert_called_once()
                self.assertFalse(pending_reading(root, []))

    def test_valid_negative_whole_papers_are_not_reextracted_and_steps_are_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            manifest = [{"key": f"10.test/{i}", "text_file": f"{i}.txt"} for i in range(5)]
            for row in manifest: (dest / row["text_file"]).write_text("A short unrelated study.")
            with patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", return_value={"paper_relevant": False}) as extract:
                F._register_named_enzymes_from_fulltext(root, "target", dest, manifest, SimpleNamespace(budget_exhausted=False), Mock(), set(), 10)
                self.assertEqual(extract.call_count, 2)
                F._register_named_enzymes_from_fulltext(root, "target", dest, manifest, SimpleNamespace(budget_exhausted=False), Mock(), set(), 10)
                self.assertEqual(extract.call_count, 4)
                self.assertEqual(len(reading_state(root)["extractions"]), 4)
