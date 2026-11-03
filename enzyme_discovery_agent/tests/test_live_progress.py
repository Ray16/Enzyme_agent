"""Live views follow saved evidence without changing discovery state."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.output.live_progress import snapshot, publish, trajectory_figure


class LiveProgressTest(unittest.TestCase):
    def seed(self, root, acc, status, **metadata):
        update_candidate_registry(root, acc, "discovered", {"protein_name": "Synthetic catalyst", **metadata})
        if status != "discovered": update_candidate_registry(root, acc, status)

    def test_new_candidates_join_ranking_and_rejection_removes_them(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.seed(root, "P12345", "evaluated")
            self.seed(root, "Q12345", "screened")
            before = (root / "candidate_registry.json").read_bytes()
            state = snapshot(root)
            self.assertEqual((root / "candidate_registry.json").read_bytes(), before)
            self.assertEqual(state["ranked_count"], 1)
            self.assertEqual(state["rows"][0]["accession"], "P12345")
            self.assertEqual(state["rows"][1]["rank"], "")
            update_candidate_registry(root, "Q12345", "evaluated")
            self.assertEqual(snapshot(root)["ranked_count"], 2)
            update_candidate_registry(root, "P12345", "rejected", {"rejection_reason": "Contradicted native reaction"})
            state = snapshot(root)
            self.assertEqual(state["ranked_count"], 1)
            self.assertEqual(state["rows"][0]["accession"], "Q12345")
            self.assertEqual(state["rows"][-1]["reason"], "Contradicted native reaction")
            self.assertEqual(state["rows"][-1]["rank"], "")

    def test_pending_and_failed_validation_are_not_ranked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.seed(root, "P12345", "evidence_pending")
            self.seed(root, "Q12345", "evaluated", claim_validation={"promotion_eligible": False})
            self.assertEqual(snapshot(root)["ranked_count"], 0)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "Plotly required for report rendering")
    def test_exploration_and_literature_metrics_are_observational(self):
        from enzyme_discovery_agent.models.event_log import append_event
        from enzyme_discovery_agent.state_store import atomic_write_json, write_tool_artifact
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "synthetic search"},
                         status="success", count=1, elapsed_s=1.2)
            write_tool_artifact(root, "literature", "search_pubmed", {"result": {
                "status": "success", "data": [{"title": "Synthetic paper", "doi": "10.test/example"}]}})
            atomic_write_json(root / "agent_state/literature_reading.json", {"downloads": {}, "extractions": {
                "paper": {"status": "empty", "chunks_read": 1}}})
            atomic_write_json(root / "token_usage.json", {"total_tokens": 123, "accounting_complete": True, "model": "synthetic"})
            self.seed(root, "P12345", "evidence_pending", literature_name="Example lead")
            before = (root / "candidate_registry.json").read_bytes()
            state = publish(root)
            self.assertEqual((root / "candidate_registry.json").read_bytes(), before)
            self.assertEqual(state["literature"]["unique_papers"], 1)
            self.assertEqual(state["literature"]["extractions"]["empty"], 1)
            self.assertEqual(state["literature"]["literature_linked_candidates_or_leads"], 1)
            self.assertEqual(state["tokens"], 123)
            self.assertEqual(state["searches"][0]["query"], "synthetic search")
            page = (root / "candidate_progress.html").read_text()
            self.assertIn("Literature: retrieval to evidence", page)
            self.assertIn("Counts alone do not establish", page)

    def test_resume_budget_and_serving_model_are_not_misreported(self):
        from enzyme_discovery_agent.models.event_log import append_event
        from enzyme_discovery_agent.state_store import atomic_write_json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_write_json(root / "token_usage.json", {"total_tokens": 1200, "model": "old"})
            atomic_write_json(root / "agent_state/invocation_resources.json", {"starting_tokens": 1000, "token_budget": 2000})
            append_event(root, "llm_call", backend="first", model="a")
            append_event(root, "llm_model_switch", to_model="second:b")
            state = snapshot(root)
            self.assertEqual(state["tokens"], 1200)
            self.assertEqual(state["invocation_tokens"], 200)
            self.assertEqual(state["token_budget"], 2000)
            self.assertEqual(state["model"], "second:b")
            atomic_write_json(root / "agent_state/experiment_budget.json", {
                "prior_attempt_tokens": 75, "total_token_allowance": 5000})
            state = snapshot(root)
            self.assertEqual(state["experiment_tokens"], 1275)
            self.assertEqual(state["experiment_token_budget"], 5000)

    def test_private_site_publisher_bounds_data_and_rejects_other_origins(self):
        from enzyme_discovery_agent.output.site_publisher import bounded_snapshot, site_endpoint
        self.assertEqual(site_endpoint("https://example.chatgpt.site"), "https://example.chatgpt.site/api/state")
        for url in ("http://example.chatgpt.site", "https://example.chatgpt.site.evil.test",
                    "https://example.chatgpt.site/other", "https://user@example.chatgpt.site"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                site_endpoint(url)
        with patch("enzyme_discovery_agent.output.site_publisher.snapshot", return_value={
            "rows": [{}] * 400, "transitions": [], "frontier": {"overview": {}, "modeling": [{"private": "omitted"}]}}):
            state, payload = bounded_snapshot("unused")
        self.assertEqual(len(state["rows"]), 300)
        self.assertNotIn("modeling", state["frontier"])
        self.assertNotIn(b"omitted", payload)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "Plotly required for report rendering")
    def test_render_escapes_labels_and_cumulative_coverage_counts_distinct_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.seed(root, "P12345", "evaluated", gene_names=["<script>unsafe</script>"])
            self.seed(root, "Q12345", "rejected")
            state = publish(root)
            figure = trajectory_figure(state)
            self.assertEqual(list(figure.data[0].y)[-1], 2)
            self.assertEqual(list(figure.data[1].y)[-1], 1)
            self.assertEqual(list(figure.data[2].y)[-1], 1)
            with (root / "candidate_ranking.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            text = (root / "candidate_progress.html").read_text()
            self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", text)
            self.assertNotIn("<script>unsafe</script>", text)
            self.assertIn("Pending and unresolved candidates", text)
            self.assertIn("Recorded lifecycle transition", text)
            self.assertFalse((root / "pass1_ranking.json").exists())


if __name__ == "__main__":
    unittest.main()


def test_private_publisher_enriches_family_names_without_mislabeling_groups():
    from unittest.mock import patch
    from enzyme_discovery_agent.output.site_publisher import enrich_family_names
    state = {"rows": [{"family": "IPR123456", "family_id": "IPR123456"}], "frontier": {
        "families": [{"family": "ipr:IPR123456"}, {"family": "gene:example"}]}}
    cache = {}
    with patch("enzyme_discovery_agent.output.site_publisher._fetch_family_name", return_value="Example family") as fetch:
        enrich_family_names(state, cache)
        enrich_family_names(state, cache)
    fetch.assert_called_once_with("IPR123456")
    assert state["frontier"]["families"][0]["family_name"] == "Example family"
    assert state["frontier"]["families"][0]["interpro_id"] == "IPR123456"
    assert "family unresolved" in state["frontier"]["families"][1]["family_name"]
    assert state["rows"][0]["family"] == "Example family"
