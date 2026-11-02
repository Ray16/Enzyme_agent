"""Campaign memory survives compaction, provider changes and fresh readers."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.context import build_model_context, build_research_state
from enzyme_discovery_agent.llm_client import LLMClient
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.models.research_memory import recall_memory
from enzyme_discovery_agent.tools.reasoning import get_research_memory


class ResearchMemoryTest(unittest.TestCase):
    def write(self, root, name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_search_outcomes_are_paginated_queryable_and_survive_a_new_reader(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(15):
                append_event(root, "tool_call", tool="search_pubmed", args={"query": f"synthetic scope {i}"},
                             status="error" if i == 0 else "empty", count=0)
            append_event(root, "tool_call", tool="get_notebook", args={}, status="success")
            first = recall_memory(root, "searches", limit=6)
            second = recall_memory(Path(str(root)), "searches", offset=first["next_offset"], limit=6)
            self.assertEqual(first["total"], 15)
            self.assertEqual(first["records"][0]["arguments"]["query"], "synthetic scope 14")
            self.assertTrue(set(r["event_sequence"] for r in first["records"]).isdisjoint(
                r["event_sequence"] for r in second["records"]))
            failed = get_research_memory("searches", query="synthetic scope 0", run_dir=str(root))
            self.assertEqual(failed["data"]["records"][0]["status"], "error")
            self.assertIn("not evidence of biological absence", failed["data"]["authority"])
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "synthetic scope 0"}, status="success", count=2)
            updated = recall_memory(root, "searches", query="synthetic scope 0")
            self.assertEqual(updated["total"], 1)
            self.assertEqual(updated["records"][0]["status"], "success")

    def test_unresolved_leads_keep_paper_provenance_and_resolved_names_are_not_pending(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write(root, "artifacts/fulltext/papers_enzymes.json", [{
                "doi": "10.test/primary", "enzymes": [
                    {"name": "Synthetic ZyxQ", "organism": "Synthetic organism", "accessions": [], "lookup_failed": True},
                    {"name": "Resolved ZyxR", "organism": "Synthetic organism", "accessions": []},
                    {"name": "Resolved ZyxR", "organism": "Synthetic organism", "accessions": ["P12345"]}],
                "homologs": []}])
            memory = recall_memory(root, "leads")
            self.assertEqual(memory["total"], 1)
            self.assertEqual(memory["records"][0]["doi"], "10.test/primary")
            self.assertEqual(memory["records"][0]["name"], "Synthetic ZyxQ")
            self.assertIn("unverified", memory["records"][0]["evidence_status"])

    def test_recent_notes_and_unresolved_work_survive_repeated_model_switches(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write(root, "notebook.json", {"sections": {"decisions": "old draft " * 1000 + "Latest correction: scope remains unresolved."}})
            self.write(root, "artifacts/fulltext/papers_enzymes.json", [{
                "doi": "10.test/primary", "enzymes": [{"name": "Synthetic ZyxQ", "accessions": []}], "homologs": []}])
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "failed route"}, status="error")
            messages = [{"role": "system", "content": "constraints"}, {"role": "user", "content": "target"}]
            for _ in range(50):
                messages += [{"role": "assistant", "content": "old narrative " * 1000}, {"role": "user", "content": "continue"}]
            messages += [{"role": "assistant", "_llm_model": "gemini:source", "tool_calls": [{
                "id": "one", "type": "function", "function": {"name": "get_research_memory", "arguments": '{"topic":"leads"}'},
                "extra_content": {"google": {"thought_signature": "opaque"}}}]},
                {"role": "tool", "tool_call_id": "one", "content": "saved lead evidence"}]
            original = copy.deepcopy(messages)
            projected = build_model_context(messages, root)
            # The research state trails the history now, so locate it by its header
            # rather than by a fixed index.
            block = next(m for m in projected
                         if isinstance(m.get("content"), str)
                         and m["content"].startswith("CURRENT RESEARCH STATE"))
            state = json.loads(block["content"].split("\n", 1)[1])
            self.assertIn("Latest correction", state["agent_notes_not_primary_evidence"]["decisions"])
            self.assertEqual(state["research_memory"]["searches"]["records"][0]["status"], "error")
            self.assertEqual(state["research_memory"]["leads"]["records"][0]["name"], "Synthetic ZyxQ")
            for model in ("source", "middle", "small"):
                sent = LLMClient._gemini_messages(projected, model)
                # The state block trails the history; assert the conversion preserves
                # both it and the tool transaction rather than their former indices.
                self.assertTrue(any(isinstance(m.get("content"), str)
                                    and m["content"].startswith("CURRENT RESEARCH STATE") for m in sent))
                self.assertIn("one", [m.get("tool_call_id") for m in sent])
            self.assertEqual(messages, original)
            self.assertLess(len(json.dumps(projected)), 90000)
            recalled = recall_memory(root, "notes", query="old draft")
            self.assertGreater(recalled["total"], 1)

    def test_older_decisions_are_retrievable_and_recent_updates_are_projected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            decisions = {f"family-{i}": {"decision": "expand", "reason": f"evidence {i}", "ts": f"{i:03}"}
                         for i in range(50)}
            decisions["family-0"] = {"decision": "stop", "reason": "recent reconsideration", "ts": "999"}
            self.write(root, "family_decisions.json", {"decisions": decisions})
            with patch("enzyme_discovery_agent.models.family_learning.refresh_family_learning", return_value={
                    "families": {}, "priority_order": [], "candidates": {}}):
                state = build_research_state(root)
            self.assertEqual(len(state["family_decisions"]), 12)
            self.assertEqual(state["family_decisions"]["family-0"]["decision"], "stop")
            old = recall_memory(root, "decisions", query="evidence 1")
            self.assertGreater(old["total"], 0)
            self.assertIn("not primary evidence", old["records"][0]["evidence_status"])

    def test_memory_is_scoped_to_the_explicit_run_and_large_records_are_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            root, other = Path(d) / "one", Path(d) / "two"
            root.mkdir(); other.mkdir()
            append_event(root, "tool_call", tool="search_pubmed", args={"query": "private campaign evidence"}, status="success")
            self.assertEqual(recall_memory(other, "searches")["total"], 0)
            append_event(other, "tool_call", tool="search_pubmed", args={"terms": ["a" * 10000] * 10}, status="empty")
            page = recall_memory(other, "searches")
            self.assertLess(len(json.dumps(page)), 6500)
            self.assertTrue(page["records"][0]["truncated"])
            for options in ({"offset": -1}, {"limit": 0}, {"limit": 21}):
                self.assertEqual(get_research_memory("searches", run_dir=str(root), **options)["status"], "error")


if __name__ == "__main__":
    unittest.main()
