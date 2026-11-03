"""Discovery must change routes when a weak model or a keyword search fails."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from enzyme_discovery_agent.models.literature_strategy import complementary_actions, target_routes
from enzyme_discovery_agent.models.literature_reading import pending_reading, save_reading_state
from enzyme_discovery_agent.models.literature_leads import sequence_name_leads
from enzyme_discovery_agent.models.recovery import next_recovery_action, record_recovery_attempt
from enzyme_discovery_agent.tools import literature as L, fulltext as F
from enzyme_discovery_agent.tool_registry import check_search_duplicate, clear_search_history, record_search


class StrategyTest(unittest.TestCase):
    def prepare(self, root):
        (root / "target_reasoning.json").write_text(json.dumps({
            "reaction_direction": "cleavage", "bond_types": ["glycosidic"], "substrate_topology": "free_small_molecule",
            "reactive_groups": ["hydroxyl"]}))
        (root / "target_context.json").write_text(json.dumps({
            "query": "Find enzymes that cleave glycosidic bonds for carbohydrate recycling",
            "substrate_smiles": ""}))

    def test_chemistry_and_application_routes_require_no_named_answer_or_model_family(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.prepare(root)
            routes = target_routes(root)
            queries = [r[0] for r in routes]
            self.assertIn("carbohydrate recycling enzymatic glycosidic cleavage", queries)
            self.assertIn("enzymatic glycosidic cleavage", queries)
            self.assertTrue(any("hydroxyl" in q for q in queries))
            self.assertFalse(any("ATP" in q for q in queries))

    def test_unrelated_reaction_types_do_not_inherit_a_preferred_mechanism(self):
        for direction, bonds, group in [("formation", ["carbon-carbon"], "carbonyl"),
                                        ("redox", ["carbonyl"], "hydroxyl"),
                                        ("transfer", ["phosphoryl"], "phosphate"),
                                        ("isomerization", [], "stereocenter")]:
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "target_reasoning.json").write_text(json.dumps({
                    "reaction_direction": direction, "bond_types": bonds, "reactive_groups": [group]}))
                (root / "target_context.json").write_text(json.dumps({"query": "Find enzymes for the supplied reaction"}))
                queries = [row[0] for row in target_routes(root)]
                self.assertTrue(any(direction in q for q in queries))
                self.assertTrue(any(group in q for q in queries))
                self.assertFalse(any("ATP" in q or "ligase" in q or "synthetase" in q for q in queries))

    def test_negative_search_moves_to_another_source_and_restrictive_terms_are_relaxed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.prepare(root)
            available = {"search_pubmed", "search_openalex", "search_crossref"}
            first = next_recovery_action(root, available)
            record_recovery_attempt(root, first, "started")
            record_recovery_attempt(root, first, "empty")
            second = next_recovery_action(root, available)
            self.assertEqual(second["args"]["query"], first["args"]["query"])
            self.assertNotEqual(second["tool"], first["tool"])
            hypothesis = {"name": "Untested fold", "promise": "low", "literature_terms": ["enzyme substrate context exact unusual structure"]}
            actions = complementary_actions(root, available, [hypothesis])
            self.assertTrue(any(a["lane"] == "hypothesis" for a in actions))
            self.assertTrue(any(a["args"]["query"] == "enzyme substrate context" for a in actions))
            self.assertTrue(any(a["args"].get("sort_by_citations") is False for a in actions))

    def test_recent_search_and_distinct_paper_citations_are_not_deduplicated(self):
        clear_search_history()
        try:
            old = {"query": "enzyme molecular transformation", "sort_by_citations": True}
            record_search("search_openalex", old, {"status": "success"})
            self.assertIsNone(check_search_duplicate("search_openalex", {**old, "sort_by_citations": False}))
            record_search("follow_paper_citations", {"doi": "10.test/one", "query": "same target"}, {})
            self.assertIsNone(check_search_duplicate("follow_paper_citations", {"doi": "10.test/two", "query": "same target"}))
        finally:
            clear_search_history()

    def test_raw_retrieval_remains_readable_after_semantic_reranking_discards_a_paper(self):
        with tempfile.TemporaryDirectory() as d:
            raw = [{"title": "Old paper", "doi": "10.test/old", "source": "test"},
                   {"title": "Recent target application", "doi": "10.test/new", "source": "test"}]
            with patch.object(L, "_get_rerank_model", return_value=object()), \
                 patch.object(L, "_semantic_scores", return_value=[.9, .01]):
                kept = L._semantic_rerank("target", raw, 1, min_score=.2, run_dir=d)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(L._recover_literature_papers(d)), 2)
            trace = next((Path(d) / "artifacts/literature/retrieval").glob("*.json"))
            self.assertEqual(json.loads(trace.read_text())["funnel"], {"retrieved": 2, "retained": 1, "limit": 1})

    def test_citation_traversal_returns_and_persists_reference_papers(self):
        # Synthetic references exercise plumbing without supplying a biological answer.
        reference = {"doi": "10.test/child", "pmid": "123456",
                     "title": "Characterization of two enzymes", "abstract": "An enzyme transforms a substrate."}
        with patch.object(L, "request_with_retry", return_value=b'{"results": []}'), \
             patch.object(L, "_s2_snowball", return_value=[reference]), \
             patch.object(L, "_europepmc_snowball", return_value=[]), \
             patch.object(L, "_pubmed_snowball", return_value=[]), \
             patch.object(L, "search_pubmed") as keyword:
            result = L.follow_paper_citations(doi="10.test/parent", title="Parent study")
        keyword.assert_not_called()
        self.assertEqual(result["data"]["reference_papers"], [reference])
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "artifacts/literature"; directory.mkdir(parents=True)
            (directory / "citation.json").write_text(json.dumps({"result": result}))
            papers = L._recover_literature_papers(d)
            self.assertIn(reference, papers)

    def test_reading_continues_beyond_the_first_batch_without_new_search_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            papers = [{"title": f"Enzyme characterization {i}", "doi": f"10.test/{i}"} for i in range(7)]
            directory = root / "artifacts/literature"; directory.mkdir(parents=True)
            (directory / "search.json").write_text(json.dumps({"result": {"data": papers}}))
            def fetch(paper, dest, basename=None):
                return {"key": paper["doi"], "title": paper["title"], "text_file": None}
            with patch.object(F.config, "FULLTEXT_HARVEST_MAX_PAPERS", 3), \
                 patch.object(F.config, "LITERATURE_DISCOVERY_PAPERS_PER_STEP", 3), \
                 patch.object(F, "fetch_one", side_effect=fetch) as acquisition, \
                 patch("enzyme_discovery_agent.tools.literature_extraction.select_papers_to_read", side_effect=lambda run, ps, *args, **kwargs: ps), \
                 patch.object(F, "_register_named_enzymes_from_fulltext", return_value=[]):
                F.mine_and_register_fulltext_accessions(root, "target", llm_client=object())
                self.assertEqual(len(pending_reading(root, papers)), 4)
                F.mine_and_register_fulltext_accessions(root, "target", llm_client=object())
                self.assertEqual(len(pending_reading(root, papers)), 1)
                F.mine_and_register_fulltext_accessions(root, "target", llm_client=object())
            self.assertEqual(acquisition.call_count, 7)
            self.assertFalse(pending_reading(root, papers))

    def test_empty_model_extraction_preserves_explicit_sequence_identifier_leads(self):
        text = "Genes encoding ZyxQ (GenBank: AAA12345.1) were cloned, expressed and purified."
        self.assertEqual(sequence_name_leads(text)[0]["name"], "ZyxQ")
        self.assertFalse(sequence_name_leads("References: ZyxQ (GenBank: AAA12345.1)."))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            (dest / "paper.txt").write_text(text)
            metadata = {"key": "10.test/paper", "doi": "10.test/paper", "text_file": "paper.txt", "title": "Characterization"}
            llm = SimpleNamespace(budget_exhausted=False)
            registered = Mock()
            with patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", return_value={}), \
                 patch.object(F, "_resolve_embl_accession", return_value=["Q12345"]), \
                 patch("enzyme_discovery_agent.models.candidate_registry.update_candidate_registry") as update:
                results = F._register_named_enzymes_from_fulltext(root, "target", dest, [metadata], llm, registered, set(), 10)
            self.assertEqual(results, ["Q12345"])
            lead_updates = [call.args[3] for call in update.call_args_list if len(call.args) > 3 and "literature_identifier_lead" in call.args[3]]
            self.assertFalse(lead_updates[0]["literature_identifier_lead"]["activity_validated"])

    def test_extraction_backlog_is_not_limited_to_the_first_fifteen_papers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); dest = root / "artifacts/fulltext"; dest.mkdir(parents=True)
            manifest = [{"key": f"10.test/{i}", "text_file": f"{i}.txt"} for i in range(18)]
            for row in manifest: (dest / row["text_file"]).write_text("An enzyme was characterized.")
            save_reading_state(root, {"downloads": {}, "extractions": {row["key"]: {"status": "done", "attempts": 1} for row in manifest[:17]}})
            with patch("enzyme_discovery_agent.tools.literature_extraction.extract_from_paper", return_value={}) as extraction:
                F._register_named_enzymes_from_fulltext(root, "target", dest, manifest, SimpleNamespace(budget_exhausted=False), Mock(), set(), 10)
            self.assertEqual(extraction.call_count, 1)

    def test_related_review_record_resolves_its_explicit_parent_article(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); directory = root / "artifacts/literature"; directory.mkdir(parents=True)
            paper = {"doi": "10.test/review", "title": "Review for an experiment", "type": "peer-review",
                     "related_identifiers": {"is-review-of": [{"id-type": "doi", "id": "10.test/primary"}]}}
            (directory / "search.json").write_text(json.dumps({"result": {"data": [paper]}}))
            action = next_recovery_action(root, {"fetch_paper_metadata"})
            self.assertEqual(action["args"], {"doi": "10.test/primary"})
            with patch.object(L, "request_with_retry", return_value=json.dumps({"message": {
                "DOI": "10.test/primary", "title": ["Primary experiment"], "type": "journal-article"}}).encode()):
                result = L.fetch_paper_metadata(**action["args"])
            self.assertEqual(result["data"][0]["title"], "Primary experiment")
            self.assertEqual(result["status"], "success")

    def test_resume_retries_failed_routes_without_repeating_successful_work(self):
        from enzyme_discovery_agent.models.recovery import resume_failed_routes
        from enzyme_discovery_agent.models.literature_reading import reading_state
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); state = root / "agent_state"; state.mkdir()
            (state / "discovery_recovery.json").write_text(json.dumps({
                "failed": {"status": "error", "attempts": 2}, "done": {"status": "success", "attempts": 1}}))
            save_reading_state(root, {"downloads": {"10.test/unavailable": {"status": "unavailable", "attempts": 2}},
                                      "extractions": {"10.test/done": {"status": "done", "attempts": 1}}})
            resume_failed_routes(root)
            ledger = json.loads((state / "discovery_recovery.json").read_text())
            self.assertEqual(ledger["failed"]["attempts"], 0)
            self.assertEqual(ledger["failed"]["previous_attempts"], 2)
            self.assertEqual(ledger["done"], {"status": "success", "attempts": 1})
            self.assertEqual(reading_state(root)["downloads"]["10.test/unavailable"]["attempts"], 0)
            self.assertEqual(reading_state(root)["extractions"]["10.test/done"]["status"], "done")

    def test_title_only_retrieved_citation_gets_a_bounded_metadata_followup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'artifacts/literature';folder.mkdir(parents=True)
            paper={'pmid':'123456789','title':'Synthetic characterized transformation','abstract':''}
            (folder/'search.json').write_text(json.dumps({'result':{'data':[paper]}}))
            action=next_recovery_action(root,{'fetch_paper_metadata'})
            self.assertEqual(action['args'],{'pmid':'123456789'})
            with patch.object(L,'_efetch_pubmed_papers',return_value=[{**paper,'abstract':'Synthetic native chemistry is described.'}]) as fetch:
                result=L.fetch_paper_metadata(**action['args'])
            fetch.assert_called_once_with(['123456789'])
            self.assertEqual(result['data'][0]['abstract'],'Synthetic native chemistry is described.')


if __name__ == "__main__":
    unittest.main()
