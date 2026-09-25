"""Offline tests for startup and tool registration."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.bootstrap import register_builtin_tools
from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.output.finalizer import RunFinalizer
from enzyme_discovery_agent.output.report_generator import _first_gene_label
from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
from enzyme_discovery_agent.tools.structural_homolog_search import _parse_foldseek_results
from enzyme_discovery_agent.import_structural_evidence import structural_evidence_matrix
from enzyme_discovery_agent.models.target_profile import normalize_target_profile
from enzyme_discovery_agent.tool_registry import get_all_tools, get_parallel_safe_tools, get_tool_info
from enzyme_discovery_agent.tool_registry import execute_tool
from enzyme_discovery_agent.gates.ec_triage import _evidence_tier
from enzyme_discovery_agent.gates.ec_triage import ECTriageGate
from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.llm_client import LLMClient


class TestToolBootstrap(unittest.TestCase):
    def test_registration_is_idempotent_and_populates_schemas(self):
        register_builtin_tools()
        first = get_all_tools()
        register_builtin_tools()
        second = get_all_tools()

        self.assertGreater(len(first), 10)
        self.assertEqual(first, second)
        names = {tool["function"]["name"] for tool in first}
        self.assertIn("search_uniprot", names)
        self.assertIn("evaluate_candidate", names)
        self.assertIn("search_uniprot", get_parallel_safe_tools())

    def test_unknown_tool_arguments_are_rejected_before_execution(self):
        register_builtin_tools()
        result = execute_tool("search_uniprot", {"query": "amidase", "organism": "invalid"})
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid arguments", result["summary"])

    def test_empty_notebook_update_is_safely_skipped(self):
        register_builtin_tools()
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_tool(
                "update_notebook", {"section": "enzyme_landscape"},
                injected_args={"run_dir": tmp},
            )
        self.assertEqual(result["status"], "partial")
        self.assertIn("skipped", result["summary"])

    def test_uniprot_gene_name_search_does_not_require_free_text_query(self):
        register_builtin_tools()
        schema = get_tool_info("search_uniprot")["schema"]["function"]["parameters"]
        self.assertNotIn("query", schema["required"])
        self.assertIn("gene_name", schema["properties"])

    def test_evidence_tiers_are_explicit_and_non_composite(self):
        self.assertEqual(_evidence_tier({"direct_target_evidence": ["paper"]}), 0)
        self.assertEqual(_evidence_tier({"structural_homology": {"tm_score": 0.7}}), 1)
        self.assertEqual(_evidence_tier({}), 2)


class TestLLMClient(unittest.TestCase):
    def test_call_uses_native_per_request_timeout(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return "response"

        class FakeClient:
            def __init__(self):
                self.timeout = None
                self.completions = FakeCompletions()
                self.chat = type("Chat", (), {"completions": self.completions})()

            def with_options(self, *, timeout):
                self.timeout = timeout
                return self

        client = object.__new__(LLMClient)
        client.model = "test-model"
        client.client = FakeClient()

        response = client.call([{"role": "user", "content": "test"}], timeout=17, max_retries=1)

        self.assertEqual(response, "response")
        self.assertEqual(client.client.timeout, 17)

    def test_extract_tool_calls_passthrough(self):
        """Structured tool_calls are returned unchanged, recovered=False."""
        native = [SimpleNamespace(id="c1")]
        msg = SimpleNamespace(tool_calls=native, content="some reasoning")
        calls, content, recovered = LLMClient.extract_tool_calls(msg)
        self.assertIs(calls, native)
        self.assertEqual(content, "some reasoning")
        self.assertFalse(recovered)

    def test_extract_tool_calls_recovers_leaked_tool_use(self):
        """tool_use dicts leaked into text content are recovered and stripped."""
        leaked = (
            "I will search now. "
            "{'id': 'toolu_123', 'type': 'tool_use', 'name': 'search_uniprot', "
            "'input': {'query': 'nylon hydrolase'}}"
        )
        msg = SimpleNamespace(tool_calls=None, content=leaked)
        calls, content, recovered = LLMClient.extract_tool_calls(msg)
        self.assertTrue(recovered)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "search_uniprot")
        self.assertIn("nylon hydrolase", calls[0]["function"]["arguments"])
        # The leaked serialization is stripped from the human-readable content.
        self.assertNotIn("toolu_", content)
        self.assertIn("I will search now.", content)

    def test_extract_tool_calls_no_marker_is_passthrough(self):
        msg = SimpleNamespace(tool_calls=None, content="just prose, no tools")
        calls, content, recovered = LLMClient.extract_tool_calls(msg)
        self.assertFalse(recovered)
        self.assertEqual(calls, [])
        self.assertEqual(content, "just prose, no tools")


class TestStageStateSync(unittest.TestCase):
    def test_registry_exposes_human_readable_enzyme_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "Q79F77", "screened", metadata={
                "protein_name": "6-aminohexanoate-oligomer endohydrolase",
                "gene_names": ["nylC"],
            })
            entry = load_candidate_registry(run_dir)["candidates"]["Q79F77"]

        self.assertEqual(entry["enzyme_name"], "nylC")

    def test_sync_uses_registry_family_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "screened", metadata={
                "protein_name": "Example amidohydrolase",
            })
            agent = object.__new__(EnzymeAgent)
            agent.run_dir = run_dir
            agent._gate_state = SimpleNamespace()
            agent._accumulated_papers = []
            agent._search_log = []

            agent._sync_gate_state()

        self.assertEqual(agent._candidate_count, 1)
        self.assertEqual(agent._gate_state.candidate_count, 1)

    def test_sync_after_screen_makes_candidates_visible_before_next_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent = object.__new__(EnzymeAgent)
            agent.run_dir = run_dir
            agent._gate_state = SimpleNamespace()
            agent._accumulated_papers = []
            agent._search_log = []
            agent._candidate_count = 0
            update_candidate_registry(run_dir, "Q79F77", "screened", metadata={
                "protein_name": "6-aminohexanoate-oligomer endohydrolase",
                "gene_names": ["nylC"],
            })
            agent._sync_gate_state()

        self.assertEqual(agent._candidate_count, 1)


class TestScreenToEvaluationTransition(unittest.TestCase):
    def test_screened_candidates_trigger_evaluation_without_llm_eval_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(run_dir=Path(tmp), candidate_count=1)
            self.assertTrue(ECTriageGate().should_fire(state, has_eval_calls=False, messages=[]))


class TestRunFinalizer(unittest.TestCase):
    def test_load_evaluations_accepts_wrapped_and_legacy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            candidate_dir = run_dir / "artifacts" / "candidates"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "evaluate_candidate_001.json").write_text(
                json.dumps({"result": {"status": "success", "data": {"accession": "P1"}}})
            )
            (candidate_dir / "evaluate_candidate_002.json").write_text(
                json.dumps({"status": "success", "data": {"accession": "P2"}})
            )

            finalizer = RunFinalizer(run_dir, llm=None, log=lambda _: None)
            evaluations = finalizer._load_evaluations()

        self.assertEqual([item["data"]["accession"] for item in evaluations], ["P1", "P2"])

    def test_load_evaluations_excludes_function_claim_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            candidate_dir = run_dir / "artifacts" / "candidates"
            candidate_dir.mkdir(parents=True)
            for accession, eligible in (("SUPPORTED", True), ("BLOCKED", False)):
                (candidate_dir / f"evaluate_candidate_{accession}.json").write_text(json.dumps({
                    "status": "success",
                    "data": {
                        "accession": accession,
                        "is_enzyme": True,
                        "claim_validation": {
                            "promotion_eligible": eligible,
                            "blocking_claims": [] if eligible else ["substrate_topology"],
                        },
                    },
                }))

            evaluations = RunFinalizer(
                run_dir, llm=None, log=lambda _: None,
            )._load_evaluations()

        self.assertEqual([item["data"]["accession"] for item in evaluations], ["SUPPORTED"])

    def test_finalization_stage_writes_recovery_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            finalizer = RunFinalizer(Path(tmp), llm=None, log=lambda *_args, **_kwargs: None)
            finalizer._run_finalization_stage("test_stage", lambda: None)
            status = json.loads((Path(tmp) / "finalization_status.json").read_text())

        self.assertEqual(status["stages"]["test_stage"]["status"], "completed")

    def test_report_gene_label_tolerates_missing_annotations(self):
        self.assertEqual(_first_gene_label({"gene_names": []}), "?")
        self.assertEqual(_first_gene_label({}), "?")
        self.assertEqual(_first_gene_label({"gene_names": ["nylC"]}), "nylC")


class TestStructuralHandoff(unittest.TestCase):
    def test_handoff_preserves_target_and_structural_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            update_candidate_registry(run_dir, "P12345", "evaluated", metadata={
                "protein_name": "Example enzyme",
                "structural_homology": {"seed_accession": "Q99999", "tm_score": 0.71},
            })
            path = write_structural_modeling_handoff(
                run_dir,
                [{"data": {"accession": "P12345", "organism": "Example organism"}}],
                {"query": "target reaction", "substrate_smiles": "CCO"},
            )
            manifest = json.loads(path.read_text())

        self.assertEqual(manifest["target"]["substrate_smiles"], "CCO")
        self.assertEqual(
            manifest["candidates"][0]["evidence"]["structural_homology"]["tm_score"],
            0.71,
        )


class TestStructuralHomologParsing(unittest.TestCase):
    def test_parser_deduplicates_seed_and_orders_by_tm_score(self):
        payload = {"results": [{"db": "afdb50", "alignments": [[
            {"theader": "AF-P12345-F1 seed", "eval": "0.99"},
            {"theader": "AF-Q11111-F1 high", "eval": "0.72", "seqId": "0.23"},
            {"theader": "AF-Q22222-F1 low", "eval": "0.55"},
        ]]}]}
        hits = _parse_foldseek_results(payload, "P12345")
        self.assertEqual([hit["accession"] for hit in hits], ["Q11111", "Q22222"])
        self.assertEqual(hits[0]["tm_score"], 0.72)


class TestStructuralEvidenceRanking(unittest.TestCase):
    def test_matrix_keeps_observed_components_separate(self):
        matrix, status = structural_evidence_matrix({
            "catalytic_site": {"catalytic_center_score": 0.8, "pocket_engagement_score": None},
            "cofold": {"confidence_score": 0.6, "substrate_sasa_burial": 0.9},
        })
        self.assertEqual(status, "incomplete_structural_evidence")
        self.assertEqual(matrix["catalytic_center"]["value"], 0.8)
        self.assertFalse(matrix["pocket_engagement"]["observed"])
        self.assertEqual(matrix["ligand_burial"]["value"], 0.9)

    def test_profile_controls_required_checks_without_ranking(self):
        profile = normalize_target_profile({
            "required_evidence": ["ligand_burial"],
        })
        matrix, status = structural_evidence_matrix({
            "catalytic_site": {"catalytic_center_score": 0.1},
            "cofold": {"substrate_sasa_burial": 0.9},
        }, profile)
        self.assertEqual(status, "ready_for_review")
        self.assertFalse(matrix["catalytic_center"]["required"])
        self.assertEqual(matrix["ligand_burial"]["value"], 0.9)

    def test_profile_rejects_unknown_automatic_checks(self):
        with self.assertRaises(ValueError):
            normalize_target_profile({"required_evidence": ["mechanism_plausibility"]})

    def test_matrix_ignores_invalid_values(self):
        matrix, status = structural_evidence_matrix({
            "catalytic_site": {"catalytic_center_score": 0.8},
            "cofold": {"substrate_sasa_burial": "not a score"},
        }, {"required_evidence": ["ligand_burial"]})
        self.assertEqual(status, "incomplete_structural_evidence")
        self.assertFalse(matrix["ligand_burial"]["observed"])
