"""Offline regressions for target scope, recovery and shared evidence state."""

import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry, update_candidate_registry, discovery_status,
)
from enzyme_discovery_agent.models.claim_validation import validate_uniprot_claims
from enzyme_discovery_agent.models.literature_evidence import build_literature_evidence
from enzyme_discovery_agent.output.finalizer import RunFinalizer
from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
from enzyme_discovery_agent.tools.reasoning import evaluate_candidate
from enzyme_discovery_agent.state_store import atomic_write_json, write_tool_artifact
from enzyme_discovery_agent.tools import mechanism_families as mf


def _writer(root, worker, cpu):
    os.sched_setaffinity(0, {cpu})
    from enzyme_discovery_agent.models.event_log import append_event
    for index in range(6):
        update_candidate_registry(Path(root), "P12345", "screened",
                                  metadata={f"worker_{worker}_{index}": index})
        append_event(root, "test_update", worker=worker, index=index)
        write_tool_artifact(root, "literature", "search_pubmed",
                            {"worker": worker, "index": index})


class _Client:
    def __init__(self):
        self.chat = self
        self.completions = self

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        message = SimpleNamespace(content="finished", tool_calls=None)
        message.model_dump = lambda: {"role": "assistant", "content": "finished"}
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))


class TestSharedState(unittest.TestCase):
    def test_process_updates_preserve_metadata_events_and_artifacts(self):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2 or os.getloadavg()[0] + 2 > os.cpu_count():
            self.skipTest("two CPU slots required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "artifacts/literature/search_pubmed_900.json"
            atomic_write_json(old, {"original": True})
            ctx = multiprocessing.get_context("spawn")
            workers = [ctx.Process(target=_writer, args=(tmp, i, cpus[i])) for i in range(2)]
            for worker in workers:
                worker.start()
            try:
                for worker in workers:
                    worker.join(30)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
            entry = load_candidate_registry(root)["candidates"]["P12345"]
            self.assertEqual(len(entry["statuses"]), 12)
            self.assertEqual(len(entry["metadata"]), 12)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len({event["seq"] for event in events}), 12)
            self.assertEqual([e["seq"] for e in events], sorted(e["seq"] for e in events))
            artifacts = list(old.parent.glob("search_pubmed_*.json"))
            self.assertEqual(len(artifacts), 13)
            self.assertEqual(json.loads(old.read_text()), {"original": True})

    def test_failed_atomic_publish_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            atomic_write_json(path, {"version": 1})
            with patch("enzyme_discovery_agent.state_store.os.replace", side_effect=OSError("failed")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"version": 2})
            self.assertEqual(json.loads(path.read_text()), {"version": 1})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_corrupt_registry_cannot_be_overwritten_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidate_registry.json").write_text("{")
            with self.assertRaises(json.JSONDecodeError):
                update_candidate_registry(root, "P12345", "discovered")
            self.assertEqual((root / "candidate_registry.json").read_text(), "{")

    def test_modeling_update_preserves_discovery_progress(self):
        entry = {"statuses": [{"status": "screened"}, {"status": "evaluated"},
                              {"status": "structurally_evaluated"}]}
        self.assertEqual(discovery_status(entry), "evaluated")


class TestCurrentEligibility(unittest.TestCase):
    def setUp(self):
        guard=patch('enzyme_discovery_agent.models.target_review.eligible',return_value=True)
        guard.start();self.addCleanup(guard.stop)
    def test_later_rejection_excludes_artifact_and_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation = {"status": "success", "data": {"accession": "P12345", "is_enzyme": True}}
            write_tool_artifact(root, "candidates", "evaluate_candidate", {"result": evaluation})
            update_candidate_registry(root, "P12345", "evaluated", {"claim_validation": {
                "promotion_eligible": True, "claims": [{"claim_type": "accession_identity", "subject": "P12345", "verdict": "supported", "evidence": {"source": "UniProtKB"}}]}})
            original = write_structural_modeling_handoff(root, [evaluation], {"query": "target"})
            self.assertEqual(len(json.loads(original.read_text())["candidates"]), 1)
            update_candidate_registry(root, "P12345", "rejected")
            self.assertEqual(RunFinalizer(root, None, lambda *a: None)._load_evaluations(), [])
            path = write_structural_modeling_handoff(root, [evaluation], {"query": "target"})
            self.assertEqual(json.loads(path.read_text())["candidates"], [])

    def test_preexisting_report_is_not_counted_as_fresh_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "report_confidence.md"
            path.write_text("earlier report")
            finalizer = RunFinalizer(root, None, lambda *a, **k: None)
            finalizer._run_finalization_stage("confidence_report", lambda: None, produces=path)
            status = json.loads((root / "finalization_status.json").read_text())
            self.assertEqual(status["stages"]["confidence_report"]["status"], "incomplete")

    def test_latest_artifact_failure_does_not_resurrect_old_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for eligible in (True, False):
                write_tool_artifact(root, "candidates", "evaluate_candidate", {"result": {
                    "status": "success", "data": {"accession": "P12345", "is_enzyme": True,
                    "claim_validation": {"promotion_eligible": eligible}},
                }})
            self.assertEqual(RunFinalizer(root, None, lambda *a: None)._load_evaluations(), [])


class TestScope(unittest.TestCase):
    def test_carrier_family_is_target_dependent(self):
        family = {"substrates_free": False, "native_direction": "formation",
                  "predicted_substrate_topology": "carrier_protein_tethered"}
        self.assertTrue(mf.family_passes_constraints(family, "formation", "carrier_protein_tethered")[0])
        self.assertFalse(mf.family_passes_constraints(family, "formation", "free_small_molecule")[0])
        self.assertTrue(mf.family_passes_constraints(family, "formation", None)[0])

    def test_redox_consuming_water_is_not_misclassified_as_cleavage(self):
        family = {"native_direction": "redox", "native_reaction": "substrate + water -> product"}
        self.assertEqual(mf._family_native_direction(family), "redox")

    def test_unknown_topology_is_not_a_contradiction(self):
        self.assertTrue(mf.family_passes_constraints(
            {"predicted_substrate_topology": "unknown", "native_direction": "formation"},
            "formation", "carrier_protein_tethered")[0])

    def test_cache_does_not_reuse_different_topology_or_direction(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ENZYME_AGENT_CACHE_DIR": tmp}):
            family = {"name": "Example fold", "substrates_free": True}
            with patch.object(mf, "_derive_once", return_value=[family]) as derive:
                mf.derive_mechanism_families("target", "CCO", object(), reflect=False,
                                            target_direction="formation", target_topology="free_small_molecule")
                before = derive.call_count
                mf.derive_mechanism_families("target", "CCO", object(), reflect=False,
                                            target_direction="cleavage", target_topology="polymer")
                self.assertGreater(derive.call_count, before)


class TestUsageAndResume(unittest.TestCase):
    def _client(self):
        with patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=_Client()), \
             patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                 {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]):
            return LLMClient(model="test")

    def test_internal_calls_are_counted_once_and_checkpointed(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage = TokenUsage()
            client = self._client()
            client.attach_usage(usage, tmp)
            response = client.call([{"role": "user", "content": "internal extraction"}])
            usage.record("legacy_callback", response)
            client.call([{"role": "user", "content": "discovery"}], usage_label=1)
            self.assertEqual(usage.total_tokens, 20)
            self.assertEqual(len(usage.calls), 2)
            self.assertTrue(all(call.get("backend") == "test" for call in usage.calls))
            self.assertEqual(json.loads((Path(tmp) / "token_usage.json").read_text())["total_tokens"], 20)

    def test_resume_restores_literature_exact_gate_state_and_full_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_write_json(root / "manifest.json", {"status": "running"})
            atomic_write_json(root / "target_context.json", {"query": "target", "target_profile": {}})
            atomic_write_json(root / "agent_state/transcript.json", {
                "iteration": 4, "messages": [], "gate_state": {"eval_gate_fired": False},
                "family_survey_state": {"_attempts": 2, "_surveyed_folds": ["IPR000001"]},
            })
            write_tool_artifact(root, "literature", "search_pubmed", {
                "iteration": 3, "result": {"data": [{"title": "A source", "doi": "10.test/source"}]}})
            usage = TokenUsage()
            client = self._client()
            client.attach_usage(usage, root)
            client.call([{"role": "user", "content": "internal"}])
            agent = EnzymeAgent.__new__(EnzymeAgent)
            agent.run_dir = root
            agent.model_name = "test"
            agent._resume_requested = True
            agent._gate_state = AgentState(root)
            agent._family_survey_gate = FamilySurveyGate()
            restored = agent._load_resume_payload()
            self.assertEqual(restored["token_usage"].total_tokens, 10)
            self.assertEqual(len(agent._accumulated_papers), 1)
            self.assertEqual(agent._literature_search_iterations, {3})
            self.assertFalse(agent._gate_state.eval_gate_fired)
            self.assertEqual(agent._family_survey_gate._surveyed_folds, {"IPR000001"})

    def test_missing_provider_usage_is_explicit_and_not_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client()
            usage = TokenUsage()
            client.attach_usage(usage, tmp)
            response = SimpleNamespace(usage=None)
            with patch.object(client, "_call_backend", return_value=response):
                client.call([])
            usage.record("legacy_callback", response)
            self.assertEqual(usage.unmetered_calls, 1)
            self.assertFalse(usage.to_dict()["accounting_complete"])
            restored = TokenUsage.from_dict(json.loads((Path(tmp) / "token_usage.json").read_text()))
            self.assertEqual(restored.unmetered_calls, 1)

    def test_legacy_usage_remains_marked_partial_after_restore(self):
        restored = TokenUsage.from_dict({"calls": [], "total_prompt_tokens": 7,
                                        "total_completion_tokens": 3, "total_tokens": 10})
        self.assertEqual(restored.total_tokens, 10)
        self.assertFalse(restored.to_dict()["accounting_complete"])

    def test_legacy_turn_hint_does_not_end_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client()
            with patch("enzyme_discovery_agent.agent.LLMClient", LLMClient), \
                 patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=_Client()), \
                 patch("enzyme_discovery_agent.llm_client.config.llm_backends", return_value=[
                     {"name": "test", "model": "test", "base_url": "unused", "api_key": "unused"}]), \
                 patch("enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning", return_value={}), \
                 patch("enzyme_discovery_agent.agent.RunFinalizer.finalize", return_value=("done", 0)) as finalize:
                agent = EnzymeAgent(model="test", output_dir=tmp, max_iterations=1)
                result = agent.run("target", substrate_smiles="CCO")
                self.assertIn("Discovery incomplete", result)
                finalize.assert_not_called()
                self.assertGreater(agent.llm.request_counts["test"], 1)


class TestLiteratureClaims(unittest.TestCase):
    def _evidence(self, root):
        quote = "Example enzyme hydrolyzes an ester bond in the tested substrate."
        text = root / "paper.txt"
        text.write_text("Example enzyme from Example organism. " + quote)
        record = {"name": "Example enzyme", "organism": "Example organism", "evidence_quote": quote,
                  "reaction": {"direction": "cleavage", "bond_formed": "ester"}}
        return build_literature_evidence("P12345", record, {"doi": "10.test/source"}, text,
                                         identity_basis="resolved_name_and_organism")

    def _validate(self, evidence):
        return validate_uniprot_claims(
            {"primaryAccession": "P12345", "entryType": "UniProtKB unreviewed (TrEMBL)",
             "organism": {"scientificName": "Example organism"},
             "proteinDescription": {"recommendedName": {"fullName": {"value": "Example enzyme"}}}},
            expected_accession="P12345", target_direction="cleavage", target_bond_types={"ester"},
            literature_evidence=evidence)

    def test_primary_literature_fills_missing_native_reaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self._evidence(Path(tmp))
            validation = self._validate(evidence)
            self.assertTrue(validation["promotion_eligible"], validation)
            supported = [claim for claim in validation["claims"]
                         if claim["claim_type"] in {"native_reaction", "reaction_direction", "bond_type"}]
            self.assertTrue(all(c["evidence"]["source"] == "primary_literature" for c in supported))

    def test_changed_source_and_wrong_candidate_do_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self._evidence(Path(tmp))
            evidence["accession"] = "Q12345"
            self.assertFalse(self._validate(evidence)["promotion_eligible"])
            evidence["accession"] = "P12345"
            Path(evidence["text_path"]).write_text("changed source")
            self.assertFalse(self._validate(evidence)["promotion_eligible"])

    def test_literature_cannot_override_recorded_opposite_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self._evidence(Path(tmp))
            entry = {"primaryAccession": "P12345", "entryType": "UniProtKB reviewed (Swiss-Prot)",
                     "proteinDescription": {"recommendedName": {"ecNumbers": [{"value": "6.3.2.1"}]}},
                     "comments": [{"commentType": "CATALYTIC ACTIVITY", "reaction": {
                         "name": "acid + amine + ATP = product + ADP + phosphate"}}]}
            validation = validate_uniprot_claims(entry, expected_accession="P12345",
                target_direction="cleavage", literature_evidence=evidence)
            self.assertFalse(validation["promotion_eligible"])
            self.assertIn("reaction_direction", validation["blocking_claims"])

    def test_evaluation_promotes_source_grounded_protein_and_keeps_unknown_pending(self):
        from contextlib import ExitStack
        from enzyme_discovery_agent.tools import reasoning
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            evidence = self._evidence(root)
            atomic_write_json(root / "target_reasoning.json", {
                "reaction_direction": "cleavage", "bond_types": ["ester"],
                "substrate_topology": "free_small_molecule",
            })
            entry = {"primaryAccession": "P12345", "entryType": "UniProtKB unreviewed (TrEMBL)",
                     "sequence": {"length": 200}, "organism": {"scientificName": "Example organism"},
                     "proteinDescription": {"recommendedName": {"fullName": {"value": "Example enzyme"}}}}
            stack.enter_context(patch.object(reasoning, "_fetch_uniprot_entry", return_value=entry))
            stack.enter_context(patch.object(reasoning, "_count_pubmed_characterization", return_value={
                "characterization_papers": 1, "engineering_papers": 0}))
            stack.enter_context(patch.object(reasoning, "_check_pdb_availability", return_value={"pdb_ids": []}))
            stack.enter_context(patch.object(reasoning, "_check_bacterial_expression", return_value={}))
            stack.enter_context(patch.object(reasoning, "_fallback_substrate_lookup", return_value=[]))
            stack.enter_context(patch("enzyme_discovery_agent.tools.database.get_protein_domains", return_value={"data": []}))
            update_candidate_registry(root, "P12345", "screened")
            unknown = evaluate_candidate(root, "P12345", target_reaction_type="ester hydrolysis")
            self.assertFalse(unknown["data"]["claim_validation"]["promotion_eligible"])
            reg = load_candidate_registry(root)["candidates"]["P12345"]
            self.assertEqual(reg["statuses"][-1]["status"], "evidence_pending")
            update_candidate_registry(root, "P12345", None, {"literature_evidence": evidence})
            supported = evaluate_candidate(root, "P12345", target_reaction_type="ester hydrolysis")
            self.assertTrue(supported["data"]["claim_validation"]["promotion_eligible"])
            reg = load_candidate_registry(root)["candidates"]["P12345"]
            self.assertEqual(reg["statuses"][-1]["status"], "evaluated")
            self.assertEqual(supported["data"]["reaction_chemistry"]["source"], "primary_literature")


def test_literature_quote_does_not_establish_a_sequence_by_assertion_only(tmp_path):
    from enzyme_discovery_agent.models.literature_evidence import build_literature_evidence, verify_literature_evidence
    source = tmp_path / 'paper.txt'
    quote = 'Example enzyme hydrolyzes an ester bond in the tested substrate.'
    source.write_text('Example enzyme from Example organism. ' + quote)
    record = {'name': 'Example enzyme', 'organism': 'Example organism', 'evidence_quote': quote,
              'reaction': {'direction': 'cleavage', 'bond_formed': 'ester'}}
    evidence = build_literature_evidence('P12345', record, {'doi': '10.test/source'}, source,
                                         identity_basis='resolved_name_and_organism')
    mismatch = {'primaryAccession': 'P12345', 'organism': {'scientificName': 'Example organism'},
                'proteinDescription': {'recommendedName': {'fullName': {'value': 'Unrelated enzyme'}}}}
    assert not verify_literature_evidence(evidence, 'P12345', mismatch)
    mismatch['proteinDescription']['recommendedName']['fullName']['value'] = 'Example enzyme'
    assert verify_literature_evidence(evidence, 'P12345', mismatch)
    mismatch['organism']['scientificName'] = 'Other organism'
    assert not verify_literature_evidence(evidence, 'P12345', mismatch)
    mismatch['organism']['scientificName'] = 'Example organism'
    source.write_text('Example organism is discussed. ' + quote)
    evidence = build_literature_evidence('P12345', record, {'doi': '10.test/source'}, source,
                                         identity_basis='resolved_name_and_organism')
    assert not verify_literature_evidence(evidence, 'P12345', mismatch)


def test_unverified_literature_substrate_does_not_enter_fit(tmp_path):
    from contextlib import ExitStack
    from enzyme_discovery_agent.tools import reasoning
    entry = {'primaryAccession': 'P12345', 'entryType': 'UniProtKB unreviewed (TrEMBL)',
             'sequence': {'length': 200}, 'organism': {'scientificName': 'Example organism'},
             'proteinDescription': {'recommendedName': {'fullName': {'value': 'Unrelated enzyme'}}}}
    with ExitStack() as stack:
        stack.enter_context(patch.object(reasoning, '_fetch_uniprot_entry', return_value=entry))
        stack.enter_context(patch.object(reasoning, '_count_pubmed_characterization', return_value={'characterization_papers': 0, 'engineering_papers': 0}))
        stack.enter_context(patch.object(reasoning, '_check_pdb_availability', return_value={'pdb_ids': []}))
        stack.enter_context(patch.object(reasoning, '_check_bacterial_expression', return_value={}))
        stack.enter_context(patch.object(reasoning, '_fallback_substrate_lookup', return_value=[]))
        stack.enter_context(patch('enzyme_discovery_agent.tools.database.get_protein_domains', return_value={'data': []}))
        stack.enter_context(patch('enzyme_discovery_agent.tools.literature_extraction.extract_substrates_and_mechanism', return_value={
            'paper_relevant': True, 'native_substrates': [{'name': 'unverified', 'smiles': 'CCO'}], 'literature_evidence': None}))
        result = reasoning.evaluate_candidate(tmp_path, 'P12345', target_substrate_smiles='CCO', llm_client=object())
    assert all(item.get('source') != 'literature' for item in (result.get('data') or {}).get('native_substrates', []))


def test_source_linked_natural_language_reaction_is_normalized_without_target_activity_claim(tmp_path):
    from enzyme_discovery_agent.models.literature_evidence import build_literature_evidence
    quote = 'Example enzyme catalyzes amidation of substrates to give an amide product.'
    source = tmp_path / 'paper.txt'
    source.write_text('Example enzyme from Example organism. ' + quote)
    record = {'name': 'Example enzyme', 'organism': 'Example organism', 'evidence_quote': quote,
              'reaction': {'direction': 'forward (formation)', 'bond_formed': 'amide bond between partners'}}
    evidence = build_literature_evidence('P12345', record, {'doi': '10.test/source'}, source,
                                         identity_basis='resolved_name_and_organism')
    entry = {'primaryAccession': 'P12345', 'entryType': 'UniProtKB reviewed (Swiss-Prot)',
             'organism': {'scientificName': 'Example organism'},
             'proteinDescription': {'recommendedName': {'fullName': {'value': 'Example enzyme'}}}}
    result = validate_uniprot_claims(entry, expected_accession='P12345', target_direction='formation',
                                     target_bond_types={'amide'}, literature_evidence=evidence)
    claims = {c['claim_type']: c for c in result['claims']}
    assert claims['native_reaction']['verdict'] == 'supported'
    assert claims['reaction_direction']['verdict'] == 'supported'
    assert claims['bond_type']['verdict'] == 'supported'
    assert 'target_specific_activity' not in claims


def test_shared_nucleotide_accession_is_not_proof_of_protein_identity(tmp_path):
    from enzyme_discovery_agent.models.literature_evidence import build_literature_evidence, verify_literature_evidence
    source = tmp_path / 'paper.txt'
    quote = 'Example enzyme (GenBank: AAA12345) hydrolyzes an ester bond.'
    source.write_text(quote)
    record = {'name': 'Example enzyme', 'genbank': 'AAA12345', 'evidence_quote': quote,
              'reaction': {'direction': 'cleavage', 'bond_formed': 'ester'}}
    evidence = build_literature_evidence('P12345', record, {'doi': '10.test/source'}, source,
                                         identity_basis='deposited_accession')
    entry = {'primaryAccession': 'P12345', 'uniProtKBCrossReferences': [
        {'database': 'EMBL', 'id': 'AAA12345', 'properties': []}]}
    assert not verify_literature_evidence(evidence, 'P12345', entry)
    entry['uniProtKBCrossReferences'][0]['properties'] = [{'key': 'ProteinId', 'value': 'AAA12345'}]
    assert verify_literature_evidence(evidence, 'P12345', entry)
