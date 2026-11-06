"""Exercise review, checkpoint, resume and the modeling consumer together.

The provider and target derivation are synthetic; persistence, admission,
ranking, finalization and cross-agent handoff use their real implementations.
"""

import csv
import json
from unittest.mock import patch

import pytest

from enzyme_discovery_agent import config
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.claude_subscription import Response, Choice, Message, Usage
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.handoff_contract import load_document, validate_handoff
from enzyme_discovery_agent.request_efficiency import expand_evidence
from enzyme_discovery_agent.tests.test_target_review_gate import assessment
from structural_modeling_agent.agent import ModelingAgent


@pytest.mark.parametrize('factored', [False, True])
def test_review_to_checkpoint_resume_and_modeling_consumer(tmp_path, monkeypatch, factored):
    monkeypatch.setattr(config, 'TARGET_REVIEW_FACTOR_EVIDENCE', factored)
    monkeypatch.setattr(config, 'AUTO_TARGET_REVIEW', True)
    with patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning', return_value={}):
        agent = EnzymeAgent(model='claude-code:synthetic-model', output_dir=tmp_path, token_budget=1000)
        fact = {'observation': 'Synthetic evidence with unresolved target activity. ' * 30,
                'verdict': 'unknown'}
        update_candidate_registry(agent.run_dir, 'SYNTH1', 'evaluated', {
            'protein_name': 'Synthetic catalyst', 'sequence': 'ACDEFGHIK',
            'catalytic_activities': ['Synthetic native transformation'],
            'source_function_evidence': [fact] * 4,
            'claim_validation': {'promotion_eligible': True, 'claims': [
                {'claim_type': kind, 'subject': 'SYNTH1', 'verdict': 'supported'}
                for kind in ('accession_identity', 'record_active')]}})
        reviews = []

        def provider(backend, messages, tools, *args):
            names = {tool['function']['name'] for tool in tools or []}
            if 'record_target_assessment' in names:
                request = json.loads(messages[-1]['content'])
                candidates = expand_evidence(request)
                assert candidates[0]['accession'] == 'SYNTH1'
                assert candidates[0]['record']['source_function_evidence'] == [fact] * 4
                assert ('evidence_encoding' in request) == factored
                reviews.append(request)
                agent.token_budget = agent._run_token_usage.total_tokens + 10
                calls = [{'id': 'synthetic_review', 'type': 'function', 'function': {
                    'name': 'record_target_assessment', 'arguments': json.dumps(assessment())}}]
                return Response([Choice(Message(None, calls), 'tool_calls')], Usage(7, 3))
            # Initial identity/chemistry setup receives no invented evidence.
            assert len(reviews) == 0, 'Unexpected model work after the budget stop'
            return Response([Choice(Message('{}', None), 'stop')], Usage(7, 3))

        with patch.object(agent.llm, '_call_backend', side_effect=provider):
            agent.run('Synthetic transformation', substrate_smiles='CCO.CC=O')
        assert len(reviews) == 1
        root = agent.run_dir
        rows = list(csv.DictReader((root/'ranked_candidates.csv').open()))
        assert [row['accession'] for row in rows] == ['SYNTH1']
        handoff_path = root/'structural_modeling_handoff.json'
        handoff = validate_handoff(load_document(handoff_path))
        assert [row['accession'] for row in handoff['candidates']] == ['SYNTH1']
        checkpoint = json.loads((root/'agent_state/budget_checkpoint.json').read_text())
        assert checkpoint['token_overrun'] == 0
        assert checkpoint['token_accounting_complete']
        resumed = EnzymeAgent(resume_run_dir=root, token_budget=1)
        payload = resumed._load_resume_payload()
        assert payload['token_usage'].total_tokens == agent._run_token_usage.total_tokens
        saved_batches = list((root/'artifacts/target_reviews').glob('batch_*.json'))
        resumed.llm.resource_stop_reason = 'Synthetic resource boundary after resume'
        with patch.object(resumed.llm, '_call_backend', side_effect=AssertionError('Provider after resource stop')):
            resumed.run('Synthetic transformation', substrate_smiles='CCO.CC=O')
        assert list((root/'artifacts/target_reviews').glob('batch_*.json')) == saved_batches
        assert [row['accession'] for row in csv.DictReader((root/'ranked_candidates.csv').open())] == ['SYNTH1']
        consumer = ModelingAgent(task_path=handoff_path, run_dir=tmp_path/'consumer', llm_client=object())
        inspected = consumer.dispatch('inspect_candidate', {'accession': 'SYNTH1'})
        assert inspected
        assert 'Synthetic catalyst' in json.dumps(inspected)
        agent._close_logging()
        resumed._close_logging()
