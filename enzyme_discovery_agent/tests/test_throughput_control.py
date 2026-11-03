"""Synthetic queue/control tests; no biological discovery or provider calls."""
import copy
import json
from unittest.mock import Mock, patch
from enzyme_discovery_agent.agent import EnzymeAgent
from enzyme_discovery_agent.llm_client import LLMClient
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry, discovery_status
from enzyme_discovery_agent.context import compact_tool_definitions


def test_fresh_backlog_drains_before_any_breadth_or_model_call(tmp_path):
    with patch('enzyme_discovery_agent.agent.register_builtin_tools'), \
         patch('enzyme_discovery_agent.agent.get_all_tools', return_value=[]), \
         patch('enzyme_discovery_agent.models.target_reasoning.derive_target_reasoning', return_value={}), \
         patch('enzyme_discovery_agent.models.recovery.next_recovery_action', return_value=None), \
         patch('enzyme_discovery_agent.gates.ec_triage._extract_target_params', return_value={}):
        agent = EnzymeAgent(model='claude-code:claude-sonnet-5-5', output_dir=tmp_path, token_budget=100)
        for i in range(65):
            update_candidate_registry(agent.run_dir, f'SYNTH{i}', 'screened', {'gene_names': [f'group{i%5}']})
        performed = []
        def evaluate(run_dir, accession, **kwargs):
            performed.append(accession)
            update_candidate_registry(run_dir, accession, 'evaluated', {'claim_validation': {'promotion_eligible': True}})
            return {'status': 'success', 'data': {'accession': accession, 'claim_validation': {'promotion_eligible': True}}}
        def no_model(*args, **kwargs):
            assert len(performed) == 65
            assert all(discovery_status(e) == 'evaluated' for e in load_candidate_registry(agent.run_dir)['candidates'].values())
            agent.llm.resource_stop_reason = 'Synthetic validation complete'
            return None
        for gate in (agent._literature_harvest_gate, agent._mechanism_family_gate, agent._family_survey_gate,
                     agent._systematic_gate, agent._gate_pipeline.gates[3]):
            gate.should_fire = Mock(side_effect=lambda *a, **k: False if len(performed)==65 else (_ for _ in ()).throw(AssertionError('breadth before evaluation')))
        with patch('enzyme_discovery_agent.tools.reasoning.evaluate_candidate', side_effect=evaluate), \
             patch.object(agent.llm, 'call', side_effect=no_model) as model, \
             patch('enzyme_discovery_agent.gates.ec_triage.auto_screen_candidates') as screen:
            agent.run('synthetic request', substrate_smiles='CCO')
        assert len(performed) == len(set(performed)) == 65
        screen.assert_not_called()
        assert model.call_count == 1


def test_compact_tools_preserve_validation_contracts():
    tools = [{'type':'function','function':{'name':'example','description':'x'*1000,'parameters':{
        'type':'object','properties':{'mode':{'type':'string','enum':['a','b'],'description':'y'*1000}},'required':['mode']}}}]
    original=copy.deepcopy(tools)
    small=compact_tool_definitions(tools)
    assert tools == original
    assert small[0]['function']['parameters']['required'] == ['mode']
    assert small[0]['function']['parameters']['properties']['mode']['enum'] == ['a','b']
    assert len(json.dumps(small)) < len(json.dumps(tools))/2
