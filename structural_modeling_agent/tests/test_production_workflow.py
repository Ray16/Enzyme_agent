"""Production stages share one allowance and do not use a model for job polling."""
import copy
import json

import pytest

from structural_modeling_agent import production


class Agent:
    def __init__(self, folder):
        self.run_dir = folder
        self.task = {'workflow_mode': 'production_cofold', 'candidates': [{'accession': 'SOURCE:SYNTH'}]}
        self.state = {'status': 'initialized', 'usage': {'calls': [], 'total_tokens': 0},
                      'candidates': {'SOURCE:SYNTH': {'revisions': []}}}
        self.run_calls = []

    def _save(self):
        (self.run_dir / 'checkpoint.json').write_text('{}')

    def _write_report(self):
        return copy.deepcopy(self.state)

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        usage = self.state['usage']
        usage['calls'].append({'total_tokens': 25})
        usage['total_tokens'] += 25
        if len(self.run_calls) == 1:
            self.state['candidates']['SOURCE:SYNTH']['revisions'] = [{
                'execution': {'saved_input': True}, 'modeling_input': {'mechanism': 'synthetic redox'},
                'scientific_reasoning': {'selected_hypothesis_id': 'H7'},
                'rationale': 'Independent synthetic evidence', 'alternatives': []}]
            self.state['status'] = 'ready_for_native_execution'
        else:
            assert self.state['status'] in {'ensemble_ready_for_interpretation','review_existing_predictions'}
            self.state['status'] = 'inconclusive'
        self._save()
        return self._write_report()


def setup(tmp_path, monkeypatch):
    agent = Agent(tmp_path)
    monkeypatch.setattr(production, 'ModelingAgent', lambda **kwargs: agent)
    requests = []
    def execute(request):
        requests.append(request)
        proposal = json.loads(type(tmp_path)(request['proposal_path']).read_text())
        assert proposal['scientific_reasoning']['selected_hypothesis_id'] == 'H7'
        assert proposal['modeling_input']['mechanism'] == 'synthetic redox'
        agent.state['candidates']['SOURCE:SYNTH']['revisions'][0]['pose_validation'] = {'poses': [{'path': 'synthetic'}]}
        agent.state['status'] = 'ensemble_ready_for_interpretation'
        return {'status': 'ensemble_ready_for_interpretation'}
    monkeypatch.setattr(production, 'execute', execute)
    return agent, requests


def test_agent_selected_model_executes_and_returns_for_scientific_interpretation(tmp_path, monkeypatch):
    agent, requests = setup(tmp_path, monkeypatch)
    result = production.run_candidate(task_path='task.json', run_dir=tmp_path, backend_path='backend.json',
        execution_intent='submit_after_checks_pass', token_budget=100, max_turns=4)
    assert result['status'] == 'inconclusive'
    assert len(requests) == 1
    assert len(agent.run_calls) == 2
    assert agent.run_calls[1]['token_budget'] == 75
    assert agent.run_calls[1]['max_turns'] == 3
    assert result['usage']['total_tokens'] == 50


def test_exhausted_model_allowance_preserves_completed_native_measurements(tmp_path, monkeypatch):
    agent, requests = setup(tmp_path, monkeypatch)
    result = production.run_candidate(task_path='task.json', run_dir=tmp_path, backend_path='backend.json',
        execution_intent='submit_after_checks_pass', token_budget=25, max_turns=4)
    assert result['status'] == 'budget_exhausted'
    assert len(requests) == 1 and len(agent.run_calls) == 1
    assert result['candidates']['SOURCE:SYNTH']['revisions'][0]['pose_validation']['poses']
    production.run_candidate(task_path='task.json', run_dir=tmp_path, backend_path='backend.json',
        execution_intent='submit_after_checks_pass', token_budget=25, max_turns=4)
    assert len(requests) == 1 and len(agent.run_calls) == 1


def test_stalled_setup_does_not_submit_or_restart_itself(tmp_path, monkeypatch):
    agent, requests = setup(tmp_path, monkeypatch)
    agent.state['status'] = 'setup_stalled'
    assert production.run_candidate(task_path='task.json', run_dir=tmp_path, backend_path='backend.json',
        execution_intent='submit_after_checks_pass')['status'] == 'setup_stalled'
    assert requests == [] and agent.run_calls == []


def test_execution_authorization_is_explicit(tmp_path):
    with pytest.raises(ValueError, match='Declare whether'):
        production.run_candidate(task_path='task.json', run_dir=tmp_path, backend_path='backend.json',
                                 execution_intent='unspecified')


def test_existing_predictions_return_to_agent_review_without_a_new_execution_loop(tmp_path, monkeypatch):
    agent, requests = setup(tmp_path, monkeypatch)
    def reuse(request):
        requests.append(request)
        return {'status':'review_existing_predictions'}
    monkeypatch.setattr(production,'execute',reuse)
    result=production.run_candidate(task_path='task.json',run_dir=tmp_path,backend_path='backend.json',
        execution_intent='submit_after_checks_pass',token_budget=100,max_turns=4)
    assert result['status']=='inconclusive'
    assert result['reuse_review_required']
    assert len(requests)==1 and len(agent.run_calls)==2


def test_real_controller_crosses_execution_stage_without_provider_polling_turns(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import subprocess
    from test_modeling_agent import Client, propose
    from test_validation import modeling_record, write_pose
    from structural_modeling_agent.validation import file_digest

    record = modeling_record(tmp_path)
    msa = tmp_path / 'query.csv'; msa.write_text('key,sequence\n-1,AG\n-1,AA\n')
    task = tmp_path / 'task.json'
    task.write_text(json.dumps({'schema_version': '1.0', 'objective': 'model_and_validate',
        'workflow_mode': 'production_cofold', 'target': {'reaction': 'synthetic transformation'},
        'evidence_paths': [str(tmp_path / 'reference.txt'), str(msa)],
        'candidates': [{'accession': 'SYNTHETIC1', 'sequence': 'AG', 'evidence': {'msa_path': str(msa)}}]}))
    backend = tmp_path / 'backend.json'
    backend.write_text(json.dumps({'provider': 'synthetic-provider', 'engine': 'boltz2',
        'argv': ['synthetic-adapter'], 'resource_policy': {'reservation_managed': True,
        'max_jobs': 1, 'max_predictions_per_job': 1}}))
    client = Client([[("propose_model", propose(record))],
                     [("render_inputs", {'accession': 'SYNTHETIC1'})],
                     [("finish_task", {'outcome': 'inconclusive',
                                       'rationale': 'The synthetic ensemble is measured; biological applicability remains unknown.'})]])
    original_run = subprocess.run
    operations = []
    def adapter(argv, **kwargs):
        if argv != ['synthetic-adapter']:
            return original_run(argv, **kwargs)
        request = json.loads(kwargs['input']); operations.append(request['operation'])
        folder = tmp_path / 'predictions'; folder.mkdir(exist_ok=True)
        pose = write_pose(folder / 'synthetic_model_0.pdb')
        (folder / 'confidence_synthetic_model_0.json').write_text('{}')
        manifest = tmp_path / 'producer.json'
        manifest.write_text(json.dumps({'input_sha256': request['input_sha256'],
            'execution_sha256': request['execution_sha256'],
            'predictions': [{'path': str(pose), 'sha256': file_digest(pose)}]}))
        return SimpleNamespace(stdout=json.dumps({'idempotency_key': request['idempotency_key'],
            'status': 'complete', 'prediction_dir': str(folder), 'prediction_manifest': str(manifest)}))
    monkeypatch.setattr('structural_modeling_agent.agent.subprocess.run', adapter)
    result = production.run_candidate(task_path=task, run_dir=tmp_path / 'model', backend_path=backend,
        execution_intent='submit_after_checks_pass', llm_client=client, token_budget=1000, max_turns=8)
    assert result['status'] == 'inconclusive'
    assert result['candidates'][0]['pose_validation']['n_predictions'] == 1
    assert len(result['usage']['calls']) == 3
    assert operations == ['submit']
    assert (tmp_path / 'model/execution_status.json').exists()
