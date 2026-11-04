"""Modern modeled systems retain source-scoped identity and scientific uncertainty."""
import csv
import json

import pytest

from campaign_catalog.catalog import build
from campaign_catalog.lookup import lookup_predictions
import test_catalog


def test_completed_inconclusive_model_is_reusable_evidence_for_a_source_scoped_identity(tmp_path, monkeypatch):
    test = test_catalog.CatalogTest()
    campaign = test.fixture(tmp_path)
    registry = tmp_path / 'current/candidate_registry.json'
    data = json.loads(registry.read_text())
    data['candidates']['source:CAT7'] = test.entry('source:CAT7', 'evidence_pending', '2026-02-01')
    test.put(registry, data)
    model = tmp_path / 'new_model'; model.mkdir()
    input_path = model / 'input.yaml'
    input_path.write_text('sequences:\n- protein:\n    id: X\n    sequence: AGGA\n')
    pose = model / 'synthetic_model_0.pdb'; pose.write_text('synthetic coordinate artifact\n')
    test.put(model / 'confidence_synthetic_model_0.json', {'confidence': .5})
    report = test.put(model / 'report.json', {'status': 'inconclusive',
        'decision_origin': {'author': 'synthetic agent'}, 'candidates': [{
            'accession': 'source:CAT7', 'input_yaml': str(input_path),
            'execution_sha256': 'synthetic-fingerprint',
            'modeling_input': {'validation_plan': {'poses': {'expected_predictions': 1}}},
            'setup_validation': {'status': 'pass'},
            'pose_validation': {'poses': [{'path': str(pose)}]}}]})
    monkeypatch.setattr('structural_modeling_agent.evidence.modeling_evidence', lambda path: {'candidates': [{
        'accession': 'source:CAT7', 'execution_status': 'complete', 'pose_qc_status': 'fail',
        'scientific_status': 'requires_revision', 'geometry_metrics': {'n_predictions': 1}}]})
    config = json.loads((campaign / 'sources.json').read_text())
    config['structural_agent_sources'] = [{'run_id': 'new_model', 'report': str(report),
        'discovery_run_id': 'current', 'target_id': 'synthetic'}]
    test.put(campaign / 'sources.json', config)
    result = build(campaign)
    assert result['counts']['modeling_executions'] == 2
    match, = lookup_predictions(campaign, accession='source:CAT7', sequence='AGGA')['matches']
    assert match['execution']['output_presence'] == 'complete'
    assert match['execution']['scientific_status'] == 'requires_revision'
    assert match['execution']['pose_qc_status'] == 'fail'
    assert match['execution']['protein_id'].startswith('unresolved:')
    assert match['input_integrity'] == 'verified'
    config['structural_agent_sources'][0]['discovery_run_id'] = 'reference'
    test.put(campaign / 'sources.json', config)
    with pytest.raises(ValueError, match='source-scoped discovery record'):
        build(campaign)


def test_legacy_validation_bundle_is_preserved_on_refresh(tmp_path):
    test = test_catalog.CatalogTest()
    campaign = test.fixture(tmp_path)
    bundle = test.put(tmp_path / 'legacy_validation.json', {'candidates': [{
        'accession': 'P12345', 'bundle_id': 'synthetic-proxy', 'score': .5}]})
    config = json.loads((campaign / 'sources.json').read_text())
    config['modeling_sources'][0]['validation_bundle'] = str(bundle)
    test.put(campaign / 'sources.json', config)
    assert build(campaign)['counts']['validations'] == 1
    row, = test.rows(campaign, 'validations')
    assert row['protein_id'] == 'uniprot:P12345'
    assert row['limitation'] == 'Proxy application is not proof of valid setup or target activity'


def test_ranked_and_unranked_hypotheses_share_the_review_queue(tmp_path):
    test = test_catalog.CatalogTest()
    campaign = test.fixture(tmp_path)
    registry = tmp_path / 'current/candidate_registry.json'
    data = json.loads(registry.read_text())
    candidate = test.entry('source:CAT7', 'evidence_pending', '2026-02-01')
    candidate['metadata']['priority_assessment'] = {'decision': 'investigate',
        'strengths': ['Synthetic source supports a catalytic hypothesis'], 'evidence_refs': ['synthetic-source']}
    data['candidates']['source:CAT7'] = candidate
    test.put(registry, data)
    build(campaign)
    queue = list(csv.DictReader((campaign / 'views/new_modeling_review.csv').open()))
    assert {row['accession'] for row in queue} == {'Q54321', 'source:CAT7'}
    hypothesis = next(row for row in queue if row['accession'] == 'source:CAT7')
    assert hypothesis['latest_run_rank'] == ''
    assert hypothesis['submission_authorized'] == 'False'


def test_cached_review_adds_provenance_without_counting_another_prediction(tmp_path,monkeypatch):
    test=test_catalog.CatalogTest();campaign=test.fixture(tmp_path)
    before=build(campaign);execution=test.rows(campaign,'modeling_executions')[0]
    report=test.put(tmp_path/'review/report.json',{'schema_version':'modeling-agent-1.0','status':'inconclusive',
        'candidates':[{'accession':'P12345','existing_prediction_review':{
            'source_execution_id':execution['execution_id'],'applicability':'limited_transfer'}}]})
    monkeypatch.setattr('structural_modeling_agent.evidence.modeling_evidence',lambda path:{'candidates':[
        {'accession':'P12345','execution_status':'complete'}]})
    config=json.loads((campaign/'sources.json').read_text())
    config['structural_agent_sources']=[{'report':str(report),'run_id':'review',
        'discovery_run_id':'reference','target_id':'synthetic'}]
    test.put(campaign/'sources.json',config)
    after=build(campaign)
    assert after['counts']['modeling_executions']==before['counts']['modeling_executions']
    assert after['counts']['modeling_setups']==before['counts']['modeling_setups']
    run=next(row for row in test.rows(campaign,'runs') if row['run_id']=='review')
    assert run['modeled_candidates']==0 and run['reviewed_existing_candidates']==1
