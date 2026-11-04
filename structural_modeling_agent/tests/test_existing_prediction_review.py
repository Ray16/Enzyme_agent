"""A cached review verifies historical output without respecifying or resubmitting it."""
import json
from pathlib import Path

from test_modeling_agent import make_agent, propose, msa, Client
from test_validation import modeling_record, write_pose
from structural_modeling_agent.production import run_candidate
from structural_modeling_agent.validation import file_digest,fingerprint
from structural_modeling_agent.evidence import modeling_evidence


def fixture(tmp_path):
    old=tmp_path/'producer';old.mkdir()
    agent=make_agent(old,objective='model_and_validate')
    record=modeling_record(old)
    agent.dispatch('propose_model',propose(record))
    agent.dispatch('render_inputs',msa(old))
    revision=agent._current('SYNTHETIC1')
    predictions=old/'predictions';predictions.mkdir()
    pose=write_pose(predictions/'model.pdb')
    producer=old/'producer.json';producer.write_text(json.dumps({
        'input_sha256':revision['input_sha256'],'execution_sha256':fingerprint(revision['execution']),
        'predictions':[{'path':str(pose),'sha256':file_digest(pose)}]}))
    agent.dispatch('validate_predictions',{'accession':'SYNTHETIC1','directory':str(predictions),
                                         'prediction_manifest':str(producer)})
    report=agent._write_report();report_path=agent.run_dir/'report.json'
    source_hash=file_digest(report_path)
    task=tmp_path/'task.json';task.write_text(json.dumps({'schema_version':'1.0',
        'objective':'model_and_validate','workflow_mode':'production_cofold',
        'target':{'reaction':'synthetic transformation'},'candidates':[{
            'accession':'SYNTHETIC1','sequence':'AG','evidence':{
                'do_not_recofold':True,'evidence_paths':[str(pose),str(producer),str(old/'query.csv')],
                'existing_cofolds':[{'setup':{'setup_id':'setup:synthetic','declared_modeling_input':record,
                    'input_source':{'path':revision['input_yaml'],'sha256':revision['input_sha256']}},
                    'execution':{'execution_id':'execution:synthetic','source':{
                        'path':str(report_path),'sha256':source_hash},'output_presence':'complete'}}]}}]}))
    return task,report_path,pose,source_hash


def test_real_controller_reviews_cached_predictions_in_two_calls_without_a_proposal(tmp_path):
    task,source,pose,source_hash=fixture(tmp_path)
    client=Client([[("review_existing_predictions",{'accession':'SYNTHETIC1','setup_id':'setup:synthetic',
        'applicability':'limited_transfer','interpretation':'The saved synthetic molecular question is relevant but its native applicability remains unresolved.',
        'unresolved':['The synthetic structural readout does not establish activity.']})],
        [("finish_task",{'outcome':'inconclusive','rationale':'The historical ensemble is verified; target activity remains unknown.'})]])
    result=run_candidate(task_path=task,run_dir=tmp_path/'review',backend_path=None,
        execution_intent='submit_after_checks_pass',llm_client=client,token_budget=1000,max_turns=4)
    assert result['status']=='inconclusive'
    assert result['candidates'][0]['revision'] is None
    assert len(result['usage']['calls'])==2
    exported=modeling_evidence(tmp_path/'review/report.json')['candidates'][0]
    assert exported['execution_status']=='complete'
    assert exported['computation_origin']=='reused_existing_prediction'
    assert exported['geometry_metrics']
    assert not (tmp_path/'review/execution_request.json').exists()
    assert file_digest(source)==source_hash
    pose.write_text(pose.read_text()+'REMARK changed artifact\n')
    assert modeling_evidence(tmp_path/'review/report.json')['candidates'][0]['execution_status']=='incomplete'
    source.write_text(source.read_text()+' ')
    changed=modeling_evidence(tmp_path/'review/report.json')['candidates'][0]
    assert changed['execution_status']=='incomplete' and changed['review_errors']
