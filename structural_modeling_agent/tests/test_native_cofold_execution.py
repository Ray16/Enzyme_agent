"""Execution does no provider reasoning and never resubmits a recorded job blindly."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from structural_modeling_agent import cofold_execution as workflow


class Agent:
    def __init__(self,pending):
        self.task={'candidates':[{'accession':'SYNTHETIC','sequence':'AG','evidence':{}}]}
        self.state={'candidates':{'SYNTHETIC':{'revisions':[{'job':pending,'execution':{}}]}},'turn':0}
        self.calls=[]
    def dispatch(self,name,args):
        self.calls.append(name)
        if name=='check_cofold':return {'status':'complete','prediction_dir':'synthetic','prediction_manifest':'manifest.json'}
        if name=='validate_predictions':return {'status':'fail','n_predictions':50}
        raise AssertionError('Unexpected execution action: '+name)
    def _artifact(self,*args):return 'synthetic-artifact'
    def _event(self,*args,**kwargs):pass
    def _save(self):pass
    def _write_report(self):return {}


@pytest.mark.parametrize('recorded_state',['failed','unknown','queued','complete'])
def test_cached_provider_job_is_reconciled_and_measured_without_replanning(tmp_path,monkeypatch,recorded_state):
    agent=Agent({'status':recorded_state,'idempotency_key':'synthetic'})
    # A real rendered execution is truthy and remains the original saved input.
    agent.state['candidates']['SYNTHETIC']['revisions'][0]['execution']={'identity':'synthetic-input'}
    folder=tmp_path/'model';folder.mkdir();(folder/'checkpoint.json').write_text('{}')
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps({'accession':'SYNTHETIC'}))
    monkeypatch.setattr(workflow,'ModelingAgent',lambda **kwargs:agent)
    monkeypatch.setattr(workflow,'prepare',Mock(side_effect=AssertionError('Existing outputs do not require another scientific setup action')))
    monkeypatch.setattr('enzyme_discovery_agent.provenance.snapshot_source_code',lambda root:{'source_sha256':'synthetic'})
    result=workflow.execute({'task_path':'synthetic-task','run_dir':str(folder),'backend_path':'synthetic-backend',
        'proposal_path':str(proposal),'decision_origin':{'author':'synthetic test'}})
    assert agent.calls==['check_cofold','validate_predictions']
    assert result['status']=='ensemble_ready_for_interpretation'
    assert result['provider_calls_for_execution']==0 and result['n_predictions']==50
    assert result['pose_validation_status']=='fail'  # Failed poses remain usable observations.


def test_failed_setup_never_reaches_render_or_submit(tmp_path,monkeypatch):
    agent=Agent({});agent.task={'candidates':[{'accession':'SYNTHETIC'}]}
    agent.state['candidates']['SYNTHETIC']['revisions']=[]
    def dispatch(name,args):
        agent.calls.append(name)
        if name=='propose_model':
            value={'scientific_reasoning':{'selected_hypothesis_id':'H1'}}
            agent.state['candidates']['SYNTHETIC']['revisions'].append(value);return value
        if name=='validate_setup':return {'status':'abstain','reason':'Missing source-grounded assembly'}
        raise AssertionError('Unsupported setup must not execute')
    agent.dispatch=dispatch
    result=workflow.prepare(agent,{'accession':'SYNTHETIC'},execution_intent='submit_after_checks_pass')
    assert result['status']=='blocked'
    assert agent.calls==['propose_model','validate_setup']
