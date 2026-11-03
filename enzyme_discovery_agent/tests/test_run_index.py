"""The human status page observes saved campaigns without driving discovery."""
import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.output.run_index import collect_runs,publish_index


def prepare(project):
    root=project/'enzyme_discovery_agent/runs/synthetic_run'
    root.mkdir(parents=True)
    (root/'manifest.json').write_text(json.dumps({'status':'budget_exhausted','query':'Synthetic transformation'}))
    (root/'target_context.json').write_text(json.dumps({'query':'Synthetic transformation'}))
    update_candidate_registry(root,'SYNTH_A','screened',{'protein_name':'Synthetic enzyme'})
    return root


def digest_files(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl'}}


def test_status_collection_preserves_campaign_state_and_shows_the_ready_queue(tmp_path):
    root=prepare(tmp_path)
    before=digest_files(root)
    cache={}
    first=collect_runs(tmp_path,cache=cache)
    assert first['runs'][0]['ready_count']==1
    assert first['runs'][0]['status_counts']=={'screened':1}
    assert digest_files(root)==before
    second=collect_runs(tmp_path,cache=cache)
    assert second['runs'][0]==first['runs'][0]
    update_candidate_registry(root,'SYNTH_A','evaluated',{'claim_validation':{'promotion_eligible':True}})
    assert collect_runs(tmp_path,cache=cache)['runs'][0]['status_counts']=={'evaluated':1}


def test_home_page_keeps_generated_files_outside_individual_run_roots(tmp_path):
    root=prepare(tmp_path)
    before=set(root.iterdir())
    publish_index(tmp_path)
    assert (tmp_path/'index.html').is_file()
    assert set(root.iterdir())==before
    assert 'Read-only saved-state monitor' in (tmp_path/'index.html').read_text()


def test_incomplete_owner_observation_is_reported_without_claiming_liveness(tmp_path):
    root=prepare(tmp_path)
    owner=root/'agent_state/discovery_owner.json';owner.parent.mkdir(exist_ok=True)
    owner.write_text('{incomplete')
    result=collect_runs(tmp_path)
    assert result['runs'][0]['status']=='snapshot_unavailable'
    assert result['runs'][0]['worker_alive'] is None
