"""Rolling task preparation preserves current-run sources and independent hypotheses."""
import json
from pathlib import Path

from enzyme_discovery_agent.production import candidate_sources, evidence_basis, prepare_tasks, require_capacity,prepare_input_queries


def test_only_explicit_current_run_sources_are_forwarded(tmp_path):
    run = tmp_path / 'run'; source = run / 'artifacts/source.txt'
    source.parent.mkdir(parents=True); source.write_text('Synthetic source')
    unrelated = run / 'artifacts/unrelated.txt'; unrelated.write_text('Unreferenced source')
    historical = tmp_path / 'history.txt'; historical.write_text('Old report')
    sources = candidate_sources({'citations': [str(source), str(historical)], 'text': 'Not a path'}, run)
    assert sources == {str(source)}


def test_new_scientific_evidence_reopens_a_task_but_bookkeeping_does_not():
    row = {'data': {'priority_assessment': {'decision': 'investigate'}, 'last_seen': 'before'}}
    original = evidence_basis(row)
    row['data']['last_seen'] = 'after'
    assert evidence_basis(row) == original
    row['data']['priority_assessment']={'decision':'prioritize','reason':'Reworded investigation priority','recorded_at':'now'}
    assert evidence_basis(row)==original
    row['data']['mechanism_evidence'] = {'source_id': 'new-synthetic-source'}
    assert evidence_basis(row) != original


def test_preparation_batches_sequences_and_preserves_missing_annotation(tmp_path, monkeypatch):
    rows = [{'accession': 'P12345', 'data': {'priority_assessment': {'decision': 'investigate'}},
             'native_substrates_resolved': False},
            {'accession': 'Q12345', 'data': {'priority_assessment': {'decision': 'prioritize'}},
             'native_substrates_resolved': True}]
    class Response:
        def __init__(self, accession): self.accession = accession
        def raise_for_status(self): pass
        def json(self): return {'primaryAccession': self.accession, 'sequence': {'value': 'AGGA'}}
    monkeypatch.setattr('requests.get', lambda url, **kwargs: Response(url.split('/')[-1].removesuffix('.json')))
    monkeypatch.setattr('campaign_catalog.lookup.lookup_predictions', lambda *a, **k: {'matches': []})
    calls = []
    def msa(fasta, out, cache):
        calls.append(Path(fasta).read_text())
        return [{'id': '0', 'csv': 'synthetic-msa-a'}, {'id': '1', 'csv': 'synthetic-msa-b'}]
    monkeypatch.setattr('structural_modeling_agent.msa.prepare', msa)
    tasks = prepare_tasks({'cofold_pool': 'synthetic-pool', 'msa_cache': 'synthetic-cache',
        'setup_turn_limit': 8, 'modeling_target': {'reaction': 'synthetic redox'}}, tmp_path,
        rows, tmp_path / 'prepared')
    assert len(calls) == 1 and calls[0].count('AGGA') == 2
    assert len(tasks) == 2
    first = json.loads(Path(tasks[0]['task']).read_text())
    assert first['target'] == {'reaction': 'synthetic redox'}
    assert first['candidates'][0]['evidence']['native_substrates_resolved'] is False
    assert 'modeling_input' not in first['candidates'][0]
    assert first['workflow_mode'] == 'production_cofold'


def test_worker_capacity_is_checked_in_a_child_not_against_the_coordinator_binding(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def probe(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='[3,4]')
    monkeypatch.setattr('enzyme_discovery_agent.production.subprocess.run',probe)
    monkeypatch.setattr('psutil.process_iter',lambda *a,**k:[])
    monkeypatch.setattr('os.getloadavg',lambda:(0,0,0))
    monkeypatch.setattr('os.cpu_count',lambda:8)
    monkeypatch.setattr('os.sched_getaffinity',lambda pid:{1})
    require_capacity([3,4])
    assert calls[0][:3]==['taskset','-c','3,4']


def test_explicit_task_source_probes_are_bounded_and_not_repeated_on_resume(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    calls=[]
    def execute(name,args,**kwargs):
        calls.append((name,args))
        return {'status':'success','data':[{'title':'Synthetic primary source'}]}
    monkeypatch.setattr('enzyme_discovery_agent.tool_registry.execute_tool',execute)
    agent=SimpleNamespace(run_dir=tmp_path,_save_artifact=Mock())
    settings={'input_literature_queries':[{'tool':'search_crossref','arguments':{'query':'synthetic target product formation'}}]}
    prepare_input_queries(settings,tmp_path,agent)
    prepare_input_queries(settings,tmp_path,agent)
    assert len(calls)==1 and agent._save_artifact.call_count==1


def test_separate_candidate_pool_can_reuse_a_compatible_prediction_cache(tmp_path, monkeypatch):
    from unittest.mock import Mock
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'primaryAccession':'P12345','sequence':{'value':'AGGA'}}
    monkeypatch.setattr('requests.get',lambda *a,**k:Response())
    seen=[]
    match={'execution':{'execution_id':'execution:synthetic','artifacts':[]},'setup':{}}
    def lookup(pool,**kwargs):
        seen.append((pool,kwargs))
        return {'matches':[match] if pool=='historical-cache' else []}
    monkeypatch.setattr('campaign_catalog.lookup.lookup_predictions',lookup)
    msa=Mock(side_effect=AssertionError('An existing compatible ensemble needs no new MSA'))
    monkeypatch.setattr('structural_modeling_agent.msa.prepare',msa)
    rows=[{'accession':'P12345','data':{'priority_assessment':{'decision':'investigate'}},'native_substrates_resolved':False}]
    settings={'cofold_pool':'current-target-pool','prediction_cache_pools':['historical-cache'],
              'cofold_pool_target_id':'synthetic-current-target','modeling_required_ligand_smiles':['CCO'],
              'msa_cache':'sequence-cache','setup_turn_limit':8,'modeling_target':{'reaction':'synthetic'}}
    task,=prepare_tasks(settings,tmp_path,rows,tmp_path/'prepared')
    evidence=json.loads(Path(task['task']).read_text())['candidates'][0]['evidence']
    assert evidence['do_not_recofold'] and evidence['existing_cofolds']==[match]
    assert [pool for pool,_ in seen]==['current-target-pool','historical-cache']
    assert all(args['target_id']=='synthetic-current-target' and args['required_ligand_smiles']==['CCO'] for _,args in seen)


def test_candidate_pool_refresh_imports_only_its_own_discovery_run(tmp_path):
    from enzyme_discovery_agent.production import register_completed_models
    pool=tmp_path/'pool';pool.mkdir();run=tmp_path/'current-run';run.mkdir()
    (pool/'sources.json').write_text(json.dumps({'campaign_id':'synthetic','project_root':str(tmp_path),
                                              'discovery_sources':[],'modeling_sources':[]}))
    (run/'manifest.json').write_text(json.dumps({'run_id':'current-run','status':'running'}))
    (run/'target_context.json').write_text(json.dumps({'substrate_smiles':'CCO'}))
    (run/'candidate_registry.json').write_text(json.dumps({'candidates':{'P12345':{'accession':'P12345',
        'statuses':[{'status':'discovered'}],'metadata':{'protein_name':'Synthetic candidate'}}}}))
    result=register_completed_models({'cofold_pool':str(pool),'cofold_pool_target_id':'synthetic-current-target'},run)
    assert result['counts']['proteins']==1 and result['counts']['runs']==1
    sources=json.loads((pool/'sources.json').read_text())
    assert len(sources['discovery_sources'])==1
    assert sources['discovery_sources'][0]['path']==str(run)
    assert not sources['structural_agent_sources']


def test_an_exited_worker_is_recorded_before_its_backend_jobs_settle(tmp_path, monkeypatch):
    """A worker that has exited must never keep being reported as running.

    The coordinator waits for the worker's backend jobs (e.g. a queued alignment)
    to settle before it can read a final report. That wait used to leave the queue
    entry and the published status claiming a live worker.
    """
    import enzyme_discovery_agent.production as production

    root = tmp_path / 'launch'; run = tmp_path / 'run'
    (root / 'provider/backend_jobs/job1').mkdir(parents=True)
    (root / 'provider/backend_jobs/job1/receipt.json').write_text(json.dumps({'status': 'running'}))
    run.mkdir()
    (root / 'campaign_state.json').write_text(json.dumps({'run_dir': str(run)}))
    (root / 'worker.py').write_text('')  # the worker only has to exit cleanly

    row = {'accession': 'P12345', 'data': {'priority_assessment': {'decision': 'prioritize'}},
           'native_substrates_resolved': True}
    monkeypatch.setattr('enzyme_discovery_agent.models.modeling_portfolio.select_modeling_portfolio',
                        lambda *a, **k: [row])
    monkeypatch.setattr('enzyme_discovery_agent.provenance.snapshot_source_code', lambda *a, **k: None)
    monkeypatch.setattr(production, 'register_completed_models', lambda *a, **k: {})
    monkeypatch.setattr(production, 'require_capacity', lambda cpus: None)
    monkeypatch.setattr(production, 'prepare_tasks',
                        lambda settings, run, rows, directory: [{'accession': r['accession'],
                                                                 'task': str(directory / 'task.json')} for r in rows])

    settings = {'runtime_budget_seconds': 25, 'max_modeling_candidates': 1, 'protected_unannotated_slots': 0,
                'modeling_token_budget': 10_000, 'modeling_item_token_budget': 10_000, 'modeling_max_attempts': 1,
                'modeling_cpu_affinity': [0], 'feedback_instruction': 'synthetic', 'cofold_pool': str(tmp_path / 'pool'),
                'cofold_pool_target_id': 'synthetic-target', 'model': 'synthetic', 'modeling_max_turns': 1,
                'modeling_runtime_seconds': 1, 'setup_turn_limit': 1, 'msa_cache': str(tmp_path / 'cache'),
                'modeling_target': {'reaction': 'synthetic redox'}}

    # The receipt never settles, so the coordinator stays in the waiting branch
    # until its own runtime budget expires.
    production.coordinator(settings, root, tmp_path / 'config.json')

    items = [json.loads(path.read_text())
             for path in (run / 'agent_state/production_modeling').glob('*/item.json')]
    assert [item['status'] for item in items] == ['awaiting_backend_jobs']
    assert items[0]['exit_code'] == 0 and items[0]['worker_finished_at'] > 0
    status = json.loads((root / 'campaign_status.json').read_text())
    assert status['modeling_live'] is False and status['awaiting_backend_jobs'] == 1
    assert [entry['status'] for entry in status['queue']] == ['awaiting_backend_jobs']


def test_a_model_of_another_target_is_described_without_carrying_its_bulk():
    """The other-target index must not crowd the candidate's own record out of context."""
    from enzyme_discovery_agent.production import compact_other_target_match
    match = {'identity_match': 'accession', 'input_integrity': 'verified',
             'target_applicability': 'required_molecular_input_absent',
             'execution': {'execution_id': 'execution:abc', 'producer_run_id': 'run-1',
                           'scientific_status': 'requires_interpretation_or_revision',
                           'pose_qc_status': 'pass', 'expected_poses': 50,
                           'coordinate_files': [f'/pose/{i}.cif' for i in range(50)],
                           'confidence_files': [f'/pose/{i}.json' for i in range(50)],
                           'artifacts': [{'path': f'/pose/{i}.cif', 'sha256': 'f' * 64} for i in range(50)]},
             'setup': {'setup_id': 'setup:abc', 'target_id': 'other-target',
                       'declared_modeling_input': {'modeled_step': {'index': 1},
                                                   'target_substrates': [{'name': 'other substrate'}],
                                                   'cofactors': [{'ccd': 'ATP'}]},
                       'input_source': {'path': '/big', 'sha256': 'e' * 64}}}
    compact = compact_other_target_match(match, Path('/pool/lookup.json'))
    assert len(json.dumps(compact)) < len(json.dumps(match)) / 5
    # What it was, and why it does not apply here, must survive.
    assert compact['target_applicability'] == 'required_molecular_input_absent'
    assert compact['target_id'] == 'other-target' and compact['execution_id'] == 'execution:abc'
    assert compact['target_substrates'] == [{'name': 'other substrate'}]
    assert compact['full_record'] == '/pool/lookup.json'
    # The per-pose bulk must be gone, reachable only through the on-disk record.
    assert 'coordinate_files' not in json.dumps(compact) and '/pose/0.cif' not in json.dumps(compact)


def test_an_absent_other_target_index_stays_absent():
    from enzyme_discovery_agent.production import compact_other_target_match
    assert compact_other_target_match({}, Path('/pool/lookup.json')) == {'full_record': '/pool/lookup.json'}
