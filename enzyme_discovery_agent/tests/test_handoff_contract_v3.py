"""Cross-agent handoff integrity and admission tests with synthetic data."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.models.handoff_contract import (adapt_legacy_handoff, digest, load_document,
    read_bundle_source, snapshot_payload, validate_handoff)
from enzyme_discovery_agent.output.structural_modeling_handoff import write_structural_modeling_handoff
from structural_modeling_agent.agent import ModelingAgent


@pytest.fixture(autouse=True)
def target_review_precedes_serialization(monkeypatch):
    # Isolate source/identity/serialization integrity from the separately tested target gate.
    monkeypatch.setattr('enzyme_discovery_agent.models.target_review.eligible',lambda *a,**k:True)


def validation(acc):
    return {'promotion_eligible': True, 'claims': [{'claim_type': 'accession_identity',
        'subject': acc, 'verdict': 'supported', 'evidence': {'source': 'UniProtKB'}}]}


def create_run(root):
    update_candidate_registry(root, 'P12345', 'evaluated', {'protein_name': 'Current name',
        'claim_validation': validation('P12345'), 'sequence': 'ACDE'})
    update_candidate_registry(root, 'Q12345', 'evaluated', {'protein_name': 'Unverified'})
    evaluation = [{'status': 'success', 'data': {'accession': 'P12345', 'protein_name': 'Earlier name'}},
                  {'status': 'success', 'data': {'accession': 'Q12345'}}]
    return evaluation


def test_producer_admission_leads_ranking_and_readable_summary(tmp_path):
    evaluations = create_run(tmp_path)
    (tmp_path / 'pass1_ranking.json').write_text(json.dumps([{'accession': 'Z12345', 'rank': 1}]))
    path = write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic target'},
        neighbor_map={'promising': [{'accession': 'Z12345', 'neighbors': [{'accession': 'R12345'}]}]},
        discovery_status_value='incomplete')
    doc = validate_handoff(load_document(path))
    assert doc['candidates'][0]['accession'] == 'P12345'
    assert doc['candidates'][0]['selection']['rank'] is None
    assert doc['candidates'][0]['selection']['admission'] == 'provisional'
    assert doc['candidates'][0]['evidence']['identity']['protein_name'] == 'Current name'
    assert 'protein_name' in doc['candidates'][0]['evidence']['source_conflicts']
    assert {lead['accession'] for lead in doc['unresolved_leads']} == {'Q12345', 'R12345'}
    assert 'P12345' in path.with_suffix('.md').read_text()
    assert doc['integrity']['target_sha256'] == digest(doc['target'])
    assert doc['integrity']['snapshot_sha256'] == digest(snapshot_payload(doc))


def test_mutated_content_is_rejected(tmp_path):
    doc = load_document(write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'}))
    doc['candidates'][0]['selection']['reason'] = 'changed'
    with pytest.raises(ValueError, match='snapshot fingerprint'):
        validate_handoff(doc)


def test_rejection_removes_old_candidate_from_refreshed_handoff(tmp_path):
    evaluations = create_run(tmp_path)
    path = write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic'})
    update_candidate_registry(tmp_path, 'P12345', 'rejected')
    doc = load_document(write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic'}))
    assert not doc['candidates']
    assert any(row['accession'] == 'P12345' for row in doc['excluded_candidates'])
    assert validate_handoff(doc)


def test_bundle_integrity_relocation_and_escape(tmp_path):
    root = tmp_path / 'bundle'; root.mkdir()
    folder = root / 'sources'; folder.mkdir()
    data = b'synthetic source evidence'
    path = folder / 'source.txt'; path.write_bytes(data)
    ref = {'bundle_relative_path': 'sources/source.txt', 'sha256': hashlib.sha256(data).hexdigest()}
    assert read_bundle_source(root, ref) == data
    moved = tmp_path / 'moved'; root.rename(moved)
    assert read_bundle_source(moved, ref) == data
    with pytest.raises(ValueError):
        read_bundle_source(moved, {**ref, 'bundle_relative_path': '../outside'})
    path = moved / 'sources' / 'source.txt'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        read_bundle_source(moved, ref)
    path.unlink(); path.symlink_to(tmp_path / 'outside')
    with pytest.raises(OSError):
        read_bundle_source(moved, ref)


def test_unreviewed_state_change_cannot_become_ready():
    from enzyme_discovery_agent.models.modeling_handoff import build_modeling_input, is_execution_ready
    target = {'substrate_smiles': 'CCO', 'target_reasoning': {'modeling_specification': {
        'components': [{'source_component_index': 1, 'role': 'substrate',
                        'modeling_smiles': 'CC[O-]', 'state_rationale': 'asserted'}],
        'geometry': {'kind': 'none'}}}, 'reviewed_modeling_substrates': [
            {'source_component_index': 1, 'modeling_smiles': 'CC[O-]'}]}
    proposal = build_modeling_input({'modeling_specification': {'enzyme_chains': [{'source': 'full_sequence'}]}}, target)
    assert not is_execution_ready(proposal)[0]
    assert proposal['target_substrates'][0]['review_status'] == 'required'


def test_legacy_adapter_and_empty_consumer(tmp_path):
    legacy = {'schema_version': '2.0', 'target': {'query': 'synthetic'}, 'candidates': [
        {'accession': 'P12345', 'evidence': {}, 'modeling_input': {}, 'selection': {}}]}
    adapted = adapt_legacy_handoff(legacy, 'abc')
    assert adapted['candidates'] == [] and adapted['unresolved_leads'][0]['accession'] == 'P12345'
    task = tmp_path / 'legacy.json'; task.write_text(json.dumps(legacy))
    agent = ModelingAgent(task_path=task, run_dir=tmp_path / 'consumer', llm_client=object())
    result = agent.run(max_turns=1)
    assert result['status'] == 'no_admitted_candidates'
    assert result['discovery_handoff']['unresolved_leads'][0]['accession'] == 'P12345'


def test_duplicate_json_keys_and_invalid_identity(tmp_path):
    path = tmp_path / 'bad.json'; path.write_text('{"schema_version":"3.0","schema_version":"2.0"}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        load_document(path)
    doc = load_document(write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'}))
    doc['candidates'][0]['identity']['accession'] = 'OTHER'
    doc['integrity']['snapshot_sha256'] = digest(snapshot_payload(doc))
    with pytest.raises(ValueError, match='identity'):
        validate_handoff(doc)


def test_source_bundle_is_verified_and_available_to_consumer(tmp_path):
    evaluations = create_run(tmp_path)
    source = tmp_path / 'artifacts' / 'evidence' / 'source.txt'
    source.parent.mkdir(parents=True)
    source.write_text('synthetic evidence passage')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    update_candidate_registry(tmp_path, 'P12345', None, {'literature_evidence': {
        'text_path': str(source), 'text_sha256': source_hash, 'evidence_quote': 'synthetic evidence passage'}})
    path = write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic target'})
    handoff = validate_handoff(load_document(path))
    assert handoff['evidence_sources'][0]['availability'] == 'available'
    assert handoff['candidates'][0]['evidence']['literature_reaction']['source_ref'] == source_hash
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'modeling', llm_client=object())
    inspected = agent.dispatch('inspect_candidate', {'accession': 'P12345'})
    assert json.loads(agent.messages[1]['content'])['handoff_context']['discovery_status'] == 'unknown'
    source_info = inspected['evidence_sources'][0]
    assert agent.dispatch('read_evidence', {'path': source_info['local_path']})['text'] == 'synthetic evidence passage'
    page = agent.dispatch('read_candidate_section', {'accession': 'P12345', 'section': 'evidence'})
    assert 'synthetic evidence passage' in page['text']


def test_source_changed_after_handoff_is_unavailable_to_consumer(tmp_path):
    evaluations = create_run(tmp_path)
    source = tmp_path / 'artifacts' / 'evidence' / 'source.txt'
    source.parent.mkdir(parents=True); source.write_text('synthetic evidence passage')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    update_candidate_registry(tmp_path, 'P12345', None, {'literature_evidence': {
        'text_path': str(source), 'text_sha256': source_hash}})
    path = write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic'})
    doc = load_document(path)
    (tmp_path / doc['evidence_sources'][0]['bundle_relative_path']).write_text('modified')
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object())
    info = agent.state['bundle_evidence'][0]
    assert info['availability'] == 'unavailable'
    assert 'local_path' not in info


def test_handoff_schema_1_is_rejected_but_standalone_1_remains_distinct(tmp_path):
    legacy = tmp_path / 'old.json'
    legacy.write_text(json.dumps({'schema_version': '1.0', 'source_run': 'legacy',
                                  'candidates': [{'accession': 'P12345'}]}))
    with pytest.raises(ValueError, match='unsupported discovery handoff'):
        ModelingAgent(task_path=legacy, run_dir=tmp_path / 'consumer', llm_client=object())


def test_conflicting_cached_sequence_remains_a_lead(tmp_path):
    evaluations = create_run(tmp_path)
    folder = tmp_path / 'embedding_store'; folder.mkdir()
    (folder / 'sequences.json').write_text(json.dumps({'P12345': 'ACDF'}))
    doc = load_document(write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic'}))
    assert not doc['candidates']
    lead = next(item for item in doc['unresolved_leads'] if item['accession'] == 'P12345')
    assert lead['identity_conflicts'] == ['sequence_cache']


def test_legacy_adapter_rejects_duplicate_ids():
    row = {'accession': 'P12345', 'evidence': {}, 'selection': {}, 'modeling_input': {}}
    with pytest.raises(ValueError, match='unique'):
        adapt_legacy_handoff({'schema_version': '2.0', 'target': {}, 'candidates': [row, row]}, 'sha')


def test_blocked_validation_remains_unresolved_not_rejected(tmp_path):
    update_candidate_registry(tmp_path, 'P12345', 'evidence_pending', {'claim_validation': {
        'promotion_eligible': False, 'blocking_claims': ['native_reaction'],
        'claims': [{'claim_type': 'native_reaction', 'subject': 'P12345', 'verdict': 'insufficient_evidence'}]}})
    doc = load_document(write_structural_modeling_handoff(tmp_path,
        [{'data': {'accession': 'P12345'}}], {'query': 'synthetic'}))
    assert doc['unresolved_leads'][0]['accession'] == 'P12345'
    assert doc['excluded_candidates'] == []


def test_unresolved_source_reference_is_rejected(tmp_path):
    doc = load_document(write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'}))
    doc['candidates'][0]['evidence']['extra'] = {'source_ref': 'missing'}
    doc['integrity']['snapshot_sha256'] = digest(snapshot_payload(doc))
    with pytest.raises(ValueError, match='source reference'):
        validate_handoff(doc)


def test_fifo_source_is_rejected_without_blocking(tmp_path):
    import os
    fifo = tmp_path / 'fifo'; os.mkfifo(fifo)
    with pytest.raises(ValueError, match='regular file'):
        read_bundle_source(tmp_path, {'bundle_relative_path': 'fifo', 'sha256': '0' * 64})


def test_execution_override_and_malformed_v3_are_rejected(tmp_path):
    doc = load_document(write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'}))
    doc['objective'] = 'prepare_inputs'
    doc['integrity']['snapshot_sha256'] = digest(snapshot_payload(doc))
    with pytest.raises(ValueError, match='unsupported handoff fields'):
        validate_handoff(doc)
    path = tmp_path / 'malformed.json'
    path.write_text(json.dumps({'schema_version': '3.0', 'candidates': [{'accession': 'P12345'}]}))
    with pytest.raises(ValueError, match='expected a discovery handoff'):
        ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object())


@pytest.mark.parametrize('status', ['failed', 'error', 'partial', None])
def test_only_successful_evaluation_can_be_admitted(tmp_path, status):
    evaluation = create_run(tmp_path)
    evaluation[0]['status'] = status
    doc = load_document(write_structural_modeling_handoff(tmp_path, evaluation, {'query': 'synthetic'}))
    assert not doc['candidates']
    assert any(x['accession'] == 'P12345' for x in doc['unresolved_leads'])


def test_explicit_freshness_check_reports_advanced_registry(tmp_path):
    root = tmp_path / 'discovery'; root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'run_id': root.name, 'status': 'running'}))
    (root / 'target_context.json').write_text(json.dumps({'query': 'synthetic'}))
    path = write_structural_modeling_handoff(root, create_run(root), {'query': 'synthetic'})
    update_candidate_registry(root, 'P12345', 'rejected')
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object(),
                          discovery_run_path=root)
    assert agent.state['discovery_currentness']['status'] == 'stale'
    assert 'candidate_registry' in agent.state['discovery_currentness']['changed']


def test_empty_or_wrong_run_cannot_be_assumed_current(tmp_path):
    root = tmp_path / 'discovery'; root.mkdir()
    path = write_structural_modeling_handoff(root, [], {'query': 'synthetic'})
    wrong = tmp_path / 'wrong'; wrong.mkdir()
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object(),
                          discovery_run_path=wrong)
    assert agent.state['discovery_currentness']['status'] == 'unknown'


def test_explicit_resume_rechecks_freshness(tmp_path):
    root = tmp_path / 'discovery'; root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'run_id': root.name, 'status': 'running'}))
    (root / 'target_context.json').write_text(json.dumps({'query': 'synthetic'}))
    path = write_structural_modeling_handoff(root, create_run(root), {'query': 'synthetic'})
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object(),
                          discovery_run_path=root)
    assert agent.state['discovery_currentness']['status'] == 'unchanged_checked_artifacts'
    update_candidate_registry(root, 'P12345', 'rejected')
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True, llm_client=object(), discovery_run_path=root)
    # Stop before a provider call while retaining the resumed task and fresh audit.
    resumed.llm = type('NoReply', (), {'call': lambda self, **kwargs: None})()
    report = resumed.run(max_turns=1)
    assert report['discovery_handoff']['currentness']['status'] == 'stale'


def test_unverified_sequence_blocks_render_and_retrieval_conflict(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import requests
    path = write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'})
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object())
    state = agent.state['candidates']['P12345']
    state['revisions'] = [{'modeling_input': {}}]
    blocked = agent.dispatch('render_inputs', {'accession': 'P12345', 'msa_index': {}})
    assert blocked['status'] == 'blocked' and 'identity' in blocked['reason']
    entry = {'primaryAccession': 'P12345', 'entryType': 'active', 'sequence': {'value': 'ACDF'}}
    response = SimpleNamespace(text=json.dumps(entry), json=lambda: entry, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: response)
    result = agent._retrieve_reference('uniprot', 'P12345')
    assert result['identity_status'] == 'conflict'
    assert agent._candidate('P12345')['sequence'] == 'ACDE'
    assert 'conflict' in agent._sequence_identity_error('P12345')


def test_matching_retrieved_identity_is_fingerprinted_and_rechecked(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import requests
    path = write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'})
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object())
    entry = {'primaryAccession': 'P12345', 'entryType': 'active', 'sequence': {'value': 'ACDE'}}
    response = SimpleNamespace(text=json.dumps(entry), json=lambda: entry, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: response)
    source = agent._retrieve_reference('uniprot', 'P12345')
    assert agent._sequence_identity_error('P12345') is None
    Path(source['path']).write_text('changed')
    assert 'changed' in agent._sequence_identity_error('P12345')


def test_builder_requires_sequence_source_and_exact_fasta(tmp_path):
    from structural_modeling_agent.scripts.build_step1_inputs import sequence_identity_errors
    candidate = {'accession': 'P12345', 'sequence': 'ACDE'}
    assert sequence_identity_errors(candidate, 'ACDE', None, tmp_path)
    record = tmp_path / 'identity.json'
    record.write_text(json.dumps({'primaryAccession': 'P12345', 'sequence': {'value': 'ACDE'}}))
    ref = {'path': str(record), 'sha256': hashlib.sha256(record.read_bytes()).hexdigest()}
    assert sequence_identity_errors(candidate, 'ACDE', ref, tmp_path) == []
    assert sequence_identity_errors(candidate, 'GGGG', ref, tmp_path)


def test_investigation_leads_are_readable_without_admission(tmp_path):
    from structural_modeling_agent.handoff_inspect import inspect_handoff
    path = write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'})
    summary = inspect_handoff(path)
    assert summary['admitted_candidates'] == 1
    page = inspect_handoff(path, 'Q12345')
    assert page['admission'] == 'investigation_only'
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object())
    assert agent.dispatch('inspect_investigation_lead', {'accession': 'Q12345'})['admission'] == 'investigation_only'


def test_v3_builder_does_not_render_mismatched_fasta(tmp_path):
    import subprocess
    import sys
    path = write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'})
    fasta = tmp_path / 'seq.fa'; fasta.write_text('>P12345\nGGGG\n')
    msa = tmp_path / 'seq.a3m'; msa.write_text('>P12345\nGGGG\n')
    index = tmp_path / 'msa.tsv'; index.write_text(f'P12345\tignored\t{msa}\n')
    source = tmp_path / 'identity.json'; source.write_text(json.dumps({'primaryAccession': 'P12345', 'sequence': {'value': 'ACDE'}}))
    ids = tmp_path / 'identity-index.json'; ids.write_text(json.dumps({'P12345': {'path': str(source),
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}))
    out = tmp_path / 'rendered'
    result = subprocess.run([sys.executable, '-m', 'structural_modeling_agent.scripts.build_step1_inputs',
        '--handoff', str(path), '--fasta', str(fasta), '--msa-index', str(index), '--out', str(out),
        '--sequence-identity-index', str(ids), '--allow-unreviewed-exploratory'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / 'campaign_manifest.json').read_text())
    assert 'P12345' in report['blocked']
    assert not list(out.glob('*.yaml'))


def test_incomplete_snapshot_refreshes_prior_handoff(tmp_path, monkeypatch):
    from enzyme_discovery_agent.output import report_generator, finalizer
    evaluations = create_run(tmp_path)
    (tmp_path / 'target_context.json').write_text(json.dumps({'query': 'synthetic'}))
    path = write_structural_modeling_handoff(tmp_path, evaluations, {'query': 'synthetic'})
    assert load_document(path)['candidates']
    update_candidate_registry(tmp_path, 'P12345', 'rejected')
    monkeypatch.setattr(report_generator, '_family_relevance_partition', lambda *a: ([], [], []))
    monkeypatch.setattr(report_generator, '_family_rank_map', lambda *a: {})
    monkeypatch.setattr(report_generator, '_deterministic_mechanistic_ranking', lambda *a, **k: [])
    monkeypatch.setattr(report_generator, '_write_deterministic_report',
                        lambda root, name, *a, **k: (root / name).write_text('interim'))
    monkeypatch.setattr(finalizer, 'generate_candidates_csv', lambda *a, **k: None)
    runner = finalizer.RunFinalizer(tmp_path, None, lambda *a, **k: None)
    monkeypatch.setattr(runner, '_load_evaluations', lambda: evaluations)
    runner.snapshot(query='synthetic', reason='test checkpoint')
    doc = validate_handoff(load_document(path))
    assert doc['producer']['discovery_status'] == 'incomplete'
    assert not doc['candidates']
    assert any(x['accession'] == 'P12345' for x in doc['excluded_candidates'])


def test_cache_change_makes_snapshot_stale(tmp_path):
    root = tmp_path / 'discovery'; root.mkdir()
    (root / 'manifest.json').write_text(json.dumps({'run_id': root.name, 'status': 'running'}))
    (root / 'target_context.json').write_text(json.dumps({'query': 'synthetic'}))
    path = write_structural_modeling_handoff(root, create_run(root), {'query': 'synthetic'})
    folder = root / 'embedding_store'; folder.mkdir()
    (folder / 'sequences.json').write_text(json.dumps({'P12345': 'ACDF'}))
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / 'consumer', llm_client=object(),
                          discovery_run_path=root)
    assert agent.state['discovery_currentness']['status'] == 'stale'
    assert 'sequence_cache' in agent.state['discovery_currentness']['changed']


def test_reviewed_report_can_drive_validated_v3_builder(tmp_path):
    import importlib.util
    import subprocess
    import sys
    from structural_modeling_agent.validation import fingerprint
    discovery = tmp_path / 'discovery'; discovery.mkdir()
    update_candidate_registry(discovery, 'P12345', 'evaluated', {'claim_validation': validation('P12345'), 'sequence': 'AG'})
    handoff = write_structural_modeling_handoff(discovery,
        [{'status': 'success', 'data': {'accession': 'P12345'}}], {'query': 'synthetic'})
    modeling = tmp_path / 'modeling'; sources = modeling / 'sources'; sources.mkdir(parents=True)
    helper_path = Path(__file__).resolve().parents[2] / 'structural_modeling_agent' / 'tests' / 'test_validation.py'
    spec = importlib.util.spec_from_file_location('handoff_test_validation', helper_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    record = module.modeling_record(sources)
    report = modeling / 'report.json'
    report.write_text(json.dumps({'schema_version': 'modeling-agent-1.0',
        'task_sha256': fingerprint(load_document(handoff)),
        'candidates': [{'accession': 'P12345', 'modeling_input': record}]}))
    fasta = tmp_path / 'seq.fa'; fasta.write_text('>P12345\nAG\n')
    msa = tmp_path / 'seq.csv'; msa.write_text('key,sequence\n0,AG\n')
    index = tmp_path / 'msa.tsv'; index.write_text(f'P12345\tignored\t{msa}\n')
    source = tmp_path / 'identity.json'; source.write_text(json.dumps({'primaryAccession': 'P12345', 'sequence': {'value': 'AG'}}))
    ids = tmp_path / 'identity-index.json'; ids.write_text(json.dumps({'P12345': {'path': str(source),
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}))
    out = tmp_path / 'rendered'
    result = subprocess.run([sys.executable, '-m', 'structural_modeling_agent.scripts.build_step1_inputs',
        '--handoff', str(handoff), '--fasta', str(fasta), '--msa-index', str(index), '--out', str(out),
        '--sequence-identity-index', str(ids), '--reviewed-decisions', str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / 'campaign_manifest.json').read_text())
    assert manifest['blocked'] == {}
    assert manifest['executions'][0]['sequence_identity_source']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (out / 'step1_P12345.yaml').exists()


def test_identity_evidence_rechecked_when_report_is_exported(tmp_path):
    from structural_modeling_agent.evidence import modeling_evidence
    model = tmp_path / 'model'; sources = model / 'sources'; sources.mkdir(parents=True)
    source = sources / 'identity.txt'
    source.write_text(json.dumps({'primaryAccession': 'P12345', 'sequence': {'value': 'AG'}}))
    report = model / 'report.json'
    report.write_text(json.dumps({'schema_version': 'modeling-agent-1.0', 'sequence_identity_requirement': 'exact_uniprot_primary',
        'candidates': [{'accession': 'P12345', 'sequence_sha256': hashlib.sha256(b'AG').hexdigest(),
            'sequence_identity_source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                         'source_id': 'uniprot:P12345'}}]}))
    before = modeling_evidence(report)['candidates'][0]['review_errors']
    assert not any('sequence identity' in error for error in before)
    source.write_text('modified')
    after = modeling_evidence(report)['candidates'][0]['review_errors']
    assert any('sequence identity' in error for error in after)


@pytest.mark.parametrize('mutate', [
    lambda d: d.pop('created_at'),
    lambda d: d['producer'].pop('run_id'),
    lambda d: d['producer'].pop('source_artifact_sha256'),
    lambda d: d.update(ranking={}),
    lambda d: d['candidates'][0]['selection'].update(rank='not-a-rank'),
    lambda d: d['candidates'][0]['selection'].update(admission='admitted'),
])
def test_incomplete_v3_metadata_is_rejected(tmp_path, mutate):
    doc = load_document(write_structural_modeling_handoff(tmp_path, create_run(tmp_path), {'query': 'synthetic'},
                                                          discovery_status_value='incomplete'))
    mutate(doc)
    doc['integrity']['snapshot_sha256'] = digest(snapshot_payload(doc))
    with pytest.raises(ValueError):
        validate_handoff(doc)
