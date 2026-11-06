"""Memory recovery preserves scientific settings and never reuses a failed tier."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from structural_modeling_agent import memory_retry as retry
from structural_modeling_agent import family_cofold as family
from structural_modeling_agent.lambda_boltz_backend import eligible_gpus, memory_failure


@pytest.fixture
def failed(tmp_path, monkeypatch):
    native = tmp_path / 'old_provider/backend_jobs/job'
    native.mkdir(parents=True)
    receipt = {'status': 'failed', 'failure_kind': 'gpu_out_of_memory',
        'idempotency_key': 'job', 'host': 'synthetic_host', 'pid': 999999,
        'process_start_ticks': '123', 'started_at': 100,
        'gpu_memory_total_mib': 32768, 'provider_state_dir': str(native.parents[1])}
    (native / 'receipt.json').write_text(json.dumps(receipt))
    (native / 'partial_output').write_text('preserve')
    folder = tmp_path / 'cohort/jobs/job'
    folder.mkdir(parents=True)
    (folder / 'receipt.json').write_text(json.dumps(receipt))
    (folder / 'failure.json').write_text(json.dumps({'error': 'old failure'}))
    backend = {'argv': ['provider', '--state-dir', str(tmp_path / 'new_provider')],
        'resource_policy': {'execution_host': 'synthetic_host', 'memory_retry_cooldown_seconds': 0}}
    monkeypatch.setattr(retry, 'gpu_inventory', lambda policy: {0: 32768, 1: 95830})
    return folder, native, receipt, backend


def test_zero_exit_boltz_oom_is_retryable_but_unrelated_errors_are_not(tmp_path):
    output = tmp_path / 'outputs'
    (tmp_path / 'predictor.log').write_text('WARNING: ran out of memory, skipping batch\nNumber of failed examples: 1')
    receipt = {'status': 'failed', 'error': 'Incomplete prediction: rc=0, coordinates=0, confidence=0',
               'argv': ['predict', '--out_dir', str(output)]}
    assert retry.is_memory_failure(receipt)
    assert memory_failure((tmp_path / 'predictor.log').read_text())
    assert not retry.is_memory_failure({'status': 'failed', 'error': 'Invalid substrate identity'})
    assert not retry.is_memory_failure({**receipt, 'status': 'complete'})


def test_undersized_worker_defers_without_deleting_or_submitting(failed, monkeypatch):
    folder, native, receipt, backend = failed
    monkeypatch.setattr(retry, 'gpu_inventory', lambda policy: {0: 32768})
    with pytest.raises(retry.MemoryRetryPending, match='Waiting for GPU'):
        retry.retry_backend(folder, backend, lambda receipt: False)
    assert (native / 'partial_output').read_text() == 'preserve'
    assert (folder / 'receipt.json').exists()
    state = json.loads((folder / 'memory_retry.json').read_text())
    assert state['status'] == 'waiting_for_memory'
    assert state['minimum_gpu_memory_mib'] > 32768


def test_eligible_retry_archives_failed_attempt_and_filters_actual_device(failed):
    folder, native, receipt, backend = failed
    original = copy.deepcopy(backend)
    result = retry.retry_backend(folder, backend, lambda receipt: False)
    assert backend == original
    assert result['resource_policy']['allowed_gpu_indices'] == [1]
    assert result['resource_policy']['minimum_gpu_memory_mib'] > 32768
    state = json.loads((folder / 'memory_retry.json').read_text())
    assert Path(state['attempts'][0]['archived_native'], 'partial_output').read_text() == 'preserve'
    assert list((folder / 'memory_attempts').glob('*/receipt.json'))
    assert not (folder / 'failure.json').exists()
    # A reservation race can only choose devices that still satisfy the floor.
    assert eligible_gpus([0, 1], {0: 32768, 1: 95830}, result['resource_policy']) == [1]


def test_live_attempt_cannot_be_archived(failed):
    folder, native, receipt, backend = failed
    with pytest.raises(retry.MemoryRetryPending, match='still owns'):
        retry.retry_backend(folder, backend, lambda receipt: True)
    assert native.exists() and (folder / 'receipt.json').exists()


def test_polling_does_not_consume_attempts_and_repeated_ooms_are_bounded(failed):
    folder, native, receipt, backend = failed
    for _ in range(5):
        state = retry.record_failure(folder, receipt, backend['resource_policy'])
    assert len(state['attempts']) == 1
    for index in [1, 2]:
        state = retry.record_failure(folder, {**receipt, 'started_at': 100 + index}, backend['resource_policy'])
    assert state['status'] == 'exhausted'
    with pytest.raises(retry.MemoryRetryPending, match='limit reached'):
        retry.retry_backend(folder, backend, lambda receipt: False)


def test_cooldown_cannot_consume_another_attempt(failed, monkeypatch):
    folder, native, receipt, backend = failed
    backend['resource_policy']['memory_retry_cooldown_seconds'] = 60
    with pytest.raises(retry.MemoryRetryPending, match='cooldown'):
        retry.retry_backend(folder, backend, lambda receipt: False)
    assert len(json.loads((folder / 'memory_retry.json').read_text())['attempts']) == 1


def test_memory_retry_submits_identical_input_msa_and_fifty_samples(failed, monkeypatch):
    folder, native, receipt, backend = failed
    job = {'id': 'job', 'input_yaml': '/synthetic/input.yaml', 'input_sha256': 'input-hash',
           'msa_path': '/synthetic/prepared.csv', 'msa_sha256': 'msa-hash'}
    calls = []
    def submit(argv, **kwargs):
        calls.append(json.loads(kwargs['input']))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'complete'}))
    monkeypatch.setattr(family, 'provider_owner_live', lambda receipt: False)
    monkeypatch.setattr(family.subprocess, 'run', submit)
    assert family.execute_job(job, backend, folder)['status'] == 'complete'
    request = calls[0]
    assert request['input_yaml'] == job['input_yaml']
    assert request['input_sha256'] == job['input_sha256']
    assert request['expected_predictions'] == 50
    assert request['msa_evidence'] == [{'path': job['msa_path'], 'sha256': job['msa_sha256']}]
    assert request['resource_policy']['allowed_gpu_indices'] == [1]


def test_full_saved_ensemble_blocks_refolding_until_validation(failed):
    folder, native, receipt, backend = failed
    saved = folder.parents[1] / 'workers/peer/provider/backend_jobs/job'
    outputs = saved / 'outputs';outputs.mkdir(parents=True)
    (saved / 'receipt.json').write_text(json.dumps({'status': 'failed'}))
    (saved / 'request.json').write_text(json.dumps({'input_sha256': 'same-input'}))
    for i in range(50):
        (outputs / f'prediction_model_{i}.pdb').touch()
        (outputs / f'confidence_prediction_model_{i}.json').touch()
    with pytest.raises(retry.MemoryRetryPending, match='producer validation'):
        retry.retry_backend(folder, backend, lambda receipt: False, {'input_sha256': 'same-input'})
    assert native.exists()
    assert json.loads((folder / 'memory_retry.json').read_text())['status'] == 'coordinates_need_validation'


def test_unreachable_owner_is_resource_deferral_and_cannot_authorize_replacement(monkeypatch):
    import subprocess
    monkeypatch.setattr(family.socket, 'gethostname', lambda: 'controller')
    def unreachable(*args, **kwargs):
        raise subprocess.CalledProcessError(255, args[0], stderr='SSH temporarily unavailable')
    monkeypatch.setattr(family.subprocess, 'run', unreachable)
    with pytest.raises(retry.MemoryRetryPending, match='preserve attempt'):
        family.provider_owner_live({'host': 'remote', 'pid': 1234, 'process_start_ticks': '5678'})
