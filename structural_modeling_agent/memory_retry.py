"""Durable, bounded GPU-memory recovery without changing scientific inputs."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import socket
import subprocess
import time

from enzyme_discovery_agent.state_store import atomic_write_json, state_lock


class MemoryRetryPending(RuntimeError):
    """Resource deferral, not a failed scientific setup or worker failure."""


def is_memory_failure(receipt):
    if receipt.get('status') != 'failed':
        return False
    if receipt.get('failure_kind') == 'gpu_out_of_memory':
        return True
    text = str(receipt.get('error', '')).lower()
    argv = receipt.get('argv', [])
    if '--out_dir' in argv:
        log = Path(argv[argv.index('--out_dir') + 1]).parent / 'predictor.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 262144))
                text += stream.read().decode(errors='replace').lower()
    return any(value in text for value in ('cuda out of memory', 'cuda error: out of memory',
                                          'ran out of memory, skipping batch', 'torch.outofmemoryerror'))


def gpu_inventory(policy):
    """Query installed memory, including busy devices; reservations happen later."""
    argv = ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits']
    host = policy.get('execution_host', 'lambda13')
    if host != socket.gethostname().split('.')[0]:
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, shlex.join(argv)]
    result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=25)
    inventory = {int(line.split(',')[0]): int(line.split(',')[1])
                 for line in result.stdout.splitlines() if line.strip()}
    allowed = policy.get('allowed_gpu_indices')
    if allowed is not None:
        inventory = {index: value for index, value in inventory.items() if index in allowed}
    if policy.get('gpu_index') is not None and not policy.get('allow_any_free_gpu'):
        inventory = {index: value for index, value in inventory.items() if index == policy['gpu_index']}
    return inventory


def record_failure(directory, receipt, policy):
    """Count distinct actual OOM attempts, never polling or resource contention."""
    directory = Path(directory)
    path = directory / 'memory_retry.json'
    identity = hashlib.sha256(json.dumps({k: receipt.get(k) for k in
        ('provider_state_dir', 'host', 'pid', 'process_start_ticks', 'started_at')}, sort_keys=True).encode()).hexdigest()
    with state_lock(directory / 'memory_retry'):
        state = json.loads(path.read_text()) if path.exists() else {'attempts': []}
        if not any(row['identity'] == identity for row in state['attempts']):
            measured = max(receipt.get('peak_reserved_memory') or 0, receipt.get('peak_allocated_memory') or 0)
            capacity = receipt.get('gpu_memory_total_mib') or math.ceil(measured / 1024**2)
            # Legacy receipts may lack memory measurements. Require an explicit
            # larger tier rather than silently repeating on the failed device.
            if not capacity:
                previous_policy = dict(policy, execution_host=receipt['host'],
                    gpu_index=receipt.get('gpu'), allow_any_free_gpu=False)
                previous_policy.pop('allowed_gpu_indices', None)
                capacity = gpu_inventory(previous_policy).get(receipt.get('gpu'))
            if not capacity:
                raise MemoryRetryPending('Cannot verify failed GPU capacity; memory retry remains pending')
            state['attempts'].append({'identity': identity, 'receipt': receipt,
                'failed_gpu_memory_mib': capacity, 'recorded_at': time.time()})
            state['minimum_gpu_memory_mib'] = max(state.get('minimum_gpu_memory_mib', 0), math.ceil(capacity * 1.25))
            state['not_before'] = time.time() + policy.get('memory_retry_cooldown_seconds', 60)
        state['max_oom_attempts'] = policy.get('max_oom_attempts', 3)
        state['status'] = 'exhausted' if len(state['attempts']) >= state['max_oom_attempts'] else 'waiting_for_memory'
        atomic_write_json(path, state)
    return state


def retry_backend(directory, backend, owner_live, job=None):
    """Defer undersized workers; archive a dead attempt before an eligible retry."""
    directory = Path(directory)
    path = directory / 'memory_retry.json'
    receipt_path = directory / 'receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if is_memory_failure(receipt):
        state = record_failure(directory, receipt, backend['resource_policy'])
    elif path.exists():
        state = json.loads(path.read_text())
    else:
        return backend
    # A pending/recovered producer must be reconciled by the ordinary receipt path.
    if receipt.get('status') in {'queued', 'running', 'unknown', 'complete'}:
        return backend
    if state['status'] == 'exhausted':
        raise MemoryRetryPending('GPU-memory retry limit reached; attempts retained')
    # Another worker may have written a full ensemble despite a failed wrapper.
    # Preserve and validate those coordinates before spending another GPU attempt.
    for native_receipt in directory.parents[1].glob('workers/*/provider/backend_jobs/'+receipt.get('idempotency_key', directory.name)+'/receipt.json'):
        native = json.loads(native_receipt.read_text())
        request_path = native_receipt.parent / 'request.json'
        if not request_path.exists():
            continue
        if job and json.loads(request_path.read_text()).get('input_sha256') != job['input_sha256']:
            continue
        if native.get('status') == 'complete':
            native['provider_state_dir'] = str(native_receipt.parents[2])
            atomic_write_json(receipt_path, native)
            return backend
        outputs = native_receipt.parent / 'outputs'
        if (len(list(outputs.rglob('*_model_*.pdb'))) == 50
                and len(list(outputs.rglob('confidence_*_model_*.json'))) == 50):
            state.update(status='coordinates_need_validation', saved_receipt=str(native_receipt))
            atomic_write_json(path, state)
            raise MemoryRetryPending('Full saved coordinates require producer validation before another cofold')
    if time.time() < state['not_before']:
        raise MemoryRetryPending('GPU-memory retry cooldown')
    policy = backend['resource_policy']
    inventory = gpu_inventory(policy)
    minimum = max(state['minimum_gpu_memory_mib'], policy.get('minimum_gpu_memory_mib', 0))
    eligible = [index for index, size in inventory.items() if size >= minimum]
    if not eligible:
        raise MemoryRetryPending(f"Waiting for GPU with at least {state['minimum_gpu_memory_mib']} MiB memory")
    if receipt:
        if receipt.get('status') != 'failed' or not is_memory_failure(receipt):
            raise MemoryRetryPending('Prior attempt requires reconciliation before memory retry')
        if owner_live(receipt):
            raise MemoryRetryPending('Previous producer still owns its attempt')
        native_root = Path(receipt['provider_state_dir'])
        native = native_root / 'backend_jobs' / receipt['idempotency_key']
        if native.exists():
            current = json.loads((native / 'receipt.json').read_text())
            if current.get('status') != 'failed':
                raise MemoryRetryPending('Native producer state changed; reconcile before retry')
            archive = native_root / 'memory_attempts' / receipt['idempotency_key'] / str(time.time_ns())
            archive.parent.mkdir(parents=True, exist_ok=True)
            native.rename(archive)
            state['attempts'][-1]['archived_native'] = str(archive)
        archive = directory / 'memory_attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        receipt_path.rename(archive / 'receipt.json')
        failure = directory / 'failure.json'
        if failure.exists():
            failure.rename(archive / 'failure.json')
    result = copy.deepcopy(backend)
    result['resource_policy'].update(minimum_gpu_memory_mib=minimum, allowed_gpu_indices=eligible)
    state['status'] = 'retrying'
    atomic_write_json(path, state)
    return result
