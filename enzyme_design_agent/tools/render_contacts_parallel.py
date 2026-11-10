"""Bounded CPU-only rendering launcher for the validated M3/M4 contact audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--first-cpu', type=int, default=44)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--m4-workers', type=int, default=6)
    p.add_argument('--m3-workers', type=int, default=4)
    a = p.parse_args()
    n = a.m4_workers+a.m3_workers
    if a.threads < 1 or a.m4_workers < 1 or a.m3_workers < 1:
        raise RuntimeError('Positive thread and shard counts required')
    slots = 1+n*a.threads
    cpus = set(range(a.first_cpu, a.first_cpu+slots))
    if slots > 32 or not cpus <= os.sched_getaffinity(0):
        raise RuntimeError('Allocation exceeds allowed affinity or the 32-CPU project budget')
    if os.getloadavg()[0]+slots > os.cpu_count():
        raise RuntimeError('Current load plus requested CPU slots exceeds capacity')
    # The coordinator has a separate CPU; rendering processes inherit disjoint sets.
    os.sched_setaffinity(0, {a.first_cpu})
    env = os.environ.copy()
    caps = ['OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
    env.update({k: '1' for k in caps})
    logs = a.audit/'render_logs'
    logs.mkdir(exist_ok=True)
    workers, streams = [], []
    source = Path(__file__).with_name('render_contact_audit.py')
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    interpreter=sys.executable, cpu_count=os.cpu_count(), initial_load=os.getloadavg(),
                    coordinator_cpu=a.first_cpu, allocated_slots=slots, thread_caps={k: env[k] for k in caps},
                    gpu_use=False, workers=[])
    start = time.monotonic()
    try:
        for camp, count in [('M4', a.m4_workers), ('M3', a.m3_workers)]:
            for shard in range(count):
                first = a.first_cpu+1+len(workers)*a.threads
                assigned = list(range(first, first+a.threads))
                cmd = ['taskset', '-c', ','.join(map(str, assigned)), sys.executable, '-u', '-m',
                       'enzyme_design_agent.tools.render_contact_audit', 'render', '--audit', str(a.audit),
                       '--campaign', camp, '--shards', str(count), '--shard', str(shard), '--threads', str(a.threads)]
                log = (logs/f'{camp}_{shard}.log').open('w')
                streams.append(log)
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                worker = dict(campaign=camp, shard=shard, cpus=assigned, pid=process.pid, command=cmd)
                workers.append((process, worker))
                manifest['workers'].append(worker)
        path = a.audit/'render_launch.json'
        path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Launched {n} render shards; {slots} total reserved CPU slots', flush=True)
        verified = False
        while any(proc.poll() is None for proc, _ in workers):
            time.sleep(2)
            for proc, worker in workers:
                if proc.poll() is not None:
                    if proc.returncode:
                        raise RuntimeError(f'Render worker failed: {worker}')
                    continue
                affinity = sorted(os.sched_getaffinity(proc.pid))
                threads = len(list(Path(f'/proc/{proc.pid}/task').iterdir()))
                worker['verified_affinity'] = affinity
                worker['observed_threads'] = max(threads, worker.get('observed_threads', 0))
                if affinity != worker['cpus'] or threads > a.threads:
                    raise RuntimeError(f'Worker resource contract violated: {worker}, threads={threads}')
            if not verified and time.monotonic()-start > 8:
                print('Verified live worker affinities and thread counts', flush=True)
                verified = True
                path.write_text(json.dumps(manifest, indent=2)+'\n')
        for proc, worker in workers:
            if proc.wait():
                raise RuntimeError(f'Render worker failed: {worker}')
        manifest['elapsed_seconds'] = time.monotonic()-start
        manifest['status'] = 'complete'
        path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'All render shards complete in {manifest["elapsed_seconds"]:.1f} s', flush=True)
    except BaseException:
        for proc, _ in workers:
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in workers:
            proc.wait()
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
