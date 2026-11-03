"""One reserved GPU, resumable atomic task claims, and retained failed attempts."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback

BOOT = time.monotonic()
import mdtraj as md
import torch
from bioemu2 import System, EmuSession, get_msa, load_model, map_cif_atom_order, check_physicality, save_frames
from core import digest, write_json, frame_metrics
IMPORT_SECONDS = time.monotonic() - BOOT


def process_identity():
    return dict(host=socket.gethostname(),pid=os.getpid(),
        process_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        thread_caps={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})


def complete(root, task):
    p=root/'tasks'/task['id']/'result.json'
    if not p.is_file():
        return False
    r=json.loads(p.read_text())
    if r.get('task') != task or len(r.get('files',[])) != 4:
        raise RuntimeError('Result identity/coverage mismatch: '+str(p))
    return all(Path(x['path']).is_file() and digest(x['path'])==x['sha256'] for x in r['files'])


def watch_gpu(root, identity, stop):
    """Stop only this worker if a foreign resident process overlaps its GPU."""
    physical = identity['gpu_visibility']
    uuid=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    while not stop.wait(10):
        try:
            output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            foreign=[int(row.split(',')[1]) for row in output.splitlines()
                if row.split(',')[0].strip()==uuid and int(row.split(',')[1])!=os.getpid()]
            if foreign:
                write_json(root/'workers'/f'{identity["host"]}_{os.getpid()}_conflict.json',dict(identity=identity,foreign_pids=foreign))
                os.kill(os.getpid(),signal.SIGTERM)
                return
        except (subprocess.SubprocessError,ValueError):
            # Loss of visibility means the no-contention guarantee cannot be verified.
            os.kill(os.getpid(),signal.SIGTERM)
            return


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--only')
    args=parser.parse_args()
    root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    ident=process_identity()
    if not ident['gpu_visibility'] or ',' in ident['gpu_visibility'] or not torch.cuda.is_available():
        raise RuntimeError('Exactly one externally reserved CUDA GPU is required')
    for key,value in ident['thread_caps'].items():
        if value!='1': raise RuntimeError('Missing one-thread cap: '+key)
    if max(os.getloadavg())+len(ident['cpu_affinity']) > os.cpu_count():
        raise RuntimeError('CPU capacity unavailable')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    worker_path=root/'workers'/f'{ident["host"]}_{os.getpid()}.json'
    write_json(worker_path,dict(identity=ident,status='loading',import_seconds=IMPORT_SECONDS))
    indexed={f['path']:f['sha256'] for f in manifest['files']}
    checkpoint=manifest['checkpoint']
    for path in [checkpoint,str(Path(checkpoint).with_name('config.yaml'))]:
        if digest(path)!=indexed[path]: raise RuntimeError('Checkpoint/config changed')
    stop=threading.Event()
    threading.Thread(target=watch_gpu,args=(root,ident,stop),daemon=True).start()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('terminated')))
    load_begin=time.monotonic()
    model=load_model(checkpoint_path=Path(checkpoint),device='cuda',precision='32-true')
    torch.cuda.synchronize()
    load_seconds=time.monotonic()-load_begin
    tasks=[t for t in manifest['tasks'] if not args.only or t['id']==args.only]
    if not tasks: raise ValueError('No matching tasks')
    consecutive=0
    session=None
    current_enzyme=None
    completed=0
    owned_enzyme=None
    enzyme_claim=None
    def release_enzyme():
        nonlocal owned_enzyme,enzyme_claim
        if enzyme_claim is not None:
            (enzyme_claim/'owner.json').unlink(missing_ok=True)
            enzyme_claim.rmdir()
        owned_enzyme=None
        enzyme_claim=None
    try:
        while True:
            claimable=0
            held=0
            for task in tasks:
                if complete(root,task): continue
                dest=root/'tasks'/task['id']
                dest.mkdir(parents=True,exist_ok=True)
                attempts=list(dest.glob('attempt_*/failure.json'))
                if len(attempts)>=2: continue
                if attempts and time.time()-max(p.stat().st_mtime for p in attempts)<30: continue
                # Keep all four endpoints of an enzyme together to reuse its costly
                # embedding, while retaining independently resumable task chunks.
                if task['enzyme']!=owned_enzyme:
                    release_enzyme()
                    group=root/'enzyme_claims'/task['enzyme']
                    group.parent.mkdir(parents=True,exist_ok=True)
                    try: group.mkdir()
                    except FileExistsError:
                        held+=1
                        continue
                    enzyme_claim=group
                    owned_enzyme=task['enzyme']
                    write_json(group/'owner.json',dict(identity=ident,enzyme=owned_enzyme,claimed_at=time.time()))
                claim=root/'claims'/task['id']
                claim.parent.mkdir(parents=True,exist_ok=True)
                try: claim.mkdir()
                except FileExistsError:
                    held+=1
                    continue
                write_json(claim/'owner.json',dict(identity=ident,task=task['id'],claimed_at=time.time()))
                claimable+=1
                attempt=dest/f'attempt_{len(list(dest.glob("attempt_*")))+1}'
                attempt.mkdir()
                try:
                    if complete(root,task): continue
                    for path in [task['system'],task['msa'],task['initial']]:
                        if digest(path)!=indexed[path]: raise RuntimeError('Input changed: '+path)
                    fresh_embedding=task['enzyme']!=current_enzyme
                    if fresh_embedding:
                        session=None
                        gc.collect()
                        torch.cuda.empty_cache()
                        system=System.from_yaml(Path(task['system']))
                        msa=get_msa(system,source={'A':Path(task['msa'])})[:128]
                        session=EmuSession(system,model=model,mode='equilibrium',msa=msa)
                        current_enzyme=task['enzyme']
                    start=map_cif_atom_order(system,Path(task['initial']))
                    torch.cuda.reset_peak_memory_stats()
                    begin=time.monotonic()
                    write_json(worker_path,dict(identity=ident,status='predicting',completed=completed,
                        last_task=task['id'],generation_started_epoch=time.time(),
                        startup_to_generation_seconds=begin-BOOT))
                    print('PREDICTION_START',task['id'],json.dumps(ident),flush=True)
                    endpoint=session.propagate(start,lag_time_ns=task['lag_ns'],batch_size=1,seed=task['seed'])
                    torch.cuda.synchronize()
                    generation_seconds=time.monotonic()-begin
                    if endpoint.n_frames!=1 or endpoint.n_atoms!=start.n_atoms:
                        raise RuntimeError('Incomplete endpoint')
                    physicality=check_physicality(endpoint,system=system)
                    save_frames(endpoint,attempt/'raw',physicality=physicality)
                    roundtrip=md.load_xtc(str(attempt/'raw/samples.xtc'),top=endpoint.topology)
                    if roundtrip.n_frames!=1 or roundtrip.n_atoms!=start.n_atoms:
                        raise RuntimeError('Saved trajectory coverage mismatch')
                    metrics=frame_metrics(start,roundtrip)
                    # This is a geometry measurement, not a ligand chemistry validation.
                    metrics.update(protein_physicality_passed=bool(physicality.passed[0]),
                        protein_physicality_reason=physicality.reasons[0])
                    write_json(attempt/'metrics.json',metrics)
                    supporting=[attempt/'raw/samples.xtc',attempt/'raw/topology.cif',
                        attempt/'raw/physicality.json',attempt/'metrics.json']
                    result=dict(task=task,identity=ident,metrics=metrics,
                        files=[dict(path=str(p),sha256=digest(p)) for p in supporting],
                        timing={'import_seconds':IMPORT_SECONDS,'model_load_seconds':load_seconds,
                            'generation_entrypoint_seconds':generation_seconds,
                            'entrypoint_includes_first_embedding':fresh_embedding,
                            'startup_to_generation_seconds':begin-BOOT,
                            'process_elapsed_seconds':time.monotonic()-BOOT,
                            'scheduler_wait_seconds':None,'warmup_seconds':None,
                            'average_generation_seconds_per_completed_endpoint':generation_seconds},
                        resources={'device':torch.cuda.get_device_name(),'precision':'32-true','batch_size':1,
                            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                            'protein_length':task['sequence_length'],'requested_samples':1,'completed_samples':1})
                    write_json(dest/'result.json',result)
                    if not complete(root,task): raise RuntimeError('Result verification failed')
                    completed+=1
                    consecutive=0
                    print('PREDICTION_COMPLETE',task['id'],generation_seconds,metrics,flush=True)
                    write_json(worker_path,dict(identity=ident,status='running',completed=completed,last_task=task['id']))
                except Exception:
                    consecutive+=1
                    write_json(attempt/'failure.json',dict(task=task,identity=ident,error=traceback.format_exc(),time=time.time()))
                    print(traceback.format_exc(),flush=True)
                    session=None
                    current_enzyme=None
                    if consecutive>=3: raise RuntimeError('Three consecutive prediction failures; worker circuit breaker')
                finally:
                    (claim/'owner.json').unlink(missing_ok=True)
                    claim.rmdir()
            release_enzyme()
            remaining=[t for t in tasks if not complete(root,t)]
            if not remaining:
                state='complete'
                break
            retryable=[t for t in remaining if len(list((root/'tasks'/t['id']).glob('attempt_*/failure.json')))<2]
            if not retryable:
                state='terminal_failures'
                break
            if not claimable and held==len(retryable):
                state='remaining_work_held_by_peers'
                break
            time.sleep(5)
    finally:
        release_enzyme()
        stop.set()
    write_json(worker_path,dict(identity=ident,status=state,completed=completed,elapsed_seconds=time.monotonic()-BOOT))


if __name__=='__main__':
    main()
