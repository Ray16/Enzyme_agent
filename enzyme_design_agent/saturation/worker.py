"""Reservation-owned resumable folding worker with output-based completion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from .evaluate import verified_paths, verify_outputs
from .msa import valid_cache


def atomic_json(path,data):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def run(manifest_path,cache,max_sequence_length=None):
    from .predict import capped_source
    import inspect,textwrap
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    import torch
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:
        raise RuntimeError('Worker must run inside a single-GPU reservation')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    capped_source(textwrap.dedent(inspect.getsource(AtomDiffusion.sample)))
    m=json.loads(Path(manifest_path).read_text());out=Path(m['output_dir'])
    for name in ('claims','attempts','logs','results','archive','workers'):(out/name).mkdir(exist_ok=True)
    owner={'host':socket.gethostname(),'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],
           'cpu_affinity':sorted(os.sched_getaffinity(0)),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
           'python':sys.executable,'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
           'torch_version':torch.__version__,'sampler_source_sha256':hashlib.sha256(inspect.getsource(AtomDiffusion.sample).encode()).hexdigest()}
    worker_record=out/'workers'/f"{owner['host']}_{owner['gpu']}_{owner['pid']}.json"
    atomic_json(worker_record,owner)
    child=None;current_claim=None
    def stop(sig,frame):
        nonlocal child
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        if current_claim and current_claim.exists():
            (current_claim/'owner.json').unlink(missing_ok=True);current_claim.rmdir()
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    jobs=sorted(m['jobs'],key=lambda j:(j['mutation']!='WT',j['id'].split('_',1)[-1],j['candidate']))
    if max_sequence_length is not None:
        jobs = [j for j in jobs if len(j['sequence']) <= max_sequence_length]
    candidates = {c['id']: c for c in m['candidates']}
    failures=0
    while True:
        ran=False;cooldown=False
        for job in jobs:
            ident=job['id'];config=out/'inputs'/(ident+'.yaml')
            if verified_paths(out,job,m['n_samples']) is not None:continue
            if not config.exists():continue
            cfg=json.loads(config.read_text());protein=next(e['protein'] for e in cfg['sequences'] if 'protein' in e)
            if protein['sequence']!=job['sequence'] or not valid_cache(Path(protein['msa']),job['sequence']):
                raise RuntimeError('Invalid sequence-specific input')
            ap=out/'attempts'/(ident+'.json')
            previous=json.loads(ap.read_text()) if ap.exists() else {'count':0,'retry_after':0}
            if previous['count']>=3:continue
            if previous['retry_after']>time.time():cooldown=True;continue
            claim=out/'claims'/ident
            try:claim.mkdir()
            except FileExistsError:continue
            current_claim=claim
            atomic_json(claim/'owner.json',owner)
            try:
                if verify_outputs(out,job,candidates[job['candidate']],m['n_samples']):continue
                result=out/'results'/ident
                attempt=previous['count']+1
                if result.exists():
                    archive=out/'archive'/f'{ident}_before_attempt{attempt}_{time.time_ns()}'
                    result.rename(archive)
                    atomic_json(archive/'replacement_reason.json',{'reason':'Incomplete output set','next_attempt':attempt})
                argv=[sys.executable,'-u','-m','enzyme_design_agent.saturation.predict','predict',str(config),
                      '--out_dir',str(result),'--cache',cache,'--diffusion_samples',str(m['n_samples']),
                      '--max_parallel_samples','1','--output_format','pdb','--accelerator','gpu','--devices','1',
                      '--num_workers','0','--preprocessing-threads','1','--seed',str(m['seed']),
                      '--recycling_steps','3','--sampling_steps','200']
                record={**owner,'count':attempt,'argv':argv,'started':time.time(),'retry_after':0,
                        'input_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
                atomic_json(ap,record)
                print('START',ident,'attempt',attempt,flush=True)
                with (out/'logs'/f'{ident}.attempt{attempt}.log').open('w') as log:
                    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    record['predictor_pid']=child.pid;atomic_json(ap,record)
                    rc=child.wait()
                child=None;ran=True
                complete=verify_outputs(out,job,candidates[job['candidate']],m['n_samples'])
                record.update(finished=time.time(),returncode=rc,complete=complete,retry_after=0 if complete else time.time()+120)
                atomic_json(ap,record)
                failures=0 if complete else failures+1
                print('DONE' if complete else 'FAILED',ident,'rc',rc,flush=True)
                if failures>=3:
                    print('CIRCUIT_BREAK',flush=True)
                    atomic_json(worker_record,{**owner,'status':'circuit_break','time':time.time()})
                    return 4
            finally:
                if current_claim:
                    (claim/'owner.json').unlink(missing_ok=True);claim.rmdir();current_claim=None
        if not ran:
            # Release the GPU; a coordinator may retry cooled-down or newly ready work.
            atomic_json(worker_record,{**owner,'status':'idle_exit','cooldown':cooldown,'time':time.time()})
            return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--cache',required=True)
    p.add_argument('--max-sequence-length',type=int)
    a=p.parse_args();sys.exit(run(a.manifest,a.cache,a.max_sequence_length))
