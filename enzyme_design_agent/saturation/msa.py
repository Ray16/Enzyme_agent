"""Fresh sequence-specific MSA batches with content-addressed server caches."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path


def _aligned_length(sequence):
    # A3M insertions are ASCII lowercase. Regex avoids a Python loop over every
    # character of every alignment on repeated readiness checks.
    return len(re.sub('[a-z]', '', sequence))


def transient_service_error(exc):
    parts=[];seen=set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc));parts.append(type(exc).__name__+': '+str(exc))
        exc=exc.__cause__ or exc.__context__
    message=' '.join(parts).lower()
    return any(x in message for x in ['connectionerror','connectionpool','nameresolution','name resolution','gaierror',
        'timeouterror','timeout','timed out','temporary failure','too many failed attempts','maintenance',
        '429','502','503','504','msa generation request'])


def request_batch(fetch,sequences,prefix,max_attempts=8,retry_seconds=60,sleep=time.sleep,on_retry=None):
    for attempt in range(1,max_attempts+1):
        try:return fetch(sequences,prefix,use_env=True,use_pairing=False)
        except Exception as exc:
            if not transient_service_error(exc) or attempt==max_attempts:raise
            delay=min(900,retry_seconds*2**(attempt-1))
            if on_retry:on_retry(attempt,delay,exc)
            print(f'MSA service retry {attempt}/{max_attempts}: {type(exc).__name__}: {exc}; waiting {delay}s',flush=True)
            sleep(delay)


def alignment_csv(a3m, query):
    seqs = []
    for line in a3m.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            seqs.append('')
        elif seqs:
            seqs[-1] += line.strip()
        else:
            raise ValueError('Malformed alignment')
    if not seqs or seqs[0] != query:
        raise ValueError('MSA query does not match variant sequence')
    if any(_aligned_length(s) != len(query) for s in seqs):
        raise ValueError('MSA alignment length mismatch')
    return 'key,sequence\n' + ''.join('-1,'+s+'\n' for s in seqs[:16384])


def valid_cache(path, seq):
    try:
        with path.open() as f:
            reader=csv.DictReader(f)
            first=next(reader)
            return first['sequence']==seq and first['key']=='-1' and all(
                _aligned_length(r['sequence'])==len(seq)
                for r in reader)
    except (OSError, StopIteration, KeyError,TypeError,csv.Error):
        return False


def run(manifest_path, chunk=101,max_attempts=8,retry_seconds=60):
    if not 1 <= chunk <= 101:
        raise ValueError('Batch size must be between 1 and 101')
    from boltz.data.msa.mmseqs2 import run_mmseqs2
    m=json.loads(Path(manifest_path).read_text()); out=Path(m['output_dir'])
    cache=out/'msa'; cache.mkdir(exist_ok=True)
    configs=out/'inputs'; configs.mkdir(exist_ok=True)
    candidates={c['id']:c for c in m['candidates']}
    # WT first, then interleave scaffolds to enable early progress on each.
    jobs=sorted(m['jobs'],key=lambda j:(j['mutation']!='WT',j['id'].split('_',1)[-1],j['candidate']))
    unique={j['sequence_sha256']:j['sequence'] for j in jobs}
    ready={h for h,s in unique.items() if valid_cache(cache/(h+'.csv'),s)}
    missing=[(h,s) for h,s in unique.items() if h not in ready]
    print(f"{len(unique)} unique sequences; {len(missing)} need fresh MSAs",flush=True)
    def write_ready(hashes):
        for j in jobs:
            if j['sequence_sha256'] not in hashes:continue
            p=cache/(j['sequence_sha256']+'.csv')
            cfg=copy.deepcopy(candidates[j['candidate']]['boltz_input'])
            protein=next(e['protein'] for e in cfg['sequences'] if 'protein' in e)
            protein.update(sequence=j['sequence'],msa=str(p))
            # JSON is valid YAML; avoid a serialization dependency.
            dest=configs/(j['id']+'.yaml'); tmp=dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(cfg,indent=2)+'\n');tmp.replace(dest)
    write_ready(ready)
    status_path=out/'msa_status.json'
    def status(**extra):
        data=dict(pid=os.getpid(),host=socket.gethostname(),updated=time.time(),ready_sequences=len(ready),total_sequences=len(unique),**extra)
        tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(status_path)
    status(state='running')
    for start in range(0,len(missing),chunk):
        batch=missing[start:start+chunk]
        ident=hashlib.sha256('\n'.join(h for h,s in batch).encode()).hexdigest()
        print(f'Batch {start//chunk+1}: {len(batch)} sequences, identity {ident}',flush=True)
        def retry(attempt,delay,exc):status(state='retrying',batch_identity=ident,attempt=attempt,retry_after=time.time()+delay,error=type(exc).__name__+': '+str(exc))
        status(state='running',batch_identity=ident,batch_size=len(batch))
        try:result=request_batch(run_mmseqs2,[s for h,s in batch],str(cache/('batch_'+ident)),max_attempts,retry_seconds,on_retry=retry)
        except Exception as exc:
            status(state='failed',batch_identity=ident,error=type(exc).__name__+': '+str(exc),transient=transient_service_error(exc));raise
        aligns=result[0] if isinstance(result,tuple) else result
        if len(aligns)!=len(batch):raise ValueError('Server returned wrong alignment count')
        # Validate the entire server response before publishing any input.
        contents=[alignment_csv(a,s) for (h,s),a in zip(batch,aligns)]
        for (h,s),content in zip(batch,contents):
            p=cache/(h+'.csv');tmp=p.with_suffix('.tmp')
            tmp.write_text(content);tmp.replace(p)
            ready.add(h)
        write_ready({h for h,s in batch})
        status(state='running',batch_identity=ident)
        print(f'Ready inputs: {len(list(configs.glob("*.yaml")))}/{len(jobs)}',flush=True)
    (out/'msa_complete.json').write_text(json.dumps({'sequences':len(unique),'inputs':len(jobs)})+'\n')
    status(state='complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('--chunk',type=int,default=101)
    args=ap.parse_args();run(args.manifest,args.chunk)
