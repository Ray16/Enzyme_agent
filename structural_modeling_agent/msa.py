"""Resumable batched alignment retrieval with sequence-bound cached outputs."""
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time

import requests

SERVER='https://api.colabfold.com'
ALIGNMENTS=('uniref.a3m','bfd.mgnify30.metaeuk30.smag30.a3m')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_bytes(content if isinstance(content,bytes) else content.encode())
    os.replace(temporary,path)


def save(path,data):
    write(path,json.dumps(data,indent=2)+'\n')


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def valid_csv(path,sequence):
    try:
        path=Path(path)
        metadata=json.loads(path.with_suffix('.json').read_text())
        if metadata.get('alignment_policy')!='full_server_alignment_v1':return False
        if metadata.get('csv_sha256')!=digest(path.read_bytes()) or metadata.get('sequence_sha256')!=digest(sequence.encode()):
            return False
        rows=list(csv.DictReader(Path(path).open()))
        return len(rows)>1 and rows[0].get('sequence')==sequence and all(
            row.get('key')=='-1' and bool(row.get('sequence')) for row in rows)
    except (OSError,ValueError):return False


def alignment_csvs(archive,sequences):
    """Read only named alignment members; never extract arbitrary archive paths."""
    results={101+i:[] for i in range(len(sequences))}
    with tarfile.open(archive,'r:gz') as tar:
        for filename in ALIGNMENTS:
            member=tar.getmember(filename)
            if not member.isfile() or member.size>512*1024*1024:
                raise ValueError('Invalid alignment archive member')
            text=tar.extractfile(member).read().decode()
            current=None;new_block=True
            for raw in text.splitlines():
                if '\x00' in raw:new_block=True
                line=raw.replace('\x00','').strip()
                if not line:continue
                if line.startswith('>'):
                    if new_block:
                        current=int(line[1:]);new_block=False
                    continue
                if current not in results:raise ValueError('Unexpected alignment query identifier')
                results[current].append(line)
    outputs=[]
    for index,sequence in enumerate(sequences,101):
        rows=results[index]
        if not rows or rows[0]!=sequence:raise ValueError('Alignment query does not match its exact source sequence')
        # Keep a complete valid query; server homolog absence is an explicit observation.
        outputs.append('key,sequence\n'+''.join('-1,'+row+'\n' for row in dict.fromkeys(rows)))
    return outputs


def request_json(session,method,url,*,data=None):
    """Retry safe connection failures; an ambiguous POST is never resubmitted."""
    for attempt in range(4):
        try:
            response=session.request(method,url,data=data,timeout=(30,180))
            if response.status_code in {429,502,503,504} and method=='GET':
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.ConnectTimeout:
            if attempt==3:raise
        except (requests.ReadTimeout,requests.ConnectionError,requests.HTTPError):
            if method!='GET' or attempt==3:raise
        time.sleep(min(10*2**attempt,40))


def fetch_batch(sequences,cache,*,session=None,timeout=1800,server=SERVER):
    if not 1<=len(sequences)<=101:raise ValueError('A batch must contain 1..101 distinct sequences')
    if len(set(sequences))!=len(sequences):raise ValueError('Deduplicate sequences before submission')
    cache=Path(cache);identity=digest(json.dumps({'sequences':sequences,'server':server,'mode':'env'}).encode())
    folder=cache/'batches'/identity;folder.mkdir(parents=True,exist_ok=True)
    state_path=folder/'request.json';archive=folder/'out.tar.gz';session=session or requests.Session()
    with lock(folder/'lock'):
        state=json.loads(state_path.read_text()) if state_path.exists() else {'identity':identity,'status':'new'}
        if not archive.exists():
            if state.get('status') in {'submission_unknown','submitting'} and not state.get('ticket'):
                raise RuntimeError('Prior submission response was ambiguous; resolve it before resubmitting')
            if not state.get('ticket'):
                query=''.join(f'>{101+i}\n{seq}\n' for i,seq in enumerate(sequences))
                state.update(status='submitting',requested_at=time.time(),sequence_hashes=[digest(s.encode()) for s in sequences])
                save(state_path,state)
                try:
                    reply=request_json(session,'POST',server+'/ticket/msa',data={'q':query,'mode':'env'})
                except requests.ConnectTimeout:
                    state.update(status='connection_failed');save(state_path,state);raise
                except Exception:
                    state.update(status='submission_unknown');save(state_path,state);raise
                if not reply.get('id'):
                    state.update(status='server_rejected',response=reply);save(state_path,state)
                    raise RuntimeError('Alignment server did not accept the batch: '+str(reply.get('status')))
                state.update(ticket=reply['id'],status=reply.get('status','PENDING'));save(state_path,state)
            deadline=time.monotonic()+timeout
            while state['status']!='COMPLETE':
                if state['status'] in {'ERROR','MAINTENANCE'}:
                    raise RuntimeError('Alignment server returned '+state['status'])
                if time.monotonic()>=deadline:raise TimeoutError('Alignment batch checkpointed while waiting for '+state['ticket'])
                reply=request_json(session,'GET',server+'/ticket/'+state['ticket'])
                state.update(status=reply['status'],checked_at=time.time());save(state_path,state)
                print(json.dumps({'batch':identity[:12],'ticket':state['ticket'],'status':state['status']}),flush=True)
                if state['status']!='COMPLETE':time.sleep(10)
            response=session.get(server+'/result/download/'+state['ticket'],timeout=(30,180))
            response.raise_for_status();write(archive,response.content)
        outputs=alignment_csvs(archive,sequences)
        state.update(status='validated',archive_sha256=digest(archive.read_bytes()));save(state_path,state)
        for sequence,content in zip(sequences,outputs):
            path=cache/(digest(sequence.encode())+'.csv');write(path,content)
            save(path.with_suffix('.json'),{'sequence_sha256':digest(sequence.encode()),'csv_sha256':digest(content.encode()),
                'server':server,'batch':identity,'ticket':state.get('ticket'),'aligned_rows':content.count('\n')-1,
                'alignment_policy':'full_server_alignment_v1'})
        return [cache/(digest(sequence.encode())+'.csv') for sequence in sequences]


def prepare(fasta,out,cache,*,fetcher=fetch_batch):
    records={};identifier=None
    for line in Path(fasta).read_text().splitlines():
        if line.startswith('>'):identifier=line[1:].split()[0];records[identifier]=''
        elif line.strip():
            if identifier is None:raise ValueError('FASTA sequence precedes its identifier')
            records[identifier]+=line.strip()
    if not records or any(not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',s) for s in records.values()):
        raise ValueError('Require nonempty exact protein sequences')
    cache=Path(cache).resolve();out=Path(out).resolve();cache.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    unique=list(dict.fromkeys(records.values()))
    missing=[s for s in unique if not valid_csv(cache/(digest(s.encode())+'.csv'),s)]
    for offset in range(0,len(missing),101):fetcher(missing[offset:offset+101],cache)
    rows=[]
    for identifier,sequence in records.items():
        checksum=digest(sequence.encode());path=cache/(checksum+'.csv')
        if not valid_csv(path,sequence):raise ValueError('Missing or mismatched sequence-specific alignment')
        rows.append({'id':identifier,'seqhash':checksum,'csv':str(path)})
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['id','seqhash','csv'],delimiter='\t');writer.writeheader();writer.writerows(rows)
    write(out/'index.tsv',stream.getvalue())
    return rows


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta',required=True);parser.add_argument('--out',required=True);parser.add_argument('--cache',required=True)
    args=parser.parse_args();print(json.dumps({'alignments':prepare(args.fasta,args.out,args.cache)}))
