"""Create a reusable RFD4 work manifest from a task-specific JSON config."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(config,root,wrapper):
    if root.exists() and any(root.iterdir()):raise FileExistsError('Output directory is not empty')
    batch=int(config['batch_size']);chunk=int(config['samples_per_chunk']);per_input=int(config['replicates_per_input']);seed0=int(config['seed_start'])
    if batch<1 or batch>16 or chunk<1 or per_input<1 or seed0<0:raise ValueError('Invalid batch, chunk, replicate count or seed')
    inputs=config['inputs'];ids=[x['id'] for x in inputs]
    if len(ids)!=len(set(ids)) or not ids or any(not x or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in x) for x in ids):raise ValueError('Input identifiers must be unique and filesystem safe')
    extras=config.get('overrides',[])
    reserved=('inputs=','out_dir=','batch_size=','num_replicates=','seed=','hydra.run.dir=')
    if not all(isinstance(x,str) and '=' in x and not x.lstrip('+~').startswith(reserved) for x in extras):raise ValueError('Invalid or conflicting override')
    python=Path(config['python']).expanduser().resolve()
    if not python.is_file() or not wrapper.is_file():raise ValueError('Python or profile wrapper missing')
    jobs=[]
    for i,entry in enumerate(inputs):
        inp=Path(entry['path']).expanduser().resolve()
        if not inp.is_file():raise FileNotFoundError(inp)
        for c in range(math.ceil(per_input/chunk)):
            n=min(chunk,per_input-c*chunk);name=f"{entry['id']}_chunk{c:03d}";output=root/'runs'/name;seed=seed0+len(jobs)
            if seed>=2**31:raise ValueError('Seed out of range')
            overrides=['inputs='+str(inp),'out_dir='+str(output),f'num_replicates={n}',f'seed={seed}','hydra.run.dir='+str(output/'hydra'),*extras]
            jobs.append(dict(id=name,input=str(inp),input_sha256=digest(inp),output=str(output),expected=n,seed=seed,argv=[str(python),'-u',str(wrapper),*overrides]))
    root.mkdir(parents=True,exist_ok=True)
    for p in ('claims','runs','logs','complete','failed'):(root/p).mkdir()
    manifest=dict(schema_version=1,status='prepared_not_submitted',batch_size=batch,replicates_per_input=per_input,samples_per_chunk=chunk,
                  expected_designs=per_input*len(inputs),jobs=jobs,wrapper_sha256=digest(wrapper),config=config)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    m=prepare(json.loads(a.config.read_text()),a.output.resolve(),Path(__file__).with_name('batch_profile.py').resolve())
    print(json.dumps(dict(jobs=len(m['jobs']),expected_designs=m['expected_designs'],batch_size=m['batch_size'])))
if __name__=='__main__':main()
