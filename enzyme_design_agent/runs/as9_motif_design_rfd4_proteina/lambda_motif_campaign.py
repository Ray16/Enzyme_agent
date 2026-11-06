"""Run the frozen AS9 M0/M1 campaign through reserved, continuously guarded GPUs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback

THREADS = ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,data):
    p = Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp = p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(p)


def identity():
    return dict(pid=os.getpid(),host=socket.gethostname(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        cpu_affinity=sorted(os.sched_getaffinity(0)),started=time.time())


def chemistry(file,chain):
    """Exact exported heavy-atom chemistry, independent of hydrogen inventories."""
    from biotite.structure.io import pdbx
    b = pdbx.CIFFile.read(file).block
    site = b['atom_site']; selected=site['auth_asym_id'].as_array(str)==chain
    codes = set(site['auth_comp_id'].as_array(str)[selected])
    if len(codes)!=1: raise ValueError('Expected one component on ligand chain')
    code = codes.pop(); ac=b['chem_comp_atom']; bc=b['chem_comp_bond']
    atoms = {}
    for comp,name,element,charge in zip(ac['comp_id'].as_array(str),ac['atom_id'].as_array(str),
                                      ac['type_symbol'].as_array(str),ac['charge'].as_array(int)):
        if comp==code and element!='H': atoms[name]=(element,int(charge))
    bonds = []
    for comp,a,z,order,aromatic in zip(bc['comp_id'].as_array(str),bc['atom_id_1'].as_array(str),
        bc['atom_id_2'].as_array(str),bc['value_order'].as_array(str),bc['pdbx_aromatic_flag'].as_array(str)):
        if comp==code and a in atoms and z in atoms:
            bonds.append((min(a,z),max(a,z),order,aromatic))
    return atoms,sorted(bonds)


def score(root,job,path):
    import numpy as np
    from biotite.structure.io import pdbx
    from audit_motif_outputs import audit
    from rdkit import Chem
    source=root/'inputs'/(job['input_id']+'.cif')
    meta=json.loads(source.with_suffix('.metadata.json').read_text())
    reference=pdbx.get_structure(pdbx.CIFFile.read(source),model=1)
    atoms=pdbx.get_structure(pdbx.CIFFile.read(path),model=1)
    geom={'center':{'chain':'B','atom':'PA'},'angle_reference':{'chain':'B','atom':'O3A'},
        'attack_candidates':{'chain':'D','element':'O'},'expected_attack_candidates':4}
    row=audit(path,reference,meta,geom)
    fatal=[s for s in row['integrity_issues'] if not s.startswith('Peptide C-N')]
    if not row['requested_length_match']: fatal.append('Requested length516 not met')
    chemical={}
    for chain in ('B','C','D'):
        expected=chemistry(source,chain); actual=chemistry(path,chain)
        if expected!=actual: fatal.append('Heavy-atom chemistry mismatch chain '+chain)
        chemical[chain]=sum(x[1] for x in actual[0].values())
        if chain=='C': continue
        # Independently derive carbon stereochemistry from generated coordinates.
        inventory,bonds=actual; mol=Chem.RWMol(); names=list(inventory); lookup={}
        for name in names:
            element,charge=inventory[name];a=Chem.Atom(element);a.SetFormalCharge(charge)
            lookup[name]=mol.AddAtom(a)
        for a,z,order,aromatic in bonds:
            kind={'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,
                  'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}[order]
            mol.AddBond(lookup[a],lookup[z],kind)
        mol=mol.GetMol(); Chem.SanitizeMol(mol)
        conf=Chem.Conformer(len(names))
        for name,i in lookup.items():
            xyz=atoms.coord[(atoms.chain_id==chain)&(atoms.atom_name==name)]
            if len(xyz)!=1: raise ValueError('Ambiguous ligand atom')
            conf.SetAtomPosition(i,tuple(map(float,xyz[0])))
        mol.AddConformer(conf)
        Chem.AssignStereochemistryFrom3D(mol,replaceExistingTags=True)
        for a in mol.GetAtoms():
            if a.GetSymbol()!='C':a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if Chem.MolToSmiles(mol)!=meta['ligand_smiles'][chain]:
            fatal.append('Ligand carbon stereochemistry/graph mismatch '+chain)
    quality=list(row['integrity_issues'])
    mapped=row['motif_source_to_output_residue'].get('311')
    if mapped is not None:
        donor=atoms[(atoms.chain_id=='A')&(atoms.res_id==mapped)]
        name=str(donor.res_name[0]);row['mapped311_identity']=name
        if name=='PRO': quality.append('Mapped311 is PRO; required backbone NH absent')
        n=donor.coord[donor.atom_name=='N'][0]
        distal_names=['O11','O13']
        distal=atoms.coord[(atoms.chain_id=='D')&np.isin(atoms.atom_name,distal_names)]
        row['donor_N_to_distal_oxygen_min_A']=float(np.linalg.norm(distal-n,axis=1).min())
    row.update(input_id=job['input_id'],execution_structure_valid=not fatal,
        fatal_issues=fatal,quality_issues=quality,generation_QC_pass=not fatal and not quality,
        ligand_formal_charges=chemical,scope='Conditioned generation; independent cofolding pending',
        AME_status='pending',batch_size=1)
    return row


def worker(root,job_id):
    m=json.loads((root/'manifest.json').read_text())
    producer=root/m.get('producer_dir','producer')
    job=next(j for j in m['jobs'] if j['id']==job_id)
    for name,h in m['payload_sha256'].items():
        if sha(root/name)!=h:raise RuntimeError('Frozen payload changed: '+name)
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').isdigit():
        raise RuntimeError('Missing gate-owned GPU')
    claim=root/'claims'/job_id;claim.mkdir()
    owner=identity();owner['gpu']=os.environ['CUDA_VISIBLE_DEVICES'];save(claim/'owner.json',owner)
    child=None;out=root/'runs'/job_id
    def terminate(sig,frame):
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        if out.exists():raise RuntimeError('Preserve partial output; do not rerun silently')
        env=dict(os.environ,CCD_MIRROR_PATH=str(root/'ccd'),HF_HUB_OFFLINE='1',WANDB_MODE='disabled')
        for k in THREADS:env[k]='1'
        argv=[m['python'],'-u',str(producer/'cuda_motif_generate.py'),
            'inputs='+str(root/'inputs'/(job['input_id']+'.cif')),'out_dir='+str(out),
            'num_replicates='+str(job['expected']),'seed='+str(job['seed']),
            'hf_ckpt_path='+m['checkpoint_path'],'recycle_strategy=medium/midpeak',
            'generation.args.partial_t_backbone=0.0','generation.args.partial_t_latent=0.0',
            'prevalidate_inputs=true','compress_outputs=false','hydra.run.dir='+str(out/'hydra')]
        with (root/'logs'/(job_id+'.generation.log')).open('x') as log:
            started=time.time();child=subprocess.Popen(argv,env=env,cwd=producer/'model_source',stdout=log,stderr=subprocess.STDOUT)
            save(claim/'predictor.json',dict(pid=child.pid,argv=argv,started=started))
            print('GENERATION_START',job_id,child.pid,flush=True)
            rc=child.wait();child=None
        files=sorted(out.rglob('*_generated.cif'))
        if rc or len(files)!=job['expected']:
            raise RuntimeError(f'Generator rc={rc}; files={len(files)}/{job["expected"]}')
        rows=[score(root,job,p) for p in files]
        timing=json.loads((out/'timing.json').read_text())
        save(root/'complete'/(job_id+'.json'),dict(id=job_id,owner=owner,rows=rows,timing=timing,
            generation_process_wall_seconds=time.time()-started,finished=time.time()))
        print('GENERATION_COMPLETE',job_id,len(rows),flush=True)
    except BaseException:
        save(root/'failed'/(job_id+'.json'),dict(id=job_id,owner=owner,error=traceback.format_exc(),finished=time.time()))
        raise
    finally:
        (claim/'predictor.json').unlink(missing_ok=True);(claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


def occupied(exclude):
    cpus=set()
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if int(p.name)==exclude or p.stat().st_uid<1000 or not (p/'cmdline').read_bytes():continue
            slots=os.sched_getaffinity(int(p.name))
            if len(slots)<os.cpu_count():cpus|=slots
        except (OSError,PermissionError):pass
    return cpus


def publish(root,m,active,status):
    records=[json.loads(p.read_text()) for p in (root/'complete').glob('*.json')]
    rows=[r for record in records for r in record['rows']]
    generated=list((root/'runs').glob('*/inference_samples/**/*_generated.cif'))
    claimed={}
    for p in (root/'claims').glob('*/owner.json'):
        try:
            o=json.loads(p.read_text());claimed[o['host']+'/gpu'+str(o['gpu'])]=p.parent.name
        except (OSError,KeyError,json.JSONDecodeError):pass
    save(root/'status.json',dict(status=status,expected_draws=100,generated_files_present=len(generated),
        assessed_designs=len(rows),generation_QC_pass=sum(r['generation_QC_pass'] for r in rows),
        failed_chunks=[p.stem for p in (root/'failed').glob('*.json')],
        completed_chunks=[r['id'] for r in records],active=claimed,
        scope='Generation only; independent cofold assessment pending',updated=time.time()))


def controller(root,host=None):
    m=json.loads((root/'manifest.json').read_text()); owner=identity()
    host=host or m['host']
    producer=root/m.get('producer_dir','producer')
    if socket.gethostname().split('.')[0]!=host:raise RuntimeError('Wrong controller host')
    save(root/'controllers'/(host+'.owner.json'),owner);active={};began=time.monotonic()
    while time.monotonic()-began<m['controller_max_seconds']:
        for gpu,item in list(active.items()):
            if item['process'].poll() is None:continue
            rc=item['process'].wait();item['log'].close();jid=item['job']['id']
            if (root/'claims'/jid).exists():
                claim_owner=root/'claims'/jid/'owner.json'
                try:o=json.loads(claim_owner.read_text())
                except (OSError,json.JSONDecodeError):
                    save(root/'controllers'/(host+'.stop.json'),dict(reason='Unresolved claim owner; inspect before recovery',job=jid));return
                if o.get('host','').split('.')[0]==host and str(o.get('gpu'))==str(gpu):
                    save(root/'controllers'/(host+'.stop.json'),dict(reason='Worker exited with its own unresolved claim; inspect descendants',job=jid,returncode=rc));return
            if rc and not (root/'failed'/(jid+'.json')).exists():
                save(root/'resource_events'/f'{jid}_{time.time_ns()}.json',
                     dict(reason='Gate or guard startup refused; no scientific attempt recorded',returncode=rc))
            del active[gpu]
        done={p.stem for p in (root/'complete').glob('*.json')}
        failed={p.stem for p in (root/'failed').glob('*.json')}
        pilot=m['pilot_job']
        if pilot in failed or len(failed)>=3:
            publish(root,m,active,'stopped_failure_gate');return
        if pilot in done:
            pilot_rows=json.loads((root/'complete'/(pilot+'.json')).read_text())['rows']
            if not all(r['execution_structure_valid'] for r in pilot_rows):
                publish(root,m,active,'stopped_pilot_structure_gate');return
        pending=[j for j in m['jobs'] if j['id'] not in done|failed
                 and not (root/'claims'/j['id']).exists()
                 and j['id'] not in {v['job']['id'] for v in active.values()}
                 and (pilot in done or (j['id']==pilot and host==m.get('pilot_host',host)))]
        if len(done|failed)==len(m['jobs']) and not active:
            publish(root,m,active,'finished');return
        gate=subprocess.run([m['gpu_reserve'],'list','--hosts',host],text=True,capture_output=True,timeout=30)
        if gate.returncode:raise RuntimeError('GPU availability gate failed')
        free={int(line.split()[1]) for line in gate.stdout.splitlines() if line.startswith(host+' ')}
        for allocation in m['allocations']:
            if allocation.get('host',m['host'])!=host:continue
            gpu=allocation['gpu']
            if not pending or gpu not in free or gpu in active:continue
            slots=set(allocation['cpus'])
            if slots & occupied(os.getpid()):continue
            if max(os.getloadavg())+len(slots)>os.cpu_count():continue
            while pending and (root/'claims'/pending[0]['id']).exists():pending.pop(0)
            if not pending:break
            job=pending.pop(0);jid=job['id']
            env=dict(os.environ)
            for k in THREADS:env[k]='1'
            argv=['taskset','-c',','.join(map(str,allocation['cpus'])),m['gpu_reserve'],'run',str(gpu),
                '--label','AS9_M0M1_'+jid,'--',m['python'],str(producer/'guard_single_gpu.py'),
                '--gpu-index',str(gpu),'--gpu-uuid',allocation['uuid'],
                '--monitor-cpu',str(allocation['cpus'][0]),
                '--child-cpus',','.join(map(str,allocation['cpus'][1:])),
                '--max-seconds',str(m['worker_max_seconds']),
                '--record',str(root/'guards'/(jid+'.json')),'--',
                m['python'],'-u',str(producer/'lambda_motif_campaign.py'),str(root),'--job',jid]
            log=(root/'logs'/f'{jid}.launch_{time.time_ns()}.log').open('x')
            child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=dict(job=job,process=child,log=log)
            save(root/'launches'/f'{jid}_{host}_{child.pid}.json',dict(job=jid,host=host,gpu=gpu,cpus=allocation['cpus'],
                gate_pid=child.pid,argv=argv,launched=time.time()))
            time.sleep(4)
        publish(root,m,active,'generating' if pilot in done else 'validating_first_draw')
        time.sleep(10)
    publish(root,m,active,'controller_time_limit_workers_retain_ownership')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--job');parser.add_argument('--controller-host')
    args=parser.parse_args();root=args.root.resolve()
    if args.job:worker(root,args.job)
    else:controller(root,args.controller_host)
