"""Compile shared family protocols and execute cofolds without per-member LLM calls."""
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import string
import socket
import shlex
import subprocess
import sys
import time

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from .modeling_reasoning.chemistry import characterize_smiles
from .modeling_reasoning.literature import LiteratureKnowledge
from .modeling_reasoning.reaction import ReactionCoordinates
from .modeling_reasoning.resolve_handoff import resolve_candidate, _load_family_metal
from .msa import valid_csv
from .scripts.build_step1_inputs import enzyme_chains, modeled_substrates, yaml_for


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_key(smiles):
    from rdkit import Chem
    return digest([Chem.MolToSmiles(Chem.MolFromSmiles(value),isomericSmiles=True) for value in smiles])


def register_protocol(path,registry):
    """Cache an explicit shared target protocol, never candidate discoveries."""
    path=Path(path);registry=Path(registry);spec=json.loads(path.read_text())
    spec.pop('fleet_pilots',None)  # Operational cohort selectors never enter a reusable chemistry protocol.
    key=target_key([c['source_smiles'] for c in spec['components']])
    directory=registry.parent/'protocols';directory.mkdir(parents=True,exist_ok=True)
    content=(json.dumps(spec,indent=2)+'\n').encode()
    destination=directory/(hashlib.sha256(content).hexdigest()+'.json')
    if not destination.exists():destination.write_bytes(content)
    with state_lock(registry):
        data=json.loads(registry.read_text()) if registry.exists() else {}
        data[key]={'path':str(destination.resolve()),'sha256':file_hash(destination),'scope':'User-supplied conditional shared protocol; not experimentally demonstrated activity.'}
        atomic_write_json(registry,data)


def adopt_registered_protocol(run,path):
    from enzyme_discovery_agent.models.target_review import target_contract
    registry=Path(run).parent/'.family_protocols/index.json'
    if not registry.exists():return False
    target=[s for s in (target_contract(run).get('substrate_smiles') or '').split('.') if s]
    if not target:return False
    reference=json.loads(registry.read_text()).get(target_key(target))
    if not reference:return False
    source=Path(reference['path'])
    if file_hash(source)!=reference['sha256']:raise ValueError('Registered protocol changed')
    atomic_write_json(path,json.loads(source.read_text()));return True


def compile_protocol(mode, specification):
    """Resolve both steps once for a mechanism/unit/target protocol."""
    coordinates=(ReactionCoordinates.load(specification['reaction_coordinates_path'])
                 if specification.get('reaction_coordinates_path') else ReactionCoordinates.load())
    key,coordinate,_=coordinates.select(activation_mode=mode)
    if coordinate is None:raise ValueError('No shared reaction coordinate for '+str(mode))
    template=specification['mechanism_templates'][mode]
    metal=specification.get('metal_override') or template.get('metal')
    components=specification['components'];stages={}
    knowledge=LiteratureKnowledge.load()
    for stage in coordinate['steps']:
        index=stage['index']
        partners=[];directives=[]
        for number,component in enumerate(components,1):
            state=component['states'][mode][str(index)]
            included=state['include_in_model'];smiles=state['modeling_smiles'];role=state['role']
            partners.append({'component_index':number,'source_smiles':component['source_smiles'],
                'modeling_smiles':smiles,'role':role,'include_in_model':included})
            directives.append({'match':{'component_index':number},'include_in_model':included,
                'modeling_smiles':smiles,'role':role,'review_status':'reviewed',
                'origin':'user_protocol','rationale':'Shared stage-specific representation from the supplied protocol.'})
        nucleotide=template.get('nucleotide_ccd')
        candidate={'accession':'shared-protocol','evidence':{},'modeling_input':{
            'activation':{'status':'assigned','activation_mode':mode,'nucleotide':nucleotide},
            'target_substrates':partners,'cofactors':copy.deepcopy(template.get('step_cofactors',{}).get(str(index),template.get('cofactors',[])))}}
        declaration={'step':{'index':index,'mechanism':key,'chosen_because':specification['step_rationale'][str(index)]},
            'scoring_policy':'unrestrained','resolutions':{
                'enzyme_state':{'chains':[{'source':'full_sequence'}],'copies':specification.get('protein_copies',1),
                    'state':'intact','origin':'shared_protocol','rationale':specification['assembly_basis']},
                'substrates':directives,'catalytic_metal':({'strategy':'explicit','ccd':metal['metal'],
                    'per_site_metal_count':metal['per_site_metal_count'],'uniprot_cross_check':False,
                    'origin':'existing_mechanism_protocol','rationale':metal['rationale']} if metal else {'strategy':'omit'})}}
        resolved=resolve_candidate(candidate,declaration,knowledge,step=index,mechanism_override=key,coordinates=coordinates)
        if resolved['status']!='ready':raise ValueError(str(resolved['errors']))
        stages[str(index)]=candidate['modeling_input']
    return {'mechanism':mode,'coordinate':key,'steps':stages,
        'scope':'Shared conditional structural test, not established target activity.',
        'assembly_basis':specification['assembly_basis'],'setup_llm_calls':0}


def _family(data):
    value=data.get('interpro_family_id') or data.get('screening_family_id') or data.get('interpro_family')
    return json.dumps(value,sort_keys=True) if isinstance(value,(list,dict)) else str(value or 'unresolved')


def build_handoff(run_dir, specification):
    """Group published candidates; direct contradictory evidence always splits groups."""
    root=Path(run_dir);registry=load_candidate_registry(root,strict=True)['candidates']
    rows=list(csv.DictReader((root/'ranked_candidates.csv').open()))
    assignments=[];known=defaultdict(set)
    coordinates=(ReactionCoordinates.load(specification['reaction_coordinates_path'])
                 if specification.get('reaction_coordinates_path') else ReactionCoordinates.load())
    for row in rows:
        meta=(registry.get(row['accession']) or {}).get('metadata') or {}
        data={**(meta.get('evaluation_data') or {}),**meta};family=_family(data)
        text=' '.join(data.get('catalytic_activities') or [])
        matched=[rule['mode'] for rule in specification.get('assignment_rules',[])
                 if all(re.search(pattern,text,re.I) for pattern in rule.get('required',[]))
                 and not any(re.search(pattern,text,re.I) for pattern in rule.get('excluded',[]))
                 and rule.get('required')]
        basis='recorded_reaction'
        if len(set(matched))==1:
            mode=matched[0];known[family].add(mode)
        else:
            mode=((data.get('modeling_input') or {}).get('activation') or {}).get('activation_mode')
            basis='source_activation_annotation'
            if mode not in specification['mechanism_templates']:
                label=str(data.get('interpro_family') or row.get('mechanism_family') or '')
                modes={rule['mode'] for rule in specification.get('assignment_rules',[])
                       if rule.get('family_regex') and re.search(rule['family_regex'],label,re.I)}
                mode=next(iter(modes)) if len(modes)==1 else None
                basis='conditional_family_transfer'
            if not mode:
                ids=[family] if family.startswith('IPR') else []
                _,coordinate,_=coordinates.select(family_ids=ids)
                modes=(coordinate or {}).get('keys',{}).get('activation_mode',[])
                mode=modes[0] if len(modes)==1 else None
                basis='conditional_family_coordinate'
        assignments.append((row,data,family,mode,basis))
    protocols={};groups={};pending=[];invalid={}
    for row,data,family,mode,basis in assignments:
        if not mode and len(known[family])==1:
            mode=next(iter(known[family]));basis='conditional_family_member_transfer'
        member={'accession':row['accession'],'enzyme_name':row.get('protein_name') or row.get('enzyme_name') or data.get('protein_name'),
            'rank':int(row.get('rank') or len(rows)),'assignment_basis':basis,
            'recorded_activities':data.get('catalytic_activities') or []}
        if mode not in specification['mechanism_templates']:
            pending.append({**member,'family_id':family,'reason':'No applicable shared mechanism protocol; resolve this family once.'});continue
        shared={**specification,**specification.get('family_overrides',{}).get(family,{})}
        protocol_id=digest({'mode':mode,'specification':shared})[:20]
        if protocol_id not in protocols and protocol_id not in invalid:
            try:protocols[protocol_id]=compile_protocol(mode,shared)
            except (KeyError,ValueError) as exc:invalid[protocol_id]=str(exc)
        if protocol_id in invalid:
            pending.append({**member,'family_id':family,'reason':'Shared protocol needs review: '+invalid[protocol_id]});continue
        group_id=family+':'+mode
        group=groups.setdefault(group_id,{'family_id':family,'protocol_id':protocol_id,'members':[]})
        group['members'].append(member)
    return {'schema_version':'family-cofold-1','discovery_run_id':root.name,'source_run':str(root.resolve()),'specification_sha256':digest(specification),
        'protocols':protocols,'families':list(groups.values()),'pending_families':pending,
        'ranked_candidates':len(rows),'setup_llm_calls':0}


def _prepared_candidates(root):
    result={}
    for path in (root/'agent_state/production_modeling').glob('*/item.json'):
        item=json.loads(path.read_text())
        task_path=Path(item.get('task',''))
        if task_path.is_file():
            task=json.loads(task_path.read_text())
            for candidate in task.get('candidates',[]):result[candidate['accession']]=candidate
    return result


def render_jobs(run_dir, handoff, directory,candidates=None):
    """Only sequence/identity/MSA expansion is per member; no scientific replanning."""
    root=Path(run_dir);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    candidates=candidates if candidates is not None else _prepared_candidates(root);jobs=[];pending=[]
    for family in handoff['families']:
        protocol=handoff['protocols'][family['protocol_id']]
        for member in family['members']:
            accession=member['accession'];candidate=candidates.get(accession)
            try:
                if not candidate:raise ValueError('Exact sequence/MSA preparation pending')
                evidence=candidate['evidence'];sequence=candidate['sequence']
                identity=Path(evidence['identity_source_path'])
                if file_hash(identity)!=evidence['identity_source_sha256']:raise ValueError('Sequence source changed')
                record=json.loads(identity.read_text())
                if record.get('primaryAccession')!=accession or record.get('sequence',{}).get('value')!=sequence:raise ValueError('Sequence identity mismatch')
                msa=Path(evidence['msa_path'])
                if not valid_csv(msa,sequence):raise ValueError('A validated exact-sequence MSA is required')
                for step,model in protocol['steps'].items():
                    chains=enzyme_chains(sequence,str(msa),model['enzyme_state']['chains'])
                    text=yaml_for(chains,model['cofactors'],modeled_substrates(model))
                    job_id=digest({'yaml':text,'msa_sha256':file_hash(msa),'samples':50,'profile':'exact','seed':42})
                    path=directory/(job_id+'.yaml')
                    if not path.exists():path.write_text(text)
                    if path.read_text()!=text:raise ValueError('Existing rendered input changed')
                    labels=iter(string.ascii_uppercase);entities={}
                    for _ in chains:next(labels)
                    for cofactor in model['cofactors']:
                        for _ in range(cofactor['count']):entities.setdefault(cofactor['ccd'],[]).append(next(labels))
                    for substrate in modeled_substrates(model):entities[substrate['role']]=next(labels)
                    jobs.append({**member,'id':job_id,'family_id':family['family_id'],'protocol_id':family['protocol_id'],
                        'discovery_run_id':root.name,'discovery_run_dir':str(root.resolve()),
                        'step':int(step),'input_yaml':str(path.resolve()),'input_sha256':file_hash(path),'msa_path':str(msa),
                        'msa_sha256':file_hash(msa),'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),
                        'expected_predictions':50,'entities':entities,'modeling_input':model})
            except (KeyError,ValueError,OSError) as exc:pending.append({**member,'reason':str(exc)})
    return jobs,pending


def write_input_index(jobs,directory):
    """Expose readable member/stage aliases without changing fingerprinted inputs."""
    from rdkit import Chem
    directory=Path(directory);rows=[]
    for job in jobs:
        name=job['accession'] if re.fullmatch('[A-Za-z0-9_.-]+',job['accession']) else digest(job['accession'])[:20]
        folder=directory/'inputs/by_accession'/name;folder.mkdir(parents=True,exist_ok=True)
        alias=folder/('step_'+str(job['step'])+'.yaml')
        temporary=alias.with_name(alias.name+'.'+str(os.getpid())+'.tmp')
        temporary.symlink_to(Path(job['input_yaml']).resolve());os.replace(temporary,alias)
        model=job['modeling_input']
        rows.append({'discovery_run_id':job.get('discovery_run_id',directory.parent.name),
            'enzyme_name':job['enzyme_name'],'accession':job['accession'],'family_id':job['family_id'],
            'step':job['step'],'input_yaml':str(alias),
            'cofactors':'; '.join(f"{v['ccd']} x {v['count']}" for v in model['cofactors']),
            'ligands':'; '.join(f"{v['role']}: {v['modeling_smiles']} (charge {Chem.GetFormalCharge(Chem.MolFromSmiles(v['modeling_smiles']))})" for v in model['target_substrates'] if v.get('include_in_model'))})
    if rows:
        path=directory/'input_index.csv';temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        with temporary.open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)


def provider_owner_live(receipt):
    """A terminal receipt still needs PID/start-time reconciliation before retry."""
    code="from pathlib import Path;import json,sys;r=json.load(sys.stdin);p=Path('/proc/'+str(r['pid'])+'/stat');print(json.dumps(p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==r['process_start_ticks']))"
    if receipt.get('host')==socket.gethostname():
        path=Path('/proc/'+str(receipt['pid'])+'/stat')
        return path.exists() and path.read_text().rsplit(')',1)[1].split()[19]==receipt['process_start_ticks']
    command=shlex.join(['/nfs/lambda_stor_01/homes/rzhu/boltz2-venv/bin/python','-c',code])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',receipt['host'],command],
        input=json.dumps(receipt),text=True,capture_output=True,timeout=20,check=True)
    return json.loads(result.stdout)


def execute_job(job, backend, directory,resource_attempt=0):
    """Use the existing reserved native provider, without a modeling-agent session."""
    directory=Path(directory);receipt_path=directory/'receipt.json'
    state_dir=Path(backend['argv'][backend['argv'].index('--state-dir')+1])
    if not receipt_path.exists():
        for path in directory.parents[1].glob('workers/*/provider/backend_jobs/'+job['id']+'/receipt.json'):
            native=json.loads(path.read_text());request_path=path.parent/'request.json'
            if native.get('status') not in {'queued','running','unknown','complete'} or not request_path.exists():continue
            if json.loads(request_path.read_text()).get('input_sha256')!=job['input_sha256']:continue
            native['provider_state_dir']=str(path.parents[2]);atomic_write_json(receipt_path,native);break
    if receipt_path.exists():
        previous=json.loads(receipt_path.read_text())
        if previous.get('status')=='failed' and previous.get('provider_state_dir') not in {None,str(state_dir)}:
            attempts=directory/'backend_attempts';attempts.mkdir(exist_ok=True)
            os.rename(receipt_path,attempts/(str(time.time_ns())+'.json'))
    request={'operation':'submit','idempotency_key':job['id'],'input_yaml':job['input_yaml'],
        'input_sha256':job['input_sha256'],'expected_predictions':50,'resource_policy':backend['resource_policy'],
        'execution_sha256':digest({'job':job['id'],'input':job['input_sha256'],'msa':job['msa_sha256'],'profile':'exact','samples':50}),
        'msa_evidence':[{'path':job['msa_path'],'sha256':job['msa_sha256']}]}
    if not receipt_path.exists():
        result=subprocess.run(backend['argv'],input=json.dumps(request),text=True,capture_output=True,timeout=90)
        if result.returncode:raise RuntimeError(result.stderr[-2000:])
        receipt=json.loads(result.stdout);receipt['provider_state_dir']=str(state_dir);atomic_write_json(receipt_path,receipt)
    else:receipt=json.loads(receipt_path.read_text())
    producer_state=Path(receipt.get('provider_state_dir',state_dir))
    live_receipt=producer_state/'backend_jobs'/job['id']/'receipt.json'
    while receipt.get('status') in {'queued','running','unknown'}:
        if live_receipt.exists():receipt={**json.loads(live_receipt.read_text()),'provider_state_dir':str(producer_state)}
        atomic_write_json(receipt_path,receipt)
        if receipt.get('status') not in {'queued','running','unknown'}:break
        time.sleep(5)
    if (receipt.get('status')=='failed' and receipt.get('error')=='Foreign resident compute process appeared'
            and resource_attempt<backend['resource_policy'].get('max_resource_retries',6)):
        for _ in range(6):
            if not provider_owner_live(receipt):break
            time.sleep(2)
        else:raise RuntimeError('Stopped native worker still owns its attempt; reconciliation required')
        stamp=str(time.time_ns());native=producer_state/'backend_jobs'/job['id']
        archive=producer_state/'resource_stops'/job['id']/stamp;archive.parent.mkdir(parents=True,exist_ok=True)
        os.rename(native,archive)
        local=directory/'resource_stops'/stamp;local.mkdir(parents=True,exist_ok=True)
        os.rename(receipt_path,local/'receipt.json')
        atomic_write_json(local/'reason.json',{'reason':'Actual foreign GPU contention; no completed ensemble replaced.',
            'native_archive':str(archive),'resource_attempt':resource_attempt+1})
        time.sleep(5)
        return execute_job(job,backend,directory,resource_attempt+1)
    if receipt.get('status')!='complete':raise RuntimeError(receipt.get('error') or str(receipt))
    return receipt


def find_reusable(job,run,specification):
    """Consult completed family results and the existing registered prediction pools."""
    import yaml
    from campaign_catalog.lookup import physical_system_key,lookup_exact_system
    document=yaml.safe_load(Path(job['input_yaml']).read_text());wanted=physical_system_key(document)
    for path in Path(run).parent.glob('*/family_cofold/jobs/*/complete.json'):
        if path.resolve()==(Path(run)/'family_cofold/jobs'/job['id']/'complete.json').resolve():continue
        saved=json.loads(path.read_text());old=saved.get('reuse_source_job') or saved['job']
        old.setdefault('discovery_run_id',Path(saved.get('reuse_source') or path).parents[3].name)
        input_path=Path(old['input_yaml'])
        if not input_path.is_file() or file_hash(input_path)!=old['input_sha256']:continue
        if physical_system_key(yaml.safe_load(input_path.read_text()))!=wanted:continue
        if digest(old['modeling_input']['geometry'])!=digest(job['modeling_input']['geometry']):
            return {'status':'exact_coordinates_available_analysis_pending','source':str(path),'saved':saved}
        analysis=analyze_outputs(old,saved['receipt'])
        return {'status':'reused','source':str(path),'job':old,'receipt':saved['receipt'],'analysis':analysis}
    for pool in specification.get('prediction_pools',[]):
        if not (Path(pool)/'catalog/current.json').is_file():continue
        result=lookup_exact_system(pool,document)
        if result['exact_system_matches']:
            return {'status':'exact_coordinates_available_analysis_pending','pool':str(pool),
                'matches':result['exact_system_matches'],'physical_system_key':wanted}
    return None


def publish_results(directory,jobs):
    directory=Path(directory);rows=[]
    for job in jobs:
        folder=directory/'jobs'/job['id'];complete=folder/'complete.json';pending=folder/'reuse_pending_analysis.json'
        saved=json.loads(complete.read_text()) if complete.exists() else {}
        score=(saved.get('analysis') or {}).get('geometry') or {}
        rows.append({'discovery_run_id':job.get('discovery_run_id',directory.parent.name),
            'producer_discovery_run_id':(saved.get('reuse_source_job') or job).get('discovery_run_id',directory.parent.name),
            'enzyme_name':job['enzyme_name'],'accession':job['accession'],'family_id':job['family_id'],
            'step':job['step'],'status':'complete' if saved else 'cached_analysis_pending' if pending.exists() else 'pending',
            'n_poses':score.get('n_poses'),'weighted_attack_angle_deg':score.get('weighted_attack_angle_deg'),
            'harmonic_distance_A':score.get('harmonic_distance_A'),'reused':bool(saved.get('reuse_source')),
            'numerical_profile':saved.get('receipt',{}).get('numerical_profile'),
            'input_yaml':job['input_yaml'],'result':str(complete) if saved else str(pending) if pending.exists() else ''})
    if rows:
        path=directory/'family_results.csv';temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        with temporary.open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary,path)
    from .family_analysis import refresh
    try:refresh(directory)
    except ImportError as exc:
        atomic_write_json(directory/'analysis_backend_pending.json',{'error':str(exc),'scope':'Plotting unavailable on this frontend; completed coordinates remain available for the central analyzer.'})


def analyze_outputs(job,receipt):
    """Measure every pose with the existing geometry metric; retain identity failures."""
    from .validation import _read_structure,_protein_completeness
    from .proxies.attack_angle import weighted_attack_angle
    import numpy as np
    import gemmi
    source=json.loads(Path(receipt['prediction_manifest']).read_text())
    if source['input_sha256']!=job['input_sha256']:raise ValueError('Producer input identity changed')
    predictions=source['predictions']
    if len(predictions)!=50:raise ValueError('Require the complete 50-pose ensemble')
    poses=[];geometry=job['modeling_input']['geometry'];entities=job['entities'];ccd_atoms={}
    for ccd,reference in source.get('chemical_component_sources',{}).items():
        if file_hash(reference['path'])!=reference['sha256']:raise ValueError('CCD source changed')
        block=gemmi.cif.read_file(reference['path']).sole_block()
        ccd_atoms[ccd]={gemmi.cif.as_string(name):gemmi.cif.as_string(element).upper()
                        for name,element in zip(block.find_values('_chem_comp_atom.atom_id'),block.find_values('_chem_comp_atom.type_symbol'))
                        if gemmi.cif.as_string(element).upper() not in {'H','D'}}
    def references(participant):
        spec=geometry[participant]
        if spec['kind']=='cofactor':return [(entities[spec['ccd']][0],spec['atom'])]
        chain=entities[spec['role']];mapping=source['ligand_atom_maps'][chain]['atoms']
        return [(chain,mapping[str(index)]['name']) for index in spec['reactive_atom_indices']]
    for prediction in predictions:
        path=Path(prediction['path'])
        if file_hash(path)!=prediction['sha256']:raise ValueError('Prediction changed')
        atoms,sequences=_read_structure(path);issues=[]
        for chain,sequence in sequences.items():
            if hashlib.sha256(sequence.encode()).hexdigest()!=job['sequence_sha256']:issues.append('protein_sequence:'+chain)
            if any(c['status']!='pass' for c in _protein_completeness(atoms,chain,{})):issues.append('protein_heavy_atoms:'+chain)
        if len(sequences)!=len(job['modeling_input']['enzyme_state']['chains']):issues.append('protein_copy_count')
        observed={k[0] for k in atoms};expected=set(sequences)
        for entity in entities.values():expected.update(entity if isinstance(entity,list) else [entity])
        if observed!=expected:issues.append('entity_inventory')
        for ccd,chains in ((k,v) for k,v in entities.items() if isinstance(v,list)):
            for chain in chains:
                actual={k[2]:v['element'] for k,v in atoms.items() if k[0]==chain}
                if actual!=ccd_atoms.get(ccd):issues.append('cofactor_identity:'+chain)
        for chain,definition in source.get('ligand_atom_maps',{}).items():
            actual={k[2]:v['element'] for k,v in atoms.items() if k[0]==chain}
            desired={v['name']:v['element'] for v in definition['atoms'].values()}
            if actual!=desired:issues.append('ligand_atom_inventory:'+chain)
        def coordinate(ref):
            matches=[v['xyz'] for key,v in atoms.items() if key[0]==ref[0] and key[2]==ref[1]]
            if len(matches)!=1:raise ValueError('Missing/ambiguous reaction atom '+str(ref))
            return matches[0]
        centers=references('electrophile');leaving=references('leaving_group')
        if len(centers)!=1 or len(leaving)!=1:raise ValueError('Reaction center is ambiguous')
        center=coordinate(centers[0]);axis=coordinate(leaving[0])-center
        donors=[(ref,coordinate(ref)) for ref in references('nucleophile')]
        ref,donor=min(donors,key=lambda pair:np.linalg.norm(pair[1]-center))
        vector=donor-center;distance=float(np.linalg.norm(vector))
        angle=math.degrees(math.acos(float(np.clip(np.dot(vector,axis)/(distance*np.linalg.norm(axis)),-1,1))))
        poses.append({'path':str(path),'sha256':prediction['sha256'],'distance_A':distance,'angle_deg':angle,
                      'identity_issues':issues,'nucleophile':ref})
    score=weighted_attack_angle([p['distance_A'] for p in poses],[p['angle_deg'] for p in poses])
    return {'poses':poses,'geometry':score,'identity_pass_poses':sum(not p['identity_issues'] for p in poses),
        'step':job['step'],'ideal_angle':geometry.get('ideal_angle'),'scope':'Raw step-specific structural observations; no enzyme-activity threshold or cross-step activity ranking.'}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--backend',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pilot-accession');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--pilot-steps',default='1,2')
    parser.add_argument('--max-protein-length',type=int)
    parser.add_argument('--expand-after-pilot',action='store_true')
    parser.add_argument('--worker-id',default='worker-0')
    parser.add_argument('--msa-cache',type=Path)
    parser.add_argument('--watch-seconds',type=int,default=0)
    args=parser.parse_args(argv);args.out.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch('[A-Za-z0-9_.-]+',args.worker_id):raise ValueError('Unsafe worker identity')
    failed=0;seen=set()
    with state_lock(args.out/'workers'/args.worker_id/'lock'):
        atomic_write_json(args.out/'workers'/args.worker_id/'owner.json',{'host':socket.gethostname(),'pid':os.getpid(),
            'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'started_at':time.time()})
        while True:
            if not args.specification.exists():
                if adopt_registered_protocol(args.run,args.specification):continue
                atomic_write_json(args.out/'workers'/args.worker_id/'current.json',{
                    'status':'awaiting_shared_family_protocol','path':str(args.specification),'setup_llm_calls':0})
                if not args.watch_seconds:return
                time.sleep(args.watch_seconds);continue
            specification=json.loads(args.specification.read_text());handoff=build_handoff(args.run,specification)
            atomic_write_json(args.out/'family_handoff.json',handoff)
            from .family_sequences import prepare_members
            candidates=prepare_members(handoff,_prepared_candidates(args.run),args.out/'sequences',
                args.msa_cache or args.run.parent/'.sequence_msa_cache')
            jobs,pending=render_jobs(args.run,handoff,args.out/'inputs',candidates)
            atomic_write_json(args.out/'queue.json',{'items':jobs,'pending':pending,'setup_llm_calls':0})
            write_input_index(jobs,args.out)
            publish_results(args.out,jobs)
            if args.prepare_only:return
            pilot_steps={int(value) for value in args.pilot_steps.split(',')}
            pilot_jobs=[j for j in jobs if j['accession']==args.pilot_accession and j['step'] in pilot_steps]
            pilot_passed=bool(pilot_jobs) and all((args.out/'jobs'/j['id']/'complete.json').is_file() for j in pilot_jobs)
            if pilot_passed:
                pilot_passed=all(json.loads((args.out/'jobs'/j['id']/'complete.json').read_text())['analysis']['identity_pass_poses']==50 for j in pilot_jobs)
            selected=[j for j in jobs if not args.pilot_accession or (j['accession']==args.pilot_accession and j['step'] in pilot_steps) or (args.expand_after_pilot and pilot_passed)]
            backend_definition=json.loads(args.backend.read_text())
            provider_root=Path(backend_definition['argv'][backend_definition['argv'].index('--state-dir')+1])
            owned_native={p.parent.name for p in (provider_root/'backend_jobs').glob('*/receipt.json')
                if json.loads(p.read_text()).get('status') in {'queued','running','unknown','complete'}
                and not (args.out/'jobs'/p.parent.name/'complete.json').exists()}
            if owned_native:selected=[j for j in selected if j['id'] in owned_native]
            if args.max_protein_length:
                import yaml
                selected=[j for j in selected if max(len(e['protein']['sequence']) for e in yaml.safe_load(Path(j['input_yaml']).read_text())['sequences'] if e.get('protein'))<=args.max_protein_length]
            selected.sort(key=lambda j:(j['rank'],j['step']))
            for job in selected:
                if job['id'] in seen:continue
                folder=args.out/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
                complete=folder/'complete.json'
                if complete.exists():
                    saved=json.loads(complete.read_text())
                    if saved['job']['input_sha256']!=job['input_sha256']:raise ValueError('Cached input identity changed')
                    analyze_outputs(saved.get('reuse_source_job') or job,saved['receipt'])
                    seen.add(job['id']);continue
                reused=find_reusable(job,args.run,specification)
                if reused:
                    if reused['status']=='reused':
                        atomic_write_json(complete,{'job':job,'reuse_source_job':reused['job'],'receipt':reused['receipt'],
                            'analysis':reused['analysis'],'reuse_source':reused['source']})
                    else:atomic_write_json(folder/'reuse_pending_analysis.json',reused)
                    seen.add(job['id']);continue
                claim=args.out/'claims'/job['id'];claim.parent.mkdir(exist_ok=True)
                try:claim.mkdir()
                except FileExistsError:
                    owner_path=claim/'owner.json'
                    if owner_path.exists():
                        owner=json.loads(owner_path.read_text())
                        if owner.get('host')==socket.gethostname():
                            try:actual=Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]
                            except FileNotFoundError:actual=None
                            native=[json.loads(p.read_text()) for p in args.out.glob('workers/*/provider/backend_jobs/'+job['id']+'/receipt.json')]
                            if actual!=owner.get('process_start_ticks') and not any(r.get('status') in {'queued','running','unknown'} for r in native):
                                archive=args.out/'stale_claims';archive.mkdir(exist_ok=True)
                                try:os.rename(claim,archive/(job['id']+'_'+str(time.time_ns())))
                                except FileNotFoundError:pass
                    continue
                atomic_write_json(claim/'owner.json',{'pid':os.getpid(),'host':socket.gethostname(),'worker_id':args.worker_id,
                    'process_start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]})
                atomic_write_json(args.out/'workers'/args.worker_id/'current.json',{'status':'running','accession':job['accession'],
                    'enzyme_name':job['enzyme_name'],'step':job['step'],'job':job['id'],'setup_llm_calls':0})
                shared_claim=None
                try:
                    from .family_inflight import acquire,release
                    shared_claim=acquire(args.run,job,args.worker_id)
                    if shared_claim is None:
                        atomic_write_json(folder/'inflight_wait.json',{'status':'waiting_for_exact_existing_system',
                            'discovery_run_id':args.run.name,'job':job['id']})
                        continue
                    receipt=execute_job(job,json.loads(args.backend.read_text()),folder)
                    analysis=analyze_outputs(job,receipt)
                    atomic_write_json(complete,{'job':job,'receipt':receipt,'analysis':analysis,'scope':'Complete ensemble with identity checks and step-specific geometry; not experimental activity.'})
                    publish_results(args.out,jobs)
                    failed=0;seen.add(job['id'])
                except Exception as exc:
                    atomic_write_json(folder/'failure.json',{'error':str(exc),'job':job})
                    failed+=1
                    if (args.pilot_accession and not pilot_passed) or failed>=3:raise
                finally:
                    if shared_claim is not None:release(shared_claim)
                    (claim/'owner.json').unlink();claim.rmdir()
            atomic_write_json(args.out/'workers'/args.worker_id/'current.json',{'status':'waiting_for_new_ranked_candidates' if args.watch_seconds else 'finished',
                'completed_jobs':len(seen),'setup_llm_calls':0})
            if not args.watch_seconds:return
            time.sleep(args.watch_seconds)


if __name__=='__main__':main()
