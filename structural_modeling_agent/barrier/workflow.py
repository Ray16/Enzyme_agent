"""Agent-owned barrier studies, with immutable plans and durable job receipts.

The LLM selects a reviewed scientific protocol. This module builds typed work,
checks artifact identity, and calls a reservation-aware provider; it never runs
a calculator in the controller process.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid

from ..validation import file_digest, fingerprint
from enzyme_discovery_agent.state_store import atomic_write_json

ACTIVE = {'submission_pending','queued','running','unknown'}
TOOLS = {'inspect_barrier_contract','prepare_barrier_inputs','plan_barrier_calculation',
         'submit_barrier_calculation','check_barrier_calculation','inspect_barrier_result',
         'record_barrier_assessment'}
READERS = {'inspect_barrier_contract','check_barrier_calculation','inspect_barrier_result'}


def tool_schemas():
    string=lambda description:dict(type='string',description=description)
    def schema(name,description,properties,required):
        return dict(type='function',function=dict(name=name,description=description,
            parameters=dict(type='object',properties=properties,required=required)))
    identity=dict(accession=string('Candidate identifier'),study_id=string('Saved barrier study ID'))
    return [
        schema('plan_barrier_calculation','Save a chemically reviewed, bounded barrier study. Embedded DFT/MM is the main assessment route; cluster/UMA and UMA-xTB are exploratory alternatives requiring a stated rationale. Does not submit compute.',{
            'accession':string('Candidate identifier'),
            'preparation_path':string('Hashed preparation.json returned by prepare_barrier_inputs'),
            'label':string('Safe label for this explicit protocol version; use a new version after failures or changed inputs'),
            'method':dict(type='string',enum=['dft','uma','uma-xtb']),
            'rationale':string('Scientific question, why this method is informative, and why its environmental approximation is appropriate'),
            'setup_review':dict(type='object',description='Explicit reviewed setup and conditional microstate; required true checks and supporting hashed artifacts',properties={
                **{name:dict(type='boolean') for name in ('sequence_mapping_reviewed','molecular_graph_reviewed','protonation_reviewed','metal_shell_reviewed','parameters_reviewed')},
                'conditional_state':string('Exact proton/metal/spin state being tested'),
                'sources':dict(type='array',minItems=1,items=dict(type='object',properties={
                    'path':string('Authorized reviewed setup artifact'),'sha256':string('Actual artifact SHA-256')},required=['path','sha256']))},
                required=['sequence_mapping_reviewed','molecular_graph_reviewed','protonation_reviewed','metal_shell_reviewed','parameters_reviewed','conditional_state','sources']),
            'settings':dict(type='object',description='Explicit images, max_steps and wall_seconds; optional route, basis, xc, model and ECP',properties={
                'images':dict(type='integer',minimum=3,maximum=41),
                'max_steps':dict(type='integer',minimum=1,maximum=2000),
                'wall_seconds':dict(type='integer',minimum=1,maximum=86400),
                'route':dict(type='string',enum=['concerted','form-first','break-first']),
                'basis':string('DFT basis'), 'xc':string('DFT functional'), 'model':string('UMA model identifier'),
                'ecp':dict(type=['string','object'])},required=['images','max_steps','wall_seconds'])},
            ['accession','preparation_path','label','method','rationale','setup_review','settings']),
        schema('submit_barrier_calculation','Submit the saved study once through the configured reservation-aware barrier backend. Missing backend/resources remain explicit gaps; no automatic method fallback.',identity,['accession','study_id']),
        schema('check_barrier_calculation','Reconcile the existing job receipt and verify returned output hashes; do not resubmit an uncertain job.',identity,['accession','study_id']),
        schema('record_barrier_assessment','Record interpretation of a completed calculation, its limitations and whether it answers the declared exploratory/static-energy question. Does not certify an activation free energy.',{
            **identity,'interpretation':string('Interpret the numerical output against the scientific question'),
            'limitations':dict(type='array',minItems=1,items=dict(type='string')),
            'answers_question':dict(type='boolean'), 'next_action':string('Next scientific action or explicit stopping conclusion')},
            ['accession','study_id','interpretation','limitations','answers_question','next_action'])]


def studies(agent,accession):
    agent._candidate(accession)
    return agent.state['candidates'][accession].setdefault('barrier_studies',{})


def _candidate_identity(agent,accession):
    candidate=agent._candidate(accession)
    sequence=candidate.get('sequence')
    if not isinstance(sequence,str) or not re.fullmatch('[A-Z]+',sequence):
        raise ValueError('Resolve the exact candidate sequence before planning a barrier')
    return dict(sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                target_sha256=fingerprint(agent.task.get('target')))


def _verify_files(files,authorize=None):
    if not isinstance(files,dict) or not files:raise ValueError('Missing artifact fingerprints; regenerate preparation')
    for name,digest in files.items():
        path=authorize(name) if authorize else Path(name).resolve(strict=True)
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or file_digest(path)!=digest:
            raise ValueError(f'Barrier input or output changed: {name}')


def _settings(method,mode,settings):
    if mode not in {'cluster','embedded'}:raise ValueError('Unknown prepared barrier mode')
    if mode=='cluster' and method!='uma' or mode=='embedded' and method not in {'dft','uma-xtb'}:
        raise ValueError('Method does not match preparation mode; prepare the requested molecular model explicitly')
    if not isinstance(settings,dict):raise ValueError('Explicit bounded calculation settings required')
    allowed={'images','max_steps','wall_seconds','route','basis','xc','model','ecp'}
    if settings.keys()-allowed:raise ValueError('Unknown barrier settings')
    for name,upper in [('images',41),('max_steps',2000),('wall_seconds',86400)]:
        lower=3 if name=='images' else 1
        if type(settings.get(name)) is not int or not lower<=settings[name]<=upper:
            raise ValueError(f'Explicit bounded {name} required')
    values=dict(route='concerted',basis='def2-svp',xc='wb97m-v',model='uma-s-1p2') | settings
    if values['route'] not in {'concerted','form-first','break-first'}:
        raise ValueError('Unknown path route')
    if mode=='cluster' and values['route']=='break-first':
        raise ValueError('Cluster break-first is not implemented; use an embedded path or a supported cluster route')
    for name in ('basis','xc','model'):
        if not isinstance(values[name],str) or not values[name].strip():raise ValueError(f'Invalid {name}')
    return values


def plan(agent,args):
    accession=args['accession'];records=studies(agent,accession)
    identity=_candidate_identity(agent,accession)
    identity_gap=agent._sequence_identity_error(accession)
    if identity_gap:raise ValueError(identity_gap)
    label=args['label']
    if not isinstance(label,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',label):
        raise ValueError('Study label must be a safe file component')
    if any(s['plan']['label']==label for s in records.values()):
        raise ValueError('Study label already exists; inspect that study or declare a new explicit protocol version')
    rationale=args.get('rationale')
    if not isinstance(rationale,str) or not rationale.strip():raise ValueError('Scientific method rationale required')
    review=args['setup_review']
    for name in ('sequence_mapping_reviewed','molecular_graph_reviewed','protonation_reviewed','metal_shell_reviewed','parameters_reviewed'):
        if review.get(name) is not True:raise ValueError(f'Setup review unresolved: {name}')
    if not isinstance(review.get('conditional_state'),str) or not review['conditional_state'].strip():
        raise ValueError('Declare the conditional molecular state')
    sources=review.get('sources')
    if not isinstance(sources,list) or not sources:raise ValueError('Hashed setup-review sources required')
    source_files={s['path']:s['sha256'] for s in sources};_verify_files(source_files,agent._authorized_evidence)
    prep_path=agent._authorized_evidence(args['preparation_path']);preparation=json.loads(prep_path.read_text())
    if preparation.get('status')!='inputs_prepared_requires_chemical_review':raise ValueError('Preparation is not complete')
    mode=preparation['mode'];settings=_settings(args['method'],mode,args['settings'])
    inputs={**preparation.get('input_sha256',{}),**preparation.get('output_sha256',{})}
    _verify_files(preparation.get('input_sha256'),agent._authorized_evidence)
    _verify_files(preparation.get('output_sha256'),agent._authorized_evidence)
    inputs.update(source_files);inputs[str(prep_path)]=file_digest(prep_path)
    identifier=f'{label}-{uuid.uuid4().hex[:12]}'
    root=agent.run_dir/'artifacts'/'barriers'/accession/identifier;root.mkdir(parents=True,exist_ok=False)
    from .provider import freeze_producer
    producer,producer_hashes=freeze_producer(root/'producer')
    specification=dict(schema_version='barrier-study-1.0',id=identifier,label=label,accession=accession,
        protein_length=len(agent._candidate(accession)['sequence']),
        **identity,mode=mode,method=args['method'],settings=settings,
        rationale=rationale,setup_review=review,preparation_path=str(prep_path),
        prepared_directory=preparation['output_dir'],input_sha256=inputs,
        producer_directory=str(producer),producer_sha256=producer_hashes,
        work_dir=str(root),result_path=str(root/'result.json'),
        quantity='constrained_electronic_energy_gap' if args['method']=='dft' else 'constrained_reference_energy_gap',
        validated_activation_free_energy=False)
    path=root/'plan.json';atomic_write_json(path,specification,allow_nan=False)
    records[identifier]=dict(plan=specification,plan_path=str(path),plan_sha256=file_digest(path),status='planned')
    return dict(study_id=identifier,plan_path=str(path),mode=mode,method=args['method'],
                status='planned',next_action='submit_barrier_calculation',quantity=specification['quantity'])


def _checked(agent,accession,identifier):
    record=studies(agent,accession)[identifier];plan=record['plan']
    if file_digest(record['plan_path'])!=record['plan_sha256']:raise ValueError('Barrier plan changed')
    if any(plan[k]!=v for k,v in _candidate_identity(agent,accession).items()):
        raise ValueError('Candidate sequence or target changed; declare a new barrier study')
    _verify_files(plan['input_sha256'])
    _verify_files({str(Path(plan['producer_directory'])/p):h for p,h in plan['producer_sha256'].items()})
    return record


def backend_config(agent):
    config=getattr(agent,'barrier_backend',None) or agent.state.get('barrier_backend')
    if config is None and isinstance(agent.backend,dict):config=agent.backend.get('barrier')
    if not isinstance(config,dict):raise ValueError('No barrier backend configured; use --barrier-backend or backend.barrier')
    policy=config.get('resource_policy') or {}
    if policy.get('reservation_managed') is not True:raise ValueError('Barrier backend must manage reservations')
    for field in ('max_jobs','max_wall_seconds'):
        if type(policy.get(field)) is not int or policy[field]<1:raise ValueError(f'Explicit positive barrier resource_policy.{field} required')
    if not config.get('provider'):raise ValueError('Barrier provider is missing')
    methods=config.get('supported_methods')
    if not isinstance(methods,list) or not methods or any(m not in {'dft','uma','uma-xtb'} for m in methods):
        raise ValueError('Barrier backend must declare supported_methods')
    if config['provider']!='local_reserved':
        argv=config.get('argv')
        if not isinstance(argv,list) or not argv or not all(isinstance(v,str) and v for v in argv):
            raise ValueError('External barrier adapter requires an explicit argv list')
    else:
        from .provider import validate_config
        validate_config(config)
    return config


def _accept_result(record,reply):
    plan=record['plan'];path=Path(plan['result_path'])
    if reply.get('result_sha256')!=file_digest(path):raise ValueError('Barrier result fingerprint is missing or changed')
    output=json.loads(path.read_text())
    if not isinstance(output,dict) or output.get('plan_sha256')!=record['plan_sha256'] or output.get('status')!='calculation_complete':
        raise ValueError('Barrier result is not bound to the completed plan')
    if any(output.get(key)!=plan[key] for key in ('mode','method','quantity')):
        raise ValueError('Barrier result method or scientific quantity differs from the plan')
    outputs=output.get('output_sha256');_verify_files(outputs)
    if any(not Path(name).resolve().is_relative_to(Path(plan['work_dir']).resolve()) for name in outputs):
        raise ValueError('Barrier output lies outside its declared work directory')
    gap=output.get('dE_act_kcal')
    if (gap is not None and (type(gap) not in (int,float) or not math.isfinite(gap))) or (
            output.get('calculation_status')=='converged_profile' and gap is None):
        raise ValueError('Completed profile has no admissible energy gap')
    record['result']=dict(path=str(path),sha256=file_digest(path),quantity=output.get('quantity'),
        dE_act_kcal=output.get('dE_act_kcal'),calculation_status=output.get('calculation_status'),
        validated_activation_free_energy=False)


def reconcile(agent,accession,identifier,operation):
    record=_checked(agent,accession,identifier);plan=record['plan'];config=backend_config(agent)
    if plan['method'] not in config['supported_methods']:
        raise ValueError('Configured barrier backend does not support this method; do not silently substitute another potential')
    if plan['settings']['wall_seconds']>config['resource_policy']['max_wall_seconds']:
        raise ValueError('Barrier calculation exceeds the configured wall budget')
    key=fingerprint(dict(plan_sha256=record['plan_sha256'],backend=config))
    job=record.get('job')
    if job and job['idempotency_key']!=key:raise ValueError('Barrier provider/configuration changed; retain this job and create a new study')
    if operation=='submit' and job and job['status']!='not_found':
        return dict(study_id=identifier,status='already_requested',job=job,next_action='check_barrier_calculation')
    if operation=='status' and not job:return dict(study_id=identifier,status='not_submitted')
    if operation=='submit':
        if agent.task.get('workflow_mode')=='production_cofold' and agent.task.get('objective') not in {'evaluate_barriers','prepare_sampling'}:
            from ..workflow_policy import baseline_errors
            errors=baseline_errors(agent._current(accession))
            if errors:raise ValueError('Inspect the primary cofold before barrier compute: '+'; '.join(errors))
        active=sum(s.get('job',{}).get('status') in ACTIVE for c in agent.state['candidates'].values()
                   for s in c.get('barrier_studies',{}).values())
        if active>=config['resource_policy']['max_jobs']:raise ValueError('Barrier job budget is occupied; inspect existing jobs')
        if config['provider']=='local_reserved':
            from .provider import validate_config,verify_cpu_capacity
            validate_config(config)
            verify_cpu_capacity(config)
            # Keep two CPUs available for bounded preparation operators even
            # while a previous study runs; the queue owns the final CPU.
            controller=set(config['controller_cpus'][:-1])
            if not controller<=set(os.sched_getaffinity(0)):
                raise ValueError('Configured controller CPUs are outside current affinity')
            os.sched_setaffinity(0,controller)
        record['job']=dict(idempotency_key=key,status='submission_pending')
        record['status']='submission_pending';agent._save()
    request=dict(operation=operation,idempotency_key=key,plan_path=record['plan_path'],
                 plan_sha256=record['plan_sha256'],provider=config['provider'],config=config)
    argv=config.get('argv')
    env=os.environ.copy()
    if config['provider']=='local_reserved':
        argv=[sys.executable,'-m','structural_modeling_agent.barrier.provider']
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
    try:
        result=subprocess.run(argv,input=json.dumps(request,allow_nan=False),text=True,capture_output=True,
            cwd=str(agent.run_dir),env=env,timeout=min(float(config.get('timeout_seconds',30)),60),check=True)
        reply=json.loads(result.stdout)
        if reply.get('idempotency_key')!=key or reply.get('status') not in {'queued','running','complete','failed','unknown','not_found'}:
            raise ValueError('Invalid or mismatched barrier job receipt')
    except (subprocess.SubprocessError,ValueError,OSError) as exc:
        # The adapter may have launched before losing its reply. Keep ownership
        # uncertain and reconcile; never turn this into automatic resubmission.
        record['job'].update(status='unknown',last_error=str(exc));record['status']='unknown';agent._save()
        raise
    record['job']=reply;record['status']=reply['status']
    if reply['status']=='complete':
        try:_accept_result(record,reply)
        except (ValueError,OSError,KeyError) as exc:
            record['job'].update(status='unknown',last_error=f'Output verification failed: {exc}')
            record['status']='unknown';agent._save();raise
    agent._save()
    return dict(study_id=identifier,status=reply['status'],job=reply,result=record.get('result'))


def assess(agent,args):
    record=_checked(agent,args['accession'],args['study_id'])
    result=record.get('result')
    if record.get('status')!='complete' or not result:raise ValueError('Inspect a completed, fingerprinted calculation first')
    _verify_files({result['path']:result['sha256']})
    output=json.loads(Path(result['path']).read_text());_verify_files(output['output_sha256'])
    for field in ('interpretation','next_action'):
        if not isinstance(args.get(field),str) or not args[field].strip():raise ValueError(f'Explicit {field} required')
    if type(args.get('answers_question')) is not bool:raise ValueError('Explicit answers_question required')
    limits=args.get('limitations')
    if not isinstance(limits,list) or not limits or not all(isinstance(v,str) and v.strip() for v in limits):
        raise ValueError('Explicit scientific limitations required')
    if args['answers_question'] and output.get('calculation_status')!='converged_profile':
        raise ValueError('Unresolved calculations cannot answer a barrier-energy question')
    assessment={key:args[key] for key in ('interpretation','next_action','answers_question','limitations')}
    assessment.update(result_sha256=result['sha256'],validated_activation_free_energy=False)
    record['assessment']=assessment;record['assessment_artifact']=agent._artifact('barrier_assessment',assessment)
    return dict(study_id=args['study_id'],status='reviewed',**assessment)


def completion_errors(agent,accession,required=False):
    records=studies(agent,accession);errors=[]
    if required and not records:return ['no barrier study has been planned']
    for identifier,record in records.items():
        try:
            _checked(agent,accession,identifier)
            if record.get('status')!='complete' or not record.get('assessment',{}).get('answers_question'):
                errors.append(f'{identifier}: barrier execution or scientific assessment remains unresolved')
                continue
            result=record['result'];_verify_files({result['path']:result['sha256']})
            output=json.loads(Path(result['path']).read_text());_verify_files(output['output_sha256'])
            assessment=json.loads(Path(record['assessment_artifact']).read_text())
            if assessment!=record['assessment'] or assessment['result_sha256']!=result['sha256']:
                errors.append(f'{identifier}: barrier assessment changed')
        except (OSError,ValueError,KeyError) as exc:errors.append(f'{identifier}: {exc}')
    return errors


def summary(records):
    return {identifier:dict(status=record.get('status'),mode=record['plan']['mode'],
        method=record['plan']['method'],rationale=record['plan']['rationale'],plan_path=record['plan_path'],
        job=record.get('job'),result=record.get('result'),assessment=record.get('assessment'))
        for identifier,record in records.items()}
