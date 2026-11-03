"""Let discovery record compact, source-linked catalytic proposals in batches."""
from collections import Counter
import csv
import io
import json
from pathlib import Path

from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
from enzyme_discovery_agent.models.native_annotation import native_annotation_for_report
from enzyme_discovery_agent.models.target_review import digest, identity_verified, target_contract
from enzyme_discovery_agent.state_store import atomic_write_json, atomic_write_text, state_lock


ROLE = {'type':'object','required':['component_index','role','basis','recognition_change'],
        'properties':{'component_index':{'type':'integer','minimum':1},
            'role':{'type':'string','minLength':1},'basis':{'type':'string','minLength':1},
            'recognition_change':{'type':'string'}}}
PROPOSAL = {'type':'object','required':['accessions','tier','family','hypothesis','stage',
    'support_basis','target_roles','evidence','structural_question','unknowns','variant_rationale'],
    'properties':{'accessions':{'type':'array','minItems':1,'maxItems':8,'items':{'type':'string'}},
        'tier':{'type':'string','enum':['primary','exploratory','supporting','deferred']},
        'family':{'type':'string'},'hypothesis':{'type':'string'},
        'stage':{'type':'string','enum':['requested_transformation','coupled_route','later_extension','auxiliary']},
        'support_basis':{'type':'string','enum':['experimental_scope','native_role_analogy','catalytic_architecture','family_only']},
        'target_roles':{'type':'array','items':ROLE},
        'evidence':{'type':'array','minItems':1,'maxItems':4,'items':{'type':'object',
            'required':['path','quote','supports'],'properties':{
                'path':{'type':'string'},'quote':{'type':'string','minLength':1},
                'supports':{'type':'string','minLength':1}}}},
        'structural_question':{'type':'string'},'unknowns':{'type':'array','items':{'type':'string'}},
        'variant_rationale':{'type':'string'}}}


def record_proposals(run_dir, proposals):
    root=Path(run_dir).resolve(); registry=load_candidate_registry(root,strict=True)['candidates']
    target=target_contract(root); target_hash=digest(target)
    components=[s for s in (target.get('substrate_smiles') or '').split('.') if s]
    accepted=[]; errors=[]
    for index, supplied in enumerate(proposals):
        try:
            proposal=dict(supplied)
            if proposal.get('stage') not in {'requested_transformation','coupled_route','later_extension','auxiliary'}:
                raise ValueError('Unknown reaction stage')
            if not isinstance(proposal.get('unknowns'),list):
                raise ValueError('Preserve remaining uncertainties as an explicit list')
            if proposal.get('tier') not in {'primary','exploratory','supporting','deferred'}:
                raise ValueError('Unknown proposal tier')
            accessions=proposal.get('accessions') or []
            if not 1<=len(accessions)<=8 or len(set(accessions))!=len(accessions):
                raise ValueError('Use one to eight distinct registered sequences per proposal')
            if len(accessions)>1 and not str(proposal.get('variant_rationale') or '').strip():
                raise ValueError('Explain the information gained from additional related sequences')
            for key in ('family','hypothesis','structural_question'):
                if not str(proposal.get(key) or '').strip():
                    raise ValueError('Missing '+key)
            roles=proposal.get('target_roles') or []
            for role in roles:
                if not all(str(role.get(k) or '').strip() for k in ('role','basis')):
                    raise ValueError('Reacting roles require a role and its evidence or inference basis')
            if proposal['tier']=='primary':
                if proposal.get('stage') not in {'requested_transformation','coupled_route'}:
                    raise ValueError('Later-stage and auxiliary hypotheses cannot fill the primary panel')
                if proposal.get('support_basis') not in {'experimental_scope','native_role_analogy','catalytic_architecture'}:
                    raise ValueError('A family label alone is insufficient for a primary catalytic proposal')
                if not components or {r.get('component_index') for r in roles} != set(range(1,len(components)+1)):
                    raise ValueError('A primary proposal must account for every supplied target component')
            sources=[]
            for item in proposal.get('evidence') or []:
                path=(root/item['path']).resolve()
                if not path.is_relative_to(root/'artifacts') or not path.is_file():
                    raise ValueError('Evidence must be a retrieved artifact in this campaign')
                if any(part in {'discovery_proposals','discovery_quality_proposals','shortlist_decisions'} for part in path.parts):
                    raise ValueError('A generated nomination cannot serve as its own supporting source')
                text=path.read_text()
                if not item.get('quote') or item['quote'] not in text or not item.get('supports'):
                    raise ValueError('Cite an exact saved passage and state what it supports')
                sources.append({**item,'path':str(path.relative_to(root)),
                                'sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest()})
            if not sources:
                raise ValueError('At least one independently retrieved evidence passage is required')
            proposal.update(evidence=sources,target_sha256=target_hash,
                scope='Agent-selected catalytic hypothesis; native and target activity remain separate observations.')
            for accession in accessions:
                meta=(registry.get(accession) or {}).get('metadata') or {}
                if not identity_verified(root,accession,meta):
                    raise ValueError('Resolve this exact sequence identity before nomination: '+accession)
            path=root/'artifacts/discovery_quality_proposals'/(digest(proposal)+'.json')
            if not path.exists():atomic_write_json(path,proposal)
            accepted.append((proposal,str(path.relative_to(root))))
        except (KeyError,TypeError,ValueError,OSError) as exc:
            errors.append({'proposal_index':index,'error':str(exc)})
    state_path=root/'agent_state/discovery_quality.json'
    with state_lock(state_path):
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('target_sha256')!=target_hash:state={'target_sha256':target_hash,'candidates':{}}
        for proposal,path in accepted:
            for accession in proposal['accessions']:
                state['candidates'][accession]={'proposal_path':path,'proposal_sha256':digest(proposal)}
        atomic_write_json(state_path,state)
    summary=write_primary_portfolio(root)
    return {'status':'success' if accepted else 'error','summary':
        f'Recorded {len(accepted)} source-linked group proposals; {summary["selected"]} primary candidates.',
        'data':summary,'errors':errors}


def write_primary_portfolio(run_dir):
    root=Path(run_dir); target_hash=digest(target_contract(root))
    state_path=root/'agent_state/discovery_quality.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    registry=load_candidate_registry(root,strict=True)['candidates']; rows=[]; other=[]
    for accession,ref in (state.get('candidates',{}) if state.get('target_sha256')==target_hash else {}).items():
        path=(root/ref['proposal_path']).resolve()
        if not path.is_relative_to((root/'artifacts/discovery_quality_proposals').resolve()) or not path.is_file():continue
        proposal=json.loads(path.read_text())
        if digest(proposal)!=ref['proposal_sha256'] or proposal.get('target_sha256')!=target_hash:continue
        if any(not (root/e['path']).is_file() or __import__('hashlib').sha256((root/e['path']).read_bytes()).hexdigest()!=e['sha256']
               for e in proposal['evidence']):continue
        meta=(registry.get(accession) or {}).get('metadata') or {}
        if not identity_verified(root,accession,meta):continue
        data={**(meta.get('screening_annotation') or {}),**(meta.get('evaluation_data') or {}),**meta}
        annotation=native_annotation_for_report(root,accession,data)
        name=meta.get('literature_name') or meta.get('protein_name') or (meta.get('screening_annotation') or {}).get('protein_name') or 'Unresolved protein name'
        row={'accession':accession,'enzyme_name':name,'enzyme':name+' ('+accession+')',
            **{k:proposal[k] for k in ('tier','family','hypothesis','stage','support_basis','structural_question')},
            'target_roles':json.dumps(proposal['target_roles']), 'unknowns':'; '.join(proposal['unknowns']),
            'native_activity_evidence':annotation['native_activity_evidence'],
            'native_activity_sources':'; '.join(sorted({str(e.get('source'))+':'+str(e.get('id'))
                for r in annotation['reactions'] for e in r['evidences'] if e.get('source') and e.get('id')})),
            'proposal_source':ref['proposal_path'],'target_activity':'unestablished'}
        (rows if proposal['tier']=='primary' else other).append(row)
    rows.sort(key=lambda row:(row['family'],row['proposal_source'],row['accession']))
    fields=list((rows or other)[0]) if rows or other else ['enzyme_name','accession','tier','hypothesis']
    for filename,values in [('primary_candidates.csv',rows),('exploratory_candidates.csv',other)]:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)
        atomic_write_text(root/filename,stream.getvalue())
    summary={'selected':len(rows),'families':dict(Counter(r['family'] for r in rows)),
             'other_proposals':len(other),'candidates':rows,
             'scope':'Primary agent-selected modeling hypotheses, not demonstrated target activity. Counts do not establish adequacy.'}
    atomic_write_json(root/'primary_modeling_handoff.json',summary)
    return summary


@tool(description='Record compact source-grounded catalytic proposals for one or several retrieved sequences per group. Primary proposals account for every target component and the immediate requested transformation. Family-only, later-stage and auxiliary hypotheses remain separate. This records discovery reasoning without a separate per-sequence activity review.',
      category='reasoning',parallel_safe=False,injected_params=['run_dir'],params={
          'run_dir':{'type':'string'},'proposals':{'type':'array','minItems':1,'maxItems':8,'items':PROPOSAL}})
def record_discovery_proposals(run_dir, proposals):
    return record_proposals(run_dir, proposals)
