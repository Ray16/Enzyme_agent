"""Campaign-local predictions, evidence and executed research decisions.

The controller verifies provenance and execution, not scientific entailment.
Inferences stay separate from observations; a successful retrieval is not a
successful biological test. No model-written thought is promoted to evidence.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, discovery_status, family_label
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _root(run_dir):
    return Path(run_dir) / 'agent_state' / 'research_decisions'


def action_kind(tool_name):
    """Changing priorities alone supplies no new functional evidence."""
    if tool_name in {'set_family_focus','set_candidate_focus'}:
        return 'priority_update'
    if tool_name in {'assess_next_batch', 'evaluate_candidate'}:
        return 'candidate_assessment'
    if tool_name.startswith(('search_', 'fetch_', 'lookup_', 'extract_', 'read_')):
        return 'evidence_retrieval'
    return 'evidence_inspection_or_research_action'


def inspection_epoch(run_dir):
    state = _read(_root(run_dir)/'state.json', {})
    return _hash([state.get('last_decision'), _signal(observations(run_dir)[0])])


def record_inspection(run_dir):
    path = _root(run_dir)/'state.json'
    epoch = inspection_epoch(run_dir)
    with state_lock(path):
        state = _read(path, {})
        state['inspected_epoch'] = epoch
        state['review_inspections'] = state.get('review_inspections',0)+1
        atomic_write_json(path, state)


INSPECTION_TOOLS = frozenset({'get_decision_context','get_exploration_state','get_candidate_evidence','get_research_memory','inspect_discovery_source'})


def review_limits(run_dir):
    """Bound deliberation work, never the scientific conclusion it must reach."""
    from enzyme_discovery_agent import config
    state = _read(_root(run_dir)/'state.json',{})
    inspections=state.get('review_inspections',0)
    actions=state.get('review_actions',0)
    return {'inspection_count':inspections, 'action_count':actions,
            'inspection_limit':config.RESEARCH_REVIEW_MAX_INSPECTIONS,
            'action_limit':config.RESEARCH_REVIEW_MAX_ACTIONS,
            'inspections_exhausted':inspections>=config.RESEARCH_REVIEW_MAX_INSPECTIONS,
            'conclusion_required':actions>=config.RESEARCH_REVIEW_MAX_ACTIONS}


def review_action_allowed(run_dir, tool_name):
    limits=review_limits(run_dir)
    if tool_name in {'interpret_research_outcome', 'interpret_saved_outcome'}:
        return bool(_read(_root(run_dir)/'state.json',{}).get('awaiting_interpretation'))
    if tool_name=='conclude_research_review':
        return True
    return not limits['conclusion_required'] and not (tool_name in INSPECTION_TOOLS and limits['inspections_exhausted'])


def inspection_completed(run_dir):
    state = _read(_root(run_dir)/'state.json', {})
    return state.get('inspected_epoch') == inspection_epoch(run_dir)


def _literature_observations(root):
    """Make retrieved passages citable before any candidate has been evaluated."""
    rows,seen=[],set()
    for path in sorted((root/'artifacts/literature').rglob('*.json'),
                       key=lambda path:(path.stat().st_mtime_ns,str(path)),reverse=True):
        payload=_read(path,{})
        result=payload.get('result') or {}
        data=result.get('data')
        papers=data if isinstance(data,list) else [paper for key in ('seed_papers','reference_papers')
            for paper in (data.get(key) or [])] if isinstance(data,dict) else []
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for index,paper in enumerate(papers):
            if not isinstance(paper,dict) or not paper.get('title'):
                continue
            key=paper.get('doi') or paper.get('pmid') or paper['title']
            if key in seen:
                continue
            seen.add(key)
            rows.append({'kind':'retrieved_literature_evidence','family':None,
                'facts':{'paper_key':str(key),'title':paper['title'],'doi':paper.get('doi'),
                    'pmid':paper.get('pmid'),'abstract_excerpt':str(paper.get('abstract') or '')[:1000],
                    'scope':'Retrieved title/abstract, not a verified enzyme identity or demonstrated target activity.'},
                'source':{'path':str(path.relative_to(root)),'sha256':digest,'paper_index':index}})
    return rows


def _native_facts(accession, data, metadata):
    """The complete facts cited by native-function observations; work priority is labeled separately."""
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    from enzyme_discovery_agent.models.ortholog import tentative_label_key
    accommodation=(data.get('substrate_similarity') or {}).get('accommodation') or {}
    roles=role_transfer_summary(data)
    priority=data.get('priority_assessment') or {}
    claims=data.get('claim_validation') or {}
    biological={'native_reaction','reaction_direction','bond_type','substrate_topology'}
    group=tentative_label_key(data)
    return {'accession':accession,'native_reaction':(data.get('catalytic_activities') or [])[:1],
            'protein_name':data.get('protein_name'),'screening_family_id':metadata.get('screening_family_id'),
            'annotation_scope':'annotation:'+group if group else None,
            'scope_limit':'Annotation groups support reversible work allocation, not biological independence or identity.',
            'unmatched_target_inputs':accommodation.get('unaccommodated_targets'),
            'all_target_inputs_accommodated':accommodation.get('all_target_substrates_accommodated'),
            'role_transfer_status':roles['status'],'participant_roles':roles,
            'agent_priority_assessment':{key:priority.get(key) for key in ('decision','reason','weaknesses','unknowns','revisit_when')} if priority else None,
            'native_evidence_signature':_hash({key:data.get(key) for key in ('catalytic_activities','native_substrates','literature_evidence')}),
            'indirect_donor_roles':[{key:item.get(key) for key in ('native_substrate','target_smiles','role')}
                for item in accommodation.get('indirect_donor_role_mappings') or []],
            'contradicted_native_claims':sorted(c.get('claim_type','') for c in claims.get('claims') or []
                if c.get('verdict')=='contradicted' and c.get('claim_type') in biological),
            'missing_required_claims':claims.get('blocking_claims') or [],'claim_status':claims.get('verdict'),
            'target_activity':'unestablished by native annotation or structural resemblance'}


def _scientific_native_facts(facts):
    return {key:value for key,value in facts.items() if key!='agent_priority_assessment'}


def observations(run_dir):
    """Project retrieved facts, preserving native/target and unknown distinctions."""
    root = Path(run_dir)
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    target = _read(run_file(root, 'target_context.json'), {})
    rows = [{'kind': 'user_target', 'family': None, 'facts': {
        k: target.get(k) for k in ('query', 'substrate_smiles')},
        'source': {'path': str(run_file(root, 'target_context.json').relative_to(root))}}]
    assessed, seen = 0, set()
    for acc, entry in sorted(load_candidate_registry(root, strict=True)['candidates'].items()):
        if discovery_status(entry) not in {'evaluated', 'ranked', 'evidence_pending', 'rejected'}:
            continue
        assessed += 1
        metadata = entry.get('metadata') or {}
        data = {**metadata, **(metadata.get('evaluation_data') or {})}
        accommodation = (data.get('substrate_similarity') or {}).get('accommodation') or {}
        roles = role_transfer_summary(data)
        priority=data.get('priority_assessment') or {}
        priority_view={key:priority.get(key) for key in ('decision','reason','weaknesses','unknowns','revisit_when')} if priority else None
        claims = data.get('claim_validation') or {}
        biological = {'native_reaction','reaction_direction','bond_type','substrate_topology'}
        contradicted = sorted(c.get('claim_type', '') for c in claims.get('claims') or []
                              if c.get('verdict') == 'contradicted' and c.get('claim_type') in biological)
        family = family_label(metadata, acc)
        from enzyme_discovery_agent.models.ortholog import tentative_label_key
        group = tentative_label_key(data)
        unmatched = accommodation.get('unaccommodated_targets')
        topology = (data.get('reaction_chemistry') or {}).get('substrate_topology')
        signature = (family, group, tuple(unmatched or []), tuple(contradicted),
                     accommodation.get('all_target_substrates_accommodated'), topology,
                     tuple(sorted(claims.get('blocking_claims') or [])), _hash(roles),_hash(priority_view),
                     _hash({key:data.get(key) for key in ('catalytic_activities','native_substrates','literature_evidence')}))
        duplicate_native = signature in seen
        seen.add(signature)
        rows.append({'kind': 'native_analogue_evidence', 'family': family,
            'facts': _native_facts(acc,data,metadata),
            'source': {'path': str(run_file(root, 'candidate_registry.json').relative_to(root)),
                       'accession': acc, 'record_sha256': _hash(data)}})
        source = rows[-1]['source']
        if duplicate_native:
            rows.pop()
        from enzyme_discovery_agent.models.modeling_feedback import modeling_feedback_summary
        modeled = modeling_feedback_summary(acc, data, source['path'])
        if modeled:
            rows.append({'kind': 'modeling_observation', 'family': family,
                         'facts': {'modeling_feedback': modeled,
                                   'target_activity': 'not established by a structural prediction'},
                         'source': source})
    rows.extend(_literature_observations(root))
    return rows, assessed


def _signal(rows):
    # Repeated variants with the same qualitative evidence do not create new
    # review work. New role/claim outcomes do, including a later compatible member.
    return _hash(sorted({_hash({'family': row['family'], 'kind': row['kind'],
        'unmatched': row['facts'].get('unmatched_target_inputs'),
        'accommodated': row['facts'].get('all_target_inputs_accommodated'),
        'annotation_scope': row['facts'].get('annotation_scope'),
        'missing': row['facts'].get('missing_required_claims'),
        'indirect_roles':row['facts'].get('indirect_donor_roles'),
        'participant_roles':row['facts'].get('participant_roles'),
        'native_evidence_signature':row['facts'].get('native_evidence_signature'),
        'literature':row['facts'].get('paper_key'),
        'modeling': row['facts'].get('modeling_feedback'),
        'contradicted': row['facts'].get('contradicted_native_claims')})
        for row in rows if row['kind'] != 'user_target'}))


def review_due(run_dir, minimum_assessed=20):
    rows, assessed = observations(run_dir)
    state = _read(_root(run_dir) / 'state.json', {})
    if state.get('awaiting_interpretation'):
        return True
    if not state.get('initial_sources_reviewed') and any(row['kind']=='retrieved_literature_evidence' for row in rows):
        return True
    model_change = any(row['kind'] == 'modeling_observation' for row in rows)
    role_change = any((row['facts'].get('participant_roles') or {}).get('review_required') for row in rows)
    consequential = any(row['facts'].get('indirect_donor_roles') or row['facts'].get('missing_required_claims') or row['facts'].get('unmatched_target_inputs') or
                        row['facts'].get('contradicted_native_claims') for row in rows)
    return ((model_change or role_change or assessed >= minimum_assessed and consequential)
            and _signal(rows) != state.get('reviewed_signal'))


def decision_context(run_dir, limit=6):
    rows, assessed = observations(run_dir)
    base = _root(run_dir)
    state = _read(base / 'state.json', {})
    # Include challenges before routine positive native annotations. Every
    # observation stays available in the content-addressed campaign catalogue.
    target_rows = [row for row in rows if row['kind'] == 'user_target']
    models = [row for row in rows if row['kind'] == 'modeling_observation']
    challenges = [row for row in rows if row['kind'] == 'native_analogue_evidence' and
                  ((row['facts'].get('participant_roles') or {}).get('review_required') or row['facts'].get('indirect_donor_roles') or row['facts'].get('unmatched_target_inputs') or row['facts'].get('contradicted_native_claims'))]
    contrasts = [row for row in rows if row['kind'] == 'native_analogue_evidence' and row not in challenges]
    held=[row for row in rows if row['kind']=='native_analogue_evidence' and
          (row['facts'].get('agent_priority_assessment') or {}).get('decision') in {'exclude','deprioritize'}]
    challenges=[row for row in challenges if row not in held]
    contrasts=[row for row in contrasts if row not in held]
    ranking=_read(run_file(Path(run_dir),'pass1_ranking.json'),[])
    ranks={row.get('accession'):row.get('rank',1000000) for row in ranking if isinstance(row,dict)} if isinstance(ranking,list) else {}
    contrasts.sort(key=lambda row:ranks.get(row['facts'].get('accession'),1000000))
    if held:
        offset=(len(list((base/'reviews').glob('*.json')))+state.get('grounding_deferrals',0))%len(held)
        held=held[offset:]+held[:offset]
    papers = [row for row in rows if row['kind']=='retrieved_literature_evidence']
    ordered = target_rows + models[:1]
    for i in range(max(len(challenges), len(contrasts),len(held))):
        for group in (challenges, contrasts,held):
            if i < len(group):
                ordered.append(group[i])
    # Reserve a source passage in the bounded packet, including before the first
    # evaluated candidate. The complete catalogue remains available on disk.
    if papers:
        ordered.insert(min(len(ordered),2),papers[0])
    rows = ordered + papers[1:] + models[1:]
    previews = []
    for row in rows:
        identifier = 'evidence:' + _hash(row)
        path = base / 'evidence' / (identifier.split(':')[1] + '.json')
        if not path.exists():
            atomic_write_json(path, row)
        if len(previews) < limit:
            previews.append({'id': identifier, **row})
    previous = None
    if state.get('last_decision'):
        folder = base / state['last_decision']
        plan = _read(folder / 'plan.json', {})
        outcome = _read(folder / 'outcome.json', {})
        actual = _read(folder/'result.json', {})
        payload = actual.get('data')
        examples = []
        assessment = None
        if isinstance(payload, list):
            hits = []
            for hit in payload:
                if isinstance(hit, dict) and 'tool_name' in hit and isinstance(hit.get('data'), list):
                    hits.extend({**record, 'source_tool':hit['tool_name']}
                                for record in hit['data'][:3] if isinstance(record, dict))
                elif (isinstance(hit, dict) and 'tool_name' in hit and isinstance(hit.get('data'), dict)
                      and hit['data'].get('accession')):
                    hits.append({**hit['data'], 'source_tool': hit['tool_name'], 'source_status': hit.get('status')})
                else:
                    hits.append(hit)
            for hit in hits[:24]:
                if isinstance(hit, dict):
                    examples.append({key:hit.get(key) for key in
                        ('accession','protein_name','name','gene_names','ec_numbers','organism','reviewed','is_swissprot',
                         'pmid','doi','title','year','url','source_tool','source_status','abstract',
                         'catalytic_activities','function_comments','native_substrates',
                         'critical_warnings') if hit.get(key) is not None})
        elif isinstance(payload, dict) and payload.get('accession'):
            examples.append({key: payload[key] for key in ('accession','protein_name','organism','reviewed','is_swissprot',
                'catalytic_activities','function_comments','native_substrates','critical_warnings') if key in payload})
        elif isinstance(payload, dict) and 'actual_sample' in payload:
            registry = load_candidate_registry(Path(run_dir))['candidates']
            records = []
            for item in payload.get('actual_sample', []):
                metadata = (registry.get(item.get('accession')) or {}).get('metadata') or {}
                data = {**metadata, **(metadata.get('evaluation_data') or {})}
                validation = data.get('claim_validation') or {}
                records.append({'accession':item.get('accession'), 'status':item.get('status'),
                                'protein_name':data.get('protein_name'),
                                'native_reactions':data.get('catalytic_activities') or [],
                                'contradicted_claims':validation.get('contradicted_claims') or [],
                                'missing_required_claims':validation.get('blocking_claims') or [],
                                'native_reaction_available':bool(data.get('catalytic_activities'))})
            assessment = {key:payload.get(key) for key in ('attempted','supported','evidence_pending','rejected','failed','sampled_family_groups')}
            assessment['records'] = records
            assessment['limit'] = 'Missing annotation is not evidence of inactivity; untested mechanisms remain unresolved.'
        previous = {'id': state['last_decision'], 'hypothesis': plan.get('hypothesis'),
                    'alternative': plan.get('alternative'),
                    'prediction': plan.get('prediction'), 'action': plan.get('tool_name'),
                    'contrasting_prediction': plan.get('contrasting_prediction'),
                    'revision_trigger': plan.get('revision_trigger'),
                    'action_kind': plan.get('action_kind', action_kind(plan.get('tool_name', ''))),
                    'result_examples': examples,
                    'assessment': assessment,
                    'primary_passage': ({'paper':payload.get('paper'),
                        'passage':str(payload.get('passage') or '')[:600],
                        'accessions':payload.get('mentioned_uniprot_accessions'),
                        'source_path':payload.get('source_path'),'text_sha256':payload.get('text_sha256')}
                        if isinstance(payload,dict) and payload.get('text_available') else None),
                    'outcome': {k: outcome.get(k) for k in ('status', 'summary', 'evidence_id')},
                    'comparison': 'Compare the observation with the prediction; tool success is not biological support.'}
    from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
    from enzyme_discovery_agent.models.ortholog import tentative_label_key
    registry = load_candidate_registry(Path(run_dir))['candidates']
    ready = fresh_evaluation_backlog(run_dir)
    work = {}
    for accession in ready:
        metadata = registry[accession].get('metadata') or {}
        view = {**(metadata.get('screening_annotation') or {}), **metadata}
        scope = tentative_label_key(view) or f'accession:{accession}'
        row = work.setdefault(scope, {'annotation_scope': 'annotation:'+scope,
            'protein_name': view.get('protein_name'), 'screened_records': 0, 'example_accessions': []})
        row['screened_records'] += 1
        if len(row['example_accessions']) < 2:
            row['example_accessions'].append(accession)
    new_leads = sorted(((acc, entry) for acc, entry in registry.items() if discovery_status(entry) == 'discovered'
                        or discovery_status(entry)=='evidence_pending' and (entry.get('metadata') or {}).get('is_literature_lead')
                        and (entry.get('metadata') or {}).get('identity_resolution_status')!='resolved'),
                       key=lambda item: item[1].get('first_seen', ''), reverse=True)
    import re
    lead_views=[]
    for acc,entry in new_leads:
        meta=entry.get('metadata') or {};name=meta.get('literature_name') or meta.get('protein_name') or ''
        matches=[{'paper_key':row['facts'].get('paper_key'),'title':row['facts'].get('title'),
                  'doi':row['facts'].get('doi'),'pmid':row['facts'].get('pmid'),
                  'evidence_id':'evidence:'+_hash(row),'source_path':row['source']['path']}
                 for row in rows if row['kind']=='retrieved_literature_evidence' and name
                 and re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',row['facts'].get('title',''),re.I)]
        lead_views.append({'accession':acc,'name':name,'source':meta.get('source_tool'),
            'identity_status':meta.get('identity_resolution_status'),'primary_title_matches':matches[:2],
            'first_seen':entry.get('first_seen',''),'organism':meta.get('organism'),
            'source_paper':((meta.get('literature_lead') or {}).get('paper') or {}).get('doi')
                           or ((meta.get('literature_lead') or {}).get('paper') or {}).get('pmid')
                           or (meta.get('literature_lead') or {}).get('text_path') or acc,
            'next_question':'Resolve the paper-scoped sequence identity before assessment; no activity conclusion follows from a name match.'})
    targeted=sorted([row for row in lead_views if row['primary_title_matches']],key=lambda row:row['first_seen'])
    def diversify(items,key):
        groups={}
        for item in items:groups.setdefault(key(item),[]).append(item)
        return [group[i] for i in range(max([len(v) for v in groups.values()] or [0]))
                for group in groups.values() if i<len(group)]
    targeted=diversify(targeted,lambda row:(row['name'].casefold(),str(row.get('organism') or '').casefold()))
    exploratory=diversify([row for row in lead_views if not row['primary_title_matches']],lambda row:row['source_paper'])
    frontier=[]
    for i in range(8):
        for bucket in (targeted,exploratory):
            if i<len(bucket) and len(frontier)<8:frontier.append(bucket[i])
    useful = []
    plans = [_read(path,{}) for path in base.glob('*/plan.json')]
    plans.sort(key=lambda plan:plan.get('created_at',''),reverse=True)
    for plan in plans:
        if len(useful) >= 2:
            break
        name = plan.get('tool_name','')
        if name not in {'search_pubmed','search_openalex','search_europepmc','search_semantic_scholar','extract_reaction_mechanism','read_primary_paper'}:
            continue
        result = _read(base/plan['id']/'result.json',{})
        data = result.get('data')
        if name == 'read_primary_paper' and isinstance(data,dict) and data.get('text_available'):
            outcome=_read(base/plan['id']/'outcome.json',{})
            useful.append({'decision_id':plan['id'],'evidence_id':outcome.get('evidence_id'),
                'tool':name,'paper':data.get('paper'),'passage':str(data.get('passage') or '')[:400],
                'mentioned_accessions':data.get('mentioned_uniprot_accessions'),
                'source_path':data.get('source_path'),'text_sha256':data.get('text_sha256')})
            continue
        if not isinstance(data,list):
            continue
        papers = [{key:hit.get(key) for key in ('pmid','doi','title','year','url') if hit.get(key)}
                  for hit in data[:3] if isinstance(hit,dict) and (hit.get('title') or hit.get('doi') or hit.get('pmid'))]
        if papers:
            outcome=_read(base/plan['id']/'outcome.json',{})
            useful.append({'decision_id':plan['id'],'evidence_id':outcome.get('evidence_id'),'tool':name,'papers':papers})
    return {'review_due': review_due(run_dir), 'assessed_records': assessed,
            'awaiting_interpretation':bool(state.get('awaiting_interpretation')),
            'review_limits':review_limits(run_dir),
            'last_concluded_review':_read(base/'reviews'/(state['last_review']+'.json'),{}) if state.get('last_review') else None,
            'last_outcome_interpretation':_read(base/'interpretations'/(state['last_interpretation']+'.json'),{}) if state.get('last_interpretation') else None,
            'observations': previews, 'previous_test': previous,
            'ready_work_examples': list(work.values())[:8], 'ready_records': len(ready),
            'recent_unassessed_leads':frontier,
            'frontier_scope_note':'Primary-title matching, name/taxon labels and source-paper groups diversify work allocation only; they do not establish identity, activity or mechanistic independence.',
            'recent_paper_findings':useful,
            'limits': 'Integrity checks do not prove scientific entailment or target activity. Alternative explanations remain hypotheses.'}


def proposal_repair_context(run_dir, proposal=None):
    """Return current citation and revision contracts after a rejected proposal."""
    packet=decision_context(run_dir,limit=4)
    previous=packet.get('previous_test') or {}
    stale=[]
    refs=(proposal or {}).get('evidence_refs') or []
    if isinstance(refs,str):
        try:refs=json.loads(refs)
        except ValueError:refs=[]
    for ref in refs if isinstance(refs,list) else []:
        if not isinstance(ref,str) or not ref.startswith('evidence:'):continue
        identifier=ref.removeprefix('evidence:')
        if len(identifier)!=64 or any(c not in '0123456789abcdef' for c in identifier):continue
        evidence=_read(_root(run_dir)/'evidence'/(identifier+'.json'),{})
        source=evidence.get('source') or {};accession=source.get('accession')
        if source.get('record_sha256') and accession:
            entry=load_candidate_registry(Path(run_dir),strict=True)['candidates'].get(accession,{})
            meta=entry.get('metadata') or {};current={**meta,**(meta.get('evaluation_data') or {})}
            if _hash(current)!=source['record_sha256']:
                stale.append({'previous_evidence_id':ref,'accession':accession,
                              'current_candidate_evidence':candidate_priority_repair_context(run_dir,accession)['candidate_evidence']})
    return {'revises':previous.get('id'),
            'awaiting_interpretation':packet.get('awaiting_interpretation',False),
            'previous_outcome_evidence_id':(previous.get('outcome') or {}).get('evidence_id'),
            'available_evidence_refs':[{'id':row['id'],'kind':row['kind'],
                'accession':(row.get('facts') or {}).get('accession')} for row in packet['observations']],
            'changed_candidate_records':stale,
            'instruction':'Use exact saved IDs. Interpret the previous outcome only when awaiting_interpretation is true. Otherwise take the next grounded action or conclude the review. These references do not establish a biological conclusion.'}


def candidate_priority_repair_context(run_dir, accession):
    """Return current candidate-linked sources without choosing a scientific priority."""
    decision_context(run_dir,limit=0)
    current=[{'id':'evidence:'+_hash(row),'kind':row['kind'],'source':row.get('source'),
              'facts':row.get('facts')} for row in observations(run_dir)[0]
             if (row.get('source') or {}).get('accession')==accession
             or (row.get('facts') or {}).get('accession')==accession]
    return {'candidate_evidence':current,
            'instruction':'These are current recorded candidate observations, not activity conclusions. Interpret them and cite the relevant record in a revised priority decision; an empty or deduplicated lookup is not the candidate record.'}


def interpret_outcome(run_dir, *, decision_id, evidence_refs, interpretation, belief_update):
    """Record an explicit interpretation without inventing another evidence action."""
    base=_root(run_dir)
    state=_read(base/'state.json',{})
    if not state.get('awaiting_interpretation') or decision_id!=state.get('last_decision'):
        raise ValueError('Interpret the current pending decision; inspect its exact ID and outcome first')
    if not isinstance(interpretation,str) or not 20<=len(interpretation.strip())<=1800:
        raise ValueError('State a concise interpretation of the actual observation and its limits')
    plan=_read(base/decision_id/'plan.json',{})
    outcome=_read(base/decision_id/'outcome.json',{})
    result=_read(base/decision_id/'result.json',None)
    identifier=outcome.get('evidence_id','').removeprefix('evidence:')
    observation=_read(base/'evidence'/(identifier+'.json'),{})
    if result is None or _hash(result)!=(observation.get('facts') or {}).get('result_sha256'):
        raise ValueError('The saved outcome changed; preserve the pending interpretation for review')
    proposal=validate_plan(run_dir,{**{key:plan.get(key) for key in
        ('hypothesis','alternative','prediction','contrasting_prediction','revision_trigger','scope')},
        'tool_name':'interpret_saved_outcome','arguments':{},'why_this_test':interpretation,
        'evidence_refs':evidence_refs,'revises':decision_id,'belief_update':belief_update},
        {'interpret_saved_outcome'})
    record={**proposal,'id':uuid.uuid4().hex,'created_at':datetime.now(timezone.utc).isoformat(),
            'interpretation':interpretation,'decision_id':decision_id,
            'scope_limit':'Interpretation of a saved observation; no new experiment or activity validation.'}
    path=base/'interpretations'/(record['id']+'.json')
    initial_source_interpreted = any(row['kind']=='retrieved_literature_evidence'
                                     for row in observations(run_dir)[0])
    with state_lock(base/'state.json'):
        current=_read(base/'state.json',{})
        if current.get('last_decision')!=decision_id or not current.get('awaiting_interpretation'):
            raise ValueError('Pending decision changed during interpretation')
        atomic_write_json(path,record)
        current.update(awaiting_interpretation=False,last_interpretation=record['id'])
        if initial_source_interpreted:
            current['initial_sources_reviewed']=True
        atomic_write_json(base/'state.json',current)
    from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
    capture_scientific_record(run_dir,'outcome_interpreted',path)
    return {'status':'success','summary':'Saved outcome interpreted; research can continue without a new evidence action.',
            'data':record}


def defer_ungrounded_review(run_dir, ready_records):
    """Suspend model deliberation while preserving independent evaluation work."""
    base=_root(run_dir)
    signal=_signal(observations(run_dir)[0])
    with state_lock(base/'state.json'):
        state=_read(base/'state.json',{})
        state.update(awaiting_interpretation=False,reviewed_signal=signal,
                     review_actions=0,review_inspections=0,initial_sources_reviewed=True,
                     grounding_deferrals=state.get('grounding_deferrals',0)+1)
        atomic_write_json(base/'state.json',state)
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(run_dir,'research_review_deferred',ready_records=ready_records,
        reason='Model proposals could not be grounded; preserve uncertainty and evaluate the ready queue.',
        scope='Controller work allocation only; scientific claims and family priorities are unchanged.')


BRIEF_FIELD_LIMIT = 1800
REVIEW_CONCLUSION_LIMIT = 8000


MAX_PLANNED_ACTIONS = 8


def planned_actions(plan):
    """Every action a registered plan executes, primary first.

    One hypothesis may fund several retrievals. A plan that judges four families
    relevant should be able to search all four, because a family judged relevant and
    never searched is an untested family, and the deliberation that selected it costs
    far more than the extra lookups do.
    """
    actions = [{'tool_name': plan.get('tool_name'), 'arguments': plan.get('arguments') or {}}]
    actions += [dict(entry) for entry in (plan.get('additional_actions') or [])]
    return actions


def _validate_additional_actions(run_dir, proposal, allowed_tools, primary_name):
    """Admit a fan-out only where concurrent execution is already sanctioned.

    Restricting the fan-out to registry ``parallel_safe`` actions keeps every
    consequential mutation -- priority changes, family focus, evaluation batches --
    on the original one-action-per-hypothesis contract, so a single prediction can
    still never record two separate verdicts.
    """
    extra = proposal.get('additional_actions')
    if extra in (None, ''):
        return {k: v for k, v in proposal.items() if k != 'additional_actions'}
    if not isinstance(extra, list):
        raise ValueError('additional_actions must be a list of {tool_name, arguments} objects')
    if len(extra) + 1 > MAX_PLANNED_ACTIONS:
        raise ValueError(f'A plan may execute at most {MAX_PLANNED_ACTIONS} actions')
    # The fan-out boundary is read from the tool registry, so validation must not
    # depend on which modules an importing caller happened to load first.
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.tool_registry import get_parallel_safe_tools
    register_builtin_tools()
    concurrent = get_parallel_safe_tools()
    if primary_name not in concurrent:
        raise ValueError(f'{primary_name} must be planned on its own; only independently repeatable '
                         'retrieval actions may share one hypothesis')
    prefix = 'mcp__discovery__'
    seen = {(primary_name, _hash(proposal.get('arguments') or {}))}
    cleaned = []
    for entry in extra:
        if not isinstance(entry, dict):
            raise ValueError('Each additional action must be a {tool_name, arguments} object')
        name = entry.get('tool_name')
        if isinstance(name, str) and name.startswith(prefix) and name[len(prefix):] in allowed_tools:
            name = name[len(prefix):]
        arguments = entry.get('arguments') or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError('Each additional action needs a tool_name and an arguments object')
        if name not in allowed_tools or name not in concurrent:
            raise ValueError(f'{name} is not an available independently repeatable retrieval action')
        if not review_action_allowed(run_dir, name):
            raise ValueError('This review has exhausted its work allowance; conclude_research_review '
                             'preserves uncertainty and resumes evaluation')
        key = (name, _hash(arguments))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({'tool_name': name, 'arguments': arguments})
    if not cleaned:
        return {k: v for k, v in proposal.items() if k != 'additional_actions'}
    return {**proposal, 'additional_actions': cleaned}


def validate_plan(run_dir, proposal, allowed_tools, *, field_limits=None, prospective=True):
    if not isinstance(proposal, dict):
        raise ValueError('A decision must be an object')
    from enzyme_discovery_agent.tool_registry import _coerce_arg_types
    proposal = _coerce_arg_types(proposal, {'arguments':{'type':'object'}, 'evidence_refs':{'type':'array'},
                                            'additional_actions':{'type':'array'}})
    if not prospective and proposal.get('tool_name')!='resume_routine_scheduling':
        raise ValueError('Only a review closure may omit a new prospective test record')
    scientific_fields=('hypothesis','alternative','prediction','contrasting_prediction','revision_trigger')
    required = scientific_fields+('why_this_test','tool_name') if prospective else ('why_this_test','tool_name')
    for key in required:
        if not isinstance(proposal.get(key), str) or not proposal[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
        maximum = (field_limits or {}).get(key, BRIEF_FIELD_LIMIT)
        if len(proposal[key]) > maximum:
            raise ValueError(f'{key} exceeds {maximum} characters (received {len(proposal[key])}); retain the full evidence in its source artifact and shorten this field')
    for key in scientific_fields if not prospective else ():
        if proposal.get(key) and (not isinstance(proposal[key],str) or len(proposal[key])>BRIEF_FIELD_LIMIT):
            raise ValueError(f'{key} must be a string of at most {BRIEF_FIELD_LIMIT} characters when supplied')
    if (proposal.get('hypothesis') and proposal.get('alternative') and
            proposal['hypothesis'].strip().casefold() == proposal['alternative'].strip().casefold()):
        raise ValueError('State a distinct competing explanation')
    if (proposal.get('prediction') and proposal.get('contrasting_prediction') and
            proposal['prediction'].strip().casefold() == proposal['contrasting_prediction'].strip().casefold()):
        raise ValueError('Competing explanations need distinguishable predicted observations')
    name = proposal['tool_name']
    prefix = 'mcp__discovery__'
    if name.startswith(prefix) and name[len(prefix):] in allowed_tools:
        proposal = {**proposal, 'tool_name': name[len(prefix):]}
        name = proposal['tool_name']
    if name not in allowed_tools or name in {'plan_research_step', 'conclude_research_review', 'interpret_research_outcome', 'get_notebook', 'update_notebook'}:
        raise ValueError('Choose an available substantive evidence or priority action')
    if name!='resume_routine_scheduling' and not review_action_allowed(run_dir,name):
        raise ValueError('This review has exhausted its work allowance; conclude_research_review preserves uncertainty and resumes evaluation')
    if not isinstance(proposal.get('arguments'), dict):
        raise ValueError('arguments must match the selected tool schema')
    proposal = _validate_additional_actions(run_dir, proposal, allowed_tools, name)
    refs = proposal.get('evidence_refs')
    if not isinstance(refs, list) or not refs or len(refs) > 8:
        raise ValueError('Cite one to eight recorded evidence identifiers')
    canonical = []
    for reference in refs:
        if isinstance(reference, str) and len(reference) == 32 and all(c in '0123456789abcdef' for c in reference):
            completed = _read(_root(run_dir)/reference/'outcome.json', {})
            if completed.get('evidence_id'):
                reference = completed['evidence_id']
        if reference not in canonical:
            canonical.append(reference)
    proposal = {**proposal, 'evidence_refs': canonical}
    refs = canonical
    cited, historical, unchanged_scopes = [], [], []
    for ref in refs:
        if not isinstance(ref, str) or not ref.startswith('evidence:'):
            raise ValueError('Use evidence identifiers from get_decision_context')
        digest = ref.split(':', 1)[1]
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid evidence identifier')
        evidence = _read(_root(run_dir) / 'evidence' / (digest + '.json'), None)
        if evidence is None or _hash(evidence) != digest:
            raise ValueError('Missing or changed evidence; retrieve the current decision context')
        source = evidence.get('source') or {}
        if source.get('sha256'):
            try:
                path=(Path(run_dir)/source['path']).resolve()
                path.relative_to(Path(run_dir).resolve())
                intact=path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
            except (KeyError,ValueError,OSError):
                intact=False
            if not intact:
                raise ValueError('The cited source artifact changed; retrieve current evidence before planning')
        if source.get('record_sha256') and source.get('accession'):
            entry = load_candidate_registry(Path(run_dir), strict=True)['candidates'].get(source['accession'])
            metadata = (entry or {}).get('metadata') or {}
            current = {**metadata, **(metadata.get('evaluation_data') or {})}
            if not entry or _hash(current) != source['record_sha256']:
                same_native_facts=(entry and evidence.get('kind')=='native_analogue_evidence'
                    and _scientific_native_facts(evidence['facts'])==
                        _scientific_native_facts(_native_facts(source['accession'],current,metadata)))
                if same_native_facts:
                    unchanged_scopes.append({'evidence_id':ref,'accession':source['accession'],
                        'scope':'Cited native-function facts unchanged; work priority and unrelated metadata are not empirical observations.',
                        'current_record_sha256':_hash(current)})
                else:
                    historical.append((ref, source.get('accession'), _hash(current) if entry else None))
        cited.append(evidence)
    if review_due(run_dir) and all(row['kind'] == 'user_target' for row in cited):
        raise ValueError('Address the new observations, not only the original target')
    previous = proposal.get('revises') or ''
    state = _read(_root(run_dir) / 'state.json', {})
    if state.get('awaiting_interpretation') and previous != state.get('last_decision'):
        latest = state.get('last_decision')
        outcome = _read(_root(run_dir)/latest/'outcome.json', {}) if latest else {}
        if not previous and outcome.get('evidence_id') in refs:
            previous = latest
            proposal = {**proposal, 'revises':latest, 'revision_link_inferred_from_citation':True}
        else:
            raise ValueError('Interpret the previous recorded test outcome before proposing the next step')
    if previous:
        if not isinstance(previous, str) or len(previous) != 32 or any(c not in '0123456789abcdef' for c in previous):
            raise ValueError('Invalid previous decision identifier')
        outcome = _read(_root(run_dir) / previous / 'outcome.json', {})
        if not outcome or outcome.get('evidence_id') not in refs:
            raise ValueError('A revision must cite the recorded outcome of its previous test')
    if name == 'set_family_focus' and state.get('last_decision'):
        prior = _read(_root(run_dir) / state['last_decision'] / 'plan.json', {})
        if prior.get('tool_name') == 'set_family_focus':
            fresh_functional_evidence = any(reference not in prior.get('evidence_refs', []) and
                row['kind'] in {'native_analogue_evidence', 'modeling_observation', 'experimental_observation'}
                for reference, row in zip(refs, cited))
            if not fresh_functional_evidence:
                proposal = {**proposal, 'update_basis': 'reassessment_of_existing_evidence'}
    if historical:
        prior = _read(_root(run_dir) / previous / 'plan.json', {}) if previous else {}
        for ref, accession, current_hash in historical:
            has_current_version = any((row.get('source') or {}).get('accession') == accession
                and (row.get('source') or {}).get('record_sha256') == current_hash for row in cited)
            if ref not in prior.get('evidence_refs', []) or not has_current_version:
                raise ValueError(f'Changed candidate evidence for {accession}: cite its current record from repair_context.changed_candidate_records. To compare historically, revise the earlier decision and cite both versions.')
    update = proposal.get('belief_update', 'retain')
    if update not in {'retain', 'revise', 'suspend'}:
        raise ValueError('belief_update must be retain, revise or suspend')
    scope = proposal.get('scope')
    if scope not in {'native_function', 'target_compatibility', 'model_applicability', 'retrieval', 'search_strategy'}:
        raise ValueError('State which scientific or retrieval hypothesis is being tested')
    def functional_content(row):
        if row['kind'] in {'native_analogue_evidence', 'experimental_observation'}:
            return True
        if not row['facts'].get('nonempty_result'):
            return False
        source = row.get('source') or {}
        source_decision = Path(source.get('path', '')).parent.name
        recorded = _read(_root(run_dir)/source_decision/'plan.json', {}) if len(source_decision) == 32 else {}
        return bool(recorded) and action_kind(recorded.get('tool_name', '')) != 'priority_update'
    if (scope in {'native_function', 'target_compatibility'} and update == 'revise'
            and not any(functional_content(row) for row in cited)):
        raise ValueError('A retrieval outcome alone cannot revise a biological claim; obtain functional evidence or suspend judgment')
    if unchanged_scopes:proposal={**proposal,'unchanged_cited_fact_scopes':unchanged_scopes}
    return proposal


def conclude_review(run_dir, *, conclusion, evidence_refs, hypothesis='', alternative='',
                    prediction='', contrasting_prediction='', revision_trigger='', scope='auto',
                    belief_update='suspend', revises=''):
    """Close deliberation, preserving uncertainty and reopening on changed facts.

    A review closure authorizes routine scheduling, not a biological verdict or
    discovery completion. The same source-validation contract as a test applies.
    """
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise ValueError('conclusion must be a nonempty string')
    if len(conclusion) > REVIEW_CONCLUSION_LIMIT:
        raise ValueError(f'conclusion exceeds {REVIEW_CONCLUSION_LIMIT} characters (received {len(conclusion)})')
    if scope=='auto':
        state=_read(_root(run_dir)/'state.json',{})
        prior=_read(_root(run_dir)/state.get('last_decision','')/'plan.json',{})
        scope=prior.get('scope') or 'search_strategy'
    proposal = validate_plan(run_dir, {
        'hypothesis':hypothesis, 'alternative':alternative, 'why_this_test':conclusion,
        'prediction':prediction, 'contrasting_prediction':contrasting_prediction,
        'revision_trigger':revision_trigger, 'scope':scope, 'belief_update':belief_update,
        'evidence_refs':evidence_refs, 'revises':revises,
        'tool_name':'resume_routine_scheduling', 'arguments':{},
    }, {'resume_routine_scheduling'}, field_limits={'why_this_test': REVIEW_CONCLUSION_LIMIT},prospective=False)
    base = _root(run_dir)
    identifier = uuid.uuid4().hex
    record = {**proposal, 'id':identifier, 'created_at':datetime.now(timezone.utc).isoformat(),
              'conclusion':conclusion, 'scientific_entailment':'not automatically verified',
              'effect':'Resume pending evaluation under existing family priorities; target activity remains unproven.',
              'review_only':True,'new_prospective_test_registered':False}
    signal = _signal(observations(run_dir)[0])
    atomic_write_json(base/'reviews'/(identifier+'.json'),record)
    from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
    capture_scientific_record(run_dir,'review_concluded',base/'reviews'/(identifier+'.json'))
    with state_lock(base/'state.json'):
        state = _read(base/'state.json',{})
        state.update(awaiting_interpretation=False, reviewed_signal=signal, last_review=identifier,
                     review_actions=0,review_inspections=0)
        if any(row['kind']=='retrieved_literature_evidence' for row in observations(run_dir)[0]):
            state['initial_sources_reviewed']=True
        atomic_write_json(base/'state.json',state)
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(run_dir,'research_review_concluded',review_id=identifier,
                 revises=proposal.get('revises'),belief_update=belief_update,conclusion=conclusion)
    return {'status':'success','summary':'Review concluded; routine evaluation can resume. Changed evidence reopens review.',
            'data':record}


def start_decision(run_dir, proposal, allowed_tools, *, model_provenance=None):
    requested_name = proposal.get('tool_name')
    requested_refs = proposal.get('evidence_refs')
    proposal = validate_plan(run_dir, proposal, allowed_tools)
    identifier = uuid.uuid4().hex
    record = {**proposal, 'id': identifier, 'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'prediction_registered_before_action', 'scientific_entailment': 'not automatically verified',
              'input_signal': _signal(observations(run_dir)[0])}
    record['action_kind'] = action_kind(proposal['tool_name'])
    record['requested_tool_name'] = requested_name
    record['requested_evidence_refs'] = requested_refs
    record['model_provenance'] = model_provenance or {'status':'not_captured_by_caller'}
    registry = load_candidate_registry(Path(run_dir), strict=True)['candidates']
    historical = []
    for reference in proposal['evidence_refs']:
        evidence = _read(_root(run_dir) / 'evidence' / (reference.split(':')[1] + '.json'), {})
        source = evidence.get('source') or {}
        if source.get('record_sha256') and source.get('accession'):
            metadata = (registry.get(source['accession']) or {}).get('metadata') or {}
            if _hash({**metadata, **(metadata.get('evaluation_data') or {})}) != source['record_sha256']:
                historical.append(reference)
    record['historical_evidence_refs'] = historical
    atomic_write_json(_root(run_dir) / identifier / 'plan.json', record)
    from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
    capture_scientific_record(run_dir,'prediction_registered',_root(run_dir)/identifier/'plan.json')
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(run_dir, 'research_prediction_registered', decision_id=identifier,
                 plan_sha256=_hash(record), action=proposal['tool_name'])
    return identifier


def finish_decision(run_dir, identifier, result):
    base = _root(run_dir)
    record = {'kind': 'tool_outcome', 'family': None,
              'facts': {'status': result.get('status'), 'summary': result.get('summary'),
                        'nonempty_result': bool(result.get('data')) and result.get('status') in {'success', 'ok', 'partial'},
                        'biological_support': 'not implied by tool completion', 'result_sha256': _hash(result)},
              'source': {'path': str((base / identifier / 'result.json').relative_to(Path(run_dir)))}}
    evidence_id = 'evidence:' + _hash(record)
    atomic_write_json(base / identifier / 'result.json', result)
    atomic_write_json(base / 'evidence' / (evidence_id.split(':')[1] + '.json'), record)
    atomic_write_json(base / identifier / 'outcome.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(), 'status': result.get('status'),
        'summary': str(result.get('summary') or '')[:500], 'evidence_id': evidence_id})
    from enzyme_discovery_agent.models.reasoning_trace import capture_scientific_record
    capture_scientific_record(run_dir,'observation_recorded',base/identifier/'outcome.json')
    plan = _read(base / identifier / 'plan.json', {})
    with state_lock(base / 'state.json'):
        state=_read(base/'state.json',{})
        state.update(last_decision=identifier,reviewed_signal=plan.get('input_signal'),awaiting_interpretation=True,
                     review_actions=state.get('review_actions',0)+1)
        atomic_write_json(base / 'state.json',state)
    if plan.get('tool_name') in {'get_decision_context','get_exploration_state','get_candidate_evidence','get_research_memory'}:
        record_inspection(run_dir)
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(run_dir, 'research_decision', decision_id=identifier,
                 hypothesis=plan.get('hypothesis'), prediction=plan.get('prediction'),
                 action=plan.get('tool_name'), revises=plan.get('revises'),
                 outcome_status=result.get('status'), evidence_id=evidence_id,
                 reasoning=plan.get('why_this_test'))
    return evidence_id


def decision_summary(run_dir, limit=8):
    """Read-only human-facing view; counts describe records, not scientific merit."""
    base = _root(run_dir)
    plans = [_read(path, {}) for path in base.glob('*/plan.json')]
    plans.sort(key=lambda row: row.get('created_at', ''), reverse=True)
    by_id = {row.get('id'): row for row in plans}
    rows, executed, evidence_actions, changed, noninformative = [], 0, 0, 0, 0
    for plan in plans:
        outcome = _read(base/plan['id']/'outcome.json', {})
        executed += bool(outcome)
        result=_read(base/plan['id']/'result.json',{})
        has_observation=bool(outcome) and result.get('status') in {'success','ok','partial','empty'} and not result.get('deduplicated')
        if result.get('data',{}) and isinstance(result.get('data'),dict) and result['data'].get('new_evidence') is False:
            has_observation=False
        if plan.get('tool_name')=='assess_next_batch' and not result.get('count'):
            has_observation=False
        evidence_actions += has_observation and action_kind(plan.get('tool_name', '')) != 'priority_update'
        noninformative += bool(outcome) and not has_observation
        previous = by_id.get(plan.get('revises')) or {}
        def action(record):
            arguments = {key: value for key, value in (record.get('arguments') or {}).items()
                         if key not in {'reasons', 'reason', 'rationale', 'notes'}}
            return record.get('tool_name'), arguments
        action_changed = bool(previous and action(plan) != action(previous))
        changed += action_changed
        if len(rows) >= limit:
            continue
        row = {key: plan.get(key) for key in ('id', 'created_at', 'scope', 'tool_name',
              'arguments', 'revises', 'belief_update', 'historical_evidence_refs', 'evidence_refs')}
        row.update({key: str(plan.get(key) or '')[:900] for key in (
            'hypothesis', 'alternative', 'prediction', 'contrasting_prediction', 'why_this_test', 'revision_trigger')})
        row.update(outcome=outcome, action_changed=action_changed)
        row['action_kind'] = plan.get('action_kind', action_kind(plan.get('tool_name', '')))
        rows.append(row)
    reviews = [_read(path,{}) for path in (base/'reviews').glob('*.json')]
    reviews.sort(key=lambda item:item.get('created_at',''),reverse=True)
    return {'registered_predictions': len(plans), 'completed_actions': executed,
            'concluded_reviews':len(reviews), 'latest_concluded_review':reviews[0] if reviews else None,
            'evidence_actions': evidence_actions,
            'noninformative_actions':noninformative,
            'decisions_with_changed_actions': changed, 'records': rows,
            'scope': 'Source integrity and execution are checked. Scientific entailment and discovery benefit require separate validation.'}
