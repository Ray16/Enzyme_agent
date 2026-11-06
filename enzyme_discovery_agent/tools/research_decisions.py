"""Controller-mediated research decisions and their campaign-local evidence."""
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.models.research_decisions import BRIEF_FIELD_LIMIT, REVIEW_CONCLUSION_LIMIT


_BRIEF = {'type':'string','minLength':1,'maxLength':BRIEF_FIELD_LIMIT}
_OPTIONAL_BRIEF = {'type':'string','maxLength':BRIEF_FIELD_LIMIT,'default':''}


@tool(description='Interpret the current pending test outcome without executing another test. Cite its exact outcome evidence and distinguish observations from uncertainty.',
      category='reasoning',parallel_safe=False,injected_params=['run_dir'],params={
          'run_dir':{'type':'string'},'decision_id':{'type':'string'},
          'evidence_refs':{'type':'array','items':{'type':'string'}},
          'interpretation':{**_BRIEF,'minLength':20},
          'belief_update':{'type':'string','enum':['retain','revise','suspend']}})
def interpret_research_outcome(run_dir, **kwargs):
    from enzyme_discovery_agent.models.research_decisions import interpret_outcome,proposal_repair_context
    try:
        return interpret_outcome(run_dir,**kwargs)
    except (TypeError,ValueError) as exc:
        return {'status':'error','summary':str(exc),'repair_context':proposal_repair_context(run_dir,proposal=kwargs)}


@tool(description='Record a holistic, revisable candidate priority or remove it from the suggested shortlist while retaining its evidence history.',
      category='reasoning',parallel_safe=False,injected_params=['run_dir','research_decision_id'],params={
          'run_dir':{'type':'string'},'research_decision_id':{'type':'string'},
          'accession':{'type':'string'},'decision':{'type':'string','enum':['prioritize','investigate','deprioritize','exclude']},
          'reason':{'type':'string'},'strengths':{'type':'array','items':{'type':'string'}},
          'weaknesses':{'type':'array','items':{'type':'object','required':['finding','basis','severity'],
              'properties':{'finding':{'type':'string'},'basis':{'type':'string','enum':['observed_mismatch','missing_evidence']},
                            'severity':{'type':'string','enum':['minor','major']}}}},
          'unknowns':{'type':'array','items':{'type':'string'}},'revisit_when':{'type':'string'}})
def set_candidate_focus(run_dir, accession, decision, reason, strengths, weaknesses, unknowns, revisit_when,
                        research_decision_id=''):
    import hashlib,json,re,uuid
    from pathlib import Path
    from datetime import datetime,timezone
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry,update_candidate_registry
    from enzyme_discovery_agent.state_store import atomic_write_json
    from enzyme_discovery_agent.models.event_log import append_event
    root=Path(run_dir)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',accession) or accession in {'.','..'}:
        return {'status':'error','summary':'Invalid candidate identifier'}
    if not re.fullmatch(r'[0-9a-f]{32}',research_decision_id):
        return {'status':'error','summary':'Candidate priority changes require plan_research_step with candidate evidence references.'}
    plan_path=root/'agent_state/research_decisions'/research_decision_id/'plan.json'
    plan=json.loads(plan_path.read_text())
    if plan.get('tool_name')!='set_candidate_focus' or (plan.get('arguments') or {}).get('accession')!=accession:
        return {'status':'error','summary':'The recorded prediction must address this candidate priority action.'}
    if accession not in load_candidate_registry(root,strict=True)['candidates']:
        return {'status':'error','summary':'Candidate is not registered in this campaign.'}
    if decision not in {'prioritize','investigate','deprioritize','exclude'} or not reason.strip() or not revisit_when.strip():
        return {'status':'error','summary':'State a valid decision, scientific rationale and reopening condition.'}
    if decision in {'deprioritize','exclude'} and not any(w.get('basis')=='observed_mismatch' for w in weaknesses):
        return {'status':'error','summary':'Missing evidence alone cannot justify deprioritization or exclusion; preserve it as an investigation question.'}
    if decision=='prioritize' and not strengths:
        return {'status':'error','summary':'A priority hypothesis requires recorded strengths as well as uncertainty.'}
    associated=False
    for reference in plan.get('evidence_refs') or []:
        evidence=json.loads((root/'agent_state/research_decisions/evidence'/(reference.removeprefix('evidence:')+'.json')).read_text())
        source=evidence.get('source') or {}
        associated|=source.get('accession')==accession or (evidence.get('facts') or {}).get('accession')==accession
        if evidence.get('kind')=='tool_outcome' and source.get('path'):
            result=json.loads((root/source['path']).read_text()).get('data')
            values=result if isinstance(result,list) else [result]
            associated|=any(isinstance(value,dict) and value.get('accession')==accession for value in values)
    if not associated:
        from enzyme_discovery_agent.models.research_decisions import candidate_priority_repair_context
        return {'status':'error','summary':'Cite the current candidate record or a recorded result containing its accession.',
                'repair_context':candidate_priority_repair_context(root,accession)}
    record={'accession':accession,'decision':decision,'reason':reason,'strengths':strengths,
            'weaknesses':weaknesses,'unknowns':unknowns,'revisit_when':revisit_when,
            'decision_id':research_decision_id,'evidence_refs':plan['evidence_refs'],
            'recorded_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Revisable target-specific work priority; not a verdict on native enzyme activity.'}
    path=root/'artifacts/shortlist_decisions'/accession/(uuid.uuid4().hex+'.json')
    atomic_write_json(path,record)
    update_candidate_registry(root,accession,None,{'priority_assessment':record,'priority_artifact':str(path.relative_to(root))})
    append_event(root,'candidate_priority_recorded',accession=accession,decision=decision,artifact=str(path.relative_to(root)))
    from enzyme_discovery_agent.output.csv_generator import refresh_ranked_candidates
    refresh_ranked_candidates(root)
    return {'status':'success','summary':'Candidate priority updated; registry history and native evidence are retained.',
            'data':record,'artifact':{'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}


@tool(description='Read the actual result of an action by decision_id; omit ID for current evidence and candidate IDs.',
      category='reasoning', injected_params=['run_dir'], params={'run_dir': {'type': 'string'},
      'decision_id': {'type': 'string', 'default': '', 'description':'Recorded action ID. Supplying it returns the full saved action result, including retrieved accessions.'}})
def get_decision_context(run_dir, decision_id=''):
    from enzyme_discovery_agent.models.research_decisions import decision_context
    if decision_id:
        from pathlib import Path
        import json
        from enzyme_discovery_agent.context import compact_for_llm
        if len(decision_id) != 32 or any(c not in '0123456789abcdef' for c in decision_id):
            return {'status': 'error', 'summary': 'Invalid decision identifier.'}
        folder = Path(run_dir)/'agent_state/research_decisions'/decision_id
        if not (folder/'plan.json').exists():
            return {'status': 'error', 'summary': 'Decision is not recorded in this campaign.'}
        data = {'plan': json.loads((folder/'plan.json').read_text()),
                'result': compact_for_llm(json.loads((folder/'result.json').read_text())) if (folder/'result.json').exists() else None}
        if len(json.dumps(data, default=str)) > 10000:
            data['result'] = {'preview': json.dumps(data['result'], default=str)[:7000],
                              'complete_result': str((folder/'result.json').relative_to(Path(run_dir)))}
        return {'status': 'success', 'summary': 'Saved prediction and actual tool result; interpretations remain revisable.', 'data': data}
    return {'status': 'success', 'summary': 'Recorded observations; scientific inferences remain revisable.',
            'data': decision_context(run_dir)}


@tool(description='Register a hypothesis, competing explanation and prediction; execute the grounded test. '
                  'One hypothesis may run several independent retrievals together via additional_actions: '
                  'when the reasoning identifies more than one candidate scope, retrieve all of them in this '
                  'step rather than spending another deliberation turn per scope.',
      category='reasoning', parallel_safe=False, params={
          'hypothesis': dict(_BRIEF), 'alternative': dict(_BRIEF),
          'scope': {'type': 'string', 'enum': ['native_function', 'target_compatibility', 'model_applicability', 'retrieval', 'search_strategy']},
          'prediction': dict(_BRIEF), 'contrasting_prediction': dict(_BRIEF),
          'why_this_test': dict(_BRIEF), 'revision_trigger': dict(_BRIEF),
          'evidence_refs': {'type': 'array', 'minItems':1,'maxItems':8,'items': {'type': 'string'}, 'description':'Full evidence IDs from the packet, or completed decision IDs resolving to their recorded outcomes.'},
          'tool_name': {'type': 'string', 'description':'Canonical action name from the permitted contracts.'},
          'arguments': {'type': 'object', 'description':'Arguments matching that action. A focused information request can use a retrieval hypothesis.'},
          'additional_actions': {'type':'array','maxItems':7,'default':[],
              'description':'Further independent retrieval actions to execute in this same step, each '
                            '{tool_name, arguments}. Use whenever the hypothesis covers several scopes so each '
                            'one is actually searched. Actions that change a priority or a stored assessment '
                            'cannot be combined and must be planned alone.',
              'items': {'type':'object','required':['tool_name','arguments'],
                        'properties':{'tool_name':{'type':'string'},'arguments':{'type':'object'}}}},
          'revises': {'type': 'string', 'default': ''},
          'belief_update': {'type': 'string', 'enum': ['retain', 'revise', 'suspend'], 'default': 'retain'},
      })
def plan_research_step(**kwargs):
    # Only the controller may validate, register and execute the enclosed action;
    # calling this registry function directly cannot bypass its tool permissions.
    return {'status': 'error', 'summary': 'Research decisions require controller-mediated execution.'}


@tool(description='Conclude the current evidence review from actual observations and citations; no new prospective test record is required.',
      category='reasoning', parallel_safe=False, injected_params=['run_dir'], params={
          'run_dir':{'type':'string'}, 'hypothesis':dict(_OPTIONAL_BRIEF), 'alternative':dict(_OPTIONAL_BRIEF),
          'conclusion':{'type':'string','minLength':1,'maxLength':REVIEW_CONCLUSION_LIMIT,'description':'What the observation supports and leaves unresolved. This does not mark discovery complete.'},
          'evidence_refs':{'type':'array','minItems':1,'maxItems':8,'items':{'type':'string'}},
          'prediction':dict(_OPTIONAL_BRIEF), 'contrasting_prediction':dict(_OPTIONAL_BRIEF),
          'revision_trigger':dict(_OPTIONAL_BRIEF),
          'scope':{'type':'string','enum':['auto','native_function','target_compatibility','model_applicability','retrieval','search_strategy'],'default':'auto'},
          'belief_update':{'type':'string','enum':['retain','revise','suspend'],'default':'suspend'},
          'revises':{'type':'string','default':''}})
def conclude_research_review(run_dir, **kwargs):
    from enzyme_discovery_agent.models.research_decisions import conclude_review
    try:
        return conclude_review(run_dir,**kwargs)
    except (TypeError,ValueError) as exc:
        from enzyme_discovery_agent.models.research_decisions import proposal_repair_context
        return {'status':'error','summary':str(exc),'repair_context':proposal_repair_context(run_dir,proposal=kwargs)}


@tool(description='Assess explicit accessions/scopes; choose ready_queue explicitly for a broad batch, not latest hits.',
      category='reasoning', parallel_safe=False, injected_params=['run_dir', 'llm_client'],
      params={'run_dir': {'type': 'string'}, 'llm_client': {'type': 'object'},
              'limit': {'type': 'integer', 'default': 20},
              'accessions': {'type': 'array', 'items': {'type': 'string'}, 'default': [], 'description':'Exact requested accessions. New discovered records are screened before evaluation.'},
              'family_scopes': {'type': 'array', 'items': {'type': 'string'}, 'default': [], 'description':'Reported InterPro or annotation scheduling scopes to sample.'},
              'scope_mode': {'type': 'string', 'enum': ['explicit', 'ready_queue'], 'default': 'explicit', 'description':'ready_queue explicitly samples the existing broad queue; it does not target latest hits.'}})
def assess_next_batch(run_dir, llm_client=None, limit=20, accessions=None, family_scopes=None, scope_mode='explicit'):
    from pathlib import Path
    from enzyme_discovery_agent.gates.ec_triage import _auto_evaluate_all, auto_screen_candidates
    from enzyme_discovery_agent.models.rolling import fresh_evaluation_backlog
    from enzyme_discovery_agent import config
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= config.EVALUATIONS_PER_STEP:
        return {'status': 'error', 'summary': 'limit must fit the configured native-evaluation batch.'}
    root = Path(run_dir)
    if not accessions and not family_scopes and scope_mode != 'ready_queue':
        return {'status':'error', 'summary':'Specify accessions or reported family_scopes, or scope_mode=ready_queue for a broad sample. A limit alone does not target newly retrieved hits.'}
    if not accessions and not family_scopes and not fresh_evaluation_backlog(root):
        auto_screen_candidates(root)
    stats = {}
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, family_label, discovery_status
    from enzyme_discovery_agent.models.ortholog import tentative_label_key
    before = load_candidate_registry(root)['candidates']
    selected = set(accessions) if accessions else None
    if family_scopes:
        scopes = set(family_scopes)
        matching = set()
        for accession, entry in before.items():
            metadata = entry.get('metadata') or {}
            view = {**(metadata.get('screening_annotation') or {}), **metadata}
            group = tentative_label_key(view)
            keys = {metadata.get('interpro_family_id'), metadata.get('screening_family_id'), family_label(metadata, accession)}
            if group:
                keys.add('annotation:'+group)
            if keys & scopes:
                matching.add(accession)
        selected = matching if selected is None else selected & matching
    screening = []
    if selected is not None:
        from enzyme_discovery_agent.tools.reasoning import screen_candidates
        new = [accession for accession in sorted(selected) if discovery_status(before.get(accession, {})) in {None,'discovered'}][:limit]
        for offset in range(0, len(new), config.SCREENING_BATCH_SIZE):
            screened = screen_candidates(root, new[offset:offset+config.SCREENING_BATCH_SIZE])
            screening.append(screened)
    _auto_evaluate_all(root, llm_client=llm_client, native_first=True, limit=limit, stats=stats,
                       fresh_only=True, candidate_accessions=selected)
    from collections import Counter
    registry = load_candidate_registry(root)['candidates']
    assessed = []
    from enzyme_discovery_agent.chemistry_logic import derive_activation_mechanism
    for accession in stats.get('accessions_attempted', []):
        entry = registry.get(accession) or {}
        metadata = entry.get('metadata') or {}
        data = metadata.get('evaluation_data') or metadata
        reactions = data.get('catalytic_activities') or []
        assessed.append({'accession': accession, 'family': family_label(metadata, accession),
                         'protein_name': metadata.get('protein_name'), 'status': discovery_status(entry),
                         'native_reaction': str(reactions[0])[:400] if reactions else None,
                         'equation_derived_activation': derive_activation_mechanism(reactions)})
    stats['actual_sample'] = assessed
    stats['sampled_family_groups'] = dict(Counter(row['family'] for row in assessed))
    stats['sampling_method'] = 'Requested recorded scopes' if selected is not None else 'Balanced ready queue; no mechanism-specific subset was requested.'
    stats['requested_family_scopes'] = family_scopes or []
    stats['screening_outcomes'] = screening
    stats['scope_limit'] = 'This batch cannot refute families or mechanisms it did not sample.'
    if selected:
        existing = []
        attempted_accessions = set(stats.get('accessions_attempted', []))
        for accession in sorted(selected - attempted_accessions):
            entry = registry.get(accession)
            if not entry:
                continue
            metadata = entry.get('metadata') or {}
            data = {**(metadata.get('evaluation_data') or {}), **metadata}
            existing.append({'accession':accession,'status':discovery_status(entry),
                             'protein_name':data.get('protein_name'),
                             'native_reactions':data.get('catalytic_activities') or [],
                             'work_status':'not_evaluated_in_this_batch',
                             'missing_claims':(data.get('claim_validation') or {}).get('blocking_claims') or []})
        if existing:
            stats['existing_assessments'] = existing
        if not stats.get('attempted') and existing and not screening:
            stats['new_evidence'] = False
            return {'status':'skipped','summary':
                'No requested candidate was evaluated in this call. Saved states distinguish completed work, '
                'unresolved evidence and candidates not yet evaluated. Inspect those states before choosing the next action. '
                'This call supplies no new evaluation evidence.',
                'count':0,'data':stats}
    return {'status': 'success' if stats.get('attempted') else 'partial',
            'summary': ('Batch assessed; native annotation does not demonstrate target activity.' if stats.get('attempted') else
                        'No eligible records sampled; the requested hypothesis remains untested.'),
            'count': stats.get('attempted', 0), 'data': stats}
