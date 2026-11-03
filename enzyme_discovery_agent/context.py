"""LLM context compaction: strip bulky fields from tool results."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from collections import Counter

from enzyme_discovery_agent import config
from enzyme_discovery_agent.run_layout import run_file

ARTIFACT_CATEGORIES = {
    "search_pubmed": "literature",
    "search_openalex": "literature",
    "search_europepmc": "literature",
    "search_semantic_scholar": "literature",
    "search_crossref": "literature",
    "deep_literature_search": "literature",
    "follow_paper_citations": "literature",
    "fetch_paper_metadata": "literature",
    "search_uniprot": "database",
    "fetch_uniprot_sequence": "database",
    "search_kegg_enzyme": "database",
    "search_interpro": "database",
    "search_modelseed_reactions": "database",
    "get_modelseed_compound": "database",
    "get_protein_domains": "database",
    "evaluate_candidate": "candidates",
    "screen_candidates": "candidates",
    # Evidence extractors: persist the full mechanism / measurement facts (with their
    # evidence, confidence, papers_tried, and provenance) as auditable per-call artifacts,
    # not only inside the conversation transcript.
    "extract_reaction_mechanism": "evidence",
    "extract_experimental_measurements": "evidence",
}

STRIP_KEYS = frozenset({"sequence", "sequence_preview", "openalex_id", "member_databases"})

TRUNCATE_KEYS = {
    "abstract": config.ABSTRACT_TRUNCATE_CHARS,
    "description": config.DESCRIPTION_TRUNCATE_CHARS,
}

CAP_LIST_KEYS = {
    "genes": 3, "papers": config.PAPER_LIST_CAP,
    "results": config.RESULTS_LIST_CAP, "entries": config.RESULTS_LIST_CAP,
    "hits": config.RESULTS_LIST_CAP, "reasoning": 3, "function_comments": 2,
    "catalytic_activities": 3, "authors": 3,
}


def compact_for_llm(obj):
    """Recursively strip bulky fields from a tool result for LLM context."""
    if isinstance(obj, dict):
        if (isinstance(obj.get('accession'),str) and isinstance(obj.get('catalytic_activities'),list)
                and ('native_substrates' in obj or 'substrate_similarity' in obj)):
            from enzyme_discovery_agent.output.report_generator import _candidate_evidence_payload
            return {key:compact_for_llm(value) for key,value in _candidate_evidence_payload(obj).items()}
        out = {}
        for k, v in obj.items():
            if k in STRIP_KEYS:
                continue
            if k in TRUNCATE_KEYS and isinstance(v, str) and len(v) > TRUNCATE_KEYS[k]:
                out[k] = v[:TRUNCATE_KEYS[k]] + "..."
            elif k in CAP_LIST_KEYS and isinstance(v, list) and len(v) > CAP_LIST_KEYS[k]:
                cap = CAP_LIST_KEYS[k]
                out[k] = [compact_for_llm(i) for i in v[:cap]]
                out[k].append(f"... +{len(v) - cap} more")
            else:
                out[k] = compact_for_llm(v)
        return out
    if isinstance(obj, list):
        return [compact_for_llm(i) for i in obj]
    return obj


def compact_tool_definitions(tools):
    """Preserve function schemas while removing repeated prose from arguments."""
    def clean(value, top=False, critical=False):
        if isinstance(value, dict):
            return {k: (str(v)[:(200 if top and critical else 100 if top else 120)] if k == 'description' else clean(v,critical=critical))
                    for k, v in value.items() if k != "description" or top or critical}
        if isinstance(value, list):
            return [clean(v,critical=critical) for v in value]
        return value
    important = {'plan_research_step','conclude_research_review','interpret_research_outcome','get_decision_context','assess_next_batch','get_exploration_state','read_primary_paper','record_reaction_roles','record_reaction_step','get_reaction_step_context'}
    return [{**tool, "function": clean(tool["function"], top=True, critical=tool['function']['name'] in important)} for tool in tools]


def strategic_tool_definitions(tools, *, include_inspections=False):
    """One strategic interface, retaining contracts for every permitted action."""
    small = compact_tool_definitions(tools)
    planner = next((copy.deepcopy(tool) for tool in small if tool['function']['name']=='plan_research_step'), None)
    if planner is None:
        return small
    controls={'plan_research_step','conclude_research_review','interpret_research_outcome','get_notebook','update_notebook'}
    actions = [tool['function'] for tool in small if tool['function']['name'] not in controls]
    properties = planner['function']['parameters']['properties']
    properties['tool_name']['enum'] = [action['name'] for action in actions]
    # Every action's full schema is already carried once in this same toolset, as its
    # own definition. Repeating all of them here as an anyOf branch list doubled the
    # tool block: measured on a production run it was 6.4k tokens of every 32k prompt,
    # resent each turn, for schemas the model can already read a few hundred tokens
    # away. The planned arguments are still validated against the selected tool's
    # schema before anything executes, so the contract is unchanged -- only the
    # duplicate copy of it is gone.
    properties['arguments'] = {'type':'object',
        'description':'Arguments for the tool named in tool_name, matching that '
                      "tool's own schema as listed in this toolset."}
    conclusion = [tool for tool in small if tool['function']['name'] in {'conclude_research_review','interpret_research_outcome'}]
    if include_inspections:
        from enzyme_discovery_agent.models.research_decisions import INSPECTION_TOOLS
        conclusion += [tool for tool in small if tool['function']['name'] in INSPECTION_TOOLS]
    return [planner]+conclusion


def stable_tool_definitions(all_tools, *, include_inspections=True):
    """The same tool schemas every turn, so the provider's cache prefix survives.

    Tool definitions are the first thing in a cached prefix, so varying them by turn
    re-bills the system prompt, the research state and the whole history behind them as
    fresh cache writes. Measured on a production run, that made 60% of the prompt cache
    creation rather than reads, at roughly ten times the necessary cost.

    What the model is *permitted* to do each turn is unchanged: the per-turn allowlist is
    still computed and enforced before execution, and stated in the turn's own message.
    This only stops the menu from being rewritten underneath the cache.
    """
    planner = strategic_tool_definitions(all_tools, include_inspections=include_inspections)
    merged = {tool['function']['name']: tool for tool in compact_tool_definitions(all_tools)}
    # The planner carries the richer nested action contracts; prefer it on collision.
    merged.update({tool['function']['name']: tool for tool in planner})
    return [merged[name] for name in sorted(merged)]


def _read_state(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def build_research_state(run_dir: Path) -> dict:
    """Project durable evidence into a bounded decision state, without new I/O services."""
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, discovery_status
    from enzyme_discovery_agent.models.expansion import select_expansion_seeds, seed_evidence
    from enzyme_discovery_agent.output.report_generator import _accommodation_score
    registry = load_candidate_registry(run_dir, strict=True).get("candidates", {})
    target = _read_state(run_file(run_dir, 'target_context.json'), {})
    chemistry = target.get("target_reasoning") or _read_state(run_file(run_dir, 'target_reasoning.json'), {})
    candidates = []
    for acc in select_expansion_seeds(registry, 12, run_dir):
        data = seed_evidence(acc, registry[acc])
        candidates.append({
            "accession": acc, "name": str(data.get("protein_name") or "")[:120],
            "genes": (data.get("gene_names") or [])[:3], "family": data.get("interpro_family_id"),
            "organism": str(data.get("organism") or "")[:120],
            "native_reactions": [str(x)[:220] for x in (data.get("catalytic_activities") or [])[:2]],
            "native_target_resemblance": _accommodation_score(data),
            "reviewed": data.get("is_swissprot"),
            "mechanism_evidence_status": (data.get("mechanism_evidence") or {}).get("status"),
            "preferred_host_expression": ((data.get("expression_data") or {}).get("preferred_host_assessment") or {}).get("level", "unknown"),
        })
    pending = []
    for acc, entry in sorted(registry.items(), key=lambda item: not bool((item[1].get("metadata") or {}).get("is_literature_lead"))):
        if discovery_status(entry) == "evidence_pending":
            data = entry.get("metadata") or {}
            pending.append({"accession": acc, "name": str(data.get("protein_name") or "")[:120],
                            "organism": str(data.get("organism") or "")[:120],
                            "identity_status": data.get("identity_resolution_status"),
                            "source_paper": (data.get("literature_lead") or {}).get("paper"),
                            "missing_claims": (data.get("claim_validation") or {}).get("blocking_claims")})
        if len(pending) >= 12:
            break
    families = _read_state(run_file(run_dir, 'mechanism_families.json'), [])
    landscape = _read_state(run_file(run_dir, 'family_landscape.json'), [])
    from enzyme_discovery_agent.models.family_learning import refresh_family_learning
    learning = refresh_family_learning(run_dir, registry)
    priorities = [learning["families"][fid] for fid in learning["priority_order"]]
    searches = _read_state(run_dir / "agent_state" / "expansion_searches.json", {})
    notes = _read_state(run_file(run_dir, 'notebook.json'), {})
    from enzyme_discovery_agent.models.literature_strategy import coverage_summary
    literature_coverage = coverage_summary(run_dir, _read_state(run_dir / "agent_state" / "discovery_recovery.json", {}))
    from enzyme_discovery_agent.models.expansion import expansion_assessments
    expansion = expansion_assessments(run_dir, registry, learning, "foldseek")
    expansion_brief = [{"accession": acc, **{k: decision.get(k) for k in (
        "decision", "reason", "family", "native_support")},
        "pending_independent_groups": len(decision.get("previous_method_outcome", {}).get("pending", [])),
        "supported_independent_groups": len(decision.get("previous_method_outcome", {}).get("supported", []))}
        for acc, decision in expansion.items()]
    from enzyme_discovery_agent.models.research_memory import recall_memory
    memory = {topic: recall_memory(run_dir, topic, limit=6 if topic!='reasoning' else 2)
              for topic in ("searches", "leads", "reasoning")}
    from enzyme_discovery_agent.models.exploration_state import exploration_state
    exploration = exploration_state(run_dir, registry=registry, learning=learning)
    modeling = sorted(exploration["modeling"], key=lambda row: (
        str(registry[row["accession"]].get("last_updated") or ""), row["accession"]), reverse=True)
    def diagnostic_preview(value):
        if isinstance(value, str):
            return value[:300] + (" [preview; inspect saved record]" if len(value) > 300 else "")
        if isinstance(value, dict):
            return {k: diagnostic_preview(v) for k, v in list(value.items())[:24]}
        if isinstance(value, list):
            return [diagnostic_preview(v) for v in value[:8]]
        return value
    from enzyme_discovery_agent.models.research_decisions import decision_context
    return {
        "authority": "Saved retrieved evidence; native-function support is not proof of target-substrate activity. Name-based group counts are tentative, not verified independence.",
        "adaptive_reasoning": (decision_context(run_dir)
                               if config.ADAPTIVE_REASONING else {'enabled': False}),
        "target": {"query": target.get("query"), "substrate_smiles": target.get("substrate_smiles"),
                   **{k: chemistry.get(k) for k in ("reaction_direction", "bond_types", "substrate_topology",
                                                   "reactive_groups", "activation_categories")}},
        "candidate_status_counts": dict(Counter(discovery_status(e) for e in registry.values())),
        "literature_coverage": literature_coverage,
        "research_memory": memory,
        "exploration_diagnostics": diagnostic_preview({"overview": exploration["overview"],
            "family_examples": exploration["families"][:4],
            "method_examples": exploration["methods"][:4], "questions": exploration["questions"][:6],
            "limits": exploration["limitations"],
            "inspection": "Use get_exploration_state with section, offset, limit and optional query to inspect the complete saved frontier; examples are not the entire opportunity set."}),
        "modeling_feedback": diagnostic_preview(modeling[:6]), "omitted_modeling_records": max(0, len(modeling)-6),
        "promising_evaluated_candidates": candidates, "evidence_pending_examples": pending,
        "mechanism_hypotheses": [{"name": f.get("name"), "terms": (f.get("literature_terms") or [])[:3]}
                                  for f in families],
        "surveyed_folds": [f.get("fold_id") for f in landscape][:24],
        "learned_family_priorities": [{k: row.get(k) for k in (
            "family", "name", "state", "label_groups_assessed", "grouping_basis", "supported", "unresolved",
            "deprioritized", "best_supported_fit", "median_supported_fit", "survey_verdict", "direction_counts",
            "survey_fit", "agent_decision", "expansion_attempts", "new_supported_groups",
        )} for row in priorities[:12]],
        "families_to_set_aside": [{"family": row["family"], "name": row["name"], "state": row["state"]}
                                 for row in priorities if row["state"] == "deprioritized"][:12],
        "family_decisions": {fid: {"decision": value.get("decision"),
                                   "reason": str(value.get("reason") or "")[:160]}
                             for fid, value in sorted(_read_state(run_file(run_dir, 'family_decisions.json'), {}).get("decisions", {}).items(),
                                                      key=lambda item: item[1].get("ts", ""), reverse=True)[:12]},
        "expansion_attempt_counts": {method: len(rows) for method, rows in searches.items()},
        "expansion_feedback": expansion_brief[:12],
        "families_to_reconsider": [{"family": row["family"], "name": row["name"],
            "reason": "explicit stop coexists with supported native-chemistry records; reassess the saved evidence"}
            for row in priorities if row.get("agent_decision") == "stop" and row.get("supported", 0)][:8],
        "agent_notes_not_primary_evidence": {k: str(v)[-800:] for k, v in (notes.get("sections") or {}).items() if v},
        "next_action": "Evaluate new literature leads and returned expansion candidates before expanding their family again. "
                       "Follow learned family priorities. Deepen supported native chemistry and reserve breadth for "
                       "unresolved families; avoid repeating unsupported substrate requirements. These priorities "
                       "are revisable and do not prove target-substrate activity. Read focused saved records "
                       "with get_candidate_evidence. Retrieve older searches, unresolved leads, decisions and notebook passages "
                       "with get_research_memory, using topic, query and offset. Do not repeat completed work or "
                       "emit full reports during discovery. Review newly imported modeling observations with "
                       "their setup and distance limitations; they do not establish activity or revise the fixed metric. "
                       "Exploration diagnostics describe unresolved questions, not global exhaustion or compute authorization.",
    }


def _adaptive_context(prefix, messages, state, run_dir):
    """Keep decision evidence and inspection memory ahead of optional history."""
    from enzyme_discovery_agent.steering import context_message
    steering = context_message(run_dir)
    required = [steering] if steering else []
    packet = state['adaptive_reasoning']
    names = {}
    for message in messages:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            names[call.get('id')] = (function.get('name'), function.get('arguments'))
    inspections = []
    for message in reversed(messages):
        if message.get('role') != 'tool':
            continue
        name, arguments = names.get(message.get('tool_call_id'), (None, None))
        if name not in {'get_decision_context', 'get_exploration_state', 'get_candidate_evidence', 'get_research_memory'}:
            continue
        if name == 'get_decision_context' and not (isinstance(arguments, str) and 'decision_id' in arguments):
            continue  # The fresh canonical packet below replaces this duplicate.
        raw = str(message.get('content') or '')
        data=None
        inspection_payload={'tool':name,'request':arguments,'tool_call_id':message.get('tool_call_id'),
                            'result_text':raw,'capture_mode':'projection_of_saved_tool_result'}
        import hashlib
        from enzyme_discovery_agent.state_store import atomic_write_json
        digest=hashlib.sha256(json.dumps(inspection_payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        inspection_path=Path(run_dir)/'artifacts/evidence/inspection_memory'/(digest+'.json')
        if not inspection_path.exists():
            atomic_write_json(inspection_path,inspection_payload)
        try:
            value = json.loads(raw)
            data = value.get('data') if isinstance(value, dict) else None
            if name == 'get_decision_context' and isinstance(data, dict) and 'result' in data:
                # Historical inspection must show the observation, rather than
                # spending its entire preview on the earlier long hypothesis.
                value = {'result':data['result'], 'plan_id':(data.get('plan') or {}).get('id')}
            raw = json.dumps(compact_for_llm(value),ensure_ascii=False,default=str)
        except (ValueError, TypeError):
            pass
        inspections.append({'tool':name, 'request':arguments, 'result':raw,
                            'source':str(inspection_path.relative_to(run_dir)),
                            'candidate_evidence':compact_for_llm(data) if name=='get_candidate_evidence' and isinstance(data,dict) else None})
        break  # Preserve one informative recent inspection, not several stubs.
    directive = {'role':'user', 'content':
        'ADAPTIVE RESEARCH DECISION: continue this existing campaign; completed inspections below must not restart unchanged. Register the next test or priority action with plan_research_step. '
        'State a hypothesis, competing explanation, contrasting predictions and revision condition. Cite the evidence IDs below. '
        'Interpret the last actual outcome; set revises to its decision ID and cite its outcome evidence. '
        'Declare explicit assessment accessions/scopes or deliberate scope_mode=ready_queue. Unsampled mechanisms remain unresolved. '
        'Priority changes are not new functional evidence; explain reinterpretations and preserve uncertainty about target activity.'}
    directive['content'] += (' Conclude with conclude_research_review when this review has reached a defensible provisional conclusion, '
        'so routine pending evaluation can resume. Keep scientific uncertainty explicit; new observations reopen the question.')
    lean_directive={'role':'user','content':
        'ADAPTIVE RESEARCH DECISION: continue this saved campaign. Use plan_research_step with source IDs, '
        'a competing explanation, contrasting predictions and a revision condition. If previous_action has an id, '
        'set revises to it and cite the saved outcome evidence ID. Interpret that observation before the next action. '
        'Completed inspections remain complete. Conclude with conclude_research_review when the review allowance is reached; '
        'retain scientific uncertainty and resume routine evaluation.'}
    for detail in (240, 100):
        def evidence_preview(value, key=''):
            if isinstance(value,dict):
                return {k:evidence_preview(v,k) for k,v in value.items()}
            if isinstance(value,list):
                return [evidence_preview(v,key) for v in value[:3]]
            if isinstance(value,str) and len(value)>detail and key not in {
                    'artifacts','path','locator','native_smiles','target_smiles','occurrence_id',
                    'id','evidence_id','decision_id'}:
                return value[:detail]+' [preview; inspect saved artifact]'
            return value
        observations = []
        for row in packet.get('observations', []):
            if row['kind'] == 'user_target':
                continue
            facts = row.get('facts') or {}
            observations.append({'id':row['id'], 'kind':row['kind'], 'family':row.get('family'),
                'facts': {key:evidence_preview(value,key)
                          for key,value in facts.items() if key in {
                    'accession','protein_name','native_reaction','annotation_scope','unmatched_target_inputs',
                    'all_target_inputs_accommodated','contradicted_native_claims','missing_required_claims','modeling_feedback',
                    'paper_key','title','doi','pmid','abstract_excerpt','scope',
                    'role_transfer_status','indirect_donor_roles','participant_roles','agent_priority_assessment'}}})
            if len(observations) == 2:
                break
        previous = packet.get('previous_test') or {}
        outcome = previous.get('outcome') or {}
        brief = {'candidate_status_counts':state.get('candidate_status_counts'),
            'review_limits':packet.get('review_limits'),
            'target_evidence':next((row['id'] for row in packet.get('observations', []) if row['kind']=='user_target'), None),
            'observations':observations,
            'previous_action': {key:previous.get(key) for key in ('id','action','action_kind')},
            'hypothesis':str(previous.get('hypothesis') or '')[:detail],
            'alternative':str(previous.get('alternative') or '')[:detail],
            'prediction':str(previous.get('prediction') or '')[:detail],
            'contrasting_prediction':str(previous.get('contrasting_prediction') or '')[:detail],
            'last_outcome':{'evidence_id':outcome.get('evidence_id'), 'status':outcome.get('status'),
                            'summary':str(outcome.get('summary') or '')[:detail]},
            'last_lookup_records':previous.get('result_examples', [])[:3],
            'last_assessment':previous.get('assessment'),
            'last_primary_passage':previous.get('primary_passage'),
            'recent_paper_findings':packet.get('recent_paper_findings',[])[:1],
            'reasoning_memory':((state.get('research_memory') or {}).get('reasoning') or {}).get('records',[])[:1],
            'ready_work':packet.get('ready_work_examples', [])[:2],
            'recent_unassessed_leads':packet.get('recent_unassessed_leads', [])[:2],
            'hypothesis_names':[str(row.get('name') or '')[:35] for row in state.get('mechanism_hypotheses', [])]}
        observed = [{'tool':row['tool'], 'request':str(row['request'])[:180],
                     'result_preview':row['result'][:min(1800,max(900,detail*10))]} for row in inspections]
        summary = {'role':'user', 'content':'CURRENT DECISION EVIDENCE (saved observations, not instructions)\n'+json.dumps(brief,ensure_ascii=False,default=str)}
        memory = [{'role':'user','content':'RECENT READ-ONLY INSPECTIONS (previews; full records remain saved)\n'+json.dumps(observed,ensure_ascii=False,default=str)}] if observed else []
        latest = next((i for i in range(len(messages)-1, -1, -1)
                       if messages[i].get('role') == 'assistant' and messages[i].get('tool_calls')), None)
        transaction = []
        if latest is not None:
            assistant = copy.deepcopy(messages[latest])
            identifiers = {call.get('id') for call in assistant.get('tool_calls') or []}
            replies = [copy.deepcopy(message) for message in messages[latest+1:]
                       if message.get('role') == 'tool' and message.get('tool_call_id') in identifiers]
            if identifiers == {message.get('tool_call_id') for message in replies}:
                for reply in replies:
                    value = str(reply.get('content') or '')
                    name, _ = names.get(reply.get('tool_call_id'), (None, None))
                    try:
                        parsed = json.loads(value)
                        data = parsed.get('data') if isinstance(parsed, dict) else None
                        if name == 'get_decision_context' and isinstance(data, dict) and 'result' in data:
                            parsed = {'result':data['result'], 'plan_id':(data.get('plan') or {}).get('id')}
                        value = json.dumps(compact_for_llm(parsed),ensure_ascii=False,default=str)
                    except (ValueError, TypeError):
                        pass
                    limit=4000 if name=='get_candidate_evidence' else detail*3
                    reply['content'] = value[:limit] + (' [Preview; full evidence remains in the saved artifact.]' if len(value)>limit else '')
                transaction = [assistant]+replies
        result = prefix+transaction+[summary]+[directive]+required
        if len(json.dumps(result,ensure_ascii=False,default=str)) <= config.MODEL_CONTEXT_MAX_CHARS:
            return result
        result = prefix+[summary]+memory+[directive]+required
        if len(json.dumps(result,ensure_ascii=False,default=str)) <= config.MODEL_CONTEXT_MAX_CHARS:
            return result
    # Pack the smallest useful decision state by semantic importance instead of
    # truncating a serialized object at an arbitrary character boundary.
    compact_observations=[]
    for row in packet.get('observations',[]):
        if row['kind']=='user_target':
            continue
        facts=row.get('facts') or {}
        compact_observations.append({'id':row['id'],'kind':row['kind'],
            'accession':facts.get('accession'),'title':str(facts.get('title') or '')[:80],
            'protein_name':str(facts.get('protein_name') or '')[:70],
            'role_transfer_status':facts.get('role_transfer_status'),
            'missing_claims':(facts.get('missing_required_claims') or [])[:3],
            'contradictions':(facts.get('contradicted_native_claims') or [])[:3],
            'unmatched_input_count':len(facts.get('unmatched_target_inputs') or [])})
    packed={'target_evidence':next((row['id'] for row in packet.get('observations',[]) if row['kind']=='user_target'),None),
        'review_limits':packet.get('review_limits'),'observations':compact_observations[:1],
        'additional_observation_refs':[{'id':row['id'],'kind':row['kind'],'accession':row.get('accession')}
                                        for row in compact_observations[1:]],
        'previous_action':{key:previous.get(key) for key in ('id','action','action_kind')},
        'hypothesis':str(previous.get('hypothesis') or '')[:80],
        'alternative':str(previous.get('alternative') or '')[:80],
        'prediction':str(previous.get('prediction') or '')[:80],
        'contrasting_prediction':str(previous.get('contrasting_prediction') or '')[:80],
        'outcome':{'evidence_id':outcome.get('evidence_id'),'status':outcome.get('status'),
                   'summary':str(outcome.get('summary') or '')[:120]},
        'inspection_receipts':[{'tool':row['tool'],'full_record':row['source'],
                                'preview':row['result'][:80]} for row in inspections],
        'scope':'Compact previews. Target activity remains unestablished. Full source facts, inspection results and reasoning remain saved; retrieve them before making a consequential inference.'}
    summary={'role':'user','content':'CURRENT DECISION EVIDENCE\n'+json.dumps(packed,ensure_ascii=False,default=str)}
    result=prefix+[summary,lean_directive]+required
    if len(json.dumps(result,ensure_ascii=False,default=str))<=config.MODEL_CONTEXT_MAX_CHARS:
        evidence=next((row.get('candidate_evidence') for row in inspections if row.get('candidate_evidence')),None)
        if evidence:
            packed['recent_candidate_evidence']={key:evidence[key] for key in (
                'accession','protein_name','native_claim_validation','reaction','function_comments',
                'reactant_occurrences','substrate_accommodation') if key in evidence}
            summary['content']='CURRENT DECISION EVIDENCE\n'+json.dumps(packed,ensure_ascii=False,default=str)
            if len(json.dumps(result,ensure_ascii=False,default=str))>config.MODEL_CONTEXT_MAX_CHARS:
                del packed['recent_candidate_evidence']
                summary['content']='CURRENT DECISION EVIDENCE\n'+json.dumps(packed,ensure_ascii=False,default=str)
        for key in ('last_assessment','last_lookup_records','last_primary_passage','ready_work','recent_unassessed_leads'):
            if not brief.get(key):
                continue
            packed[key]=brief[key]
            summary['content']='CURRENT DECISION EVIDENCE\n'+json.dumps(packed,ensure_ascii=False,default=str)
            if len(json.dumps(result,ensure_ascii=False,default=str))>config.MODEL_CONTEXT_MAX_CHARS:
                del packed[key]
                summary['content']='CURRENT DECISION EVIDENCE\n'+json.dumps(packed,ensure_ascii=False,default=str)
        return result
    raise ValueError('Mandatory instructions and adaptive evidence cannot fit the configured context allowance; evidence was not silently discarded')


def build_model_context(messages: list[dict], run_dir: Path) -> list[dict]:
    """Send current evidence and recent complete transactions; keep the full audit history on disk."""
    prefix = []
    for message in messages:
        if message.get("role") == "system" or (message.get("role") == "user" and not any(
                m.get("role") == "user" for m in prefix)):
            prefix.append(copy.deepcopy(message))
        if any(m.get("role") == "user" for m in prefix):
            break
    state = build_research_state(run_dir)
    if (state.get('adaptive_reasoning') or {}).get('review_due'):
        return _adaptive_context(prefix, messages, state, run_dir)
    def preview(value, cap):
        if isinstance(value, dict):
            return {k: preview(v, cap) for k, v in value.items()}
        if isinstance(value, list):
            return [preview(v, cap) for v in value[:cap]]
        return value[:400] if isinstance(value, str) and len(value) > 400 else value
    for cap in (6, 4, 2, 1):
        if len(json.dumps(state, default=str)) <= 8000:
            break
        state = preview(state, cap)
    summary = {"role": "user", "content": "CURRENT RESEARCH STATE\n" + json.dumps(
        state, ensure_ascii=False, default=str)}
    from enzyme_discovery_agent.steering import context_message
    steering = context_message(run_dir)
    steering_context = [steering] if steering else []
    positions = [i for i, m in enumerate(messages) if i >= len(prefix) and m.get("role") == "assistant"]
    cut = positions[-config.MODEL_CONTEXT_RECENT_ASSISTANTS] if len(positions) >= config.MODEL_CONTEXT_RECENT_ASSISTANTS else len(prefix)
    tail = copy.deepcopy(messages[cut:])
    for message in tail:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if message.get("role") == "assistant" and not message.get("tool_calls") and len(content) > 1200:
            message["content"] = "[Earlier narrative omitted; use the saved evidence above, not unverified draft claims.]"
        elif message.get("role") in {"user", "tool"}:
            limit = config.MODEL_CONTEXT_TOOL_RESULT_CHARS if message.get("role") == "tool" else 1200
            if len(content) > limit:
                message["content"] = content[:limit] + "\n[Preview only; full evidence remains in the run artifacts.]"
    def size(items):
        return len(json.dumps(items, ensure_ascii=False, default=str))
    while size(prefix + tail + [summary] + steering_context) > config.MODEL_CONTEXT_MAX_CHARS:
        next_group = next((i for i, m in enumerate(tail) if i > 0 and m.get("role") == "assistant"), None)
        if next_group is None:
            # Drop an oversized transaction as a whole, preserving tool pairing
            # and opaque provider signatures in any transactions retained.
            tail = [tail[-1]] if tail and tail[-1].get("role") == "user" else []
            break
        tail = tail[next_group:]
    if size(prefix + tail + [summary] + steering_context) > config.MODEL_CONTEXT_MAX_CHARS:
        # Keep the complete projection retrievable while selecting a bounded
        # working set by meaning. Recursively truncating large memory records
        # can leave an escaped JSON prefix instead of the facts needed next.
        import hashlib
        from enzyme_discovery_agent.state_store import atomic_write_json
        full=json.dumps(state,sort_keys=True,ensure_ascii=False,default=str)
        snapshot=Path(run_dir)/'artifacts/evidence/context_memory'/(hashlib.sha256(full.encode()).hexdigest()+'.json')
        if not snapshot.exists():
            atomic_write_json(snapshot,{'scope':'Saved derived campaign state; inspect underlying sources before a functional claim.',
                                        'state':state})
        target=state.get('target') or {}
        memory=state.get('research_memory') or {}
        reasoning=[]
        for row in (memory.get('reasoning') or {}).get('records',[])[:2]:
            item={key:row.get(key) for key in ('kind','id','accession','source','event_sequence','current_interpretation') if key in row}
            for key in ('hypothesis','alternative','prediction','revision_trigger'):
                if row.get(key):item[key]=str(row[key])[:100]
            roles=row.get('participant_roles') or {}
            step=roles.get('reaction_step') or row.get('interpretation') or {}
            if isinstance(step,dict):
                item['reaction_step']={key:(str(step[key])[:120] if isinstance(step[key],str) else step[key])
                    for key in ('local_bond_changes_match','activation_assessment','continuation_assessment') if key in step}
            reasoning.append(item)
        state={'authority':state.get('authority'),'candidate_status_counts':state.get('candidate_status_counts'),
            'target_contract':{key:target.get(key) for key in ('reaction_direction','bond_types','substrate_topology')},
            'ready_records':(state.get('adaptive_reasoning') or {}).get('ready_records'),
            'candidate_examples':[{key:row.get(key) for key in ('accession','name','family')}
                                  for row in state.get('promising_evaluated_candidates',[])[:2]],
            'pending_examples':[{key:row.get(key) for key in ('accession','identity_status','missing_claims')}
                                for row in state.get('evidence_pending_examples',[])[:2]],
            'mechanism_hypotheses':[str(row.get('name') or '')[:70] for row in state.get('mechanism_hypotheses',[])[:6]],
            'reasoning_memory':reasoning,'full_state_artifact':str(snapshot.relative_to(run_dir)),
            'recall':'Full campaign facts remain in the saved state artifact and source records. Use get_research_memory, get_candidate_evidence or inspect_discovery_source; a compact preview is not evidence of inactivity.'}
        summary["content"] = "CURRENT RESEARCH STATE\n" + json.dumps(state, ensure_ascii=False, default=str)
        if size(prefix + tail + [summary] + steering_context) > config.MODEL_CONTEXT_MAX_CHARS:
            tail=[]
    if size(prefix + tail + [summary] + steering_context) > config.MODEL_CONTEXT_MAX_CHARS:
        raise ValueError("Mandatory instructions exceed the configured model context allowance")
    return prefix + tail + [summary] + steering_context
