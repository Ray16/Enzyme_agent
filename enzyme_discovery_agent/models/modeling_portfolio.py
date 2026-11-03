"""Select source-cited hypotheses for structural review as discovery progresses."""
import json
from pathlib import Path

from .candidate_registry import discovery_status, load_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json


def select_modeling_portfolio(run_dir, *, max_candidates, protected_unannotated_slots=0):
    """Schedule review independently of native annotation or activity-rank admission."""
    if (type(max_candidates) is not int or max_candidates < 1
            or type(protected_unannotated_slots) is not int
            or not 0 <= protected_unannotated_slots <= max_candidates):
        raise ValueError('Require a positive capacity and protected slots within that capacity')
    root = Path(run_dir).resolve()
    quality_policy=root/'agent_state/discovery_quality_policy.json'
    if quality_policy.exists() and json.loads(quality_policy.read_text()).get('primary_only'):
        from enzyme_discovery_agent.tools.discovery_proposals import write_primary_portfolio
        summary=write_primary_portfolio(root); registry=load_candidate_registry(root,strict=True)['candidates']
        grouped={}
        for row in summary['candidates']:
            grouped.setdefault(row['family'],[]).append(row)
        ordered=[group[i] for i in range(max((len(g) for g in grouped.values()),default=0))
                 for group in grouped.values() if i<len(group)]
        rows=[]
        for row in ordered[:max_candidates]:
            meta=registry[row['accession']].get('metadata') or {}
            proposal=json.loads((root/row['proposal_source']).read_text())
            data={**(meta.get('evaluation_data') or {}),**meta,'discovery_proposal':proposal}
            data['priority_assessment']={'decision':'investigate','reason':proposal['hypothesis'],
                'strengths':[proposal['support_basis']],'unknowns':proposal['unknowns'],
                'source':'agent_recorded_discovery_proposal','evidence_refs':[e['path'] for e in proposal['evidence']]}
            rows.append({'accession':row['accession'],'data':data,'priority':'investigate',
                'family':row['family'],'proposal_source':row['proposal_source'],
                'native_substrates_resolved':bool(data.get('native_substrates') or data.get('catalytic_activities')),
                'modeling_setup_status':'requires_step_partner_cofactor_and_assembly_validation'})
        atomic_write_json(root/'modeling_portfolio.json',{'selection':rows,
            'policy':'Primary source-linked catalytic proposals selected by the discovery agent. Other exploration hypotheses remain separate.'})
        return rows
    from enzyme_discovery_agent import config
    from enzyme_discovery_agent.run_layout import run_file
    if (not config.AUTO_TARGET_REVIEW and
            (run_file(root,'family_landscape.json').exists() or
             (root/'agent_state/discovery_nominations.json').exists())):
        from enzyme_discovery_agent.output.discovery_portfolio import write_discovery_portfolio
        proposals=write_discovery_portfolio(root,limit=max(config.DISCOVERY_PORTFOLIO_SIZE,max_candidates))
        rows=[]
        for row in proposals['candidates']:
            source=json.loads((root/row['proposal_source']).read_text())
            data=source['source_record']
            rows.append({'accession':row['accession'],'data':data,
                         'native_substrates_resolved':bool(data.get('native_substrates') or data.get('catalytic_activities')),
                         'priority':'explore','family':row['mechanism_family'] or row['family_id'],
                         'proposal_source':row['proposal_source'],
                         'modeling_setup_status':row['modeling_setup_status']})
        available=len(rows)
        unannotated=[row for row in rows if not row['native_substrates_resolved']]
        protected=unannotated[:protected_unannotated_slots]
        protected_ids={row['accession'] for row in protected}
        rows=(protected+[row for row in rows if row['accession'] not in protected_ids])[:max_candidates]
        atomic_write_json(root/'modeling_portfolio.json',{
            'policy':'Source-backed discovery proposals. No target-activity review is required for exploration. '
                'Validate identity, reacting roles, molecular setup and reuse before compute.',
            'selection':[{key:row[key] for key in ('accession','native_substrates_resolved','priority','family','proposal_source','modeling_setup_status')} for row in rows],
            'discovery_candidates':available,'unannotated_available':len(unannotated),
            'skipped_artifacts':[]})
        return rows
    registry = load_candidate_registry(root, strict=True)['candidates']
    rows = []
    skipped_artifacts = []
    for accession, entry in registry.items():
        meta = entry.get('metadata') or {}
        data = {**(meta.get('evaluation_data') or {}), **meta}
        priority = data.get('priority_assessment') or {}
        from .target_review import eligible
        if not eligible(root,accession,meta,purpose='modeling'):continue
        if (discovery_status(entry) == 'rejected' or data.get('is_literature_lead')
                or priority.get('decision') not in {'prioritize', 'investigate'}):
            continue
        if not all(priority.get(key) for key in ('reason', 'strengths', 'evidence_refs', 'revisit_when')):
            continue
        artifact = meta.get('priority_artifact')
        if not artifact:
            continue
        path = (root / artifact).resolve()
        try:
            current = path.is_relative_to(root) and path.is_file() and json.loads(path.read_text()) == priority
        except (OSError, ValueError):
            current = False
        if not current:
            skipped_artifacts.append({'accession': accession,
                                      'reason': 'Priority source unavailable, changed, malformed or outside this run; retain for later review.'})
            continue
        known = bool(data.get('native_substrates') or data.get('catalytic_activities') or
                     (data.get('literature_evidence') or {}).get('native_substrates'))
        rows.append({'accession': accession, 'data': data, 'native_substrates_resolved': known,
                     'priority': priority['decision'],
                     'family': data.get('interpro_family_id') or data.get('screening_family_id') or accession})
    rows.sort(key=lambda row: (row['priority'] != 'prioritize', row['accession']))

    def diverse(items):
        grouped = {}
        for row in items:
            key = json.dumps(row['family'], sort_keys=True)
            grouped.setdefault(key, []).append(row)
        return [group[i] for i in range(max([len(v) for v in grouped.values()] or [0]))
                for group in grouped.values() if i < len(group)]

    unannotated = diverse([row for row in rows if not row['native_substrates_resolved']])
    chosen = unannotated[:protected_unannotated_slots]
    used = {row['accession'] for row in chosen}
    chosen += diverse([row for row in rows if row['accession'] not in used])[:max_candidates - len(chosen)]
    atomic_write_json(root / 'modeling_portfolio.json', {
        'policy': 'Source-cited hypothesis investigation; not native-function admission or an activity ranking. Family labels diversify scheduling only. Identity, molecular setup and reuse still require validation before compute.',
        'selection': [{key: row[key] for key in ('accession', 'native_substrates_resolved', 'priority', 'family')}
                      for row in chosen],
        'unannotated_available': len(unannotated), 'discovery_candidates': len(rows),
        'skipped_artifacts': skipped_artifacts})
    return chosen
