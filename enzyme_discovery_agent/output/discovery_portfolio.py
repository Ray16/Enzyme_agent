"""Source-backed exploration proposals for structural investigation, not activity ranks."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, discovery_status
from enzyme_discovery_agent.models.native_annotation import native_annotation_for_report
from enzyme_discovery_agent.models.target_review import identity_verified, target_contract, digest
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json, atomic_write_text


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write_discovery_portfolio(run_dir, *, limit=48):
    """Select distinct source-grounded hypotheses before additional close relatives.

    A family hypothesis is permission to investigate, not proof of substrate
    acceptance. Unknown native function and incomplete structural setup stay
    explicit. Name/reaction groups guide attention, never assert orthology.
    """
    if type(limit) is not int or limit<1:
        raise ValueError('Discovery portfolio capacity must be a positive integer')
    root=Path(run_dir)
    registry=load_candidate_registry(root,strict=True)['candidates']
    landscape=_read(run_file(root,'family_landscape.json'),[])
    families={row['fold_id']:row for row in landscape}
    nomination_document=_read(root/'agent_state/discovery_nominations.json',{})
    nominated=nomination_document.get('candidates',{}) if nomination_document.get('target_sha256')==digest(target_contract(root)) else {}
    groups=defaultdict(list)
    for accession,entry in registry.items():
        meta=entry.get('metadata') or {}
        data={**(meta.get('screening_annotation') or {}),**(meta.get('evaluation_data') or {}),**meta}
        assessment=meta.get('target_assessment') or {}
        nomination=nominated.get(accession) or {}
        if nominated and not nomination:
            continue
        if (discovery_status(entry)=='rejected' or meta.get('is_literature_lead')
                or assessment.get('decision')=='exclude'
                or not identity_verified(root,accession,meta)):
            continue
        family_id=data.get('screening_family_id') or data.get('interpro_family_id')
        family=families.get(family_id) or {}
        hypotheses=family.get('derived_family_hypotheses') or []
        if not hypotheses and family.get('derived_family_hypothesis'):
            hypotheses=[family['derived_family_hypothesis']]
        if not nomination and not hypotheses and assessment.get('decision') not in {'prioritize','investigate'}:
            continue
        reactions=data.get('catalytic_activities') or []
        # Keep the full source record, including conflicts, in an immutable
        # proposal. The shared family rationale is not regenerated per protein.
        annotation=native_annotation_for_report(root,accession,data)
        name=' '.join(str(data.get('protein_name') or '').lower().split())
        label=name if name and not any(word in name for word in ('uncharacterized','hypothetical','unknown')) else accession
        group=(nomination.get('mechanism_family') or str(family_id or 'unresolved'),label,
               tuple(sorted(reactions)))
        group_id=hashlib.sha256(json.dumps(group).encode()).hexdigest()[:16]
        reviewed=any(c.get('claim_type')=='reviewed_status' and c.get('verdict')=='supported'
                     for c in (data.get('claim_validation') or {}).get('claims',[]))
        row={
            'selection_order':0,'accession':accession,
            'protein_name':data.get('protein_name',''),'gene':'; '.join(data.get('gene_names') or []),
            'organism':data.get('organism',''),'family_id':family_id or 'unresolved',
            'family_name':family.get('fold_name') or data.get('interpro_family',''),
            'mechanism_family':nomination.get('mechanism_family') or ' | '.join(h.get('name','') for h in hypotheses),
            'route':nomination.get('route','family_level_hypothesis'),
            'selection_role':'group_representative',
            'annotation_group':group_id,
            'proposal_basis':'source-backed discovery nomination' if nomination else 'source-backed family hypothesis' if hypotheses else 'saved candidate hypothesis',
            'native_activity_evidence':annotation['native_activity_evidence'],
            'native_activity_sources':'; '.join(sorted({
                str(item.get('source',''))+':'+str(item.get('id',''))
                for reaction in annotation['reactions'] for item in reaction['evidences']
                if item.get('source') and item.get('id')})),
            'protein_record_status':annotation.get('record_status','unknown'),
            'native_reaction':' | '.join(reactions),
            'family_hypothesis':nomination.get('hypothesis') or ' | '.join(h.get('name','') for h in hypotheses),
            'required_catalytic_capability':' | '.join(h.get('required_catalytic_capability','') for h in hypotheses),
            'target_assessment':assessment.get('decision','not_assessed'),
            'target_roles':'; '.join(r['participant']+': '+r['role'] for r in assessment.get('target_roles',[])),
            'recorded_constraints':'; '.join(c.get('finding','') for c in assessment.get('constraints',[])),
            'unknowns':nomination.get('key_unknown') or '; '.join(assessment.get('unknowns') or []) or
                'Target substrate acceptance, reacting-partner placement and product outcome are unresolved.',
            'structural_question':'Compare the retrieved native reaction context with the target partners. '
                'Test whether both reacting partners can be accommodated with the required catalytic machinery; '
                'identify recognition differences and missing evidence. Geometry is not an activity measurement.',
            'modeling_setup_status':'requires_step_partner_and_cofactor_validation',
            'target_activity':'unestablished',
            'sequence_length':data.get('sequence_length') or data.get('seq_length') or '',
            'uniprot_url':'https://www.uniprot.org/uniprotkb/'+accession+'/entry',
            'group_alternatives':'','group_priority_status':'','proposal_source':'',
        }
        proposal={'accession':accession,'target_sha256':digest(target_contract(root)),
            'source_record':data,'native_annotation_provenance':annotation,
            'family_hypotheses':hypotheses,'selection_scope':row['proposal_basis'],
            'discovery_nomination':nomination,
            'structural_question':row['structural_question'],
            'limits':'Exploration proposal only. No independent target assessment is required for discovery admission. '
                'Identity, molecular setup and source applicability must be validated before structural compute.'}
        fingerprint=digest(proposal)
        path=root/'artifacts/discovery_proposals'/(fingerprint+'.json')
        if not path.exists():atomic_write_json(path,proposal)
        row['proposal_source']=str(path.relative_to(root))
        preference=(annotation['native_activity_evidence']!='experimental_evidence_cited',
                    not reviewed,assessment.get('decision')!='prioritize',accession)
        groups[group].append((preference,row))

    representatives=[]
    alternatives=[]
    for members in groups.values():
        members.sort(key=lambda pair:pair[0])
        rows=[pair[1] for pair in members]
        decisions={r['target_assessment'] for r in rows if r['target_assessment']!='not_assessed'}
        status='conflicting_assessments_need_comparison' if len(decisions)>1 else 'no_recorded_priority_conflict'
        for row in rows:
            row['group_priority_status']=status
            row['group_alternatives']='; '.join(r['accession'] for r in rows if r is not row)
        representatives.append((members[0][0],rows[0]))
        alternatives.extend(rows[1:])
    remaining=sorted(representatives,key=lambda item:item[0]);selected=[];used=Counter();fold_used=Counter()
    while remaining and len(selected)<limit:
        index=min(range(len(remaining)),key=lambda i:(used[remaining[i][1]['mechanism_family'] or remaining[i][1]['family_id']],
            fold_used[remaining[i][1]['family_id']],remaining[i][0]))
        _,row=remaining.pop(index);row['selection_order']=len(selected)+1
        selected.append(row);used[row['mechanism_family'] or row['family_id']]+=1
        fold_used[row['family_id']]+=1
    if nominated and len(selected)<limit:
        # Explicitly nominated sequence variants remain adjacent to their
        # group's representative. They are never presented as new mechanisms.
        group_order={row['annotation_group']:i for i,row in enumerate(selected)}
        extra=[row for row in alternatives if row['annotation_group'] in group_order]
        for row in extra[:limit-len(selected)]:
            row['selection_role']='nominated_sequence_variant'
            selected.append(row)
        selected.sort(key=lambda row:(group_order[row['annotation_group']],row['selection_role']!='group_representative',row['accession']))
        for i,row in enumerate(selected,1):row['selection_order']=i
    # Related sequences are included only through explicit nominations, never
    # invented or duplicated to meet the requested capacity.
    fields=list(selected[0]) if selected else ['selection_order','accession']
    for name,values in [('discovery_candidates.csv',selected),('discovery_alternatives.csv',alternatives)]:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)
        atomic_write_text(run_file(root,name),stream.getvalue())
    summary={'requested':limit,'selected':len(selected),'available_annotation_groups':len(groups),
             'family_ids':dict(Counter(row['family_id'] for row in selected)),
             'mechanism_families':dict(Counter(row['mechanism_family'] for row in selected)),
             'alternatives':len(alternatives),
             'policy':'Discovery proposals ordered for breadth and native-evidence provenance, not predicted target activity. '
                 'Annotation groups are scheduling hints, not demonstrated ortholog equivalence.',
             'candidates':selected}
    atomic_write_json(run_file(root,'discovery_modeling_handoff.json'),summary)
    return summary
