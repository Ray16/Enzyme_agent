"""Evidence-linked participant roles independent of enzyme class or target chemistry."""
import hashlib
import json
import re

PARTICIPANT_ROLES=frozenset({'direct_reactant','precursor','cofactor','catalyst','carrier','solvent','spectator','unresolved'})
TRANSFER_UNITS=frozenset({'whole_molecule','fragment','atom','electrons','proton','unresolved'})
SITE_RELATIONS=frozenset({'same_site','separate_site','tunneled_intermediate','carrier_tethered','unresolved'})


def evidence_basis(data):
    return hashlib.sha256(json.dumps({key:data.get(key) for key in (
        'catalytic_activities','function_comments','native_substrates','native_substrate_reaction_scope','required_cofactors')},
        sort_keys=True,default=str).encode()).hexdigest()


def validate_roles(data,roles):
    """Verify participant identity and source passages, not scientific entailment."""
    if not isinstance(roles,list) or not 1<=len(roles)<=20:
        raise ValueError('Supply one to twenty participant-role records')
    native={item.get('smiles') for item in data.get('native_substrates') or [] if item.get('smiles')}
    records=[]
    for item in roles:
        if not isinstance(item,dict):
            raise ValueError('Each participant role must be an object')
        if item.get('native_smiles') and item['native_smiles'] not in native:
            raise ValueError('Each structural participant must match a resolved native substrate in this evaluation')
        if not item.get('native_smiles') and not isinstance(item.get('participant_name'),str):
            raise ValueError('Identify a resolved native_smiles or a participant_name explicitly present in the quoted source')
        for key,choices in [('role',PARTICIPANT_ROLES),('transferred_unit',TRANSFER_UNITS),('site_relation',SITE_RELATIONS)]:
            if item.get(key) not in choices:
                raise ValueError(f'Invalid {key}; use a declared classification or unresolved')
        if not isinstance(item.get('interpretation'),str) or not item['interpretation'].strip():
            raise ValueError('Explain the mechanistic interpretation and what is uncertain')
        source=item.get('source') or {}
        locator=source.get('locator','')
        match=re.fullmatch(r'(function_comments|catalytic_activities)/(\d+)',locator)
        quote=source.get('quote')
        if not match or not isinstance(quote,str) or len(quote.strip())<20:
            raise ValueError('Cite an exact annotation passage and its function_comments/N or catalytic_activities/N locator')
        values=data.get(match[1]) or []
        index=int(match[2])
        if index>=len(values) or quote not in values[index]:
            raise ValueError('The cited passage is not present in the candidate annotation')
        if not item.get('native_smiles') and (not item['participant_name'].strip() or item['participant_name'].casefold() not in quote.casefold()):
            raise ValueError('A participant without a structure must be named in the exact cited passage')
        records.append({key:item.get(key) for key in ('native_smiles','participant_name','role','transferred_unit','site_relation','interpretation','source')})
    return {'schema_version':1,'basis_sha256':evidence_basis(data),'participants':records,
            'validation_scope':'Participant identities and quoted source integrity checked; mechanistic interpretation remains an agent assertion.'}


def apply_roles(data,role_record,accommodation):
    """Prevent whole-molecule shape from proving an atom/fragment-role transfer."""
    if not role_record:
        return accommodation
    accommodation=dict(accommodation or {})
    if accommodation.pop('participant_role_mismatches',None) and not accommodation.get('indirect_donor_role_mappings'):
        accommodation.pop('role_transfer_status',None)
    if role_record.get('basis_sha256')!=evidence_basis(data):
        return {**accommodation,'role_record_status':'stale; source evidence changed',
                'role_transfer_status':'unresolved'}
    findings=[]
    from enzyme_discovery_agent.models.reaction_step import step_summary
    step=step_summary(data)
    for item in role_record['participants']:
        for pair in accommodation.get('per_target') or []:
            if not item.get('native_smiles') or pair.get('native_smiles')!=item['native_smiles'] or pair.get('target_smiles')==pair.get('native_smiles'):
                continue
            direct_fragment=(item['transferred_unit']=='fragment' and
                             step['status']=='source_checked_hypothesis' and step['local_bond_changes_match'] is True)
            if (item['role']!='direct_reactant' or not (item['transferred_unit']=='whole_molecule' or direct_fragment) or
                    item['site_relation'] not in {'same_site'}):
                findings.append({'target_smiles':pair.get('target_smiles'),'native_smiles':item['native_smiles'],
                    'role':item['role'],'transferred_unit':item['transferred_unit'],
                    'site_relation':item['site_relation'],'interpretation':item['interpretation'],'source':item['source']})
    result={**accommodation,'role_record_status':'source_checked_interpretation'}
    if findings:
        result.update(role_transfer_status='unresolved',participant_role_mismatches=findings)
    return result


def role_transfer_summary(data):
    """Recheck role evidence for consumers; cached fit is not a role assessment."""
    accommodation=(data.get('substrate_similarity') or {}).get('accommodation') or {}
    record=data.get('reaction_roles')
    current=apply_roles(data,record,accommodation) or {}
    mappings=list(current.get('participant_role_mismatches') or [])
    mappings.extend(current.get('indirect_donor_role_mappings') or [])
    unresolved=current.get('role_transfer_status')=='unresolved'
    from enzyme_discovery_agent.models.reaction_step import step_summary
    return {'status':'unresolved' if unresolved else 'source_checked' if record else 'not_recorded',
            'reaction_step':step_summary(data),
            'artifacts':[path for path in (data.get('reaction_role_artifact'),data.get('reaction_step_artifact')) if path],
            'review_required':unresolved,
            'record_status':current.get('role_record_status'),
            'mappings':mappings,
            'next_question':('Establish which atom, fragment or molecule reaches the reacting site, '
                             'through which steps, and whether the target can fill that role.' if unresolved else None),
            'scope':'Source integrity and role assertions do not establish non-native activity.'}
