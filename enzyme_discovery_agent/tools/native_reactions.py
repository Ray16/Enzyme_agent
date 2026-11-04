"""Keep native substrate evidence within one recorded catalytic reaction."""
import re
from enzyme_discovery_agent.chemistry_logic import classify_reaction_chemistry, check_bond_type_mismatch, derive_activation_mechanism


def reaction_groups(entry, reactant_ids):
    """Retain all reactions; resolve participants separately for each one."""
    groups=[]
    for index,comment in enumerate(entry.get('comments') or []):
        if comment.get('commentType')!='CATALYTIC ACTIVITY':
            continue
        reaction=comment.get('reaction') or {}
        ec=reaction.get('ecNumber')
        facts=classify_reaction_chemistry([ec] if ec else [],[reaction.get('name','')])
        groups.append({'reaction_index':index,'reaction':reaction.get('name',''),
            'ec_number':ec,'reaction_direction':facts.get('reaction_direction'),
            'bond_types':facts.get('bond_types') or [],
            'chebi_ids':reactant_ids({'comments':[comment]}),
            'rhea_ids':[x['id'] for x in reaction.get('reactionCrossReferences') or []
                        if x.get('database')=='Rhea' and x.get('id')]})
    return groups


def select_reaction_group(groups, target_direction, target_bonds, score):
    """Choose one coherent native reaction; unknown transfer remains unresolved.

    A separate enzyme activity must not supply a missing partner for this one.
    The caller provides its existing role-aware fit function. All original groups
    remain available as evidence, independent of which reaction orders this work.
    """
    def key(group):
        direction=group.get('reaction_direction')
        direction_rank=0 if direction and direction==target_direction else 1 if not direction or not target_direction else 2
        bonds=group.get('bond_types') or []
        bond_rank=1 if not bonds or not target_bonds else 2 if check_bond_type_mismatch(
            {'bond_types':bonds},{'bond_types':list(target_bonds)}) else 0
        fit=score(group)
        return direction_rank,bond_rank,fit is None,-fit if fit is not None else 0,group['reaction_index']
    return min(groups,key=key) if groups else None


def scoped_activation(selected,all_activities):
    """Assign a reaction center within the selected native activity, retaining others."""
    activities=[selected['reaction']] if selected and selected.get('reaction') else all_activities
    result={'activities':activities,'activation':derive_activation_mechanism(activities)}
    if selected:
        result['reaction_scope']={key:selected.get(key) for key in ('reaction_index','reaction','ec_number')}
    if len(all_activities)>1:
        result['all_activity_observations']=derive_activation_mechanism(all_activities)
    return result


def substrate_occurrences(reaction, substrates):
    """Preserve explicitly recorded stoichiometry without inventing extra sites.

    A chemical species and a reactant occurrence are distinct identities. Only
    exact names on the input side justify repeated copies; ambiguous annotation
    retains one occurrence and an unresolved multiplicity for investigation.
    """
    from enzyme_discovery_agent.tools.family_reactions import _norm_chem
    from enzyme_discovery_agent.chemistry_logic import get_chebi_names
    left=reaction.split('=',1)[0] if '=' in reaction else ''
    terms=re.split(r'\s+\+\s+', left)
    counts={}
    for term in terms:
        match=re.match(r'^\s*(?:(\d+)\s+)?(.+?)\s*$', term)
        if match:
            name=_norm_chem(match[2])
            counts[name]=counts.get(name,0)+int(match[1] or 1)
    occurrences=[]
    for index, substrate in enumerate(substrates):
        labels=[substrate.get('name') or '']
        if substrate.get('chebi_id'):labels+=get_chebi_names(substrate['chebi_id'])
        matching={_norm_chem(label) for label in labels}&counts.keys()
        count=sum(counts[label] for label in matching) if matching else None
        # Large coefficients remain explicit facts rather than creating a
        # combinatorial structural assignment over a bulk reaction equation.
        for copy in range(min(count or 1, 8)):
            occurrences.append({**substrate, 'occurrence_id':f'native:{index}:{copy}',
                'stoichiometry':count, 'stoichiometry_status':'source_recorded' if count else 'unresolved',
                'stoichiometry_source':reaction})
    return occurrences


def annotate_indirect_donor_roles(entry, native_substrates, accommodation):
    """Flag whole-molecule fit used in place of an annotated transferred atom.

    The donor identity is extracted from this record's hydrolyzing qualifier,
    then cross-checked against its explicit nitrogen-source description. This
    does not use an enzyme-name blacklist or infer non-native inactivity.
    """
    if not accommodation:
        return accommodation
    names=[]
    description=entry.get('proteinDescription') or {}
    for block in [description.get('recommendedName') or {}]+list(description.get('alternativeNames') or []):
        value=(block.get('fullName') or {}).get('value')
        if value:
            names.append(value)
    precursors=[]
    for name in names:
        precursors.extend(re.findall(r'\[([^\[\]]+?)-hydroly[sz]ing\]',name,re.I))
    findings=[]
    for precursor in precursors:
        for comment in entry.get('comments') or []:
            if comment.get('commentType')!='FUNCTION':
                continue
            for text_record in comment.get('texts') or []:
                text=text_record.get('value') or ''
                source=re.search(re.escape(precursor)+r'.{0,60}?(?:source of nitrogen|nitrogen source)',text,re.I)
                if not source:
                    continue
                donors=[item for item in native_substrates if re.search(r'\b'+re.escape(precursor)+r'\b',item.get('name') or '',re.I)]
                for donor in donors:
                    for pair in accommodation.get('per_target') or []:
                        if pair.get('native_smiles')!=donor.get('smiles') or pair.get('target_smiles')==pair.get('native_smiles'):
                            continue
                        finding={'target_smiles':pair.get('target_smiles'),'native_smiles':donor.get('smiles'),
                            'native_substrate':donor.get('name'),'role':'hydrolyzed_nitrogen_source',
                            'source':{'kind':'UniProtKB annotation','locator':'proteinDescription + comments[FUNCTION]',
                                'name_qualifier':next(name for name in names if precursor.casefold() in name.casefold()),
                                'quote':text,'evidences':text_record.get('evidences') or []},
                            'inference':'Whole precursor similarity does not establish acceptance at the atom-transfer/ligation site. Non-native role mapping remains unresolved.'}
                        if finding not in findings:
                            findings.append(finding)
    if not findings:
        return accommodation
    result={**accommodation,'indirect_donor_role_mappings':findings,'role_transfer_status':'unresolved'}
    return result
