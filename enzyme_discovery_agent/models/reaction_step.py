"""Check a source-linked local bond-change hypothesis without predicting activity."""
from collections import Counter, defaultdict
import hashlib
from itertools import permutations, product
import json
import re

from enzyme_discovery_agent.models.reaction_roles import evidence_basis


def target_basis(context):
    return hashlib.sha256(json.dumps({key:context.get(key) for key in (
        'query','substrate_smiles','target_profile','target_reasoning')}, sort_keys=True, default=str).encode()).hexdigest()


def _identity(molecule):
    from rdkit import Chem
    molecule=Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _reaction(smiles):
    from rdkit import Chem
    if not isinstance(smiles,str) or len(smiles)>12000 or smiles.count('>>')!=1:
        raise ValueError('Supply a bounded atom-mapped reactants>>products reaction')
    sides=[]
    for side in smiles.split('>>'):
        molecules=[Chem.MolFromSmiles(part) for part in side.split('.')]
        if not molecules or any(mol is None for mol in molecules):
            raise ValueError('Reaction contains an invalid molecular structure')
        atoms, bonds, occurrences={}, {}, {}
        for occurrence,mol in enumerate(molecules):
            for atom in mol.GetAtoms():
                label=atom.GetAtomMapNum()
                if label<=0 or label in atoms:
                    raise ValueError('Every atom needs a unique positive map number on each reaction side')
                atoms[label]=atom.GetAtomicNum()
                occurrences[label]=occurrence
            for bond in mol.GetBonds():
                pair=tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(),bond.GetEndAtom().GetAtomMapNum())))
                bonds[pair]=bond.GetBondTypeAsDouble()
        sides.append((atoms,bonds,molecules,occurrences))
    if sides[0][0]!=sides[1][0]:
        raise ValueError('Reactants and products must conserve mapped atom identities, including leaving fragments')
    before,after=sides[0][1],sides[1][1]
    changes=[(*pair,before.get(pair,0),after.get(pair,0)) for pair in sorted(set(before)|set(after))
             if before.get(pair,0)!=after.get(pair,0)]
    if not changes:
        raise ValueError('Record actual bond changes, not only shared groups or an unchanged reaction')
    return {'atoms':sides[0][0], 'changes':changes,
            'atom_occurrences':sides[0][3],
            'reactants':[_identity(mol) for mol in sides[0][2]],
            'products':[_identity(mol) for mol in sides[1][2]]}


def _same_edits(native, target):
    """Compare connected edit roles up to relabeling, independent of scaffold."""
    groups=[]
    for reaction in (native,target):
        atoms=defaultdict(list)
        for label in sorted({n for change in reaction['changes'] for n in change[:2]}):
            atoms[reaction['atoms'][label]].append(label)
        groups.append(atoms)
    if {k:len(v) for k,v in groups[0].items()}!={k:len(v) for k,v in groups[1].items()}:
        return False
    if sum(map(len,groups[0].values()))>8:
        raise ValueError('More than eight changing atoms requires a separate step review')
    elements=sorted(groups[0])
    expected=set(target['changes'])
    for alternatives in product(*(permutations(groups[1][element]) for element in elements)):
        mapping={n:t for element,order in zip(elements,alternatives) for n,t in zip(groups[0][element],order)}
        actual={(*sorted((mapping[a],mapping[b])),old,new) for a,b,old,new in native['changes']}
        same_occurrences=all((native['atom_occurrences'][a]==native['atom_occurrences'][b]) ==
                             (target['atom_occurrences'][mapping[a]]==target['atom_occurrences'][mapping[b]])
                             for a in mapping for b in mapping)
        if actual==expected and same_occurrences:
            return True
    return False


def native_step_inputs(data, activity):
    """Use one recorded activity's participants, never a union of activities."""
    group=next((item for item in data.get('native_reaction_groups') or [] if item.get('reaction')==activity),None)
    selected=(data.get('native_substrate_reaction_scope') or {}).get('reaction')
    if group and group.get('native_substrates') is not None:
        return group['native_substrates']
    if selected and selected!=activity:
        raise ValueError('Resolve participants for this specific native activity before mapping its step')
    return data.get('native_substrates') or []


def validate_step(data, context, native_reaction, target_reaction, source,
                  interpretation, activation_assessment, continuation_assessment):
    """Validate structures and provenance; source-to-structure entailment is asserted."""
    from rdkit import Chem
    match=re.fullmatch(r'catalytic_activities/(\d+)', (source or {}).get('locator',''))
    activities=data.get('catalytic_activities') or []
    quote=(source or {}).get('quote')
    if (not match or int(match[1])>=len(activities) or not isinstance(quote,str)
            or len(quote.strip())<20 or quote not in activities[int(match[1])]):
        raise ValueError('Cite an exact catalytic_activities/N passage for this native reaction')
    selected=(data.get('native_substrate_reaction_scope') or {}).get('reaction')
    for value in (interpretation,activation_assessment,continuation_assessment):
        if not isinstance(value,str) or len(value.strip())<20:
            raise ValueError('Explain the reacting roles, activation and continuation uncertainties explicitly')
    native,target=_reaction(native_reaction),_reaction(target_reaction)
    substrates=native_step_inputs(data,activities[int(match[1])])
    if not all(item.get('occurrence_id') for item in substrates):
        from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences
        substrates=substrate_occurrences(activities[int(match[1])],substrates)
    allowed=[]
    for item in substrates:
        mol=Chem.MolFromSmiles(item.get('smiles') or '')
        if mol is None:
            raise ValueError('Native reactant structures must be resolved before recording a step')
        allowed.append(_identity(mol))
    if Counter(native['reactants'])!=Counter(allowed):
        raise ValueError('Use exactly the resolved native occurrences from this reaction; do not duplicate an unsupported participant')
    wanted=[]
    for part in str(context.get('substrate_smiles') or '').split('.'):
        mol=Chem.MolFromSmiles(part)
        if mol is None:
            raise ValueError('The target reactant structures must be resolved')
        wanted.append(_identity(mol))
    if Counter(target['reactants'])!=Counter(wanted):
        raise ValueError('Use exactly the campaign target inputs; changed or extra reactants require a separate hypothesis')
    same=_same_edits(native,target)
    return {'schema_version':1, 'basis_sha256':evidence_basis(data), 'target_sha256':target_basis(context),
        'native_reaction':native_reaction, 'target_reaction':target_reaction, 'source':source,
        'native_bond_changes':native['changes'], 'target_bond_changes':target['changes'],
        'local_bond_changes_match':same, 'target_products':target['products'],
        'native_activity':activities[int(match[1])],
        'native_inputs_sha256':hashlib.sha256(json.dumps(substrates,sort_keys=True,default=str).encode()).hexdigest(),
        'target_input_smiles':str(context.get('substrate_smiles') or '').split('.'),
        'interpretation':interpretation, 'activation_assessment':activation_assessment,
        'continuation_assessment':continuation_assessment,
        'validation_scope':'Atom identities, valence, reactant occurrence counts, local bond edits and quote integrity checked. '
            'Native product identity, source-to-structure interpretation, activation feasibility, enzyme acceptance, '
            'requested product connectivity and repeated extension remain scientific hypotheses.'}


def step_summary(data, context=None):
    record=data.get('reaction_step')
    if not record:
        return {'status':'not_recorded', 'local_bond_changes_match':None}
    current=record.get('basis_sha256')==evidence_basis(data)
    if record.get('native_inputs_sha256'):
        try:
            inputs=native_step_inputs(data,record.get('native_activity'))
            if not all(item.get('occurrence_id') for item in inputs):
                from enzyme_discovery_agent.tools.native_reactions import substrate_occurrences
                inputs=substrate_occurrences(record.get('native_activity') or '',inputs)
            current=current and hashlib.sha256(json.dumps(inputs,sort_keys=True,default=str).encode()).hexdigest()==record['native_inputs_sha256']
        except ValueError:
            current=False
    if context is not None:
        current=current and record.get('target_sha256')==target_basis(context)
    pairs=((data.get('substrate_similarity') or {}).get('accommodation') or {}).get('per_target')
    if pairs:
        current=current and sorted(pair.get('target_smiles') or '' for pair in pairs)==sorted(record.get('target_input_smiles') or [])
    return {'status':'source_checked_hypothesis' if current else 'stale',
            'local_bond_changes_match':record.get('local_bond_changes_match') if current else None,
            'interpretation':record.get('interpretation'), 'source':record.get('source'),
            'activation_assessment':record.get('activation_assessment'),
            'continuation_assessment':record.get('continuation_assessment'),
            'scope':record.get('validation_scope')}
