"""Geometry-based chemical-state observations, separate from energy ordering."""
import numpy as np


def reaction_state(parm,positions,region):
    """Observe a specified bond exchange; never assign chemical state by energy.

    The legacy phosphate classifier remains available for its stricter native
    neighbor checks. Other centers use the recorded geometric path hypothesis.
    """
    classifier=region.get('state_classifier')
    if classifier=='phosphate_transfer' or (classifier is None and 'path_definition' not in region and
            parm.atoms[region['reaction_atoms'][1]].atomic_number==15):
        return phosphate_state(parm,positions,region)
    from ase.data import covalent_radii
    from .region import METAL_NUMBERS
    definition=region.get('path_definition')
    if definition is None:
        raise ValueError('Generic bond-exchange classification requires path_definition')
    x=np.asarray(positions,float);i,j,k=region['reaction_atoms']
    df=float(np.linalg.norm(x[i]-x[j]));db=float(np.linalg.norm(x[j]-x[k]))
    form_cut=(definition['reactant_forming_A']+definition['forming_distance_A'])/2
    break_cut=(definition['reactant_breaking_A']+definition['breaking_distance_A'])/2
    bad=[]
    allowed={frozenset((i,j)),frozenset((j,k))}
    qm=set(region['indices']);numbers=[a.atomic_number for a in parm.atoms]
    expected=set()
    for at in parm.atoms:
        if at.idx not in qm:continue
        for partner in at.bond_partners:
            pair=frozenset((at.idx,partner.idx))
            expected.add(pair)
            if pair in allowed:continue
            length=float(np.linalg.norm(x[at.idx]-x[partner.idx]))
            max_length=1.4*(covalent_radii[numbers[at.idx]]+covalent_radii[numbers[partner.idx]])
            if not .4<length<max_length:bad.append(sorted(pair))
    # Unexpected covalent contacts involving the reactive center also invalidate
    # the seed. Distance observations are not a bond-order calculation.
    for atom in qm-{j}:
        if numbers[atom] in METAL_NUMBERS:continue
        pair=frozenset((atom,j))
        threshold=1.2*(covalent_radii[numbers[atom]]+covalent_radii[numbers[j]])
        if pair not in allowed|expected and np.linalg.norm(x[atom]-x[j])<threshold:
            bad.append(sorted(pair))
    state='transition_or_unresolved'
    if bad:state='off_path_or_mapping_requires_review'
    elif df>form_cut and db<break_cut:state='reactant'
    elif df<form_cut and db>break_cut:state='product'
    return dict(state=state,forming_A=df,breaking_A=db,unexpected_bond_changes=bad,
                scope='distance-based bond-exchange observation; not bond-order or transition-state proof')


def phosphate_state(parm,positions,region,bond_cutoff=1.95):
    x=np.asarray(positions);nuc,center,leaving=region['reaction_atoms']
    if parm.atoms[center].atomic_number!=15:raise ValueError('This classifier is for phosphoryl/adenylyl transfer')
    def partners(index):return {a.idx for a in parm.atoms[index].bond_partners if a.atomic_number>1}
    hetero=[i for i in region['indices'] if parm.atoms[i].atomic_number in (7,8,16)]
    current={i for i in hetero if np.linalg.norm(x[i]-x[center])<bond_cutoff}
    reactant=partners(center);product=(reactant-{leaving})|{nuc};unexpected=current-(reactant|product)
    other_changes=[]
    for p in region['indices']:
        if parm.atoms[p].atomic_number!=15 or p==center:continue
        actual={i for i in hetero if np.linalg.norm(x[i]-x[p])<bond_cutoff}
        if actual!=partners(p):other_changes.append(dict(phosphorus=p,expected=sorted(partners(p)),observed=sorted(actual)))
    carbon=[a.idx for a in parm.atoms[nuc].bond_partners if a.atomic_number==6]
    acyl_retained=bool(len(carbon)==1 and np.linalg.norm(x[nuc]-x[carbon[0]])<1.7)
    df=float(np.linalg.norm(x[nuc]-x[center]));db=float(np.linalg.norm(x[center]-x[leaving]))
    state='transition_or_unresolved'
    if current==reactant and df>2.4 and acyl_retained and not other_changes:state='reactant'
    elif current==product and db>2.6 and acyl_retained and not other_changes:state='activated_product'
    elif unexpected or other_changes or not acyl_retained:state='off_path_or_mapping_requires_review'
    return dict(state=state,forming_A=df,breaking_A=db,phosphorus_neighbors=sorted(current),
                expected_reactant_neighbors=sorted(reactant),expected_product_neighbors=sorted(product),
                unexpected_neighbors=sorted(unexpected),other_phosphorus_changes=other_changes,
                acyl_oxygen_carbon_bond_retained=acyl_retained,
                scope='distance-based heavy-atom connectivity; not bond-order or transition-state proof')
