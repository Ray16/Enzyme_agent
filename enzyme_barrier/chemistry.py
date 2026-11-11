"""Geometry-based chemical-state observations, separate from energy ordering."""
import numpy as np


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
