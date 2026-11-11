import json
from types import SimpleNamespace
import numpy as np
import pytest
from enzyme_barrier.chemistry import phosphate_state


def phosphate_fixture():
    numbers=[15,8,8,8,8,8,6,15,8,8]
    atoms=[SimpleNamespace(idx=i,atomic_number=z,bond_partners=[]) for i,z in enumerate(numbers)]
    for i,j in [(0,1),(0,2),(0,3),(0,4),(5,6),(7,8)]:
        atoms[i].bond_partners.append(atoms[j]);atoms[j].bond_partners.append(atoms[i])
    x=np.array([[0,0,0],[1.6,0,0],[-.8,1.4,0],[-.8,-1.4,0],[0,0,1.6],
                [0,0,-3],[0,0,-4.3],[10,0,0],[11.6,0,0],[20,0,0]],float)
    return SimpleNamespace(atoms=atoms),x,dict(reaction_atoms=[5,0,1],indices=list(range(10)))


def test_phosphate_identity_and_off_path_connectivity():
    p,x,r=phosphate_fixture();assert phosphate_state(p,x,r)['state']=='reactant'
    product=x.copy();product[1]=[3,0,0];product[5]=[0,0,-1.6];product[6]=[0,0,-2.9]
    assert phosphate_state(p,product,r)['state']=='activated_product'
    for change in ['otherP','extraO','lost_acyl']:
        broken=product.copy()
        if change=='otherP':broken[8]=[13,0,0]
        elif change=='extraO':broken[9]=[1,1,0]
        else:broken[6]=[0,0,-5]
        assert phosphate_state(p,broken,r)['state']=='off_path_or_mapping_requires_review'


def test_path_chemical_reference_is_not_first_energy_or_global_minimum(tmp_path,monkeypatch):
    pytest.importorskip('ase')
    from enzyme_barrier import stationary
    p,x,r=phosphate_fixture();monkeypatch.setattr(stationary,'load_snapshot',lambda _: (p,r,None,{}))
    product=x.copy();product[1]=[3,0,0];product[5]=[0,0,-1.6];product[6]=[0,0,-2.9]
    mid=x.copy();mid[1]=[2.2,0,0];mid[5]=[0,0,-2.2];mid[6]=[0,0,-3.5]
    rows=[]
    for n,(pos,energy) in enumerate([(x,5),(x,0),(mid,10),(product,-8)]):
        f=tmp_path/f'{n}.npy';np.save(f,pos)
        rows.append(dict(image=n,energy_eV=energy,positions=str(f),optimization_converged=True))
    source=tmp_path/'path.json';source.write_text(json.dumps(dict(snapshot='test',images=rows)))
    result=stationary.summarize(source,tmp_path/'review.json')
    assert result['reactant_seed']['image']==1
    assert result['product_seed']['image']==3
    assert result['saddle_seed']['image']==2
    assert result['has_interior_maximum']
    assert result['constrained_path_gap_kcal']==pytest.approx(230.6054783)
    assert not result['validated_activation_free_energy']
    rows[2]['optimization_converged']=False;source.write_text(json.dumps(dict(snapshot='test',images=rows)))
    result=stationary.summarize(source,tmp_path/'failed.json')
    assert result['status']=='path_optimization_unresolved'
    assert 'saddle_seed' not in result
