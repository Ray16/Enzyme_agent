from dataclasses import replace
import numpy as np
import pytest

pmd=pytest.importorskip('parmed')
from enzyme_barrier.md import resolve_attack_geometry
from enzyme_barrier.spec import AtomRef,ReactionSpec


def test_carbonyl_angle_uses_distinct_terminal_oxygen_and_roundtrips(tmp_path):
    p=pmd.Structure()
    for name,z in [('C',6),('OD',8),('OL',8),('P',15),('N',7)]:
        p.add_atom(pmd.Atom(name=name,atomic_number=z,mass=z), 'LIG',1)
    for i,j in [(0,1),(0,2),(2,3)]:p.bonds.append(pmd.Bond(p.atoms[i],p.atoms[j]))
    p.coordinates=np.array([[0,0,0],[1.2,0,0],[-1.3,0,0],[-2.8,0,0],[0,3,0]])
    ref=lambda name:AtomRef('LIG',name)
    spec=ReactionSpec('amide',(('LIG',0),),(),ref('N'),ref('C'),ref('OL'),carbonyl_oxygen=ref('OD'))
    assert resolve_attack_geometry(p,spec)==(4,0,2,1,'nucleophile-carbonylC-carbonylO')
    f=tmp_path/'spec.json';spec.save(f);assert ReactionSpec.load(f)==spec
    for wrong in [None,ref('OL'),ref('P')]:
        with pytest.raises(ValueError,match='Carbonyl'):resolve_attack_geometry(p,replace(spec,carbonyl_oxygen=wrong))
