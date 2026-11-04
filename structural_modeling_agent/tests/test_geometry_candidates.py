"""Predeclared reaction partners use the same nearest-atom rule in every pose."""
import numpy as np
import pytest
from structural_modeling_agent.validation import _geometry_refs


def test_closest_predeclared_atom_is_used_and_missing_candidates_do_not_disappear():
    ref=lambda atom:{'chain':'L','residue':'1','atom':atom}
    first,second,center,axis=[ref(name) for name in ['SYNTH_O1','SYNTH_O2','SYNTH_P','SYNTH_AXIS']]
    site={'nucleophile':first,'electrophile':center,'angle_reference':axis,'nucleophile_candidates':[first,second]}
    atoms={('L','1',name):{'xyz':np.array(xyz)} for name,xyz in [
        ('SYNTH_O1',[8.,0,0]),('SYNTH_O2',[3.,0,0]),('SYNTH_P',[0.,0,0]),('SYNTH_AXIS',[-1.,0,0])]}
    assert _geometry_refs(atoms,site)[0]==second
    atoms[('L','1','SYNTH_O1')]['xyz']=np.array([2.,0,0])
    assert _geometry_refs(atoms,site)[0]==first
    del atoms[('L','1','SYNTH_O2')]
    with pytest.raises(LookupError):_geometry_refs(atoms,site)
