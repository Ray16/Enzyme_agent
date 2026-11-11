import json
import numpy as np
import pytest

pytest.importorskip('ase')
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes


def test_diagnostics_preserve_geometry_identity_and_scope(tmp_path,monkeypatch):
    from enzyme_barrier import jobs,embedded
    class Harmonic(Calculator):
        implemented_properties=['energy','forces']
        def __init__(self,*args,**kwargs):super().__init__();self.last_components={}
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            self.results=dict(energy=float(np.sum(atoms.positions**2)/2),forces=-atoms.positions)
    snap=tmp_path/'snapshot';snap.mkdir();(snap/'input.json').write_text('{}')
    region=dict(indices=[0],mobile_indices=[0,1])
    monkeypatch.setattr(jobs,'load_snapshot',lambda _: (None,region,Atoms('H2',positions=[[0,0,0],[1,0,0]]),{}))
    monkeypatch.setattr(embedded,'EmbeddedDFTMM',Harmonic)
    pos=tmp_path/'pos.npy';np.save(pos,[[0,0,0],[2,0,0]])
    cfg=dict(out=str(tmp_path/'out'),snapshot=str(snap),geometries=[dict(label='reactant',positions=str(pos),sha256=jobs.sha256(pos))])
    path=tmp_path/'cfg.json';path.write_text(json.dumps(cfg))
    result=jobs.evaluate_geometries(path)
    assert result['points'][0]['energy_eV']==2
    assert result['points'][0]['max_mobile_force_eV_A']==2
    assert not result['validated_activation_free_energy']
    assert np.array_equal(np.load(tmp_path/'out/reactant.forces.npy'),[[0,0,0],[-2,0,0]])
    cfg['out']=str(tmp_path/'bad');cfg['geometries'][0]['sha256']='bad';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='changed'):jobs.evaluate_geometries(path)
    assert not (tmp_path/'bad').exists()
