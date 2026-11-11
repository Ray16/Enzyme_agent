import numpy as np
import pytest
pytest.importorskip('ase')
from enzyme_barrier.umbrella import distance_cv


def test_coupled_proton_transfer_cv_gradient():
    x=np.array([[0.,0,0],[1.,.1,0],[2.,.3,0]])
    terms=[[0,1,1],[1,2,-1]]
    value,gradient=distance_cv(x,terms)
    for i in range(3):
        for k in range(3):
            a=x.copy();b=x.copy();a[i,k]+=1e-6;b[i,k]-=1e-6
            numeric=(distance_cv(a,terms)[0]-distance_cv(b,terms)[0])/2e-6
            assert gradient[i,k]==pytest.approx(numeric,abs=1e-8)
    assert np.linalg.norm(gradient.sum(axis=0))<1e-12


def test_chunk_restart_reproduces_continuous_dynamics(tmp_path,monkeypatch):
    import json
    from ase import Atoms
    from ase.calculators.calculator import Calculator,all_changes
    from enzyme_barrier import umbrella,sampler
    class Harmonic(Calculator):
        implemented_properties=['energy','forces']
        def __init__(self,*args,**kwargs):super().__init__()
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            self.results={'energy':float(np.sum(atoms.positions**2)/2),'forces':-atoms.positions.copy()}
    snap=tmp_path/'snapshot';snap.mkdir();(snap/'input.json').write_text('{}')
    def load(_):
        a=Atoms('H2',positions=[[0,0,0],[.8,0,0]])
        return None,dict(mobile_indices=[0,1]),a,{}
    monkeypatch.setattr(umbrella,'load_snapshot',load);monkeypatch.setattr(sampler,'EmbeddedUMAXTB',Harmonic)
    for name,limit in [('continuous',1000),('resumed',1)]:
        cfg=dict(out=str(tmp_path/name),snapshot=str(snap),sampling_role='reference_only',seed=17,
                 coordinates=[[[0,1,1]]],centers=[.8],springs_kcal_A2=[10],steps=30,
                 chunk_steps=10,stride=5,equilibration_steps=0,max_chunks_per_invocation=limit)
        path=tmp_path/(name+'.json');path.write_text(json.dumps(cfg))
        for _ in range(1 if name=='continuous' else 3):result=umbrella.sample(path)
        assert result['status']=='window_complete'
    a=np.load(tmp_path/'continuous/state00002.npz');b=np.load(tmp_path/'resumed/state00002.npz')
    assert np.allclose(a['positions'],b['positions'],atol=1e-12)
    assert np.allclose(a['velocities'],b['velocities'],atol=1e-12)
