import numpy as np
import pytest
pytest.importorskip('ase')
from enzyme_barrier.umbrella import distance_cv


def test_proton_coordination_is_smooth_and_hydrogen_permutation_invariant():
    from enzyme_barrier.umbrella import coordination_cv,evaluate_cv
    x=np.array([[0.,0,0],[2.5,.1,0],[1.,.1,0],[-.4,.9,0],[-.4,-.8,.1]])
    terms=[[heavy,h,coefficient] for heavy,coefficient in [(0,1),(1,-1)] for h in [2,3,4]]
    cv=dict(kind='coordination',terms=terms,r0_A=1.25,power=12)
    value,g=evaluate_cv(x,cv)
    for i in range(len(x)):
        for j in range(3):
            a=x.copy();b=x.copy();a[i,j]+=1e-6;b[i,j]-=1e-6
            assert g[i,j]==pytest.approx((evaluate_cv(a,cv)[0]-evaluate_cv(b,cv)[0])/2e-6,abs=1e-8)
    permutation=[0,1,4,2,3];xp=x[permutation]
    vp,gp=evaluate_cv(xp,cv)
    assert vp==pytest.approx(value)
    assert np.allclose(gp,g[permutation])
    assert np.allclose(g.sum(axis=0),0)
    # At r=r0 and r=0 the switch and derivative remain finite.
    for distance,expected in [(1.25,.5),(0,1)]:
        v,grad=coordination_cv([[0,0,0],[distance,0,0]],[[0,1,1]])
        assert v==pytest.approx(expected);assert np.isfinite(grad).all()
    with pytest.raises(ValueError):coordination_cv(x,[[0,1,1],[1,0,1]])
    with pytest.raises(ValueError):evaluate_cv(x,dict(kind='typo',terms=terms))


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
