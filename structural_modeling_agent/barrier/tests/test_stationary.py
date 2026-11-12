import numpy as np
import pytest

pytest.importorskip('ase')
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.constraints import FixAtoms
from enzyme_barrier.stationary import transfer_mode,mode_diagnostics,refine_dimer


class DoubleWell(Calculator):
    implemented_properties=['energy','forces']
    def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        x=atoms.positions.copy();energy=float(np.sum(x*x));forces=-2*x
        energy+=(x[0,0]**2-1)**2-x[0,0]**2
        forces[0,0]=-4*x[0,0]*(x[0,0]**2-1)
        self.results=dict(energy=energy,forces=forces)


def test_mode_derivative_and_dimer_on_known_saddle(tmp_path):
    a=Atoms('HH',positions=[[.12,.07,0],[2,3,4]])
    a.set_constraint(FixAtoms(indices=[1]));a.calc=DoubleWell()
    mode=np.array([[1.,0,0],[0,0,0]])
    result=refine_dimer(a,mode,tmp_path/'saddle',fmax=1e-4,max_steps=70)
    assert result['stationary_candidate']
    assert np.linalg.norm(a.positions[0])<1e-4
    assert np.array_equal(a.positions[1],[2,3,4])
    assert result['mode_diagnostics']['curvature_eV_A2']==pytest.approx(-4,abs=1e-3)
    assert not result['transition_state_certified']
    assert not result['validated_activation_free_energy']
    for sign in [-1,1]:
        b=a.copy();b.positions[0,0]=sign;b.calc=DoubleWell()
        assert not mode_diagnostics(b,mode)['negative_curvature']


def test_transfer_mode_reduces_forming_and_extends_breaking_distance():
    x=np.array([[-2.5,0,0],[0,0,0],[2,0,0],[6,1,3]],float)
    v=transfer_mode(x,[0,1,2],[0,1,2]);moved=x+.01*v
    assert np.linalg.norm(moved[0]-moved[1])<2.5
    assert np.linalg.norm(moved[1]-moved[2])>2
    assert not v[3].any()
    with pytest.raises(ValueError):transfer_mode(x,[0,1,2],[3])
