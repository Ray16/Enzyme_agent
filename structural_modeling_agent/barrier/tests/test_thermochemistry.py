import numpy as np
import pytest

pytest.importorskip('ase')
from ase import Atoms
from ase.calculators.calculator import Calculator,all_changes
from ase.thermochemistry import HarmonicThermo
from ase.units import invcm
from structural_modeling_agent.barrier.thermochemistry import partial_frequencies,qrrho_correction


def test_qrrho_high_frequency_limit_matches_harmonic_partition_function():
    frequencies=np.array([800.,1400.,3000.]);temperature=300.
    actual=qrrho_correction(frequencies,temperature,cutoff_cm=.0001)
    expected=HarmonicThermo(frequencies*invcm).get_helmholtz_energy(temperature,verbose=False)
    assert actual['correction_eV']==pytest.approx(expected,abs=1e-10)
    assert actual['significant_imaginary_modes']==0


def test_partial_hessian_finds_one_negative_mode_and_restores_positions():
    class Saddle(Calculator):
        implemented_properties=['energy','forces']
        def calculate(self,atoms=None,properties=('energy',),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            sign=np.array([-1,1,1]);x=atoms.positions
            self.results=dict(energy=float(np.sum(x*x*sign)),forces=-2*x*sign)
    atoms=Atoms('HH',positions=[[.1,.2,.3],[3,4,5]]);atoms.calc=Saddle()
    original=atoms.positions.copy()
    frequencies,diagnostics=partial_frequencies(atoms,[0])
    assert np.array_equal(atoms.positions,original)
    assert diagnostics['max_asymmetry_eV_A2']<1e-10
    assert diagnostics['force_calls']==6
    assert np.count_nonzero(frequencies.imag>50)==1
    assert np.count_nonzero(frequencies.real>50)==2
    assert qrrho_correction(frequencies,300)['significant_imaginary_modes']==1
