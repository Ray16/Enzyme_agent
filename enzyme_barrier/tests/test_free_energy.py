import numpy as np
import pytest
pytest.importorskip('scipy')
from enzyme_barrier.free_energy import mbar_bias_weights,target_weights,integrated_basin_difference


def test_mbar_recovers_known_harmonic_distribution():
    rng=np.random.default_rng(12);kt=.00198720425864*303.15;spring=4.;base=1.
    centers=np.linspace(-2,2,9);n=3000
    x=np.concatenate([rng.normal(spring*c/(spring+base),np.sqrt(kt/(spring+base)),n) for c in centers])
    biases=.5*spring*(x[None,:]-centers[:,None])**2
    result=mbar_bias_weights(biases,[n]*9)
    weights=np.exp(result['log_weights']);variance=np.dot(weights,(x-np.dot(weights,x))**2)
    assert result['overlap_connected']
    assert variance==pytest.approx(kt/base,rel=.08)
    assert np.max(np.abs(result['overlap'].sum(axis=1)-1))<1e-6


def test_sparse_target_correction_accounts_for_selection_probability():
    reference=np.log([.5,.5]);delta=np.zeros(2)
    result=target_weights(reference,delta,selection_probability=[.25,1])
    assert np.exp(result['log_weights'])==pytest.approx([.8,.2])
    assert integrated_basin_difference(result['log_weights'],[True,False],[False,True])>0


def test_disconnected_umbrellas_are_flagged():
    x=np.array([-10.,-10.1,10.,10.1]);centers=np.array([-10.,10.])
    result=mbar_bias_weights(100*(x[None,:]-centers[:,None])**2,[2,2])
    assert not result['overlap_connected']
