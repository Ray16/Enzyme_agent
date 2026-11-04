"""Regression tests for mature-chain roles independent of fragment size/names."""
import numpy as np
import pytest
from structural_modeling_agent.proxies.ntn_ensemble import pairs


def chain(sequence,shift):
    names={'A':'ALA','T':'THR','G':'GLY'}
    return {i+1:{'_r':names[a],'CA':np.array([shift,i*.02,0.])} for i,a in enumerate(sequence)}


def fixture(alpha,beta):
    return {'X':chain(alpha,0.),'Z':chain(alpha,40.),
            'W':chain(beta,41.),'Y':chain(beta,1.)}


def test_large_beta_fragments_are_paired_by_contacts():
    alpha='A'*271;beta='T'+'G'*216
    assert pairs(fixture(alpha,beta),alpha,beta)==[('X','Y'),('Z','W')]
    assert pairs(fixture(alpha,beta))==[('X','Y'),('Z','W')]


def test_explicit_sequences_disambiguate_equal_lengths():
    alpha='A'*180;beta='T'+'G'*179
    assert pairs(fixture(alpha,beta),alpha,beta)==[('X','Y'),('Z','W')]
    with pytest.raises(ValueError,match='declared sequences'):
        pairs(fixture(alpha,beta))


def test_sequence_mismatch_and_non_thr_beta_fail_closed():
    alpha='A'*271;beta='T'+'G'*216;prot=fixture(alpha,beta)
    with pytest.raises(ValueError,match='two alpha'):
        pairs(prot,alpha,'T'+'G'*215)
    with pytest.raises(ValueError,match='N-terminal Thr'):
        pairs(fixture(alpha,'G'*217),alpha,'G'*217)
