"""Reaction geometry selections are explicit and reject ambiguous atom mappings."""
from types import SimpleNamespace
import numpy as np
import pytest
from enzyme_design_agent.tools.rfd4_campaign.geometry import evaluate


def atoms():
    return SimpleNamespace(chain_id=np.array(['B','B','D','D']),
        res_id=np.array([1,1,1,1]),atom_name=np.array(['PA','O3A','O1','O2']),
        element=np.array(['P','O','O','O']),
        coord=np.array([[0,0,0],[1,0,0],[-3,0,0],[0,4,0]],dtype=float))


def spec():
    return {'center':{'chain':'B','atom':'PA'},'angle_reference':{'chain':'B','atom':'O3A'},
            'attack_candidates':{'chain':'D','element':'O'},'expected_attack_candidates':2}


def test_nearest_attacking_atom_and_angle():
    result=evaluate(atoms(),spec())
    assert result['attack_atom']=='O1'
    assert result['distance_A']==pytest.approx(3)
    assert result['angle_deg']==pytest.approx(180)


def test_reject_ambiguous_or_incomplete_selection():
    s=spec();s['center']={'chain':'B','element':'O'};s['expected_attack_candidates']=3
    with pytest.raises(ValueError,match='candidate count'):evaluate(atoms(),s)
    s=spec();s['center']={'chain':'D','element':'O'}
    with pytest.raises(ValueError,match='exactly one'):evaluate(atoms(),s)
