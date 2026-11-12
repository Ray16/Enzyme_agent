import numpy as np
import pytest
from enzyme_barrier.analysis import minimax,analyze_opes


def test_minimax_does_not_bridge_unsampled_holes():
    assert minimax(np.array([[0,np.inf,2],[1,np.inf,1]]),(0,0),(0,2)) is None
    assert minimax(np.array([[0,4,2],[1,3,1]]),(0,0),(0,2))==3


def test_one_crossing_cannot_be_reported_as_valid_barrier(tmp_path):
    path=tmp_path/'COLVAR'
    rows=['#! FIELDS time d1 d2 opes.bias']
    rows += [f'{i} {3.4 if i<50 else 1.7} {1.6 if i<50 else 3.0} 0' for i in range(100)]
    path.write_text('\n'.join(rows))
    result=analyze_opes(path,discard=0)
    assert result['basin_switches']==1
    assert result['projected_barrier_kcal'] is None
    assert result['validated_activation_free_energy'] is False


def test_upper_wall_reweighting_is_accounted_for(tmp_path):
    path=tmp_path/'COLVAR';path.write_text('#! FIELDS time d1 d2 opes.bias\n'+ '\n'.join(f'{i} 4.8 1.6 0' for i in range(30)))
    result=analyze_opes(path,wall_specs=[dict(kind='upper',column='d1',at=4.7,kappa=30)])
    assert result['reconstructed_walls'][0]['kappa']==30
    assert result['reactant_frames']==15
