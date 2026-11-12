import numpy as np
import pytest
pytest.importorskip('parmed')
from enzyme_barrier.fixtures import water_pair
from enzyme_barrier.jobs import validate_geometry, _box_matrix, _min_image


def test_box_matrix_and_min_image_truncated_octahedron():
    # a truncated-octahedron cell (the AS9 solvation box geometry)
    p = water_pair()
    p.box = [40.0, 40.0, 40.0, 109.471220, 109.471220, 109.471220]
    H = _box_matrix(p)
    assert H.shape == (3, 3)
    # an atom displaced by exactly one lattice vector is the same image -> ~0
    for v in H:
        d = _min_image((v + np.array([0.3, -0.2, 0.1]))[None, :], H)[0]
        assert np.linalg.norm(d - np.array([0.3, -0.2, 0.1])) < 1e-6
    # non-periodic -> identity
    assert np.allclose(_min_image(np.array([[123.0, 0, 0]]), None), [[123.0, 0, 0]])


def test_validate_geometry_passes_clean_system():
    p = water_pair()
    r = validate_geometry(p, p.coordinates, raise_on_fail=False)
    assert r['passed'] and r['n_bad_bonds'] == 0 and r['n_clashes'] == 0


def test_validate_geometry_flags_torn_bond_unless_reaction_atom():
    p = water_pair()
    x = p.coordinates.copy()
    x[1] = x[0] + [3.0, 0, 0]           # stretch O-H1 of water 0 to 3 A
    with pytest.raises(ValueError, match='covalent bond'):
        validate_geometry(p, x)
    # same stretch is allowed when both atoms are declared reaction atoms
    r = validate_geometry(p, x, reaction_atoms=(0, 1), raise_on_fail=False)
    assert r['passed'], r['failures']


def test_validate_geometry_flags_clash():
    p = water_pair()
    x = p.coordinates.copy()
    x[3] = x[0] + [0.3, 0, 0]           # water 1's O right on top of water 0's O
    with pytest.raises(ValueError):
        validate_geometry(p, x)


def test_validate_geometry_flags_noninteger_charge():
    p = water_pair()
    p.atoms[0].charge += 0.5
    r = validate_geometry(p, p.coordinates, raise_on_fail=False)
    assert not r['passed'] and any('charge' in f for f in r['failures'])
