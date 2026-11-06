import pytest
from ase import Atoms
from structural_modeling_agent.sampling.backends import bind_electronic_state


def test_open_shell_metadata_agrees_across_model_conventions():
    atoms = Atoms("OH")
    bind_electronic_state(atoms, 0, 2)
    assert atoms.info == {"charge": 0, "spin": 2, "multiplicity": 2}
    assert atoms.info["multiplicity"] - 1 == 1


@pytest.mark.parametrize("info", [{"spin": 1}, {"multiplicity": 1}, {"charge": -1}])
def test_conflicting_model_metadata_is_rejected(info):
    atoms = Atoms("OH", info=info)
    with pytest.raises(ValueError, match="disagrees"):
        bind_electronic_state(atoms, 0, 2)


@pytest.mark.parametrize("charge,mult", [(0, 1), (0, 12), (False, 2), (0, 0)])
def test_invalid_open_shell_state_is_rejected(charge, mult):
    with pytest.raises(ValueError):
        bind_electronic_state(Atoms("OH"), charge, mult)
