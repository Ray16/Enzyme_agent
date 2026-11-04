"""Independent numerical checks for the established ensemble ranking metric."""

import pytest

from structural_modeling_agent.proxies.attack_angle import weighted_attack_angle


def test_inverse_distance_weights_and_harmonic_distance():
    result = weighted_attack_angle([1., 2.], [90., 180.])
    assert result["weighted_attack_angle_deg"] == pytest.approx(120.)
    assert result["harmonic_distance_A"] == pytest.approx(4 / 3)
    assert result["n_poses"] == 2


def test_absolute_distance_is_reported_separately_from_orientation():
    first = weighted_attack_angle([2., 4.], [100., 160.])
    second = weighted_attack_angle([20., 40.], [100., 160.])
    assert first["weighted_attack_angle_deg"] == pytest.approx(second["weighted_attack_angle_deg"])
    assert second["harmonic_distance_A"] == pytest.approx(10 * first["harmonic_distance_A"])


@pytest.mark.parametrize("distances,angles", [([], []), ([0.], [90.]), ([2.], [181.]), ([float("nan")], [90.]), ([2., 3.], [90.])])
def test_invalid_or_unpaired_ensembles_are_not_ranked(distances, angles):
    with pytest.raises(ValueError):
        weighted_attack_angle(distances, angles)
