"""Independent numerical checks for the established ensemble ranking metric."""

import pytest

from structural_modeling_agent.proxies.attack_angle import near_attack_mode, weighted_attack_angle


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


def test_near_attack_mode_ignores_distant_poses_for_the_angle():
    # Four docked poses near 115 deg and four drifted poses near 85 deg (meaningless orientation).
    distances = [3.0, 3.2, 3.4, 3.6, 8.0, 8.5, 9.0, 9.5]
    angles = [113., 115., 117., 115., 85., 88., 82., 86.]
    result = near_attack_mode(distances, angles, threshold=4.0)
    assert result["n_near_attack_poses"] == 4
    assert result["near_attack_fraction"] == pytest.approx(0.5)
    # Angle reflects only the near mode, not the full ensemble which would be dragged down.
    assert result["near_mode_weighted_angle_deg"] == pytest.approx(115., abs=1.5)
    assert result["near_mode_harmonic_distance_A"] < 4.0
    full = weighted_attack_angle(distances, angles)["weighted_attack_angle_deg"]
    assert result["near_mode_weighted_angle_deg"] > full + 5


def test_near_attack_mode_without_a_populated_mode_reports_no_angle():
    result = near_attack_mode([8., 9., 10.], [100., 90., 95.], threshold=4.0)
    assert result["near_attack_fraction"] == 0.0
    assert result["n_near_attack_poses"] == 0
    assert result["near_mode_weighted_angle_deg"] is None
    assert result["near_mode_harmonic_distance_A"] is None


def test_near_attack_mode_rejects_a_nonpositive_threshold():
    with pytest.raises(ValueError):
        near_attack_mode([3.], [110.], threshold=0.)
