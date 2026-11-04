"""Regression cases for direction, assay scope and hard acquisition eligibility."""
import math
import pytest
from structural_modeling_agent.loop.calibration import CalibrationLedger, CalibrationRecord
from structural_modeling_agent.loop.reward import combine_reward
from structural_modeling_agent.loop.acquisition import Candidate, select_batch


def test_opposite_direction_never_increases_trust():
    ledger = CalibrationLedger([CalibrationRecord(str(i), i / 19, 1 - i / 19, family="synthetic") for i in range(20)])
    assert ledger.correlation("synthetic")["spearman"] == -1
    result = ledger.calibrated_proxy(.9, "synthetic")
    assert result["confidence"] == 0 and result["estimate"] == .5
    assert result["status"] == "untrusted"
    assert result["uncertainty_kind"] == "heuristic_lack_of_trust_not_prediction_interval"
    assert result["validation_status"] == "not_established_by_this_ledger"


def test_versions_conditions_and_duplicate_enzymes_are_separate():
    ledger = CalibrationLedger([
        CalibrationRecord("a", .1, 1, "F", "v1", conditions={"endpoint": "x"}),
        CalibrationRecord("a", .1, 1, "F", "v1", conditions={"endpoint": "x"}),
        CalibrationRecord("b", .9, 9, "F", "v1", conditions={"endpoint": "x"}),
        CalibrationRecord("c", .9, 0, "F", "v2", conditions={"endpoint": "x"}),
        CalibrationRecord("d", .9, 0, "F", "v1", conditions={"endpoint": "y"}),
    ])
    assert ledger.correlation("F")["n"] == 0
    scope = dict(proxy_version="v1", conditions={"endpoint": "x"})
    assert ledger.correlation("F", **scope)["n"] == 2
    assert ledger.correlation("F", **scope)["spearman"] == pytest.approx(1)
    assert ledger.surrogate_uncertainty(None, **scope) == 1


def test_veto_cannot_be_rescued_by_exploration():
    reward = combine_reward(mechanism=1, proxy=1, developability=0)
    assert reward.reward == 0 and reward.eligible is False
    candidates = [Candidate("veto", reward.reward, 1, 1), Candidate("good", .1),
                  Candidate("explicit_veto", .9, 1, 1, eligible=False)]
    assert [c.id for c in select_batch(candidates, 3)] == ["good"]


def test_invalid_selection_inputs_fail_closed():
    with pytest.raises(ValueError):
        select_batch([Candidate("bad", math.nan)], 1)
    with pytest.raises(ValueError):
        select_batch([Candidate("a", .2), Candidate("a", .3)], 2)
    with pytest.raises(ValueError):
        select_batch([], -1)
    with pytest.raises(ValueError):
        CalibrationLedger().calibrated_proxy(120, "synthetic")


def test_small_positive_reward_is_not_rounded_into_a_veto():
    result = combine_reward(mechanism=.0001, proxy=.0001, developability=.0001)
    assert result.eligible is True and result.reward > 0
    assert select_batch([Candidate("small", result.reward, eligible=result.eligible)], 1)[0].id == "small"
