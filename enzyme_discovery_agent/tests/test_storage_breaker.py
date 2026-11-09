"""An exhausted volume must stop the run, not degrade into survivable warnings.

Run 20261106_165013 kept working for twenty minutes after its filesystem filled:
every write raised ENOSPC, each caller logged a warning and continued, and the
remaining token allowance bought results that could never be saved.
"""

import errno
import json
import types
from unittest.mock import patch

import pytest

from enzyme_discovery_agent import state_store


@pytest.fixture(autouse=True)
def _clean():
    state_store.clear_storage_fault()
    yield
    state_store.clear_storage_fault()


def _full_volume(code):
    def raise_full(*_args, **_kwargs):
        raise OSError(code, "No space left on device")
    return raise_full


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EROFS])
def test_an_unwritable_volume_is_recorded_once_and_leaves_no_partial_state(tmp_path, code):
    with patch.object(state_store.os, "fsync", _full_volume(code)):
        with pytest.raises(OSError):
            state_store.atomic_write_json(tmp_path / "state.json", {"keep": 1})
    fault = state_store.storage_fault()
    assert fault["errno"] == code
    assert fault["path"].endswith("state.json")
    # The previous snapshot must never be replaced by a truncated one, and the
    # temporary file must not survive to consume more of a full volume.
    assert not (tmp_path / "state.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_the_first_fault_is_kept_rather_than_overwritten_by_later_ones(tmp_path):
    with patch.object(state_store.os, "fsync", _full_volume(errno.ENOSPC)):
        for name in ("first.json", "second.json"):
            with pytest.raises(OSError):
                state_store.atomic_write_json(tmp_path / name, {})
    assert state_store.storage_fault()["path"].endswith("first.json")


def test_an_ordinary_write_error_is_not_treated_as_an_exhausted_volume(tmp_path):
    def permission_denied(*_a, **_k):
        raise OSError(errno.EACCES, "Permission denied")
    with patch.object(state_store.os, "fsync", permission_denied):
        with pytest.raises(OSError):
            state_store.atomic_write_json(tmp_path / "state.json", {})
    assert state_store.storage_fault() == {}


def test_the_controller_stops_and_stops_spending_once_state_cannot_be_saved():
    from enzyme_discovery_agent.agent import EnzymeAgent
    agent = types.SimpleNamespace(_budget_reason=EnzymeAgent._budget_reason.__get__(
        types.SimpleNamespace(llm=None, _run_token_usage=None, token_budget=None)))
    assert agent._budget_reason() is None
    state_store._STORAGE_FAULT.update(errno=errno.ENOSPC, strerror="No space left on device",
                                      path="/run/agent_state/candidate_registry.json")
    reason = agent._budget_reason()
    assert reason and "cannot be persisted" in reason
    assert "candidate_registry.json" in reason
    # The same reason is what the LLM client consults, so no further request is
    # issued against an allowance whose results could not be written down.
    assert isinstance(reason, str)
