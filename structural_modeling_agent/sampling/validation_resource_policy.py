"""Opt-in pytest resource policy for upstream ML/MM integration checks.

Upstream MLMMCore defaults to sixteen MM threads. The production adapter and
these integration checks explicitly use one. Load with pytest -p and record it
with the test invocation; this does not modify the upstream checkout.
"""
import functools
import pytest


@pytest.fixture(autouse=True)
def bounded_mlmm_resources(monkeypatch):
    from mlmm.backends.mlmm_calc import MLMMCore
    original = MLMMCore.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        kwargs["mm_threads"] = 1
        kwargs["xtb_ncores"] = 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MLMMCore, "__init__", init)
