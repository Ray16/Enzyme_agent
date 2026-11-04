"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_substrate_disk_cache(monkeypatch):
    """Disable the persistent substrate-property disk cache during tests.

    compute_substrate_properties memoizes to a shared on-disk cache
    (~/.cache/enzyme_discovery_agent/substrate_props) so unique molecules embed
    once across runs. In tests that would leak state between runs -- e.g. a
    molecule cached with a real 3D conformer would mask a test that forces an
    embedding failure. Point the cache at nothing so every test computes fresh.
    """
    from enzyme_discovery_agent.tools import substrate
    monkeypatch.setattr(substrate, "_DISK_CACHE_DIR", None)
