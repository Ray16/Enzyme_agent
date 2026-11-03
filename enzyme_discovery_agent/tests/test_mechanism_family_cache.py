"""Cross-run cache for mechanism-family derivation.

The derivation samples the LLM to convergence (~7 min in profiling); caching it
keyed on the reaction contract means a repeat run of the same target skips the LLM
entirely.  These tests prove miss->derive+write and hit->return-without-LLM using a
fake derivation, so they need no live run.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from enzyme_discovery_agent.tools import mechanism_families as mf


class TestMechanismFamilyCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._env = mock.patch.dict(os.environ, {"ENZYME_AGENT_CACHE_DIR": self.tmp})
        self._env.start()
        self._cache = mock.patch.object(mf.config, "ENABLE_MECHANISM_FAMILY_CACHE", True)
        self._cache.start()

    def tearDown(self):
        self._env.stop()
        self._cache.stop()

    def _derive(self, query, smiles, calls):
        def fake_once(*a, **k):
            calls["n"] += 1
            return [{"name": "FamA"}, {"name": "FamB"},
                    {"name": "FamC"}, {"name": "FamD"}]
        with mock.patch.object(mf, "_derive_once", side_effect=fake_once), \
                mock.patch.object(mf, "family_passes_constraints", return_value=(True, "")):
            return mf.derive_mechanism_families(query, smiles, object(), reflect=False)

    def test_miss_then_hit_skips_llm(self):
        calls = {"n": 0}
        fams1 = self._derive("form a bond in a molecule", "CCO", calls)
        self.assertTrue(fams1)
        self.assertGreater(calls["n"], 0)              # first time: actually derived
        n_after_first = calls["n"]

        fams2 = self._derive("form a bond in a molecule", "CCO", calls)
        self.assertEqual(fams2, fams1)                 # same result
        self.assertEqual(calls["n"], n_after_first)    # cache hit: no further LLM derivation

    def test_different_contract_misses(self):
        calls = {"n": 0}
        self._derive("form a bond in a molecule", "CCO", calls)
        n_after_first = calls["n"]
        self._derive("form a bond in a molecule", "CCCCCO", calls)  # different substrate
        self.assertGreater(calls["n"], n_after_first)  # different contract -> derived again

    def test_cache_disabled_always_derives(self):
        calls = {"n": 0}
        with mock.patch.object(mf.config, "ENABLE_MECHANISM_FAMILY_CACHE", False):
            self._derive("form a bond in a molecule", "CCO", calls)
            n_after_first = calls["n"]
            self._derive("form a bond in a molecule", "CCO", calls)
            self.assertGreater(calls["n"], n_after_first)  # no caching -> derived twice

    def test_changed_retrieved_evidence_invalidates_an_opted_in_cache(self):
        calls = []
        def derive(query, *args, **kwargs):
            calls.append(query)
            return [{"name": "Family A"}]
        with mock.patch.object(mf, "_derive_once", side_effect=derive), \
             mock.patch.object(mf, "family_passes_constraints", return_value=(True, "")):
            mf.derive_mechanism_families("target", "CCO", object(), reflect=False, max_attempts=1,
                                         retrieved_papers=[{"title": "First evidence", "abstract": "Observed activity"}])
            first_count = len(calls)
            mf.derive_mechanism_families("target", "CCO", object(), reflect=False, max_attempts=1,
                                         retrieved_papers=[{"title": "Revised evidence", "abstract": "Different activity"}])
        self.assertGreater(len(calls), first_count)
        self.assertIn("Revised evidence", calls[-1])


if __name__ == "__main__":
    unittest.main()
