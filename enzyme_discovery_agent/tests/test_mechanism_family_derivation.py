"""Mechanism-family derivation must broaden across samples to convergence.

Guard for the observed gap: one LLM sample enumerated 5 folds and derivation
stopped there, dropping the fold whose native chemistry was closest to the
target.  Derivation must keep sampling (each attempt asked for ADDITIONAL folds)
and union until the model stops offering new ones — not halt at a low floor.
"""

from __future__ import annotations

import json
import unittest

from enzyme_discovery_agent.tools.mechanism_families import derive_mechanism_families


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _Resp:
    def __init__(self, content):
        self.choices = [_Msg(content)]


class _SlicedLLM:
    """Returns a different slice of the fold space on each call, then dries up —
    simulating one-sample-sees-only-part-of-the-space."""

    def __init__(self, slices):
        self.slices = slices
        self.calls = 0

    def call(self, messages=None, **kwargs):
        i = self.calls
        self.calls += 1
        fams = self.slices[i] if i < len(self.slices) else []
        payload = {"families": [
            {"name": n, "descriptors": [f"{n} fold domain"], "literature_terms": [f"{n} synthesis"]}
            for n in fams
        ]}
        return _Resp(json.dumps(payload))


class TestDerivationBroadens(unittest.TestCase):
    def test_unions_across_samples_until_convergence(self):
        # First sample offers 3 folds; only by continuing do folds D and E appear.
        llm = _SlicedLLM([["A", "B", "C"], ["D"], ["E"], []])
        fams = derive_mechanism_families("form an amide bond", "", llm,
                                         min_families=4, max_families=10, max_attempts=6)
        names = {f["name"] for f in fams}
        self.assertEqual(names, {"A", "B", "C", "D", "E"})
        # It kept sampling past the 4-family floor and stopped once a sample was dry.
        self.assertGreaterEqual(llm.calls, 4)

    def test_does_not_exceed_max_families(self):
        llm = _SlicedLLM([[f"F{i}" for i in range(6)], [f"G{i}" for i in range(6)]])
        fams = derive_mechanism_families("x", "", llm, max_families=8, max_attempts=6)
        self.assertLessEqual(len(fams), 8)

    def test_empty_when_model_yields_nothing(self):
        self.assertEqual(derive_mechanism_families("x", "", _SlicedLLM([[], [], []])), [])


if __name__ == "__main__":
    unittest.main()
