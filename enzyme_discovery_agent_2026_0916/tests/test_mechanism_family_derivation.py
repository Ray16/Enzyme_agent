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
        self.assertTrue(all(f["validation_status"] == "hypothesis" for f in fams))
        # It kept sampling past the 4-family floor and stopped once a sample was dry.
        self.assertGreaterEqual(llm.calls, 4)

    def test_does_not_exceed_max_families(self):
        llm = _SlicedLLM([[f"F{i}" for i in range(6)], [f"G{i}" for i in range(6)]])
        fams = derive_mechanism_families("x", "", llm, max_families=8, max_attempts=6)
        self.assertLessEqual(len(fams), 8)

    def test_empty_when_model_yields_nothing(self):
        self.assertEqual(derive_mechanism_families("x", "", _SlicedLLM([[], [], []])), [])


class _AuditLLM:
    """Enumeration converges on a partial set; the COMPLETENESS-AUDIT call (a
    distinct system prompt) surfaces a fold the first pass missed.  Optionally the
    audit also proposes a wrong-direction fold that the constraint filter drops."""

    def __init__(self, enum_slices, audit_families):
        self.enum_slices = enum_slices
        self.audit_families = audit_families
        self.enum_calls = 0
        self.audit_calls = 0

    def call(self, messages=None, **kwargs):
        system = (messages or [{}])[0].get("content", "")
        if "COMPLETENESS AUDIT" in system:
            self.audit_calls += 1
            payload = {"families": self.audit_families}
        else:
            i = self.enum_calls
            self.enum_calls += 1
            fams = self.enum_slices[i] if i < len(self.enum_slices) else []
            payload = {"families": [
                {"name": n, "descriptors": [f"{n} fold domain"],
                 "literature_terms": [f"{n} synthesis"], "native_direction": "formation",
                 "substrates_free": True} for n in fams]}
        return _Resp(json.dumps(payload))


class TestCompletenessAudit(unittest.TestCase):
    def test_audit_adds_missing_fold(self):
        # First pass converges on {A,B,C}; the audit reaches a fold it overlooked.
        missing = {"name": "Overlooked", "descriptors": ["Overlooked fold domain"],
                   "literature_terms": ["Overlooked native synthesis"],
                   "native_direction": "formation", "substrates_free": True}
        llm = _AuditLLM([["A", "B", "C"], []], [missing])
        fams = derive_mechanism_families("form an amide bond", "", llm,
                                         min_families=2, max_families=10, max_attempts=6)
        names = {f["name"] for f in fams}
        self.assertIn("Overlooked", names)
        self.assertEqual(self.__audit_basis(fams, "Overlooked"),
                         "model_reasoning_completeness_audit")
        # The audit SAMPLES TO CONVERGENCE (a guard must not be one fragile draw):
        # one productive pass adds the fold, the next pass is dry and stops it.
        self.assertEqual(llm.audit_calls, 2)

    def test_audit_family_still_direction_filtered(self):
        # The audit must not smuggle in a wrong-direction (native-cleavage) fold.
        bad = {"name": "Hydrolase", "descriptors": ["Hydrolase fold domain"],
               "literature_terms": ["Hydrolase cleavage"],
               "native_reaction": "amide + H2O -> acid + amine",
               "native_direction": "cleavage", "substrates_free": True}
        llm = _AuditLLM([["A", "B", "C"], []], [bad])
        fams = derive_mechanism_families("form an amide bond", "", llm,
                                         min_families=2, max_families=10, max_attempts=6,
                                         target_direction="formation")
        self.assertNotIn("Hydrolase", {f["name"] for f in fams})

    def test_reflect_can_be_disabled(self):
        llm = _AuditLLM([["A", "B", "C"], []],
                        [{"name": "X", "descriptors": ["X d"], "native_direction": "formation",
                          "substrates_free": True}])
        derive_mechanism_families("x", "", llm, min_families=2, max_attempts=6, reflect=False)
        self.assertEqual(llm.audit_calls, 0)

    @staticmethod
    def __audit_basis(fams, name):
        return next(f for f in fams if f["name"] == name)["evidence_basis"]


if __name__ == "__main__":
    unittest.main()
