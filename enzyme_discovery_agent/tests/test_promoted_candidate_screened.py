"""Integration test: a substrate-fit-PROMOTED candidate must actually reach
evaluation, not stall at 'discovered'.

The family survey promotes the members whose native substrate resembles the
target.  That is worthless if the screening budget then drops them.  This test
patches the network/allocation seams of ``auto_screen_candidates`` and asserts
that a promoted candidate (tagged ``family_substrate_fit:*``) is (a) forwarded as
PROTECTED to the fold-relevance allocator and (b) screened — i.e. it flows into
the evaluation pipeline.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_discovery_agent.gates.ec_triage import auto_screen_candidates
from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry,
    update_candidate_registry,
)


class TestPromotedCandidateScreened(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp())
        # One literature-named candidate and one substrate-fit-promoted candidate.
        update_candidate_registry(self.run_dir, "P_LIT", "discovered",
                                  metadata={"literature_name": "namedEnz"})
        update_candidate_registry(self.run_dir, "P_FIT", "discovered",
                                  metadata={"source_tool": "family_substrate_fit:IPR053158",
                                            "substrate_fit": 0.66,
                                            "protein_name": "fit ligase"})

    def test_promoted_candidate_is_protected_and_screened(self):
        captured = {}

        def fake_allocate(run_dir, reps, protected=None, **kw):
            captured["protected"] = set(protected or set())
            captured["reps"] = {a for a, _ in reps}
            # Simulate an allocator that would drop everything (empty selection),
            # so ONLY protection can carry a candidate through.  Protected accs are
            # always selected by the real allocator, so return them.
            return set(protected or set()), []

        def fake_light(acc):
            return {"protein_name": "x", "organism": "o", "ec_numbers": [],
                    "catalytic_activities": [], "interpro_ids": ["IPR053158"],
                    "seq_length": 400}

        with mock.patch(
            "enzyme_discovery_agent.gates.family_relevance.build_relevance_and_allocate",
            side_effect=fake_allocate,
        ), mock.patch(
            "enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light",
            side_effect=fake_light,
        ):
            auto_screen_candidates(self.run_dir)

        # The promoted candidate was forwarded as PROTECTED alongside the named one.
        self.assertIn("P_FIT", captured["protected"])
        self.assertIn("P_LIT", captured["protected"])

        # And it actually advanced to 'screened' (enters evaluation).
        reg = load_candidate_registry(self.run_dir)["candidates"]
        statuses = {s["status"] for s in reg["P_FIT"]["statuses"]}
        self.assertIn("screened", statuses)


if __name__ == "__main__":
    unittest.main()
