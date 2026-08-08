"""Tests for the evidence-to-candidate selection transition."""

import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.gates import AgentState
from enzyme_discovery_agent.gates.evidence_selection import EvidenceSelectionGate
from enzyme_discovery_agent import config


class TestEvidenceSelectionGate(unittest.TestCase):
    def test_forces_bounded_screening_after_sufficient_literature(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(
                run_dir=Path(tmp),
                accumulated_papers=[{"title": f"paper {i}"} for i in range(config.EVIDENCE_SELECTION_SEARCH_THRESHOLD)],
                lit_search_count=config.MIN_LITERATURE_SEARCH_ROUNDS,
            )
            gate = EvidenceSelectionGate()

            self.assertTrue(gate.should_fire(state))
            result = gate.execute(state)

        self.assertTrue(result.intercept)
        self.assertTrue(result.state_updates["evidence_selection_prompted"])
        self.assertIn("screen_candidates", result.new_messages[0]["content"])

    def test_does_not_interrupt_after_candidate_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = AgentState(
                run_dir=Path(tmp), candidate_count=1,
                accumulated_papers=[{"title": f"paper {i}"} for i in range(config.EVIDENCE_SELECTION_SEARCH_THRESHOLD)],
            )
            self.assertFalse(EvidenceSelectionGate().should_fire(state))
