"""Regression tests for GatePipeline state propagation.

A gate that declines to intercept can still report a state transition. If the
pipeline drops those updates, one-time redirect flags are never cleared and a
downstream gate whose should_fire depends on them is locked out for the rest of
the run.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from enzyme_discovery_agent.gates import AgentState, GatePipeline, GateResult


class _RecordingGate:
    """Minimal gate stub that records how many times it executed."""

    def __init__(self, fires, result, name="gate"):
        self._fires = fires
        self._result = result
        self.name = name
        self.execute_count = 0

    def should_fire(self, state, has_eval_calls, messages):
        return self._fires(state)

    def execute(self, state, messages):
        self.execute_count += 1
        return self._result


class TestGatePipelineStatePropagation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = AgentState(run_dir=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_intercepting_updates_are_returned(self):
        gate = _RecordingGate(
            fires=lambda state: True,
            result=GateResult(
                intercept=False,
                state_updates={"discovery_coverage_incomplete": False,
                               "discovery_coverage_done": True},
            ),
        )
        result = GatePipeline([gate]).check(self.state, False, [])

        self.assertIsNotNone(result)
        self.assertFalse(result.intercept)
        self.assertEqual(
            result.state_updates,
            {"discovery_coverage_incomplete": False, "discovery_coverage_done": True},
        )

    def test_non_intercepting_updates_are_visible_to_later_gates(self):
        """Coverage clearing its flag must unlock the triage gate in the same pass."""
        coverage = _RecordingGate(
            fires=lambda state: True,
            result=GateResult(
                intercept=False,
                state_updates={"discovery_coverage_incomplete": False},
            ),
            name="coverage",
        )
        triage = _RecordingGate(
            fires=lambda state: not state.discovery_coverage_incomplete,
            result=GateResult(intercept=True, log_message="triage fired"),
            name="triage",
        )
        self.state.discovery_coverage_incomplete = True

        result = GatePipeline([coverage, triage]).check(self.state, False, [])

        self.assertIsNotNone(result)
        self.assertTrue(result.intercept)
        self.assertEqual(triage.execute_count, 1)

    def test_intercepting_gate_preserves_earlier_updates(self):
        first = _RecordingGate(
            fires=lambda state: True,
            result=GateResult(intercept=False, state_updates={"entity_extraction_done": True}),
        )
        second = _RecordingGate(
            fires=lambda state: True,
            result=GateResult(intercept=True, state_updates={"eval_gate_fired": True}),
        )
        result = GatePipeline([first, second]).check(self.state, False, [])

        self.assertTrue(result.intercept)
        self.assertEqual(
            result.state_updates,
            {"entity_extraction_done": True, "eval_gate_fired": True},
        )

    def test_returns_none_when_nothing_fires(self):
        gate = _RecordingGate(fires=lambda state: False, result=GateResult(intercept=True))
        self.assertIsNone(GatePipeline([gate]).check(self.state, False, []))
        self.assertEqual(gate.execute_count, 0)

    def test_declining_gate_does_not_re_execute_once_flag_applied(self):
        """ECTriage-style gate must not re-run auto-screening every iteration."""
        gate = _RecordingGate(
            fires=lambda state: not state.eval_gate_fired,
            result=GateResult(intercept=False, state_updates={"eval_gate_fired": True}),
        )
        pipeline = GatePipeline([gate])

        first = pipeline.check(self.state, False, [])
        for key, value in first.state_updates.items():
            setattr(self.state, key, value)
        second = pipeline.check(self.state, False, [])

        self.assertEqual(gate.execute_count, 1)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
