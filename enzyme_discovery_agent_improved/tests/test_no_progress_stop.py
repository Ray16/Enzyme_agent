"""The loop must not spin on an idle model.

Regression: `_should_continue` returns True while the candidate count is below
its exploration threshold, so once discovery plateaus the model makes no tool
calls but the loop re-invites it every iteration to the mid-run boundary — ~35
empty turns that dominated a run's token cost. `_no_progress_stop` lets the
model's own decision (no tool call) stand after one recovery nudge, once it has
results.
"""
import types
import unittest

from enzyme_discovery_agent_improved import config
from enzyme_discovery_agent_improved.agent import EnzymeAgent


def _stop(consecutive, candidates):
    a = types.SimpleNamespace(_consecutive_no_tool_turns=consecutive,
                              _candidate_count=candidates)
    return EnzymeAgent._no_progress_stop(a)


class TestNoProgressStop(unittest.TestCase):
    def test_first_silent_turn_gets_a_recovery_nudge(self):
        # one no-tool turn -> not yet stalled (give the model a chance)
        self.assertFalse(_stop(1, 19))

    def test_second_silent_turn_with_results_stops(self):
        self.assertTrue(_stop(config.MAX_NO_PROGRESS_TURNS, 19))
        self.assertTrue(_stop(config.MAX_NO_PROGRESS_TURNS + 5, 19))

    def test_no_results_keeps_nudging(self):
        # a run that never produced a candidate must not finalize empty here;
        # it keeps being nudged to begin.
        self.assertFalse(_stop(config.MAX_NO_PROGRESS_TURNS + 3, 0))

    def test_active_model_never_stalls(self):
        # tool-calling turns reset the counter to 0 -> never stalled
        self.assertFalse(_stop(0, 19))


if __name__ == "__main__":
    unittest.main()
