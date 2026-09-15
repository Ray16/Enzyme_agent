"""Prompt-history pruning bounds the re-sent prompt WITHOUT losing information
the agent needs going forward.

Guarantees pinned here:
  * old tool results are stubbed, recent ones kept full;
  * old assistant NARRATION is stubbed but its tool_calls are preserved (pairing);
  * recent assistant turns are kept full;
  * only the latest state header survives (each is a superseded cumulative snapshot);
  * non-state user messages (gate guidance) are never touched.
"""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from enzyme_discovery_agent_improved.agent import EnzymeAgent

BIG = "x" * 2000  # exceeds HISTORY_STUB_OVER_CHARS


def _prune(messages, **kw):
    # _prune_history does not use self; call it with a throwaway.
    EnzymeAgent._prune_history(types.SimpleNamespace(), messages, **kw)
    return messages


class TestHistoryPruning(unittest.TestCase):
    def test_old_tool_results_stubbed_recent_kept(self):
        msgs = [{"role": "tool", "tool_call_id": str(i), "content": BIG} for i in range(5)]
        _prune(msgs, keep_recent_tools=2)
        self.assertTrue(msgs[0]["content"].startswith("[elided older tool result"))
        self.assertEqual(msgs[-1]["content"], BIG)  # recent kept full

    def test_old_assistant_text_stubbed_but_tool_calls_preserved(self):
        calls = [{"id": "c1", "type": "function",
                  "function": {"name": "search", "arguments": "{}"}}]
        msgs = [
            {"role": "assistant", "content": BIG, "tool_calls": calls},   # old
            {"role": "assistant", "content": BIG, "tool_calls": calls},
            {"role": "assistant", "content": BIG, "tool_calls": calls},   # recent
        ]
        _prune(msgs, keep_recent_assistant=1)
        self.assertTrue(msgs[0]["content"].startswith("[earlier reasoning elided"))
        self.assertEqual(msgs[0]["tool_calls"], calls)   # pairing preserved
        self.assertEqual(msgs[-1]["content"], BIG)       # recent kept full

    def test_only_latest_state_header_survives(self):
        msgs = [
            {"role": "user", "content": "[STATE: iter=1/100, candidates=0]"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "[STATE: iter=2/100, candidates=3]"},
            {"role": "user", "content": "[STATE: iter=3/100, candidates=7]"},
        ]
        _prune(msgs)
        self.assertEqual(msgs[0]["content"], "[state superseded]")
        self.assertEqual(msgs[2]["content"], "[state superseded]")
        self.assertEqual(msgs[3]["content"], "[STATE: iter=3/100, candidates=7]")

    def test_non_state_user_guidance_is_untouched(self):
        guidance = "LITERATURE HARVEST registered 5 seeds. Expand around each. " + BIG
        msgs = [
            {"role": "user", "content": "[STATE: iter=1/100, candidates=0]"},
            {"role": "user", "content": guidance},
            {"role": "user", "content": "[STATE: iter=2/100, candidates=1]"},
        ]
        _prune(msgs)
        self.assertEqual(msgs[1]["content"], guidance)  # gate guidance preserved

    def test_short_messages_not_stubbed(self):
        msgs = [{"role": "tool", "tool_call_id": str(i), "content": "small"}
                for i in range(5)]
        _prune(msgs, keep_recent_tools=1)
        self.assertTrue(all(m["content"] == "small" for m in msgs))

    def test_idempotent(self):
        msgs = [{"role": "assistant", "content": BIG, "tool_calls": []} for _ in range(5)]
        _prune(msgs, keep_recent_assistant=1)
        once = [m["content"] for m in msgs]
        _prune(msgs, keep_recent_assistant=1)
        self.assertEqual([m["content"] for m in msgs], once)


if __name__ == "__main__":
    unittest.main()
