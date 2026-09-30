"""Event-stream durability across an interrupted run (writer side).

``read_events`` skipping a malformed line only protects the *reader*.  The writer
appends, so an unterminated fragment left by a killed process would absorb the
next event -- losing a real event and duplicating its ``seq``.  These cover the
repair, which is what makes a resumed run's event stream trustworthy.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.models.event_log import (
    _seq_counters,
    append_event,
    read_events,
)


class TruncatedTailRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.run_dir = Path(self.tmp)
        self.path = self.run_dir / "events.jsonl"

    def _fresh_process(self):
        """Forget the in-process seq counter, as a resume in a new process would."""
        _seq_counters.pop(str(self.run_dir.resolve()), None)

    def _interrupt_mid_write(self):
        with self.path.open("a") as f:
            f.write('{"seq":3,"type":"gate","gate":"interrupted')

    def test_event_after_resume_is_not_absorbed_by_a_truncated_tail(self):
        append_event(self.run_dir, "gate", gate="first")
        append_event(self.run_dir, "gate", gate="second")
        self._interrupt_mid_write()
        self._fresh_process()

        append_event(self.run_dir, "gate", gate="after_resume")

        gates = [e.get("gate") for e in read_events(self.run_dir)]
        self.assertIn("after_resume", gates, "post-resume event was absorbed by the fragment")
        self.assertEqual(gates, ["first", "second", "after_resume"])

    def test_seq_stays_unique_across_the_interruption(self):
        append_event(self.run_dir, "gate", gate="first")
        self._interrupt_mid_write()
        self._fresh_process()
        append_event(self.run_dir, "gate", gate="a")
        append_event(self.run_dir, "gate", gate="b")

        seqs = [e["seq"] for e in read_events(self.run_dir)]
        self.assertEqual(len(seqs), len(set(seqs)), f"duplicate seq in {seqs}")
        self.assertEqual(seqs, sorted(seqs), "seq must stay monotonic across a resume")

    def test_interrupted_fragment_is_preserved_not_trimmed(self):
        """Append-only contract: the damaged line stays on disk for audit."""
        append_event(self.run_dir, "gate", gate="first")
        self._interrupt_mid_write()
        self._fresh_process()
        append_event(self.run_dir, "gate", gate="after_resume")

        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("interrupted", lines[1])
        with self.assertRaises(json.JSONDecodeError):
            json.loads(lines[1])

    def test_repair_is_idempotent_and_leaves_clean_files_alone(self):
        append_event(self.run_dir, "gate", gate="first")
        before = self.path.read_text()
        for _ in range(3):
            self._fresh_process()
            read_events(self.run_dir)
        self._fresh_process()
        self.assertEqual(self.path.read_text(), before, "clean file must not be rewritten")

        append_event(self.run_dir, "gate", gate="second")
        self.assertEqual([e.get("gate") for e in read_events(self.run_dir)],
                         ["first", "second"])

    def test_empty_and_missing_files_are_handled(self):
        self.assertEqual(read_events(self.run_dir), [])   # missing
        self.path.write_text("")
        self._fresh_process()
        self.assertEqual(read_events(self.run_dir), [])   # empty
        append_event(self.run_dir, "gate", gate="only")
        self.assertEqual([e.get("gate") for e in read_events(self.run_dir)], ["only"])


if __name__ == "__main__":
    unittest.main()
