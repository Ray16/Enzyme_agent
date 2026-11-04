"""Unit tests for the async batch scheduler in esm_atlas_expand (fake transport)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import esm_atlas_expand as ex  # noqa: E402


class FakeServer:
    """A deterministic in-memory Foldseek server for the scheduler tests.

    Each seed becomes a ticket that reports PENDING for ``latency`` status probes,
    then COMPLETE. Tracks the peak number of simultaneously live tickets so the
    in-flight cap can be asserted.
    """

    def __init__(self, latency=2, fail_submit=(), fail_status=(), server_error=()):
        self.latency = latency
        self.fail_submit = set(fail_submit)
        self.fail_status = set(fail_status)
        self.server_error = set(server_error)
        self.polls = {}          # ticket -> times polled
        self.live = set()        # tickets submitted but not yet resolved
        self.peak_inflight = 0
        self._n = 0

    def submit(self, pdb_text):
        if pdb_text in self.fail_submit:
            raise RuntimeError("submit boom")
        self._n += 1
        ticket = f"T{self._n}:{pdb_text}"
        self.live.add(ticket)
        self.peak_inflight = max(self.peak_inflight, len(self.live))
        return ticket

    def status(self, ticket):
        pdb_text = ticket.split(":", 1)[1]
        if pdb_text in self.fail_status:
            raise RuntimeError("status boom")
        self.polls[ticket] = self.polls.get(ticket, 0) + 1
        if pdb_text in self.server_error:
            self.live.discard(ticket)
            return "ERROR"
        if self.polls[ticket] >= self.latency:
            self.live.discard(ticket)
            return "COMPLETE"
        return "PENDING"

    def results(self, ticket):
        pdb_text = ticket.split(":", 1)[1]
        return [{"target": f"MGYP_{pdb_text}", "prob": 1.0, "seqId": 0.4}]


class TestExpandMany(unittest.TestCase):
    def _run(self, seeds, server, **kw):
        return ex.expand_many(
            seeds, poll_s=0, timeout_s=1000,
            _submit=server.submit, _status=server.status, _results=server.results, **kw)

    def test_all_complete(self):
        server = FakeServer(latency=2)
        seeds = [(f"s{i}", f"pdb{i}") for i in range(5)]
        out = self._run(seeds, server, max_inflight=2)
        self.assertEqual(set(out), {f"s{i}" for i in range(5)})
        for rec in out.values():
            self.assertEqual(rec["status"], "COMPLETE")
            self.assertEqual(len(rec["neighbors"]), 1)

    def test_inflight_cap_respected(self):
        server = FakeServer(latency=3)
        seeds = [(f"s{i}", f"pdb{i}") for i in range(6)]
        self._run(seeds, server, max_inflight=2)
        self.assertLessEqual(server.peak_inflight, 2)

    def test_submit_failure_isolated(self):
        server = FakeServer(latency=1, fail_submit={"pdb_bad"})
        seeds = [("good", "pdb_ok"), ("bad", "pdb_bad")]
        out = self._run(seeds, server, max_inflight=2)
        self.assertEqual(out["good"]["status"], "COMPLETE")
        self.assertEqual(out["bad"]["status"], "SUBMIT_ERROR")

    def test_server_error_isolated(self):
        server = FakeServer(latency=1, server_error={"pdb_err"})
        seeds = [("ok", "pdb_ok"), ("err", "pdb_err")]
        out = self._run(seeds, server, max_inflight=2)
        self.assertEqual(out["ok"]["status"], "COMPLETE")
        self.assertEqual(out["err"]["status"], "ERROR")

    def test_poll_error_isolated(self):
        server = FakeServer(latency=2, fail_status={"pdb_p"})
        seeds = [("ok", "pdb_ok"), ("p", "pdb_p")]
        out = self._run(seeds, server, max_inflight=2)
        self.assertEqual(out["ok"]["status"], "COMPLETE")
        self.assertEqual(out["p"]["status"], "POLL_ERROR")

    def test_timeout(self):
        server = FakeServer(latency=99)  # never completes within deadline
        out = ex.expand_many(
            [("s", "pdb")], poll_s=0, timeout_s=-1,  # already past deadline on first check
            _submit=server.submit, _status=server.status, _results=server.results)
        self.assertEqual(out["s"]["status"], "TIMEOUT")


if __name__ == "__main__":
    unittest.main()
