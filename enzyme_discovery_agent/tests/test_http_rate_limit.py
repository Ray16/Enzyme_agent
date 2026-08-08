"""Tests for the shared HTTP client's proactive per-host rate limiting.

The agent issues literature searches concurrently (a thread pool), so several
requests can hit the same host at once and burst into a 429 -- most acutely on
the keyless Semantic Scholar pool.  These tests verify that requests to a
throttled host are spaced by at least the configured interval, both across many
sequential calls and across a concurrent burst, while unthrottled hosts and
distinct hosts are never serialized against each other.

A virtual clock (patched ``time.monotonic``/``time.sleep``) makes the
deterministic spacing tests instant; one test uses real threads + real time to
prove the per-host lock actually serializes a concurrent burst.
"""

import threading
import time
import unittest
import urllib.request
from unittest import mock

from enzyme_discovery_agent.tools import _http


class _VirtualClock:
    """A fake monotonic clock: sleep advances it, monotonic reads it."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        if seconds > 0:
            self.t += seconds


class _FakeResponse:
    def __init__(self):
        self.content = b"{}"
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self):
        pass


def _reset_rate_state():
    _http._RATE_STATE.clear()
    _http._RATE_LOCKS.clear()


class TestRateLimiterUnit(unittest.TestCase):
    def setUp(self):
        _reset_rate_state()

    def test_sequential_calls_are_spaced_by_interval(self):
        clock = _VirtualClock()
        with mock.patch.object(_http.config, "HTTP_RATE_LIMITS", {"s2host": 1.0}), \
                mock.patch.object(_http.time, "monotonic", clock.monotonic), \
                mock.patch.object(_http.time, "sleep", clock.sleep):
            n = 5
            for _ in range(n):
                _http._respect_rate_limit("s2host")
        # First call does not wait; each subsequent call waits one interval.
        self.assertAlmostEqual(clock.t, (n - 1) * 1.0, places=6)

    def test_unthrottled_host_never_waits(self):
        clock = _VirtualClock()
        with mock.patch.object(_http.config, "HTTP_RATE_LIMITS", {"s2host": 1.0}), \
                mock.patch.object(_http.time, "monotonic", clock.monotonic), \
                mock.patch.object(_http.time, "sleep", clock.sleep):
            for _ in range(10):
                _http._respect_rate_limit("some.other.host")  # no fragment match
        self.assertEqual(clock.t, 0.0)
        # An unthrottled host must not even allocate a lock.
        self.assertNotIn("some.other.host", _http._RATE_LOCKS)

    def test_distinct_hosts_do_not_block_each_other(self):
        clock = _VirtualClock()
        limits = {"hostA": 1.0, "hostB": 1.0}
        with mock.patch.object(_http.config, "HTTP_RATE_LIMITS", limits), \
                mock.patch.object(_http.time, "monotonic", clock.monotonic), \
                mock.patch.object(_http.time, "sleep", clock.sleep):
            _http._respect_rate_limit("hostA")
            _http._respect_rate_limit("hostB")  # different host -> no wait
        self.assertEqual(clock.t, 0.0)


class TestRateLimiterThroughRequest(unittest.TestCase):
    def setUp(self):
        _reset_rate_state()

    def test_multiple_requests_same_host_are_throttled(self):
        clock = _VirtualClock()
        fake_session = mock.Mock()
        fake_session.request.return_value = _FakeResponse()

        with mock.patch.object(_http.config, "HTTP_RATE_LIMITS", {"api.semanticscholar.org": 1.0}), \
                mock.patch.object(_http, "_get_session", return_value=fake_session), \
                mock.patch.object(_http.time, "monotonic", clock.monotonic), \
                mock.patch.object(_http.time, "sleep", clock.sleep):
            for _ in range(4):
                req = urllib.request.Request("https://api.semanticscholar.org/graph/v1/paper/search?q=x")
                _http.request_with_retry(req)

        self.assertEqual(fake_session.request.call_count, 4)
        # 4 requests -> 3 enforced gaps of 1.0s each.
        self.assertAlmostEqual(clock.t, 3.0, places=6)


class TestRateLimiterConcurrentBurst(unittest.TestCase):
    """Real threads + real time: prove the per-host lock serializes a burst."""

    def setUp(self):
        _reset_rate_state()

    def test_concurrent_requests_to_same_host_are_serialized(self):
        interval = 0.05
        n_threads = 5
        fake_session = mock.Mock()
        fake_session.request.return_value = _FakeResponse()
        errors = []

        def worker():
            try:
                req = urllib.request.Request("https://burst.test/endpoint")
                _http.request_with_retry(req)
            except Exception as e:  # pragma: no cover - surfaced via assert below
                errors.append(e)

        with mock.patch.object(_http.config, "HTTP_RATE_LIMITS", {"burst.test": interval}), \
                mock.patch.object(_http, "_get_session", return_value=fake_session):
            threads = [threading.Thread(target=worker) for _ in range(n_threads)]
            start = time.monotonic()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.monotonic() - start

        self.assertEqual(errors, [])
        self.assertEqual(fake_session.request.call_count, n_threads)
        # A concurrent burst must still be spaced: >= (n-1) intervals of wall time.
        self.assertGreaterEqual(elapsed, (n_threads - 1) * interval * 0.9)


class _RaisingResponse:
    def __init__(self, exc):
        self.content = b""
        self.status_code = exc.response.status_code
        self.headers = {}
        self._exc = exc

    def raise_for_status(self):
        raise self._exc


class TestSecretRedaction(unittest.TestCase):
    def setUp(self):
        _reset_rate_state()

    def test_api_key_is_redacted_from_raised_error(self):
        import requests
        import urllib.error

        url_with_key = ("https://api.semanticscholar.org/graph/v1/paper/search"
                        "?query=x&api_key=SECRET123")
        exc = requests.exceptions.HTTPError(
            f"429 Client Error: Too Many Requests for url: {url_with_key}")
        exc.response = mock.Mock(status_code=429, headers={})
        fake_session = mock.Mock()
        fake_session.request.return_value = _RaisingResponse(exc)

        with mock.patch.object(_http.config, "HTTP_RATE_LIMITS", {}), \
                mock.patch.object(_http, "_get_session", return_value=fake_session):
            req = urllib.request.Request(url_with_key)
            with self.assertRaises(urllib.error.HTTPError) as cm:
                _http.request_with_retry(req)

        raised = cm.exception
        self.assertNotIn("SECRET123", str(raised))
        self.assertIn("api_key=***", str(raised))
        # the stored URL attribute must be scrubbed too
        self.assertNotIn("SECRET123", str(getattr(raised, "url", "") or ""))


if __name__ == "__main__":
    unittest.main()
