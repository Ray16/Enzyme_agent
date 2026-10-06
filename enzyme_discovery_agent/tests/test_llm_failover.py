"""Availability failover in LLMClient: sticky-per-run across ordered backends,
with argo's HTTP-200 block/limit notice treated as a failure (not an answer)."""

from __future__ import annotations

import types
import unittest
from unittest import mock

from enzyme_discovery_agent import llm_client


def _resp(content):
    msg = types.SimpleNamespace(content=content, tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)


class _FakeClient:
    """Minimal stand-in for an OpenAI client: with_options().chat.completions.create()."""
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = 0

    def with_options(self, **_):
        return self

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        return self._behavior(self, kwargs)


_ARGO = "argo://x"
_OR = "or://y"

_BACKENDS = [
    {"name": "argo", "base_url": _ARGO, "api_key": "k1", "model": "argo-model"},
    {"name": "openrouter", "base_url": _OR, "api_key": "k2", "model": "or-model"},
]


class FailoverTest(unittest.TestCase):
    def _make(self, argo_behavior, or_behavior):
        self.clients = {}

        def factory(base_url=None, api_key=None):
            beh = argo_behavior if base_url == _ARGO else or_behavior
            c = _FakeClient(beh)
            self.clients[base_url] = c
            return c

        with mock.patch.object(llm_client.config, "llm_backends", return_value=[dict(b) for b in _BACKENDS]), \
                mock.patch.object(llm_client, "OpenAI", factory):
            return llm_client.LLMClient(model="argo-model")

    def test_primary_success_stays_on_primary(self):
        client = self._make(lambda c, k: _resp("hello from argo"), lambda c, k: _resp("or"))
        r = client.call([{"role": "user", "content": "hi"}], max_retries=1)
        self.assertEqual(r.choices[0].message.content, "hello from argo")
        self.assertEqual(client.last_backend, "argo")
        self.assertEqual(client._active_idx, 0)
        self.assertEqual(self.clients[_OR].calls, 0)   # fallback never touched

    def test_argo_block_notice_fails_over(self):
        # argo returns a 200 whose body is the quota/revoked NOTICE.
        blocked = lambda c, k: _resp("⚠️ IMPORTANT USAGE NOTICE FROM ARGO: ACCESS REVOKED - usage limit has been exceeded")
        client = self._make(blocked, lambda c, k: _resp("served by openrouter"))
        r = client.call([{"role": "user", "content": "hi"}], max_retries=1)
        self.assertEqual(r.choices[0].message.content, "served by openrouter")
        self.assertEqual(client.last_backend, "openrouter")
        self.assertEqual(client._active_idx, 1)
        # Request accounting counts the real HTTP attempt on each backend.
        self.assertEqual(client.request_counts.get("argo"), 1)
        self.assertEqual(client.request_counts.get("openrouter"), 1)

    def test_sticky_after_failover(self):
        blocked = lambda c, k: _resp("ACCESS REVOKED: usage limit has been exceeded")
        client = self._make(blocked, lambda c, k: _resp("or ok"))
        client.call([{"role": "user", "content": "1"}], max_retries=1)
        argo_calls_after_first = self.clients[_ARGO].calls
        client.call([{"role": "user", "content": "2"}], max_retries=1)   # should go straight to OR
        self.assertEqual(self.clients[_ARGO].calls, argo_calls_after_first)  # argo not retried
        self.assertEqual(client._active_idx, 1)

    def test_exception_fails_over(self):
        def boom(c, k):
            raise RuntimeError("connection refused")
        client = self._make(boom, lambda c, k: _resp("or rescue"))
        r = client.call([{"role": "user", "content": "hi"}], max_retries=1)
        self.assertEqual(r.choices[0].message.content, "or rescue")
        self.assertEqual(client.last_backend, "openrouter")

    def test_auth_error_fails_over_without_retry(self):
        def unauth(c, k):
            e = RuntimeError("unauthorized")
            e.status_code = 401
            raise e
        client = self._make(unauth, lambda c, k: _resp("or ok"))
        r = client.call([{"role": "user", "content": "hi"}], max_retries=3)
        self.assertEqual(r.choices[0].message.content, "or ok")
        self.assertEqual(self.clients[_ARGO].calls, 1)   # 401 -> no retry on argo

    def test_all_backends_exhausted_returns_none(self):
        def boom(c, k):
            raise RuntimeError("down")
        client = self._make(boom, boom)
        r = client.call([{"role": "user", "content": "hi"}], max_retries=1)
        self.assertIsNone(r)

    def test_single_backend_when_no_fallback_configured(self):
        with mock.patch.object(llm_client.config, "llm_backends",
                               return_value=[dict(_BACKENDS[0])]), \
                mock.patch.object(llm_client, "OpenAI",
                                  lambda base_url=None, api_key=None: _FakeClient(lambda c, k: _resp("solo"))):
            client = llm_client.LLMClient(model="argo-model")
            self.assertEqual(len(client.backends), 1)
            r = client.call([{"role": "user", "content": "hi"}], max_retries=1)
            self.assertEqual(r.choices[0].message.content, "solo")


if __name__ == "__main__":
    unittest.main()
