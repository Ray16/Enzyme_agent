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

    def test_short_structured_fallback_reserves_visible_output_budget(self):
        seen = []
        blocked = lambda c, k: _resp("ACCESS REVOKED: usage limit has been exceeded")
        def fallback(c, kwargs):
            seen.append(kwargs)
            return _resp('{"result": "ok"}')
        client = self._make(blocked, fallback)
        with mock.patch.object(llm_client.config, "OPENROUTER_STRUCTURED_REASONING_TOKENS", 512):
            client.call([{"role": "user", "content": "Return JSON"}], max_tokens=2048, max_retries=1)
            self.assertEqual(seen[-1]["extra_body"]["reasoning"]["max_tokens"], 512)
            client.call([{"role": "user", "content": "Explore"}], tools=[{"type": "function"}], max_tokens=2048, max_retries=1)
            self.assertNotIn("extra_body", seen[-1])

    def test_google_is_selected_directly_with_provider_qualified_model(self):
        endpoints = []
        def factory(base_url=None, api_key=None):
            endpoints.append(base_url)
            return _FakeClient(lambda c, k: _resp("google"))
        with mock.patch.object(llm_client.config, "GEMINI_API_KEY", "test-key"), \
             mock.patch.object(llm_client, "OpenAI", factory):
            client = llm_client.LLMClient(model="gemini:test-google-model")
        self.assertEqual([b.name for b in client.backends], ["gemini"])
        self.assertEqual(client.model, "test-google-model")
        self.assertEqual(endpoints, [llm_client.config.GEMINI_BASE_URL])

    def test_direct_google_without_key_has_actionable_error(self):
        with mock.patch.object(llm_client.config, "GEMINI_API_KEY", ""):
            with self.assertRaisesRegex(RuntimeError, "Set GEMINI_API_KEY"):
                llm_client.LLMClient(model="gemini:test-model")

    def test_google_tool_round_trip_omits_null_sdk_structs_and_preserves_signature(self):
        seen = []
        with mock.patch.object(llm_client.config, "GEMINI_API_KEY", "test-key"), \
             mock.patch.object(llm_client, "OpenAI", return_value=_FakeClient(
                 lambda c, k: seen.append(k) or _resp("finished"))):
            client = llm_client.LLMClient(model="gemini:test-model")
        messages = [{"role": "assistant", "content": None, "audio": None, "function_call": None,
                     "tool_calls": [{"id": "call-1", "type": "function",
                       "function": {"name": "example", "arguments": '{"value":null}'},
                       "extra_content": {"google": {"thought_signature": "opaque-signature"}}}]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "{}"}]
        client.call(messages, max_retries=1)
        sent = seen[0]["messages"][0]
        self.assertNotIn("audio", sent)
        self.assertNotIn("function_call", sent)
        self.assertEqual(sent["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "opaque-signature")
        self.assertEqual(sent["tool_calls"][0]["function"]["arguments"], '{"value":null}')
        self.assertIn("audio", messages[0])

    def test_google_structured_retry_delay_is_honored(self):
        error = RuntimeError("quota exceeded")
        error.body = [{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                               "retryDelay": "38s"}]}}]
        self.assertAlmostEqual(llm_client.LLMClient._retry_delay(error), 38.1)

    def test_google_pacing_spans_multiple_requests(self):
        with mock.patch.object(llm_client.config, "GEMINI_API_KEY", "test-key"), \
             mock.patch.object(llm_client, "OpenAI", return_value=_FakeClient(lambda c, k: _resp("ok"))):
            client = llm_client.LLMClient(model="gemini:test-model")
        clock = [100.0]
        with mock.patch.object(llm_client.config, "GEMINI_REQUESTS_PER_MINUTE", 5), \
             mock.patch.object(llm_client.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(client._gemini_rate_condition, "wait",
                               side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)) as wait:
            client.call([], max_retries=1)
            client.call([], max_retries=1)
            self.assertAlmostEqual(wait.call_args.args[0], 12.1)


class BackendConfigurationTest(unittest.TestCase):
    def test_google_replaces_openrouter_even_if_old_key_remains(self):
        with mock.patch.object(llm_client.config, "FALLBACK_PROVIDER", "gemini"), \
             mock.patch.object(llm_client.config, "GEMINI_API_KEY", "google-test-key"), \
             mock.patch.object(llm_client.config, "GROQ_API_KEY", ""), \
             mock.patch.object(llm_client.config, "OPENROUTER_API_KEY", "old-test-key"):
            backends = llm_client.config.llm_backends()
        self.assertEqual([b["name"] for b in backends], ["argo"] + ["gemini"] * len(llm_client.config.GEMINI_CASCADE_MODELS))
        self.assertEqual(backends[0]["model"], llm_client.config.ARGO_MODEL)

    def test_missing_google_key_does_not_silently_use_openrouter(self):
        with mock.patch.object(llm_client.config, "FALLBACK_PROVIDER", "gemini"), \
             mock.patch.object(llm_client.config, "GEMINI_API_KEY", ""), \
             mock.patch.object(llm_client.config, "GROQ_API_KEY", ""), \
             mock.patch.object(llm_client.config, "OPENROUTER_API_KEY", "old-test-key"):
            backends = llm_client.config.llm_backends()
        self.assertEqual([b["name"] for b in backends], ["argo"])


class GoogleDailyQuotaTest(unittest.TestCase):
    def _error(self, quota_ids):
        error = RuntimeError("quota exceeded")
        error.status_code = 429
        error.body = [{"error": {"details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": identifier} for identifier in quota_ids]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"},
        ]}}]
        return error

    def test_daily_and_minute_quotas_are_distinguished(self):
        daily = self._error(["GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
        minute = self._error(["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"])
        mixed = self._error(["GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                             "GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
        self.assertTrue(llm_client.LLMClient._google_daily_quota(daily))
        self.assertFalse(llm_client.LLMClient._google_daily_quota(minute))
        self.assertTrue(llm_client.LLMClient._google_daily_quota(mixed))

    def test_daily_failure_stops_retries_and_provider_failover(self):
        error = self._error(["GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
        def fail(c, kwargs):
            raise error
        primary = _FakeClient(fail)
        fallback = _FakeClient(lambda c, k: _resp("fallback"))
        with mock.patch.object(llm_client.config, "llm_backends", return_value=[
            {"name": "gemini", "base_url": "google", "api_key": "test", "model": "test"},
            {"name": "openrouter", "base_url": "fallback", "api_key": "test", "model": "test"}]), \
             mock.patch.object(llm_client, "OpenAI", side_effect=[primary, fallback]), \
             mock.patch.object(llm_client.config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             mock.patch.object(llm_client.time, "sleep") as sleep:
            client = llm_client.LLMClient(model="test")
            self.assertIsNone(client.call([]))
            self.assertIsNone(client.call([]))
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 0)
        self.assertIn("daily quota", client.resource_stop_reason)
        sleep.assert_not_called()

    def test_minute_failure_retries_and_recovers(self):
        error = self._error(["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"])
        def retry(c, kwargs):
            if c.calls == 1:
                raise error
            return _resp("recovered")
        primary = _FakeClient(retry)
        with mock.patch.object(llm_client.config, "GEMINI_API_KEY", "test"), \
             mock.patch.object(llm_client, "OpenAI", return_value=primary), \
             mock.patch.object(llm_client.config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             mock.patch.object(llm_client.time, "sleep") as sleep:
            client = llm_client.LLMClient(model="gemini:test")
            self.assertEqual(client.call([]).choices[0].message.content, "recovered")
        self.assertEqual(primary.calls, 2)
        self.assertIsNone(client.resource_stop_reason)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
