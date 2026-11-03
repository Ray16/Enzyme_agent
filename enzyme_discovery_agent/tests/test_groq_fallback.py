"""Cross-provider routing, quota reservations and intact evidence, without services."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.groq_limits import duration_seconds, quota_failure
from enzyme_discovery_agent.provenance import snapshot_config
from enzyme_discovery_agent.tests.test_llm_failover import _FakeClient, _resp, GoogleDailyQuotaTest


def error(message, status=429, headers=None):
    result = RuntimeError(message)
    result.status_code = status
    result.body = {"error": {"message": message, "code": "rate_limit_exceeded"}}
    result.response = types.SimpleNamespace(headers=headers or {})
    return result


class GroqFallbackTest(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.patches = [patch.object(config, "GEMINI_API_KEY", "unit-test-google-key"),
                        patch.object(config, "GROQ_API_KEY", "unit-test-groq-key"),
                        patch.object(config, "GROQ_FALLBACK_ENABLED", True),
                        patch.object(config, "FALLBACK_PROVIDER", "gemini"),
                        patch.object(config, "GEMINI_CASCADE_MODELS", (
                            "gemini-2.5-flash", "gemini-2.5-flash-lite"))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def make(self, behavior, model="gemini:auto"):
        def create(fake, kwargs):
            self.seen.append(copy.deepcopy(kwargs))
            return behavior(kwargs)
        with patch("enzyme_discovery_agent.llm_client.OpenAI", side_effect=lambda **kw: _FakeClient(create)):
            client = LLMClient(model=model)
        pacer = patch.object(client, "_pace_gemini", return_value="reservation")
        pacer.start()
        self.addCleanup(pacer.stop)
        return client

    def google_exhausted(self, kwargs):
        if kwargs["model"] == "gemini-2.5-flash":
            raise GoogleDailyQuotaTest()._error(["GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
        return _resp("supported output")

    def test_configuration_and_auto_insert_groq_before_final_google(self):
        expected = ["gemini-2.5-flash", config.GROQ_MODEL, "gemini-2.5-flash-lite"]
        self.assertEqual([b["model"] for b in config.free_model_backends()], expected)
        self.assertEqual([b["model"] for b in config.llm_backends()][1:], expected)
        client = self.make(self.google_exhausted)
        self.assertEqual([b.model for b in client.backends], expected)
        with patch.object(config, "GROQ_API_KEY", ""):
            self.assertEqual([b["name"] for b in config.free_model_backends()], ["gemini", "gemini"])
        with patch.object(config, "GROQ_FALLBACK_ENABLED", False):
            self.assertEqual([b["name"] for b in config.free_model_backends()], ["gemini", "gemini"])

    def test_provider_options_and_tool_history_preserved(self):
        client = self.make(self.google_exhausted)
        messages = [{"role": "assistant", "name": "source", "content": None,
                     "audio": None, "_llm_model": "gemini:gemini-2.5-flash", "tool_calls": [{
                         "id": "call", "type": "function", "function": {"name": "lookup", "arguments": "{}"},
                         "extra_content": {"google": {"thought_signature": "original-signature"}}}]},
                    {"role": "tool", "tool_call_id": "call", "name": "lookup", "content": "retrieved evidence"}]
        original = copy.deepcopy(messages)
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        self.assertIsNotNone(client.call(messages, tools=tools, max_tokens=1000, max_retries=1))
        sent = self.seen[-1]
        self.assertEqual(sent["model"], config.GROQ_MODEL)
        self.assertEqual(sent["tools"], tools)
        self.assertEqual(sent["messages"][1]["content"], "retrieved evidence")
        self.assertNotIn("extra_content", sent["messages"][0]["tool_calls"][0])
        self.assertNotIn("name", sent["messages"][1])
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertEqual(sent["extra_body"], {"reasoning_format": "hidden"})
        self.assertFalse(sent["parallel_tool_calls"])
        self.assertEqual(sent["max_completion_tokens"], 1000)
        self.assertNotIn("max_tokens", sent)
        self.assertEqual(messages, original)
        client._pace_gemini.assert_any_call(client._estimate_input_tokens(messages, tools) + 1000,
                                           client.backends[1])

    def test_subset_starting_with_groq_keeps_provider_and_model_associated(self):
        with patch.object(config, "GEMINI_CASCADE_MODELS", ("gemini-2.5-flash-lite",)):
            client = self.make(lambda k: _resp("ok"))
        self.assertEqual((client.backends[0].name, client.backends[0].model), ("groq", config.GROQ_MODEL))
        self.assertIsNotNone(client.call([], max_tokens=1000, max_retries=1))
        self.assertEqual(self.seen[0]["model"], config.GROQ_MODEL)

    def test_entire_default_ladder_exhausts_in_order_and_stops(self):
        def fail(kwargs):
            if kwargs["model"] == config.GROQ_MODEL:
                raise error("Tokens per day (TPD) exhausted")
            raise GoogleDailyQuotaTest()._error(["GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
        with patch.object(config, "GEMINI_CASCADE_MODELS", config.GEMINI_FREE_MODEL_PRIORITY):
            expected = [row["model"] for row in config.free_model_backends()]
            client = self.make(fail)
        self.assertEqual(expected[-3:], ["gemini-2.5-flash", config.GROQ_MODEL, "gemini-2.5-flash-lite"])
        self.assertIsNone(client.call([], max_tokens=1000, max_retries=1))
        self.assertEqual([row["model"] for row in self.seen], expected)
        self.assertTrue(client.budget_exhausted)
        self.assertIsNone(client.call([], max_tokens=1000, max_retries=1))
        self.assertEqual(len(self.seen), len(expected))

    def test_large_input_and_output_skip_groq_without_a_request(self):
        for text, output in (("x" * 25000, 1000), ("short", 8192)):
            with self.subTest(output=output):
                self.seen.clear()
                client = self.make(self.google_exhausted)
                messages = [{"role": "user", "content": text}]
                self.assertIsNotNone(client.call(messages, max_tokens=output, max_retries=1))
                self.assertEqual([r["model"] for r in self.seen],
                                 ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
                self.assertEqual(self.seen[-1]["messages"], messages)
                self.assertEqual(client._groq_daily_usage, [])

    def test_later_short_request_can_use_groq_after_an_oversized_request(self):
        client = self.make(self.google_exhausted)
        self.assertIsNotNone(client.call([{"role": "user", "content": "x" * 25000}],
                                         max_tokens=1000, max_retries=1))
        self.assertEqual(client.model, "gemini-2.5-flash-lite")
        self.assertIsNotNone(client.call([{"role": "user", "content": "short extraction"}],
                                         max_tokens=1000, max_retries=1))
        self.assertEqual(client.model, config.GROQ_MODEL)
        self.assertEqual([r["model"] for r in self.seen],
                         ["gemini-2.5-flash", "gemini-2.5-flash-lite", config.GROQ_MODEL])

    def test_old_google_checkpoint_can_gain_new_groq_capacity(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("enzyme_discovery_agent.models.llm_cascade.fetch_model_metadata", return_value={}):
            path = Path(directory) / "agent_state/llm_cascade.json"
            path.parent.mkdir()
            from enzyme_discovery_agent.models.llm_cascade import next_quota_reset
            path.write_text(json.dumps({"requested_model": "gemini:auto", "daily_exhausted": {
                "gemini:gemini-2.5-flash": {"reset_at": next_quota_reset()},
                "gemini:gemini-2.5-flash-lite": {"reset_at": next_quota_reset()}}}))
            client = self.make(lambda k: _resp("new evidence"))
            client.attach_usage(TokenUsage(), directory)
            self.assertIsNotNone(client.call([], max_tokens=1000, max_retries=1))
            self.assertEqual([r["model"] for r in self.seen], [config.GROQ_MODEL])

    def test_groq_daily_failure_falls_through_and_survives_resume(self):
        def behavior(kwargs):
            if kwargs["model"] == config.GROQ_MODEL:
                raise error("Tokens per day (TPD) exhausted", headers={"retry-after": "3600"})
            return self.google_exhausted(kwargs)
        with tempfile.TemporaryDirectory() as directory, \
             patch("enzyme_discovery_agent.models.llm_cascade.fetch_model_metadata", return_value={}):
            client = self.make(behavior)
            client.attach_usage(TokenUsage(), directory)
            self.assertIsNotNone(client.call([], max_tokens=1000, max_retries=1))
            self.assertEqual(client.last_backend, "gemini")
            self.assertEqual(client.model, "gemini-2.5-flash-lite")
            saved = (Path(directory) / "agent_state/llm_cascade.json").read_text()
            self.assertIn("groq:" + config.GROQ_MODEL, saved)
            self.assertNotIn("unit-test-groq-key", saved)
            self.assertNotIn("unit-test-groq-key", json.dumps(snapshot_config(config)))
            self.seen.clear()
            resumed = self.make(lambda k: _resp("new output"))
            resumed.attach_usage(TokenUsage(), directory)
            self.assertIsNotNone(resumed.call([], max_tokens=1000, max_retries=1))
            self.assertEqual([r["model"] for r in self.seen], ["gemini-2.5-flash-lite"])

    def test_minute_limit_retries_groq_before_using_final_fallback(self):
        attempts = [0]
        def behavior(kwargs):
            if kwargs["model"] == config.GROQ_MODEL:
                attempts[0] += 1
                if attempts[0] == 1:
                    raise error("Tokens per minute (TPM) exhausted", headers={"retry-after": "2"})
                return _resp("ok")
            return self.google_exhausted(kwargs)
        client = self.make(behavior)
        with patch.object(client, "_wait_for_quota") as wait:
            self.assertIsNotNone(client.call([], max_tokens=1000, max_retries=2))
        wait.assert_called_once_with(2.1, client.backends[1])
        self.assertEqual([r["model"] for r in self.seen], ["gemini-2.5-flash", config.GROQ_MODEL, config.GROQ_MODEL])
        self.assertNotIn("groq:" + config.GROQ_MODEL, client._daily_exhausted)

    def test_local_daily_tokens_prevent_sending_and_persist_reservations(self):
        response = _resp("ok")
        response.usage = types.SimpleNamespace(prompt_tokens=400, completion_tokens=600, total_tokens=1000)
        client = self.make(lambda k: response, model="groq:" + config.GROQ_MODEL)
        with tempfile.TemporaryDirectory() as directory:
            client.attach_usage(TokenUsage(), directory)
            client.model_profiles[config.GROQ_MODEL]["tokens_per_day"] = 1100
            self.assertIsNotNone(client.call([], max_tokens=1000, max_retries=1))
            self.assertEqual(client._groq_daily_usage[0]["tokens"], 1000)
            self.assertIsNone(client.call([], max_tokens=1000, max_retries=1))
            self.assertEqual(len(self.seen), 1)
            saved = json.loads((Path(directory) / "agent_state/llm_cascade.json").read_text())
            self.assertEqual(saved["groq_daily_usage"][0]["tokens"], 1000)
            restored = self.make(lambda k: _resp("ok"), model="groq:" + config.GROQ_MODEL)
            restored.attach_usage(TokenUsage(), directory)
            self.assertEqual(restored._groq_daily_usage, client._groq_daily_usage)
            self.assertTrue(restored._backend_exhausted(restored.backends[0]))

    def test_concurrent_daily_request_reservations_cannot_overspend(self):
        client = self.make(lambda k: _resp("ok"), model="groq:" + config.GROQ_MODEL)
        client.model_profiles[config.GROQ_MODEL]["requests_per_day"] = 1
        claims = []
        threads = [threading.Thread(target=lambda: claims.append(
            client._reserve_groq_daily(client.backends[0], 100))) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sum(claim is not None for claim in claims), 1)

    def test_groq_to_google_tool_trace_uses_migration_signature(self):
        def behavior(kwargs):
            if kwargs["model"] == config.GROQ_MODEL:
                raise error("unauthorized", status=401)
            return self.google_exhausted(kwargs)
        client = self.make(behavior)
        history = [{"role": "assistant", "_llm_model": "groq:" + config.GROQ_MODEL,
                    "tool_calls": [{"id": "call", "type": "function", "function": {
                        "name": "lookup", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "call", "content": "original evidence"}]
        original = copy.deepcopy(history)
        self.assertIsNotNone(client.call(history, max_tokens=1000, max_retries=1))
        self.assertEqual(self.seen[-1]["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
                         "skip_thought_signature_validator")
        self.assertEqual(history, original)

    def test_truncated_groq_response_is_counted_but_not_used(self):
        def behavior(kwargs):
            if kwargs["model"] == config.GROQ_MODEL:
                response = _resp("incomplete evidence")
                response.choices[0].finish_reason = "length"
                return response
            return self.google_exhausted(kwargs)
        client = self.make(behavior)
        self.assertIsNotNone(client.call([], max_tokens=1000, max_retries=1))
        self.assertEqual(client.model, "gemini-2.5-flash-lite")

    def test_pinned_model_requires_key_and_does_not_add_a_paid_model(self):
        with patch.object(config, "GROQ_API_KEY", ""):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                self.make(lambda k: _resp("ok"), model="groq:" + config.GROQ_MODEL)
        with self.assertRaisesRegex(ValueError, "supports only"):
            self.make(lambda k: _resp("ok"), model="groq:unapproved-model")


class GroqQuotaParsingTest(unittest.TestCase):
    def test_reset_duration_and_daily_header(self):
        self.assertAlmostEqual(duration_seconds("2m59.56s"), 179.56)
        self.assertAlmostEqual(duration_seconds("100ms"), .1)
        self.assertIsNone(duration_seconds("unknown"))
        result = quota_failure(error("rate limit exceeded", headers={
            "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2h3m"}))
        self.assertEqual(result, {"daily": True, "retry_after": 7380})

    def test_daily_tokens_do_not_use_minute_token_reset_header(self):
        result = quota_failure(error("Tokens per day (TPD) exceeded", headers={
            "x-ratelimit-reset-tokens": "2s", "retry-after": "3600"}))
        self.assertEqual(result, {"daily": True, "retry_after": 3600})
        minute = quota_failure(error("Tokens per minute (TPM) exceeded", headers={
            "x-ratelimit-reset-tokens": "3s"}))
        self.assertEqual(minute, {"daily": False, "retry_after": 3})
