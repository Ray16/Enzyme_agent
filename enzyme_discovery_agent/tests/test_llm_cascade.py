"""Synthetic model quotas, capability limits, resumability and concurrent routing."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.llm_cascade import next_quota_reset
from enzyme_discovery_agent.tests.test_llm_failover import _FakeClient, _resp, GoogleDailyQuotaTest


class CascadeTest(unittest.TestCase):
    def make(self, behavior, argo=False, models=None):
        self.seen = []
        def create(client, kwargs):
            self.seen.append(kwargs)
            return behavior(kwargs)
        models = models or ("strong-test", "mid-test", "light-test")
        backends = ([{"name": "argo", "base_url": "argo", "api_key": "unused", "model": "argo-test"}] if argo else [])
        backends += [{"name": "gemini", "base_url": "google", "api_key": "unused", "model": m} for m in models]
        with patch.object(config, "llm_backends", return_value=backends), \
             patch.object(config, "GEMINI_CASCADE_MODELS", models), \
             patch.object(config, "GROQ_API_KEY", ""), \
             patch.object(config, "GEMINI_API_KEY", "unused"), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", side_effect=lambda **kw: _FakeClient(create)):
            return LLMClient(model="argo-test" if argo else "gemini:auto")

    def daily(self):
        return GoogleDailyQuotaTest()._error(["GenerateRequestsPerDayPerProjectPerModel-FreeTier"])

    def test_available_argo_keeps_the_google_ladder_unused(self):
        client = self.make(lambda k: _resp("Argo reasoning"), argo=True)
        client.call([], max_retries=1)
        self.assertEqual([x["model"] for x in self.seen], ["argo-test"])

    def test_foreign_provider_metadata_is_removed_for_argo_but_audit_is_preserved(self):
        client = self.make(lambda k: _resp("ok"), argo=True)
        messages = [{"role": "assistant", "_llm_model": "gemini:source", "tool_calls": [{
            "id": "call", "type": "function", "function": {"name": "example", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": "opaque"}}}]}]
        original = copy.deepcopy(messages)
        client.call(messages, max_retries=1)
        self.assertNotIn("extra_content", self.seen[0]["messages"][0]["tool_calls"][0])
        self.assertEqual(messages, original)

    @patch.object(config, "GEMINI_CASCADE_REQUESTS_PER_MINUTE", 0)
    def test_argo_first_then_daily_failures_cascade_without_restarting_conversation(self):
        def behavior(kwargs):
            if kwargs["model"] == "argo-test":
                return _resp("ACCESS REVOKED: usage limit has been exceeded")
            if kwargs["model"] == "strong-test":
                raise self.daily()
            return _resp("evidence response")
        client = self.make(behavior, argo=True)
        messages = [{"role": "user", "content": "saved target and evidence"}]
        self.assertIsNotNone(client.call(messages, max_retries=1))
        self.assertEqual([x["model"] for x in self.seen], ["argo-test", "strong-test", "mid-test"])
        self.assertEqual(self.seen[-1]["messages"], messages)
        self.assertFalse(client.budget_exhausted)
        client.call(messages, max_retries=1)
        self.assertEqual(self.seen[-1]["model"], "mid-test")
        self.assertEqual([x["model"] for x in self.seen].count("strong-test"), 1)

    @patch.object(config, "GEMINI_CASCADE_REQUESTS_PER_MINUTE", 0)
    def test_all_models_exhausted_stops_and_repeated_calls_make_no_requests(self):
        def fail(kwargs):
            raise self.daily()
        client = self.make(fail)
        self.assertIsNone(client.call([], max_retries=1))
        self.assertEqual(len(self.seen), 3)
        self.assertTrue(client.budget_exhausted)
        self.assertIn("All configured", client.resource_stop_reason)
        self.assertIsNone(client.call([], max_retries=1))
        self.assertEqual(len(self.seen), 3)

    @patch.object(config, "GEMINI_CASCADE_REQUESTS_PER_MINUTE", 0)
    def test_zero_minute_quota_is_skipped_instead_of_waiting_forever(self):
        error = GoogleDailyQuotaTest()._error(["GenerateRequestsPerMinutePerModel-FreeTier"])
        error.body[0]["error"]["details"][0]["violations"][0]["quotaValue"] = "0"
        def behavior(kwargs):
            if kwargs["model"] == "strong-test":
                raise error
            return _resp("ok")
        client = self.make(behavior)
        with patch("enzyme_discovery_agent.llm_client.time.sleep") as sleep:
            self.assertIsNotNone(client.call([], max_retries=1))
        sleep.assert_not_called()
        self.assertEqual(client.model, "mid-test")

    @patch.object(config, "GEMINI_CASCADE_REQUESTS_PER_MINUTE", 0)
    def test_resume_skips_exhausted_model_until_reset_and_keeps_ladder(self):
        def behavior(kwargs):
            if kwargs["model"] == "strong-test":
                raise self.daily()
            return _resp("ok")
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.models.llm_cascade.fetch_model_metadata", return_value={}):
            client = self.make(behavior)
            client.attach_usage(TokenUsage(), d)
            client.call([], max_retries=1)
            path = Path(d) / "agent_state/llm_cascade.json"
            saved = json.loads(path.read_text())
            self.assertEqual(saved["requested_model"], "gemini:auto")
            resumed = self.make(lambda k: _resp("ok"))
            resumed.attach_usage(TokenUsage(), d)
            resumed.call([], max_retries=1)
            self.assertEqual(self.seen[0]["model"], "mid-test")
            saved["daily_exhausted"]["gemini:strong-test"]["reset_at"] = 0
            path.write_text(json.dumps(saved))
            restored = self.make(lambda k: _resp("ok"))
            restored.attach_usage(TokenUsage(), d)
            restored.call([], max_retries=1)
            self.assertEqual(self.seen[0]["model"], "strong-test")

    def test_pacing_windows_and_token_limits_are_independent_per_model(self):
        client = self.make(lambda k: _resp("ok"))
        client.model_profiles = {"strong-test": {"requests_per_minute": 2, "input_tokens_per_minute": 100},
                                 "mid-test": {"requests_per_minute": 10, "input_tokens_per_minute": 1000}}
        clock = [100.0]
        with patch("enzyme_discovery_agent.llm_client.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(client._gemini_rate_condition, "wait", side_effect=lambda s: clock.__setitem__(0, clock[0]+s)) as wait:
            client._pace_gemini(80, client.backends[0])
            client._pace_gemini(800, client.backends[1])
            wait.assert_not_called()
            client._pace_gemini(80, client.backends[0])
        self.assertAlmostEqual(clock[0], 160.1)

    @patch.object(config, "GEMINI_CASCADE_REQUESTS_PER_MINUTE", 0)
    def test_context_limit_skips_model_and_output_is_capped_to_serving_limit(self):
        client = self.make(lambda k: _resp("ok"))
        client.model_profiles = {"strong-test": {"input_token_limit": 1},
                                 "mid-test": {"input_token_limit": 10000, "output_token_limit": 40}}
        client.call([{"role": "user", "content": "native retrieved evidence"}], max_tokens=100, max_retries=1)
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.seen[0]["model"], "mid-test")
        self.assertEqual(self.seen[0]["max_tokens"], 40)

    def test_migration_keeps_tools_and_evidence_but_does_not_replay_foreign_signatures(self):
        message = {"role": "assistant", "_llm_model": "gemini:strong-test", "tool_calls": [{
            "id": "call", "type": "function", "function": {"name": "example", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": "source-model-signature"}}}]}
        original = copy.deepcopy(message)
        sent = LLMClient._gemini_messages([message], "mid-test")[0]
        self.assertNotIn("_llm_model", sent)
        self.assertEqual(sent["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "skip_thought_signature_validator")
        self.assertEqual(message, original)
        same = LLMClient._gemini_messages([message], "strong-test")[0]
        self.assertEqual(same["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "source-model-signature")

    def test_concurrent_failures_advance_only_once(self):
        client = self.make(lambda k: _resp("ok"))
        workers = [threading.Thread(target=client._advance_backend, args=(0,)) for _ in range(4)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=2)
        self.assertEqual(client._active_idx, 1)

    def test_full_free_ladder_retries_same_evidence_in_priority_order(self):
        models = config.GEMINI_FREE_MODEL_PRIORITY
        exhausted = set(models[:-1])
        def behavior(kwargs):
            if kwargs["model"] in exhausted:
                raise self.daily()
            return _resp("new evidence")
        client = self.make(behavior, models=models)
        messages = [{"role": "user", "content": "saved target and retrieved evidence"}]
        tools = [{"type": "function", "function": {"name": "retrieve", "parameters": {"type": "object"}}}]
        with patch.object(client, "_pace_gemini", return_value="reserved"):
            self.assertIsNotNone(client.call(messages, tools=tools, max_retries=1))
            self.assertEqual([row["model"] for row in self.seen], list(models))
            self.assertTrue(all(row["messages"] == messages and row["tools"] == tools for row in self.seen))
            exhausted.add(models[-1])
            self.assertIsNone(client.call(messages, max_retries=1))
        self.assertTrue(client.budget_exhausted)
        self.assertEqual([row["model"] for row in self.seen], [*models, models[-1]])

    def test_expanded_ladder_resume_uses_model_identity_not_saved_index(self):
        models = ("strong-test", "new-middle-test", "mid-test", "light-test")
        with tempfile.TemporaryDirectory() as d, \
             patch("enzyme_discovery_agent.models.llm_cascade.fetch_model_metadata", return_value={}):
            root = Path(d) / "agent_state"
            root.mkdir()
            (root / "llm_cascade.json").write_text(json.dumps({
                "requested_model": "gemini:auto", "active_index": 2,
                "ladder": ["gemini:strong-test", "gemini:mid-test", "gemini:light-test"],
                "daily_exhausted": {"gemini:strong-test": {"reset_at": next_quota_reset()},
                                    "gemini:mid-test": {"reset_at": next_quota_reset()}},
            }))
            client = self.make(lambda k: _resp("ok"), models=models)
            client.attach_usage(TokenUsage(), d)
            with patch.object(client, "_pace_gemini", return_value="reserved"):
                self.assertIsNotNone(client.call([], max_retries=1))
            self.assertEqual([row["model"] for row in self.seen], ["new-middle-test"])
            self.assertEqual(json.loads((root / "llm_cascade.json").read_text())["ladder"],
                             [f"gemini:{model}" for model in models])

    def test_configured_limits_apply_before_any_request(self):
        limits = {"strong-test": {"requests_per_minute": 5, "input_tokens_per_minute": 250,
                                  "requests_per_day": 20}}
        with patch.object(config, "GEMINI_MODEL_LIMITS", limits):
            client = self.make(lambda k: _resp("ok"))
        self.assertEqual(client.model_profiles["strong-test"]["requests_per_minute"], 5)
        self.assertEqual(client.model_profiles["strong-test"]["requests_per_day"], 20)
        with patch.object(client, "_estimate_input_tokens", return_value=300), \
             patch.object(client, "_pace_gemini", return_value="reserved"):
            self.assertIsNotNone(client.call([], max_retries=1))
        self.assertEqual([row["model"] for row in self.seen], ["mid-test"])
        self.assertEqual(client._backend_failures["gemini:strong-test"], "request exceeds model input TPM")

    def test_single_model_auto_subset_keeps_quota_checkpoints_and_profiles(self):
        def fail(kwargs):
            raise self.daily()
        with patch.object(config, "GEMINI_MODEL_LIMITS", {"strong-test": {"requests_per_minute": 5}}):
            client = self.make(fail, models=("strong-test",))
        self.assertTrue(client._cascade_enabled)
        with patch.object(client, "_pace_gemini", return_value="reserved"):
            self.assertIsNone(client.call([], max_retries=1))
        self.assertEqual(len(self.seen), 1)
        self.assertTrue(client.budget_exhausted)
        self.assertEqual(client.model_profiles["strong-test"]["requests_per_minute"], 5)

    def test_resume_preserves_observed_quotas_and_updates_configured_limits(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(config, "GEMINI_MODEL_LIMITS", {
                 "strong-test": {"requests_per_minute": 5, "input_tokens_per_minute": 250}}), \
             patch("enzyme_discovery_agent.models.llm_cascade.fetch_model_metadata", return_value={}):
            root = Path(d) / "agent_state"
            root.mkdir()
            (root / "llm_cascade.json").write_text(json.dumps({"model_profiles": {
                "strong-test": {"requests_per_minute": 2, "input_tokens_per_minute": 100,
                                "quota_sources": {"input_tokens_per_minute": "Google QuotaFailure"}}}}))
            client = self.make(lambda k: _resp("ok"))
            client.attach_usage(TokenUsage(), d)
            self.assertEqual(client.model_profiles["strong-test"]["requests_per_minute"], 5)
            self.assertEqual(client.model_profiles["strong-test"]["input_tokens_per_minute"], 100)


class FreeLadderConfigTest(unittest.TestCase):
    def test_subset_cannot_reverse_capability_priority_or_duplicate_models(self):
        chosen = config.ordered_gemini_models(
            "gemini-2.5-flash-lite,gemini-3.8-flash,gemini-3.8-flash,gemini-3-flash-preview")
        self.assertEqual(chosen, ("gemini-3.8-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"))

    def test_unapproved_and_empty_ladders_are_rejected(self):
        for value in ("gemini-4-argon", "gemini-3.1-pro-preview", "gemini-flash-latest", "", ", "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.ordered_gemini_models(value)

