"""Explicit reasoning and complete-answer capacity across supported models."""

import copy
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from enzyme_discovery_agent import config
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.tests.test_llm_failover import _FakeClient, _resp, GoogleDailyQuotaTest


class HighReasoningTest(unittest.TestCase):
    def make(self, behavior, model="gemini:auto", models=None):
        self.seen, self.options = [], []
        outer = self
        class Client(_FakeClient):
            def with_options(self, **kwargs):
                outer.options.append(kwargs)
                return self
        def create(fake, kwargs):
            self.seen.append(copy.deepcopy(kwargs))
            return behavior(kwargs)
        with patch.object(config, "GEMINI_API_KEY", "unit-test-google-key"), \
             patch.object(config, "GROQ_API_KEY", ""), \
             patch.object(config, "GEMINI_CASCADE_MODELS", models or config.GEMINI_FREE_MODEL_PRIORITY), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", side_effect=lambda **kwargs: Client(create)):
            client = LLMClient(model=model)
        client._pace_gemini = Mock(return_value="reservation")
        return client

    def test_all_seven_fallbacks_keep_high_for_internal_calls(self):
        models = config.GEMINI_FREE_MODEL_PRIORITY
        def behavior(kwargs):
            if kwargs["model"] != models[-1]:
                raise GoogleDailyQuotaTest()._error(["GenerateRequestsPerDayPerProjectPerModel-FreeTier"])
            return _resp('{"supported":true}')
        client = self.make(behavior)
        messages = [{"role": "user", "content": "Extract only grounded claims from the supplied evidence."}]
        self.assertIsNotNone(client.call(messages, max_tokens=256, max_retries=1, usage_label="internal"))
        self.assertEqual([row["model"] for row in self.seen], list(models))
        for row in self.seen:
            self.assertEqual(row["reasoning_effort"], "high")
            self.assertEqual(row["max_tokens"], 24576 + 256)
            self.assertEqual(row["messages"], messages)
        self.assertTrue(all(options["timeout"] >= 300 for options in self.options))

    def test_pinned_models_also_get_high_and_default_answer_headroom(self):
        for model in config.GEMINI_FREE_MODEL_PRIORITY:
            with self.subTest(model=model):
                client = self.make(lambda k: _resp("complete"), model="gemini:" + model)
                self.assertIsNotNone(client.call([], max_retries=1))
                self.assertEqual(self.seen[0]["reasoning_effort"], "high")
                self.assertEqual(self.seen[0]["max_tokens"], 32768)

    def test_insufficient_model_capacity_skips_without_lowering_effort(self):
        client = self.make(lambda k: _resp("complete"))
        client.model_profiles[client.backends[0].model]["output_token_limit"] = 8192
        self.assertIsNotNone(client.call([], max_tokens=512, max_retries=1))
        self.assertEqual([row["model"] for row in self.seen], [config.GEMINI_FREE_MODEL_PRIORITY[1]])
        self.assertEqual(self.seen[0]["reasoning_effort"], "high")
        self.assertEqual(self.seen[0]["max_tokens"], 24576 + 512)

    def test_truncated_high_response_is_accounted_but_not_used_as_evidence(self):
        def behavior(kwargs):
            response = _resp("partial" if kwargs["model"] == config.GEMINI_FREE_MODEL_PRIORITY[0] else "complete")
            response.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=9, total_tokens=10)
            response.choices[0].finish_reason = "length" if kwargs["model"] == config.GEMINI_FREE_MODEL_PRIORITY[0] else "stop"
            return response
        client = self.make(behavior)
        with tempfile.TemporaryDirectory() as directory, \
             patch("enzyme_discovery_agent.models.llm_cascade.fetch_model_metadata", return_value={}):
            usage = TokenUsage()
            client.attach_usage(usage, directory)
            response = client.call([], max_tokens=512, max_retries=1)
            self.assertEqual(response.choices[0].message.content, "complete")
            self.assertEqual(usage.total_tokens, 20)
            self.assertTrue(all(call["reasoning_effort"] == "high" for call in usage.calls))
            self.assertEqual(len(self.seen), 2)

    def test_unknown_pinned_models_do_not_receive_unverified_controls(self):
        client = self.make(lambda k: _resp("complete"), model="gemini:unknown-test-model")
        self.assertIsNotNone(client.call([], max_tokens=256, max_retries=1))
        self.assertNotIn("reasoning_effort", self.seen[0])
        self.assertEqual(self.seen[0]["max_tokens"], 256)

    def test_argo_sonnet_requests_high_and_retains_tool_definitions(self):
        client = self.make(lambda k: _resp("complete"), model=config.ARGO_MODEL)
        tools = [{"type": "function", "function": {"name": "retrieve", "parameters": {"type": "object"}}}]
        self.assertIsNotNone(client.call([], tools=tools, max_tokens=512, max_retries=1))
        self.assertEqual(self.seen[0]["model"], config.ARGO_MODEL)
        self.assertEqual(self.seen[0]["reasoning_effort"], "high")
        self.assertEqual(self.seen[0]["max_tokens"], 24576 + 512)
        self.assertEqual(self.seen[0]["tools"], tools)
        self.assertGreaterEqual(self.options[0]["timeout"], 300)

    def test_argo_block_falls_through_to_google_at_high(self):
        def behavior(kwargs):
            if kwargs["model"] == config.ARGO_MODEL:
                return _resp("ACCESS REVOKED: usage limit has been exceeded")
            return _resp("complete")
        client = self.make(behavior, model=config.ARGO_MODEL)
        self.assertIsNotNone(client.call([], max_tokens=512, max_retries=1))
        self.assertEqual(len(self.seen), 2)
        self.assertTrue(all(row["reasoning_effort"] == "high" for row in self.seen))

    def test_argo_rejected_effort_never_retries_without_high(self):
        def behavior(kwargs):
            if kwargs["model"] == config.ARGO_MODEL:
                problem = RuntimeError("unsupported parameter")
                problem.status_code = 400
                raise problem
            return _resp("complete")
        client = self.make(behavior, model=config.ARGO_MODEL)
        self.assertIsNotNone(client.call([], max_tokens=512, max_retries=3))
        self.assertEqual(sum(row["model"] == config.ARGO_MODEL for row in self.seen), 1)
        self.assertTrue(all(row["reasoning_effort"] == "high" for row in self.seen))
