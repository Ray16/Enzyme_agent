"""Bound model context and reserve provider token capacity without live requests."""

import copy
import json
import tempfile
import unittest
import threading
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from enzyme_discovery_agent import config
from enzyme_discovery_agent.context import build_model_context
from enzyme_discovery_agent.llm_client import LLMClient, TokenUsage
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
from enzyme_discovery_agent.tools.reasoning import get_candidate_evidence


class ContextTest(unittest.TestCase):
    def test_candidate_preview_preserves_native_chemistry_before_large_metadata(self):
        from enzyme_discovery_agent.context import compact_for_llm
        payload={'accession':'SYNTH_CHEM','protein_name':'Synthetic catalyst',
            'irrelevant_history':'old inventory '*10000,
            'catalytic_activities':['SELECTED_NATIVE_STEP uses the two declared organic reactant occurrences.'],
            'function_comments':['SOURCE_ROLE: the two native molecules supply complementary reacting ends.'],
            'native_substrates':[{'name':'native partner','smiles':'NCCCC(=O)O'}],
            'substrate_similarity':{'property_comparisons':[
                {'target_smiles':'CCO','native_smiles':'NCCCC(=O)O','mw_target':46.1,'mw_native':103.1},
                {'target_smiles':'CCN','native_smiles':'NCCCC(=O)O','mw_target':45.1,'mw_native':103.1}]}}
        projected=compact_for_llm(payload)
        self.assertIn('SELECTED_NATIVE_STEP',projected['reaction']['native_reaction'])
        self.assertIn('SOURCE_ROLE',projected['function_comments'][0])
        self.assertEqual(len(projected['substrate_accommodation']),2)
        self.assertEqual(projected['reactant_occurrences'][0]['smiles'],'NCCCC(=O)O')
        self.assertLess(len(json.dumps(projected)),4000)
        self.assertEqual(len(payload['irrelevant_history']),140000)

    def test_absence_blocked_pending_leads_are_flagged_as_resolvable(self):
        # A candidate blocked only by missing annotation must read as a lead to
        # resolve, not a rejected one, so the agent can decide to pursue it.
        from enzyme_discovery_agent.context import build_research_state
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P_ABSENT", "evidence_pending", {
                "protein_name": "Uncharacterized protein",
                "claim_validation": {"blocking_claims": ["native_reaction"],
                                     "blocked_only_by_absence": True,
                                     "blocked_by_absence": ["native_reaction"]}})
            update_candidate_registry(root, "P_WRONG", "evidence_pending", {
                "protein_name": "Opposite-direction hydrolase",
                "claim_validation": {"blocking_claims": ["reaction_direction"],
                                     "blocked_only_by_absence": False,
                                     "blocked_by_absence": []}})
            rows = {r["accession"]: r for r in build_research_state(root)["evidence_pending_examples"]}
            self.assertTrue(rows["P_ABSENT"]["blocked_only_by_absence"])
            self.assertEqual(rows["P_ABSENT"]["resolvable_missing_annotation"], ["native_reaction"])
            self.assertFalse(rows["P_WRONG"]["blocked_only_by_absence"])

    def test_old_gate_and_report_history_is_replaced_without_mutating_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "evaluated", {
                "protein_name": "Example enzyme", "catalytic_activities": ["A + B = C"],
            })
            messages = [{"role": "system", "content": "system constraints"},
                        {"role": "user", "content": "user target"}]
            for _ in range(30):
                messages += [{"role": "assistant", "content": "draft report " * 2000},
                             {"role": "user", "content": "old gate guidance " * 2000}]
            messages.append({"role": "user", "content": "take the next action"})
            original = copy.deepcopy(messages)
            projected = build_model_context(messages, root)
            self.assertEqual(messages, original)
            self.assertEqual(projected[:2], messages[:2])
            # Volatile state now trails the append-only history so the cache prefix
            # survives; assert it is present and last, not at a fixed early index.
            body = json.dumps(projected)
            self.assertIn("P12345", projected[-1]["content"])
            self.assertIn("take the next action", body)
            self.assertLess(len(json.dumps(projected)), config.MODEL_CONTEXT_MAX_CHARS)
            self.assertLess(len(json.dumps(projected)), len(json.dumps(messages)) / 10)

    def test_adaptive_review_keeps_recent_candidate_chemistry_in_long_context(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            evidence={'accession':'SYNTH_CHEM','protein_name':'Synthetic catalyst',
                'catalytic_activities':['NATIVE_STEP: the two supplied molecules couple.'],
                'function_comments':['ROLE_FACT: complementary reacting ends are retained.'],
                'native_substrates':[{'name':'native partner','smiles':'NCCCC(=O)O'}],
                'substrate_similarity':{'property_comparisons':[
                    {'target_smiles':'CCO','native_smiles':'NCCCC(=O)O','mw_target':46.1,'mw_native':103.1},
                    {'target_smiles':'CCN','native_smiles':'NCCCC(=O)O','mw_target':45.1,'mw_native':103.1}]}}
            state={'adaptive_reasoning':{'review_due':True,'observations':[],
                'previous_test':{'assessment':{'large_prior_inventory':'old facts '*10000}}}}
            messages=[{'role':'system','content':'Scientific constraints '*100},
                      {'role':'user','content':'Assess both supplied target partners.'}]
            messages += [{'role':'assistant','content':'Obsolete narrative '*1000} for _ in range(40)]
            messages += [{'role':'assistant','content':None,'tool_calls':[{
                'id':'chemistry','type':'function','function':{
                    'name':'get_candidate_evidence','arguments':'{"accession":"SYNTH_CHEM"}'}}]},
                {'role':'tool','tool_call_id':'chemistry','content':json.dumps({'status':'success','data':evidence})}]
            original=copy.deepcopy(messages)
            with patch('enzyme_discovery_agent.context.build_research_state',return_value=state):
                projected=build_model_context(messages,root)
            text=json.dumps(projected)
            for fact in ('NATIVE_STEP','ROLE_FACT','46.1','45.1','NCCCC(=O)O'):
                self.assertIn(fact,text)
            self.assertLess(len(text),config.MODEL_CONTEXT_MAX_CHARS)
            self.assertEqual(messages,original)

    def test_recent_tool_transaction_and_google_signature_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            call = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function", "function": {"name": "example", "arguments": "{}"},
                "extra_content": {"google": {"thought_signature": "opaque"}},
            }]}
            result = {"role": "tool", "tool_call_id": "c1", "content": '{"data":"evidence"}'}
            messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "target"},
                        call, result, {"role": "user", "content": "continue"}]
            projected = build_model_context(messages, Path(d))
            # The transaction must stay intact and paired; its index moved because the
            # volatile research state is now appended after the history.
            self.assertIn(call, projected)
            self.assertEqual(projected[projected.index(call) + 1], result)

    def test_saved_candidate_reader_requires_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            update_candidate_registry(root, "P12345", "evaluated", {"protein_name": "Example enzyme"})
            with patch("enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry") as network:
                result = get_candidate_evidence("P12345", run_dir=d)
            self.assertEqual(result["data"]["protein_name"], "Example enzyme")
            network.assert_not_called()


class PacingTest(unittest.TestCase):
    def client(self, limit=100):
        with patch.object(config, "GEMINI_API_KEY", "test"), \
             patch.object(config, "GEMINI_INPUT_TOKENS_PER_MINUTE", limit), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=Mock()):
            return LLMClient(model="gemini:test")

    def test_token_limit_applies_even_when_request_pacing_is_disabled(self):
        client = self.client()
        clock = [100.0]
        with patch.object(config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             patch("enzyme_discovery_agent.llm_client.time.monotonic", side_effect=lambda: clock[0]), \
             patch.object(client._gemini_rate_condition, "wait", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)) as wait:
            client._pace_gemini(80)
            client._pace_gemini(80)
        self.assertAlmostEqual(wait.call_args.args[0], 60.1)
        self.assertEqual(sum(row["tokens"] for row in client._gemini_request_window), 80)

    def test_actual_usage_releases_overestimated_capacity(self):
        client = self.client()
        with patch.object(config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             patch("enzyme_discovery_agent.llm_client.time.monotonic", return_value=100), \
             patch.object(client._gemini_rate_condition, "wait") as wait:
            reservation = client._pace_gemini(80)
            client._settle_gemini_input(reservation, 20)
            client._pace_gemini(80)
        wait.assert_not_called()

    def test_provider_input_limit_is_distinct_from_daily_or_request_limits(self):
        error = RuntimeError("quota exceeded")
        error.body = [{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                         "violations": [{"quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
                                         "quotaValue": "250000"}]}]}}]
        self.assertEqual(LLMClient._google_input_token_limit(error), 250000)
        error.body[0]["error"]["details"][0]["violations"][0]["quotaId"] = "GenerateContentInputTokensPerModelPerDay-FreeTier"
        self.assertIsNone(LLMClient._google_input_token_limit(error))

    def test_resume_restores_the_observed_limit_for_the_same_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "agent_state").mkdir()
            (root / "agent_state" / "provider_limits.json").write_text(json.dumps({
                "gemini": {"model": "test", "input_tokens_per_minute": 250000}}))
            client = self.client(limit=0)
            client.attach_usage(TokenUsage(), root)
            self.assertEqual(client._gemini_input_limit, 250000)

    def test_waiting_caller_wakes_when_actual_usage_frees_capacity(self):
        client = self.client()
        entered = threading.Event()
        finished = threading.Event()
        original_wait = client._gemini_rate_condition.wait
        def wait(seconds):
            entered.set()
            return original_wait(seconds)
        with patch.object(config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             patch("enzyme_discovery_agent.llm_client.time.monotonic", return_value=100), \
             patch.object(client._gemini_rate_condition, "wait", side_effect=wait):
            first = client._pace_gemini(80)
            def next_request():
                client._pace_gemini(80)
                finished.set()
            worker = threading.Thread(target=next_request, daemon=True)
            worker.start()
            self.assertTrue(entered.wait(1))
            client._settle_gemini_input(first, 20)
            self.assertTrue(finished.wait(1))
            worker.join(timeout=1)

    def test_quota_response_learns_and_persists_input_limit_before_retry(self):
        error = RuntimeError("quota exceeded")
        error.status_code = 429
        error.body = [{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
                                        "quotaValue": "250000"}]}]}}]
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
        provider = Mock()
        provider.with_options.return_value = provider
        provider.chat.completions.create.side_effect = [error, response]
        with tempfile.TemporaryDirectory() as d, \
             patch.object(config, "GEMINI_API_KEY", "test"), \
             patch.object(config, "GEMINI_INPUT_TOKENS_PER_MINUTE", 0), \
             patch.object(config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             patch("enzyme_discovery_agent.llm_client.OpenAI", return_value=provider), \
             patch("enzyme_discovery_agent.llm_client.time.sleep"):
            client = LLMClient(model="gemini:test")
            client.attach_usage(TokenUsage(), d)
            self.assertIs(client.call([{"role": "user", "content": "hello"}]), response)
            self.assertEqual(client._gemini_input_limit, 250000)
            saved = json.loads((Path(d) / "agent_state" / "provider_limits.json").read_text())
            self.assertEqual(saved["gemini"]["input_tokens_per_minute"], 250000)

    def test_minute_quota_retries_beyond_transient_limit_without_failover(self):
        error = RuntimeError("minute quota exceeded")
        error.status_code = 429
        error.body = {"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                 "violations": [{"quotaId": "GenerateRequestsPerMinute-FreeTier"}]}]}
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))
        client = self.client()
        provider = client.backends[0].client
        provider.with_options.return_value = provider
        provider.chat.completions.create.side_effect = [error] * 4 + [response]
        with patch.object(config, "GEMINI_REQUESTS_PER_MINUTE", 0), \
             patch.object(client, "_pace_gemini", return_value=None), \
             patch("enzyme_discovery_agent.llm_client.time.sleep"):
            self.assertIs(client.call([{"role": "user", "content": "hello"}], max_retries=1), response)
        self.assertEqual(client._active_idx, 0)
        self.assertEqual(client.request_counts["gemini"], 5)

    def test_quota_wait_honors_resource_budget_without_more_requests(self):
        client = self.client()
        checks = iter([False, True])
        client.budget_check = lambda: next(checks)
        with patch("enzyme_discovery_agent.llm_client.time.sleep") as sleep:
            client._wait_for_quota(60)
        sleep.assert_called_once_with(10.0)
        self.assertTrue(client.budget_exhausted)

    def test_daily_and_unclassified_limits_are_not_indefinitely_retryable(self):
        error = RuntimeError("quota exceeded")
        error.body = {"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                 "violations": [{"quotaId": "GenerateRequestsPerDay-FreeTier"}]}]}
        self.assertFalse(LLMClient._google_minute_quota(error))
        error.body = None
        self.assertFalse(LLMClient._google_minute_quota(error))


if __name__ == "__main__":
    unittest.main()


class CachePrefixTest(unittest.TestCase):
    """Tool schemas head the provider cache prefix; varying them re-bills everything behind."""

    def _tools(self):
        import importlib, pkgutil
        import enzyme_discovery_agent.tools as pkg
        for module in pkgutil.iter_modules(pkg.__path__):
            try:
                importlib.import_module("enzyme_discovery_agent.tools." + module.name)
            except Exception:
                pass
        from enzyme_discovery_agent.tool_registry import get_all_tools
        return get_all_tools()

    def test_schemas_are_byte_identical_whatever_the_turn_would_have_allowed(self):
        from enzyme_discovery_agent.context import stable_tool_definitions
        tools = self._tools()
        self.assertGreater(len(tools), 20)
        baseline = json.dumps(stable_tool_definitions(tools))
        # Same result for a review turn and an ordinary turn, and on repeat calls.
        self.assertEqual(baseline, json.dumps(stable_tool_definitions(tools, include_inspections=True)))
        self.assertEqual(baseline, json.dumps(stable_tool_definitions(tools)))

    def test_the_planner_keeps_every_action_contract(self):
        from enzyme_discovery_agent.context import stable_tool_definitions
        tools = self._tools()
        sent = stable_tool_definitions(tools)
        names = {tool["function"]["name"] for tool in sent}
        self.assertEqual(names, {tool["function"]["name"] for tool in tools})
        planner = next(t for t in sent if t["function"]["name"] == "plan_research_step")
        actions = planner["function"]["parameters"]["properties"]["tool_name"]["enum"]
        self.assertGreater(len(actions), 20, "the planner must still expose its action contracts")

    def test_volatile_state_trails_the_append_only_history(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            messages = [{"role": "system", "content": "constraints"},
                        {"role": "user", "content": "target"},
                        {"role": "assistant", "content": "a step"},
                        {"role": "user", "content": "continue"}]
            projected = build_model_context(messages, root)
            headers = [i for i, m in enumerate(projected)
                       if isinstance(m.get("content"), str)
                       and m["content"].startswith("CURRENT RESEARCH STATE")]
            self.assertTrue(headers, "the research state must still be sent")
            # Everything before it must be the stable prefix plus history, so that the
            # cacheable region grows by appending instead of being rewritten each turn.
            self.assertEqual(headers[0], len(projected) - 1)
