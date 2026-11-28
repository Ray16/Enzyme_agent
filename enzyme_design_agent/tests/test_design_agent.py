"""The design agent's loop, driven offline against a scripted LLM.

No network and no GPU: the point is the control flow and the bookkeeping --
that a strategy gets chosen and recorded, that tool results come back, that the
run always ends with a report on disk.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_design_agent.agent import EnzymeDesignAgent
from enzyme_design_agent.bootstrap import design_tool_names, design_tool_schemas
from enzyme_design_agent.prompts import build_state_header, build_system_prompt
from enzyme_design_agent.tests._synthetic import write_structure


class _Call:
    def __init__(self, name, args, cid="c1"):
        self.id = cid
        self.function = mock.Mock(name=name)
        self.function.name = name
        self.function.arguments = json.dumps(args)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _Response:
    def __init__(self, msg):
        self.choices = [mock.Mock(message=msg)]
        self.usage = mock.Mock(prompt_tokens=10, completion_tokens=5)


def _fake_llm(turns):
    """An LLMClient whose `call` replays a scripted list of assistant turns."""
    llm = mock.Mock()
    llm.call.side_effect = [_Response(m) for m in turns]
    llm.extract_tool_calls.side_effect = (
        lambda m: (list(m.tool_calls), m.content, False)
    )
    return llm


class ToolkitTest(unittest.TestCase):
    def test_both_strategies_are_in_the_toolkit(self):
        names = design_tool_names()
        expected = {
            "check_de_novo_design_readiness",
            "design_de_novo_enzyme",
            "summarize_de_novo_campaign",
            "check_disco_design_readiness",
            "design_enzyme_with_disco",
            "summarize_disco_campaign",
            "check_genzyme_design_readiness",
            "design_enzyme_with_genzyme",
            "get_genzyme_design_status",
            "check_rfd4_design_readiness",
            "design_enzyme_with_rfd4",
            "get_rfd4_design_status",
            "scan_design_pocket",
            "build_saturation_library",
            "prepare_saturation_campaign",
            "summarize_saturation_campaign",
            "audit_design_contacts",
            "declare_design_strategy",
        }
        self.assertTrue(expected.issubset(set(names)), f"missing: {expected - set(names)}")

    def test_discovery_only_tools_are_not(self):
        """A design loop that can also run literature search is a different agent."""
        names = design_tool_names()
        self.assertNotIn("search_literature", names)
        self.assertNotIn("search_uniprot", names)

    def test_schemas_match_the_names(self):
        self.assertEqual(
            sorted(design_tool_names()),
            sorted(s["function"]["name"] for s in design_tool_schemas()),
        )


class PromptTest(unittest.TestCase):
    def test_strategy_choice_is_stated(self):
        p = build_system_prompt()
        self.assertIn("CHOOSING A DESIGN STRATEGY", p)
        self.assertIn("GENERATIVE", p)
        self.assertIn("REDESIGN", p)

    def test_available_tools_are_listed(self):
        p = build_system_prompt(available_tools=["a_tool"])
        self.assertIn("a_tool", p)

    def test_unavailable_capabilities_are_called_out(self):
        p = build_system_prompt(capabilities={"boltz2": False, "rdkit": True})
        self.assertIn("NOT AVAILABLE", p)
        self.assertIn("boltz2", p)
        self.assertNotIn("  - rdkit", p)

    def test_state_header_is_one_line(self):
        h = build_state_header(3, 40, "redesign", 57, ["scan_design_pocket"])
        self.assertEqual(1, len(h.splitlines()))
        self.assertIn("strategy=redesign", h)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.pdb = write_structure(self.out / "holo.pdb")

    def tearDown(self):
        self.tmp.cleanup()

    def _agent(self, turns, **kw):
        with mock.patch("enzyme_design_agent.agent.LLMClient",
                        return_value=_fake_llm(turns)):
            return EnzymeDesignAgent(output_dir=self.out, query="q", **kw)

    def test_a_tool_call_then_a_report(self):
        turns = [
            _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})]),
            _Msg(content="## Design report\nRedesign chosen."),
        ]
        agent = self._agent(turns)
        report = agent.run("make it do the chemistry")
        self.assertIn("Design report", report)
        self.assertEqual("redesign", agent.strategy)
        self.assertEqual(["scan_design_pocket"], agent.tools_used)

    def test_strategy_is_recorded_in_the_manifest(self):
        turns = [
            _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})]),
            _Msg(content="done"),
        ]
        agent = self._agent(turns)
        agent.run("q")
        m = json.loads((agent.run_dir / "manifest.json").read_text())
        self.assertEqual("redesign", m["strategy"])
        self.assertEqual("completed", m["status"])
        self.assertEqual("enzyme_design_agent", m["agent"])

    def test_report_is_always_written_even_with_no_text(self):
        agent = self._agent([_Msg(content=""), _Msg(content="")])
        agent.run("q")
        self.assertTrue((agent.run_dir / "DESIGN_REPORT.md").is_file())

    def test_calling_both_strategies_is_recorded_as_both(self):
        turns = [
            _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})]),
            _Msg(tool_calls=[_Call("design_de_novo_enzyme", {}, cid="c2")]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        # Stubbed: the generative tool runs on real GPUs, and this test is about
        # strategy bookkeeping, not about the hardware under it.
        with mock.patch("enzyme_design_agent.agent.execute_tool",
                        return_value={"status": "success", "data": {"artifact_counts": {"generated": 3}}}):
            agent.run("q")
        self.assertEqual("both", agent.strategy)
        self.assertEqual(3, agent.designs)

    def test_readiness_probe_does_not_pin_strategy(self):
        turns = [
            _Msg(tool_calls=[_Call("check_de_novo_design_readiness", {})]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        with mock.patch("enzyme_design_agent.agent.execute_tool",
                        return_value={"status": "success", "data": {}}):
            agent.run("q")
        self.assertEqual("undecided", agent.strategy)

    def test_disco_genzyme_rfd4_track_strategy_and_designs_without_double_count(self):
        turns = [
            _Msg(tool_calls=[_Call("design_enzyme_with_disco", {}, cid="c1")]),
            _Msg(tool_calls=[_Call("get_rfd4_design_status", {"job_dir": "/j1"}, cid="c2")]),
            _Msg(tool_calls=[_Call("get_rfd4_design_status", {"job_dir": "/j1"}, cid="c3")]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        responses = [
            {"status": "success", "data": {"n_designs": 2}},
            {"status": "success", "data": {"manifest": "/j1/manifest.json", "designs": [{"cif": "a"}, {"cif": "b"}]}},
            {"status": "success", "data": {"manifest": "/j1/manifest.json", "designs": [{"cif": "a"}, {"cif": "b"}]}},
        ]
        with mock.patch("enzyme_design_agent.agent.execute_tool", side_effect=responses):
            agent.run("q")
        self.assertEqual("generative", agent.strategy)
        self.assertEqual(4, agent.designs)

    def test_globally_registered_non_design_tool_is_still_refused(self):
        """Even if a non-design tool is in the global registry, the design agent refuses it."""
        turns = [
            _Msg(tool_calls=[_Call("foreign_global_tool", {})]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        with mock.patch("enzyme_design_agent.agent.get_tool_info", return_value={"parallel_safe": True}), \
             mock.patch("enzyme_design_agent.agent.execute_tool") as exec_mock:
            report = agent.run("q")
        self.assertEqual("report", report)
        self.assertEqual([], agent.tools_used)
        exec_mock.assert_not_called()

    def test_malformed_json_args_returns_error_without_executing_tool(self):
        bad_call = _Call("scan_design_pocket", {})
        bad_call.function.arguments = "{not valid json"
        turns = [
            _Msg(tool_calls=[bad_call]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        with mock.patch("enzyme_design_agent.agent.execute_tool") as exec_mock:
            agent.run("q")
        exec_mock.assert_not_called()
        transcript = json.loads((agent.run_dir / "agent_state" / "transcript.json").read_text())
        tool_msgs = [m for m in transcript["messages"] if m.get("role") == "tool"]
        self.assertEqual(1, len(tool_msgs))
        self.assertIn("Malformed JSON arguments", tool_msgs[0]["content"])

    def test_state_headers_are_deduplicated_and_token_budget_finalizes(self):
        pocket = _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})])
        turns = [pocket, pocket, _Msg(content="budget report")]
        agent = self._agent(turns, token_budget=20, max_iterations=20)
        report = agent.run("q")
        self.assertEqual("budget report", report)
        transcript = json.loads((agent.run_dir / "agent_state" / "transcript.json").read_text())
        state_msgs = [
            m for m in transcript["messages"]
            if m.get("role") == "user" and str(m.get("content", "")).startswith("[STATE: ")
        ]
        self.assertEqual(1, len(state_msgs))

    def test_unknown_tool_is_refused_without_killing_the_run(self):
        turns = [
            _Msg(tool_calls=[_Call("search_literature", {})]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        report = agent.run("q")
        self.assertEqual("report", report)
        self.assertEqual([], agent.tools_used)

    def test_a_raising_tool_becomes_an_error_result(self):
        turns = [
            _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        with mock.patch("enzyme_design_agent.agent.execute_tool",
                        side_effect=RuntimeError("boom")):
            report = agent.run("q")
        self.assertEqual("report", report)

    def test_llm_failure_ends_the_run_as_failed(self):
        llm = mock.Mock()
        llm.call.return_value = None
        with mock.patch("enzyme_design_agent.agent.LLMClient", return_value=llm):
            agent = EnzymeDesignAgent(output_dir=self.out, query="q")
        agent.run("q")
        m = json.loads((agent.run_dir / "manifest.json").read_text())
        self.assertEqual("failed", m["status"])

    def test_an_empty_first_turn_is_a_nudge_not_a_finish(self):
        turns = [_Msg(content=""), _Msg(content="real report")]
        agent = self._agent(turns)
        self.assertEqual("real report", agent.run("q"))

    def test_finalization_is_forced_before_the_iteration_cap(self):
        """The run must never hit the cap with nothing written."""
        pocket = _Msg(tool_calls=[_Call("scan_design_pocket",
                                        {"structure_path": str(self.pdb)})])
        agent = self._agent([pocket] * 10, max_iterations=5)
        agent.run("q")
        m = json.loads((agent.run_dir / "manifest.json").read_text())
        self.assertLessEqual(m["iterations_used"], 5)
        self.assertTrue((agent.run_dir / "DESIGN_REPORT.md").is_file())

    def test_artifacts_are_saved_per_tool_call(self):
        turns = [
            _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        agent.run("q")
        saved = list((agent.run_dir / "artifacts" / "design").glob("*.json"))
        self.assertEqual(1, len(saved))
        self.assertEqual("scan_design_pocket",
                         json.loads(saved[0].read_text())["tool_name"])

    def test_scaffold_presence_does_not_preset_the_strategy(self):
        agent = self._agent([_Msg(content=""), _Msg(content="r")])
        agent.run("q", scaffold_path=str(self.pdb))
        self.assertEqual("undecided", agent.strategy)

    def test_prompt_prefix_stays_cache_stable_across_turns(self):
        """[STATE: ...] is appended only to the transient tail, never mutating history."""
        seen_calls: list[list[dict]] = []
        pocket = _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})])
        turns = [pocket, pocket, _Msg(content="report")]
        llm = _fake_llm(turns)
        orig_side_effect = list(llm.call.side_effect)

        def _capture(*, messages, tools, timeout):
            seen_calls.append(json.loads(json.dumps(messages, default=str)))
            return orig_side_effect[len(seen_calls) - 1]

        llm.call.side_effect = _capture
        with mock.patch("enzyme_design_agent.agent.LLMClient", return_value=llm):
            agent = EnzymeDesignAgent(output_dir=self.out, query="q")
        agent.run("q")

        self.assertEqual(3, len(seen_calls))
        # Every turn ends with exactly one [STATE: ...] header.
        for call_msgs in seen_calls:
            self.assertTrue(call_msgs[-1]["content"].startswith("[STATE: "))
            self.assertEqual(
                1,
                sum(1 for m in call_msgs if str(m.get("content", "")).startswith("[STATE: ")),
            )
        # Turn 2's prefix (everything before turn 2's new assistant/tool turn) equals
        # turn 1's non-state prefix byte-for-byte.
        self.assertEqual(seen_calls[0][:-1], seen_calls[1][:len(seen_calls[0]) - 1])
        self.assertEqual(seen_calls[1][:-1], seen_calls[2][:len(seen_calls[1]) - 1])

    def test_build_design_context_compacts_whole_groups(self):
        from enzyme_design_agent.agent import _build_design_context

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 400},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "tool_call_id": "2", "content": "y" * 200},
        ]
        compacted = _build_design_context(msgs, "[STATE: iter=3]", max_chars=450)
        self.assertEqual(compacted[:2], msgs[:2])
        self.assertEqual(compacted[-1]["content"], "[STATE: iter=3]")
        # First assistant/tool group dropped cleanly; second group kept intact.
        self.assertEqual([m["role"] for m in compacted[2:-1]], ["assistant", "tool"])
        self.assertEqual(compacted[3]["tool_call_id"], "2")

    def test_structured_handoff_is_ingested_and_emits_output_handoff(self):
        handoff_path = self.out / "design_handoff.json"
        handoff_path.write_text(json.dumps({
            "kind": "de_novo_design_request",
            "purpose": "de_novo_enzyme_design",
            "requested_tool": "rfd4-proteina",
            "source_accession": "SYNTH1",
            "target": {"name": "amide_ligation"},
            "catalytic_profile_summary": {"evolvability_class": "soft_defect", "axes": {"preorganization": "poor"}},
            "soft_defect_design_targets": ["RES42", "RES87"],
            "failure_modes": ["loop_occlusion"],
        }))
        seen_first_user: list[str] = []
        turns = [
            _Msg(tool_calls=[_Call("scan_design_pocket", {"structure_path": str(self.pdb)})]),
            _Msg(content="Handoff-informed design report."),
        ]
        llm = _fake_llm(turns)
        orig = list(llm.call.side_effect)
        idx = [0]

        def _capture(*, messages, tools, timeout):
            if idx[0] == 0:
                seen_first_user.append(messages[1]["content"])
            resp = orig[idx[0]]
            idx[0] += 1
            return resp

        llm.call.side_effect = _capture
        with mock.patch("enzyme_design_agent.agent.LLMClient", return_value=llm):
            agent = EnzymeDesignAgent(output_dir=self.out, query="q")
        agent.run("design for amide ligation", handoff_path=handoff_path)

        self.assertIn("soft_defect_design_targets", seen_first_user[0])
        self.assertIn("RES42", seen_first_user[0])
        manifest = json.loads((agent.run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["source_handoff"]["source_accession"], "SYNTH1")
        out_handoff = json.loads((agent.run_dir / "design_output_handoff.json").read_text())
        self.assertEqual(out_handoff["kind"], "design_output_handoff")
        self.assertEqual(out_handoff["strategy"], "redesign")
        self.assertEqual(out_handoff["source_handoff"]["source_accession"], "SYNTH1")

    def test_declare_design_strategy_persists_decision_and_sets_strategy(self):
        turns = [
            _Msg(tool_calls=[_Call("declare_design_strategy", {
                "strategy": "generative",
                "backend": "rfd4-proteina",
                "rationale": "No holo scaffold with plausible attack geometry.",
                "motif_or_mutable_spec": "Ser105 nucleophile + His224 base + Asp187 charge relay.",
                "falsification_criteria": "Reject if attack distance > 3.5 A or motif RMSD > 1.0 A.",
                "negative_controls": "Ser105Ala catalytic knockout.",
            })]),
            _Msg(content="Strategy declared; awaiting GPU window."),
        ]
        agent = self._agent(turns)
        agent.run("design an esterase")
        self.assertEqual("generative", agent.strategy)
        ledger = json.loads((agent.run_dir / "agent_state" / "design_strategy.json").read_text())
        self.assertEqual(ledger["current"]["backend"], "rfd4-proteina")
        self.assertEqual(len(ledger["decisions"]), 1)
        manifest = json.loads((agent.run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["strategy_decision"]["backend"], "rfd4-proteina")
        out_handoff = json.loads((agent.run_dir / "design_output_handoff.json").read_text())
        self.assertEqual(out_handoff["strategy_decision"]["revision"], 1)


if __name__ == "__main__":
    unittest.main()
