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
        self.assertIn("design_de_novo_enzyme", names)
        self.assertIn("build_saturation_library", names)
        self.assertIn("scan_design_pocket", names)

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
            _Msg(tool_calls=[_Call("check_de_novo_design_readiness", {}, cid="c2")]),
            _Msg(content="report"),
        ]
        agent = self._agent(turns)
        # Stubbed: the readiness tool probes real GPUs, and this test is about
        # strategy bookkeeping, not about the hardware under it.
        with mock.patch("enzyme_design_agent.agent.execute_tool",
                        return_value={"status": "success", "data": {}}):
            agent.run("q")
        self.assertEqual("both", agent.strategy)

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


if __name__ == "__main__":
    unittest.main()
