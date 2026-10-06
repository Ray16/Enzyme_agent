"""Tool-arg type coercion: tolerate backends that stringify int/number/bool args.

A real run on an open-weight fallback (Nemotron via OpenRouter) passed max_results
as the string "10", crashing tools that slice/compare with it.  execute_tool now
coerces args to the declared schema types before dispatch.
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent import tool_registry
from enzyme_discovery_agent.tool_registry import _coerce_arg_types, _coerce_one, execute_tool


class TestCoerceHelpers(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(_coerce_one("10", "integer"), 10)
        self.assertEqual(_coerce_one("-5", "integer"), -5)
        self.assertEqual(_coerce_one(10.0, "integer"), 10)
        self.assertEqual(_coerce_one(7, "integer"), 7)          # already int
        self.assertEqual(_coerce_one("abc", "integer"), "abc")  # uncoercible -> unchanged
        self.assertIs(_coerce_one(True, "integer"), True)       # bool left alone

    def test_number_and_boolean(self):
        self.assertEqual(_coerce_one("3.14", "number"), 3.14)
        self.assertIs(_coerce_one("true", "boolean"), True)
        self.assertIs(_coerce_one("False", "boolean"), False)
        self.assertEqual(_coerce_one("maybe", "boolean"), "maybe")

    def test_array_object_from_json_string(self):
        self.assertEqual(_coerce_one('["a","b"]', "array"), ["a", "b"])
        self.assertEqual(_coerce_one('{"k":1}', "object"), {"k": 1})
        self.assertEqual(_coerce_one("not json", "array"), "not json")

    def test_string_and_unknown_type_untouched(self):
        self.assertEqual(_coerce_one("10", "string"), "10")
        self.assertEqual(_coerce_one("10", None), "10")

    def test_coerce_arg_types_maps_by_schema(self):
        props = {"max_results": {"type": "integer"}, "q": {"type": "string"}}
        out = _coerce_arg_types({"max_results": "25", "q": "adipic"}, props)
        self.assertEqual(out, {"max_results": 25, "q": "adipic"})


class TestExecuteToolCoerces(unittest.TestCase):
    def setUp(self):
        self.name = "_coerce_probe_tool"
        tool_registry._TOOLS[self.name] = {
            "func": lambda **kw: {"status": "success",
                                  "types": {k: type(v).__name__ for k, v in kw.items()}},
            "schema": {"function": {"parameters": {
                "properties": {"max_results": {"type": "integer"},
                               "flag": {"type": "boolean"},
                               "q": {"type": "string"}},
                "required": []}}},
            "injected_params": [],
            "category": "test",
            "parallel_safe": True,
        }

    def tearDown(self):
        tool_registry._TOOLS.pop(self.name, None)

    def test_string_ints_and_bools_coerced_before_dispatch(self):
        out = execute_tool(self.name, {"max_results": "10", "flag": "true", "q": "x"})
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["types"]["max_results"], "int")
        self.assertEqual(out["types"]["flag"], "bool")
        self.assertEqual(out["types"]["q"], "str")


class TestContainerArgsRejectedNotCrashed(unittest.TestCase):
    """A declared array/object that cannot be coerced must not reach the tool.

    A real run passed `papers` as a JSON-encoded string; the tool then iterated it
    character-by-character and died with
    ``AttributeError: 'str' object has no attribute 'get'``.
    """

    def setUp(self):
        self.name = "_container_probe_tool"
        tool_registry._TOOLS[self.name] = {
            # Mimics a tool that trusts the declared type and calls .get() per item.
            "func": lambda **kw: {"status": "success",
                                  "n": sum(p.get("x", 0) for p in kw["items"])},
            "schema": {"function": {"parameters": {
                "properties": {"items": {"type": "array", "items": {"type": "object"}},
                               "opts": {"type": "object"}},
                "required": []}}},
            "injected_params": [],
            "category": "test",
            "parallel_safe": True,
        }

    def tearDown(self):
        tool_registry._TOOLS.pop(self.name, None)

    def test_json_encoded_array_is_parsed_and_dispatched(self):
        out = execute_tool(self.name, {"items": '[{"x": 2}, {"x": 3}]'})
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["n"], 5)

    def test_unparseable_array_string_errors_instead_of_crashing(self):
        out = execute_tool(self.name, {"items": "not json at all"})
        self.assertEqual(out["status"], "error")
        self.assertIn("must be a JSON array", out["summary"])
        self.assertIn("str", out["summary"])

    def test_truncated_json_array_errors_instead_of_crashing(self):
        out = execute_tool(self.name, {"items": '[{"x": 2}, {"x"'})
        self.assertEqual(out["status"], "error")
        self.assertIn("must be a JSON array", out["summary"])

    def test_wrong_container_kind_errors(self):
        out = execute_tool(self.name, {"items": [], "opts": "[1,2]"})
        self.assertEqual(out["status"], "error")
        self.assertIn("must be a JSON object", out["summary"])


if __name__ == "__main__":
    unittest.main()
