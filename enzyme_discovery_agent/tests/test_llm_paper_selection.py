"""LLM-reasoned selection of which retrieved papers to read in full.

The agent (not a keyword scorer) decides which papers are most relevant to the
DERIVED target reaction.  These tests pin the plumbing deterministically with a
fake client (index parsing, priority order, invalid-index handling) and the
fail-safe fallbacks; the actual reasoning quality is checked by the separate
LLM acceptance script.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from enzyme_discovery_agent.tools.literature_extraction import select_papers_to_read


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def call(self, messages, **kw):
        self.calls += 1
        return _Resp(self.content)


PAPERS = [
    {"title": "Broad-Spectrum Amide Bond Synthetase",
     "abstract": "We characterise an ATP-dependent amide bond synthetase that ligates acids and amines."},
    {"title": "Carbazole Aromatic Polyamide by Palladium-catalyzed Carbonylation",
     "abstract": "A chemical Pd-catalyzed synthetic route to an aromatic polymer."},
    {"title": "Regioselective biosynthesis of oligoamides for sequence-controlled co-polyamides",
     "abstract": "Enzymatic biosynthesis of oligoamides as precursors for polymers."},
]


class TestSelectPapers(unittest.TestCase):
    def _run(self):
        return Path(tempfile.mkdtemp())

    def test_returns_chosen_in_priority_order(self):
        llm = _FakeLLM(json.dumps({"selected": [2, 0]}))
        out = select_papers_to_read(self._run(), PAPERS, 6, llm, "q", fallback=lambda: [])
        self.assertEqual([p["title"] for p in out],
                         [PAPERS[2]["title"], PAPERS[0]["title"]])
        self.assertEqual(llm.calls, 1)

    def test_none_client_uses_fallback(self):
        sentinel = [{"title": "FB"}]
        out = select_papers_to_read(self._run(), PAPERS, 6, None, "q",
                                    fallback=lambda: sentinel)
        self.assertIs(out, sentinel)

    def test_bad_json_falls_back(self):
        sentinel = [{"title": "FB"}]
        out = select_papers_to_read(self._run(), PAPERS, 6, _FakeLLM("not json"), "q",
                                    fallback=lambda: sentinel)
        self.assertIs(out, sentinel)

    def test_invalid_indices_ignored(self):
        # out-of-range, non-int, and duplicate indices are dropped; booleans too.
        llm = _FakeLLM(json.dumps({"selected": [99, 1, "x", 1, True]}))
        out = select_papers_to_read(self._run(), PAPERS, 6, llm, "q", fallback=lambda: [])
        self.assertEqual([p["title"] for p in out], [PAPERS[1]["title"]])

    def test_empty_selection_falls_back(self):
        sentinel = [{"title": "FB"}]
        out = select_papers_to_read(self._run(), PAPERS, 6, _FakeLLM(json.dumps({"selected": []})),
                                    "q", fallback=lambda: sentinel)
        self.assertIs(out, sentinel)

    def test_empty_pool_returns_empty(self):
        out = select_papers_to_read(self._run(), [], 6, _FakeLLM("{}"), "q",
                                    fallback=lambda: ["x"])
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
