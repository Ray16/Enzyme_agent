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
                         [PAPERS[2]["title"], PAPERS[0]["title"], PAPERS[1]["title"]])
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
        self.assertEqual(out[0]["title"], PAPERS[1]["title"])
        self.assertEqual({p["title"] for p in out}, {p["title"] for p in PAPERS})

    def test_empty_selection_falls_back(self):
        sentinel = [{"title": "FB"}]
        out = select_papers_to_read(self._run(), PAPERS, 6, _FakeLLM(json.dumps({"selected": []})),
                                    "q", fallback=lambda: sentinel)
        self.assertIs(out, sentinel)

    def test_empty_pool_returns_empty(self):
        out = select_papers_to_read(self._run(), [], 6, _FakeLLM("{}"), "q",
                                    fallback=lambda: ["x"])
        self.assertEqual(out, [])

    def test_shrinking_unread_queue_reuses_priority_but_new_evidence_reopens_it(self):
        papers = [{"title": "Synthetic study one", "abstract": "Measured enzyme activity"},
                  {"title": "Synthetic study two", "abstract": "Measured enzyme activity"},
                  {"title": "Synthetic study three", "abstract": "Measured enzyme activity"}]
        llm = _FakeLLM(json.dumps({"selected": [2, 0]}))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = select_papers_to_read(root, papers, 3, llm, "generic transformation", fallback=lambda: papers)
            subset = [out[-1], out[0]]
            reused = select_papers_to_read(root, subset, 3, llm, "generic transformation", fallback=lambda: subset)
            self.assertEqual(reused, [out[0], out[-1]])
            self.assertEqual(llm.calls, 1)
            expanded = subset + [{"title": "Synthetic new study", "abstract": "New measurements"}]
            select_papers_to_read(root, expanded, 3, llm, "generic transformation", fallback=lambda: expanded)
            self.assertEqual(llm.calls, 2)
            select_papers_to_read(root, expanded, 3, llm, "a different transformation", fallback=lambda: expanded)
            self.assertEqual(llm.calls, 3)

    def test_empty_selection_does_not_repeat_calls_for_each_unread_subset(self):
        papers = [{"title": "Synthetic study one"}, {"title": "Synthetic study two"}]
        llm = _FakeLLM('{"selected": []}')
        with tempfile.TemporaryDirectory() as d:
            select_papers_to_read(Path(d), papers, 3, llm, "target", fallback=lambda: papers)
            select_papers_to_read(Path(d), papers[1:], 3, llm, "target", fallback=lambda: papers[1:])
        self.assertEqual(llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
