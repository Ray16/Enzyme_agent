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
import pytest
from pathlib import Path

from enzyme_discovery_agent.tools.literature_extraction import select_papers_to_read


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0
        self.requests = []

    def call(self, messages, **kw):
        self.calls += 1
        self.requests.append(messages)
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
        self.assertEqual(out[0]["title"], PAPERS[1]["title"])
        self.assertEqual(out, [PAPERS[1]])

    def test_explicit_empty_selection_defers_reading(self):
        sentinel = [{"title": "FB"}]
        out = select_papers_to_read(self._run(), PAPERS, 6, _FakeLLM(json.dumps({"selected": []})),
                                    "q", fallback=lambda: sentinel)
        self.assertEqual(out, [])

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


def test_objectives_include_fact_validation_and_evidence_changes_reopen_deferrals(tmp_path):
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
    from enzyme_discovery_agent.models.literature_reading import pending_reading
    papers = [{'title': 'Synthetic assay control study', 'doi': '10.test/controls',
               'abstract': 'Measured controls for a previously described activity.'},
              {'title': 'Synthetic background review', 'doi': '10.test/background', 'abstract': 'General context.'}]
    objective = {'index': 0, 'purpose': 'validate_claim', 'subject': 'SYNTH_A',
                 'question': 'Does the matched control distinguish enzyme activity from background?',
                 'expected_effect': 'Validate or weaken the existing activity interpretation.'}
    llm = _FakeLLM(json.dumps({'selected': [objective]}))
    select_papers_to_read(tmp_path, papers, 2, llm, 'generic target', fallback=lambda: papers)
    memo = json.loads((tmp_path/'agent_state/literature_paper_priority.json').read_text())
    assert list(memo['objectives'].values()) == [{k:v for k,v in objective.items() if k != 'index'}]
    assert len(memo['deferred']) == 1
    assert len(pending_reading(tmp_path, papers)) == 1
    # Deferred background is still available; new candidate evidence can change its value.
    update_candidate_registry(tmp_path, 'SYNTH_A', 'evaluated', {
        'protein_name': 'Synthetic catalyst', 'target_assessment': {
            'decision': 'investigate', 'strengths': ['Activity may depend on a coupled reaction'],
            'unknowns': ['Which reported controls establish the assigned step?']}})
    assert len(pending_reading(tmp_path, papers)) == 2
    select_papers_to_read(tmp_path, papers, 2, llm, 'generic target', fallback=lambda: papers)
    assert llm.calls == 2
    assert 'Which reported controls establish the assigned step?' in json.dumps(llm.requests[-1])
    assert len(pending_reading(tmp_path, papers)) == 1
    update_candidate_registry(tmp_path, 'SYNTH_A', None, {
        'function_comments': ['New source evidence arrived before the assessment was revised.']})
    assert len(pending_reading(tmp_path, papers)) == 2


def test_unavailable_selection_does_not_defer_sources(tmp_path):
    from enzyme_discovery_agent.models.literature_reading import pending_reading
    papers = [{'title': 'Synthetic primary study', 'doi': '10.test/primary'}]
    select_papers_to_read(tmp_path, papers, 1, _FakeLLM('invalid'), 'target', fallback=lambda: papers)
    assert pending_reading(tmp_path, papers)
    assert not (tmp_path/'agent_state/literature_paper_priority.json').exists()


@pytest.mark.parametrize("quote_valid", [True, False])
def test_fact_check_outcome_is_saved_without_requiring_new_enzyme_names(tmp_path, monkeypatch, quote_valid):
    from enzyme_discovery_agent.tools.fulltext import _register_named_enzymes_from_fulltext
    from enzyme_discovery_agent.models.literature_reading import reading_state
    dest = tmp_path/'artifacts/fulltext';dest.mkdir(parents=True)
    text = 'A matched inactive control produced the same signal.'
    (dest/'source.txt').write_text(text)
    objective = {'purpose': 'assay_controls', 'subject': 'SYNTH_A',
                 'question': 'Is activity distinguishable from background?',
                 'expected_effect': 'Check the previous activity claim.'}
    received = []
    def extract(*args, **kwargs):
        received.append(kwargs['reading_objective'])
        return {'paper_relevant': True, 'enzymes': [], 'reading_answer': {
            'status': 'contradicted', 'finding': 'This assay does not distinguish the contribution.',
            'evidence_quote': text if quote_valid else 'An observation absent from the supplied passage.'}}
    monkeypatch.setattr('enzyme_discovery_agent.tools.literature_extraction.extract_from_paper', extract)
    stats = {}
    records = [{'key': '10.test/control', 'doi': '10.test/control', 'text_file': 'source.txt',
                'title': 'Synthetic control study', 'reading_objective': objective}]
    _register_named_enzymes_from_fulltext(tmp_path, 'target', dest, records, _FakeLLM(''),
                                         lambda *args: None, already=set(), remaining=4, stats=stats)
    assert received == [objective]
    assert stats['reading_outcomes'][0]['status'] == ('contradicted' if quote_valid else 'unresolved')
    assert stats['reading_outcomes'][0]['quote_verified'] is quote_valid
    saved = json.loads((dest/'papers_enzymes.json').read_text())[0]
    assert saved['reading_outcomes'][0]['objective'] == objective
    assert reading_state(tmp_path)['extractions']['10.test/control']['status'] == 'done'


def test_prefilter_does_not_defer_unseen_sources(tmp_path):
    from enzyme_discovery_agent.models.literature_reading import pending_reading
    papers = [{'title': 'Synthetic study '+str(i), 'doi': '10.test/'+str(i)} for i in range(3)]
    select_papers_to_read(tmp_path, papers, 1, _FakeLLM('{"selected": []}'), 'target',
                          fallback=lambda: papers, prefilter=1)
    assert len(pending_reading(tmp_path, papers)) == 2
