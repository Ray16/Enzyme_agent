"""Bounded family surveys must cover sources before deepening one source."""
from unittest.mock import patch

from enzyme_discovery_agent.gates.family_relevance import collect_survey_folds


def test_first_batch_covers_all_hypotheses_including_those_after_eighth(tmp_path):
    hypotheses = [{'name': f'Synthetic mechanism {i}'} for i in range(10)]
    def resolve(family, *args):
        i = hypotheses.index(family)
        return [f'IPR{i:03d}{j:03d}' for j in range(12)]
    with patch('enzyme_discovery_agent.gates.family_relevance.resolve_family_folds', side_effect=resolve) as resolver:
        rows = collect_survey_folds(tmp_path, hypotheses, {}, max_folds=16)
    assert resolver.call_count == 10
    assert len(rows) == 16
    assert {row['derived_family_hypothesis']['name'] for row in rows[:10]} == {h['name'] for h in hypotheses}
    assert len({row['fold_id'] for row in rows}) == 16


def test_one_failed_resolution_does_not_skip_later_hypotheses(tmp_path):
    hypotheses = [{'name': 'Synthetic first'}, {'name': 'Synthetic second'}]
    def resolve(family, *args):
        if family == hypotheses[0]:
            raise TimeoutError('synthetic outage')
        return ['IPR000001']
    with patch('enzyme_discovery_agent.gates.family_relevance.resolve_family_folds', side_effect=resolve):
        rows = collect_survey_folds(tmp_path, hypotheses, {})
    assert rows[0]['derived_family_hypothesis']['name'] == 'Synthetic second'


def test_shared_fold_retains_all_hypothesis_links_without_overwriting_first(tmp_path):
    hypotheses = [{'name': 'Synthetic first'}, {'name': 'Synthetic second'}]
    with patch('enzyme_discovery_agent.gates.family_relevance.resolve_family_folds', return_value=['IPR000001']):
        rows = collect_survey_folds(tmp_path, hypotheses, {})
    assert len(rows) == 1
    assert rows[0]['derived_family_hypothesis']['name'] == 'Synthetic first'
    assert {h['name'] for h in rows[0]['derived_family_hypotheses']} == {h['name'] for h in hypotheses}
