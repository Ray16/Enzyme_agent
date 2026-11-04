"""The provider consults current shared records rather than a launch-time alias list."""
from structural_modeling_agent.lambda_boltz_backend import check_prediction_pool


def test_new_sequence_is_checked_against_the_supplied_pool(tmp_path, monkeypatch):
    calls = []
    def lookup(pool, **kwargs):
        calls.append((pool, kwargs))
        return {'matches': [{'execution': {'output_presence': 'complete'}}] if kwargs['sequence'] == 'AGGA' else []}
    monkeypatch.setattr('campaign_catalog.lookup.lookup_predictions', lookup)
    result = check_prediction_pool({'sequences': [
        {'protein': {'id': 'X', 'sequence': 'AGGA'}},
        {'protein': {'id': 'Y', 'sequence': 'GAGA'}},
        {'ligand': {'id': 'Z', 'smiles': 'CCO'}}]}, tmp_path)
    assert result['existing_predictions']
    assert [item[1]['sequence'] for item in calls] == ['AGGA', 'GAGA']
    assert all(item[0] == tmp_path for item in calls)
