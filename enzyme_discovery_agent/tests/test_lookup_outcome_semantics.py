"""Unavailable lookups must never become scientific negatives or broader queries."""
from unittest.mock import patch
import pytest
from enzyme_discovery_agent.tools import database


def test_interpro_outage_does_not_relax_query():
    with patch.object(database, 'request_with_retry', side_effect=TimeoutError) as fetch:
        result = database.search_interpro('specific synthetic family')
    assert fetch.call_count == 1
    assert result['status'] == 'error'
    assert result['lookup_status'] == 'unavailable'
    assert result['search_completed'] is False
    assert result['used_query'] == result['requested_query']


def test_interpro_empty_response_is_confirmed_empty():
    with patch.object(database, 'request_with_retry', return_value=b''):
        result = database.search_interpro('synthetic')
    assert result['lookup_status'] == 'empty'
    assert result['search_completed'] is True
    assert result['count'] == 0


def test_interpro_relaxes_only_after_confirmed_empty_and_records_scope():
    with patch.object(database, 'request_with_retry', side_effect=[b'', b'{"results":[{"metadata":{"accession":"IPR000001","name":"Example"}}]}']) as fetch:
        result = database.search_interpro('specific synthetic family')
    assert fetch.call_count == 2
    assert result['lookup_status'] == 'found'
    assert result['query_relaxed'] is True
    assert result['requested_query'] == 'specific synthetic family'
    assert result['used_query'] == 'specific synthetic'


def test_interpro_invalid_body_is_unavailable():
    with patch.object(database, 'request_with_retry', return_value=b'<html>error</html>'):
        result = database.get_protein_domains('P12345')
    assert result['lookup_status'] == 'unavailable'
    assert not result['search_completed']


def test_legacy_mechanism_cache_requires_an_explicit_store():
    from enzyme_discovery_agent.tools.mechanism_cache import MechanismCache
    with pytest.raises(TypeError):
        MechanismCache()


def test_chebi_transport_failure_is_retryable():
    from enzyme_discovery_agent import chemistry_logic as chemistry
    key = 'CHEBI:999999999'
    chemistry._chebi_smiles_cache.pop(key, None)
    with patch.object(chemistry, 'request_with_retry', side_effect=[TimeoutError(), b'{"_embedded":{"terms":[{"annotation":{"smiles_string":["CCO"]},"label":"Example"}]}}']) as fetch:
        assert chemistry.resolve_chebi_smiles(key) is None
        assert key not in chemistry._chebi_smiles_cache
        assert chemistry.resolve_chebi_smiles(key) == 'CCO'
        assert fetch.call_count == 2
    chemistry._chebi_smiles_cache.pop(key, None)
    chemistry._chebi_name_cache.pop(key, None)


def test_partial_expansion_can_retry_after_backoff(tmp_path):
    from enzyme_discovery_agent.models.expansion import record_seed_search, searched_seeds
    with patch('enzyme_discovery_agent.models.expansion.time.time', return_value=1000):
        record_seed_search(tmp_path, 'synthetic', 'P12345', {'status': 'partial'})
        assert 'P12345' in searched_seeds(tmp_path, 'synthetic')
    with patch('enzyme_discovery_agent.models.expansion.time.time', return_value=2000):
        assert 'P12345' not in searched_seeds(tmp_path, 'synthetic')
        record_seed_search(tmp_path, 'synthetic', 'P12345', {'status': 'error'})
    with patch('enzyme_discovery_agent.models.expansion.time.time', return_value=10000):
        assert 'P12345' not in searched_seeds(tmp_path, 'synthetic')
        record_seed_search(tmp_path, 'synthetic', 'P12345', {'status': 'empty'})
        assert 'P12345' in searched_seeds(tmp_path, 'synthetic')


def test_failed_harvest_is_not_marked_processed_and_retries_are_bounded(tmp_path):
    import json
    from types import SimpleNamespace
    from enzyme_discovery_agent.gates import AgentState
    from enzyme_discovery_agent.gates.literature_harvest import LiteratureHarvestGate
    state = AgentState(run_dir=tmp_path)
    state.accumulated_papers = [{'doi': '10.test/synthetic', 'title': 'Synthetic'}]
    gate = LiteratureHarvestGate(SimpleNamespace(budget_exhausted=False))
    with patch('enzyme_discovery_agent.tools.fulltext.mine_and_register_fulltext_accessions', side_effect=TimeoutError), \
         patch('enzyme_discovery_agent.gates.literature_harvest.time.time', return_value=1000):
        result = gate.execute(state, [])
        assert not result.intercept
        assert not (tmp_path / 'agent_state/literature_harvest_batch.json').exists()
        assert not gate.should_fire(state)
    with patch('enzyme_discovery_agent.gates.literature_harvest.time.time', return_value=2000):
        assert gate.should_fire(state)
    failure = json.loads((tmp_path / 'agent_state/literature_harvest_failure.json').read_text())
    assert failure['batch_processed'] is False
    assert failure['status'] == 'retry_pending'
