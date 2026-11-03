"""Deposited identity gaps remain pending without malformed database requests."""
from unittest.mock import patch

from enzyme_discovery_agent.gates.ec_triage import auto_screen_candidates
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry, discovery_status
from enzyme_discovery_agent.tools.reasoning import screen_candidates, evaluate_candidate


def setup_lead(root):
    accession = 'sp|P12345.1|SYNTH'
    update_candidate_registry(root, accession, 'discovered', {'is_literature_deposit': True,
        'sequence': 'ACDE', 'literature_name': 'SyntheticLead'})
    return accession


def test_auto_screen_defers_unresolved_deposit_without_lookup(tmp_path):
    acc = setup_lead(tmp_path)
    with patch('enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light') as fetch:
        auto_screen_candidates(tmp_path)
    fetch.assert_not_called()
    entry = load_candidate_registry(tmp_path)['candidates'][acc]
    assert discovery_status(entry) == 'evidence_pending'
    assert entry['metadata']['sequence'] == 'ACDE'
    assert entry['metadata']['identity_resolution_status'] == 'pending'


def test_manual_screen_and_evaluation_do_not_query_uniprot(tmp_path):
    acc = setup_lead(tmp_path)
    with patch('enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light') as light, \
         patch('enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry') as full:
        screened = screen_candidates(tmp_path, [acc])
        result = evaluate_candidate(tmp_path, acc)
    light.assert_not_called(); full.assert_not_called()
    assert screened['data'][0]['screening_decision'] == 'evidence_pending'
    assert result['status'] == 'partial'
    assert discovery_status(load_candidate_registry(tmp_path)['candidates'][acc]) == 'evidence_pending'
    assert list((tmp_path / 'artifacts/candidates').glob('screen_identifier_*.json'))


def test_rediscovery_never_demotes_a_pending_or_evaluated_candidate(tmp_path):
    from enzyme_discovery_agent.models.candidate_registry import extract_and_register_from_search
    for accession, state in [('P12345', 'evidence_pending'), ('Q12345', 'evaluated')]:
        update_candidate_registry(tmp_path, accession, state, {'protein_name': 'Example'})
        extract_and_register_from_search(tmp_path, 'search_uniprot', {'status': 'success', 'data': [
            {'accession': accession, 'name': 'Example', 'gene_names': ['example'], 'seq_length': 200}]})
        assert discovery_status(load_candidate_registry(tmp_path)['candidates'][accession]) == state


def test_quarantined_database_accession_is_not_rescreened(tmp_path):
    accession = 'P12345'
    update_candidate_registry(tmp_path, accession, 'discovered', {'is_literature_lead': True,
        'identity_resolution_status': 'conflict'})
    with patch('enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry_light') as fetch:
        auto_screen_candidates(tmp_path)
        result = screen_candidates(tmp_path, [accession])
    fetch.assert_not_called()
    assert result['data'][0]['screening_decision'] == 'evidence_pending'
