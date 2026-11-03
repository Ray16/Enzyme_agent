"""Synthetic regressions for evidence, recovery, identity, and scoring."""
import json
from pathlib import Path

import pytest

from enzyme_discovery_agent.bootstrap import register_builtin_tools
from enzyme_discovery_agent.models.claim_validation import validate_uniprot_claims
from enzyme_discovery_agent.models.literature_evidence import build_literature_evidence
from enzyme_discovery_agent.models.ortholog import ortholog_key
from enzyme_discovery_agent.tool_registry import check_search_duplicate, record_search, execute_tool
from enzyme_discovery_agent.tools.literature import dedupe_and_merge
from enzyme_discovery_agent.eval.metrics import score_run


@pytest.mark.parametrize('quote', [
    'SYNTH was purified for a separate study.',
    'SYNTH does not hydrolyze an ester bond in the tested substrate.',
    'SYNTH may hydrolyze an ester bond in the tested substrate.',
    'SYNTH hydrolyzes an ester bond in a simulation.',
    'SYNTH was purified. OTHER hydrolyzes an ester bond in the tested substrate.',
])
def test_integrity_is_not_claim_support(tmp_path, quote):
    source = tmp_path / 'source.txt'
    source.write_text(quote)
    record = {'name': 'SYNTH', 'evidence_quote': quote,
              'reaction': {'direction': 'cleavage', 'bond_formed': 'ester'}}
    evidence = build_literature_evidence('P12345', record, {'doi': '10.test/synthetic'}, source,
                                         identity_basis='deposited_accession')
    result = validate_uniprot_claims({'primaryAccession': 'P12345', 'entryType': 'reviewed'},
        expected_accession='P12345', target_direction='cleavage', target_bond_types={'ester'},
        literature_evidence=evidence)
    assert not result['promotion_eligible']
    assert 'native_reaction' in result['blocking_claims']


def test_search_retry_parameters_and_run_isolation():
    register_builtin_tools()
    first, second = [], []
    request = {'query': 'synthetic query', 'max_results': 10}
    record_search('search_pubmed', request, {'status': 'error'}, history=first)
    assert check_search_duplicate('search_pubmed', request, history=first) is None
    record_search('search_pubmed', request, {'status': 'success'}, history=first)
    assert check_search_duplicate('search_pubmed', request, history=first)['deduplicated']
    assert check_search_duplicate('search_pubmed', {**request, 'max_results': 100}, history=first) is None
    assert check_search_duplicate('search_pubmed', {**request, 'rerank_query': 'other'}, history=first) is None
    assert check_search_duplicate('search_pubmed', request, history=second) is None


@pytest.mark.parametrize('args', [None, 3, 'invalid', []])
def test_non_object_tool_arguments(args):
    assert execute_tool('search_pubmed', args)['status'] == 'error'


def test_scalar_schema_and_enum_validation():
    register_builtin_tools()
    assert execute_tool('search_pubmed', {'query': 'x', 'max_results': 'nonsense'})['status'] == 'error'
    assert execute_tool('search_pubmed', {'query': 'x', 'max_results': True})['status'] == 'error'


def test_partial_identifiers_merge_but_conflicting_identifiers_do_not():
    papers = [
        {'doi': '10.test/a', 'pmid': '123', 'title': 'Synthetic paper', 'source': 'a'},
        {'pmid': '123', 'title': 'Synthetic paper', 'source': 'b'},
        {'doi': '10.test/conflict', 'pmid': '123', 'title': 'Synthetic paper', 'source': 'c'},
    ]
    unique, stats = dedupe_and_merge(papers)
    assert stats['unique'] == 2
    assert unique[0]['sources'] == ['a', 'b']


def test_names_are_not_sequence_identity():
    assert ortholog_key({'gene_names': ['shared'], 'ec_numbers': ['1.2.3.4']}) is None
    assert ortholog_key({'sequence': 'ACDE'}) == ortholog_key({'sequence': 'acde'})
    assert ortholog_key({'sequence': 'ACDE'}) != ortholog_key({'sequence': 'ACDF'})


def test_failed_search_query_is_not_retrieval(tmp_path):
    folder = tmp_path / 'artifacts' / 'database'
    folder.mkdir(parents=True)
    (folder / 'failure.json').write_text(json.dumps({'args': {'query': 'synthetic_answer'},
        'result': {'status': 'error', 'data': []}}))
    bench = {'recall_targets': {'known': {'scope': 'artifacts', 'values': ['synthetic_answer']}},
             'primary_targets': ['known']}
    assert not score_run(tmp_path, bench)['found_answer']
    (folder / 'empty.json').write_text(json.dumps({'result': {'status': 'success',
        'data': {'query': 'synthetic_answer', 'results': []}}}))
    assert not score_run(tmp_path, bench)['found_answer']


def test_independent_adjudication_counts_unjudged_separately():
    from enzyme_discovery_agent.eval.adjudicated import score_case
    case = {'id': 'synthetic', 'adjudications': {'p': 'positive', 'n': 'negative'}}
    result = score_case(case, {'id': 'synthetic', 'retrieved': ['p', 'n', 'u'],
                               'ranked': ['u', 'p'], 'abstained': False})
    assert result['stages']['retrieved']['recall'] == 1
    assert result['stages']['retrieved']['judged_precision'] == .5
    assert result['stages']['retrieved']['unjudged'] == 1
    assert result['best_positive_rank'] == 2


def test_cost_report_separates_models_and_overlapping_tool_time(tmp_path):
    from enzyme_discovery_agent.eval.cost_analysis import analyze_run
    events = [{'type': 'llm_call', 'backend': 'b', 'model': 'm', 'prompt_tokens': 12,
               'completion_tokens': 3, 'ts': '2026-01-01T00:00:00+00:00'},
              {'type': 'tool_call', 'tool': 'search', 'status': 'success', 'elapsed_s': 4,
               'novelty': {'new_accessions': 2}, 'ts': '2026-01-01T00:00:03+00:00'}]
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(row) for row in events))
    report = analyze_run(tmp_path)
    assert report['event_span_seconds'] == 3
    assert report['models']['b:m']['prompt_tokens'] == 12
    assert report['tools']['search']['elapsed_seconds_sum'] == 4
    assert report['tools']['search']['new_accessions'] == 2


def test_historical_rejected_candidate_is_not_current_coverage(tmp_path):
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
    from enzyme_discovery_agent.eval.metrics import _candidate_text
    update_candidate_registry(tmp_path, 'P12345', 'evaluated', {
        'protein_name': 'syntheticfamily', 'claim_validation': {'promotion_eligible': True}})
    assert 'syntheticfamily' in _candidate_text(tmp_path)
    update_candidate_registry(tmp_path, 'P12345', 'rejected')
    assert 'syntheticfamily' not in _candidate_text(tmp_path)
    update_candidate_registry(tmp_path, 'Q12345', 'evaluated', {'protein_name': 'unverifiedfamily'})
    assert 'unverifiedfamily' not in _candidate_text(tmp_path)


def test_shared_name_candidates_reach_screening(tmp_path, monkeypatch):
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry
    from enzyme_discovery_agent.gates.ec_triage import auto_screen_candidates
    from enzyme_discovery_agent.gates import family_relevance
    from enzyme_discovery_agent.tools import reasoning
    for acc in ('P12345', 'Q12345'):
        update_candidate_registry(tmp_path, acc, 'discovered', {'gene_names': ['shared'], 'seq_length': 200})
    seen = []
    def allocate(root, reps, **kwargs):
        seen.extend(acc for acc, _ in reps)
        return set(), []
    monkeypatch.setattr(family_relevance, 'build_relevance_and_allocate', allocate)
    monkeypatch.setattr(reasoning, '_fetch_uniprot_entry_light', lambda acc: {'accession': acc, 'seq_length': 200})
    auto_screen_candidates(tmp_path)
    assert set(seen) == {'P12345', 'Q12345'}
