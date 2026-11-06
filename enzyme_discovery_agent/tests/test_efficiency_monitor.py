import json
from datetime import datetime, timezone

from enzyme_discovery_agent.output.efficiency_monitor import summarize, refresh


def test_only_metered_matching_profiles_contribute_to_measured_savings(tmp_path):
    (tmp_path/'token_usage.json').write_text(json.dumps({'total_tokens': 12, 'calls': [
        {'iteration': 1, 'prompt_tokens': 10, 'completion_tokens': 2, 'request_profile_id': 'ok'}]}))
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'llm_request_profile', 'request_profile_id': 'ok',
         'messages_characters_before_json_compaction': 100, 'messages_characters': 80},
        {'type': 'llm_request_profile', 'request_profile_id': 'abandoned',
         'messages_characters_before_json_compaction': 1000, 'messages_characters': 500}])+'\n')
    result = summarize(tmp_path)
    assert result['json_compaction']['saved_characters'] == 20
    assert result['json_compaction']['matched_calls'] == 1
    assert result['stages']['controller']['calls'] == 1
    assert result['tokens'] == 12


def test_unresolved_names_are_not_counted_as_verified_sequences_or_distinct_families(tmp_path):
    (tmp_path/'candidate_registry.json').write_text(json.dumps({'candidates': {
        'Synthetic lead': {'metadata': {'protein_name': 'Synthetic lead', 'is_literature_lead': True}}}}))
    result = summarize(tmp_path)
    assert result['registered_candidates'] == 1
    assert result['verified_sequence_records'] == 0
    assert result['annotated_family_groups'] == 0


def test_live_monitor_preserves_journal_and_reports_source_integrity_failure(tmp_path):
    from enzyme_discovery_agent.models.reasoning_trace import append_scientific_trace
    source = tmp_path/'source.json'; source.write_text('{}')
    import hashlib
    append_scientific_trace(tmp_path, 'test_source', {'source': {
        'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}},
        record_id='synthetic', occurred_at=datetime.now(timezone.utc).isoformat())
    journal = tmp_path/'reasoning/trace.jsonl'
    before = journal.read_bytes()
    first = refresh(tmp_path)
    assert first['reasoning']['source_files_intact']
    source.write_text('{"changed":true}')
    second = refresh(tmp_path)
    assert not second['reasoning']['source_files_intact']
    assert any('integrity' in item for item in second['alerts'])
    assert journal.read_bytes() == before
    assert second['quality_comparison'] == 'Not yet assessable'


def test_observer_does_not_reconstruct_missing_scientific_decisions(tmp_path):
    from enzyme_discovery_agent.models import research_decisions as decisions
    from enzyme_discovery_agent.tests.test_scientific_trace import proposal
    from unittest.mock import patch
    with patch('enzyme_discovery_agent.models.reasoning_trace.capture_scientific_record'):
        decisions.start_decision(tmp_path, proposal(tmp_path), {'synthetic_retrieval'})
    result = refresh(tmp_path)
    assert result['reasoning']['registered_tests'] == 0
    assert not (tmp_path/'reasoning/trace.jsonl').exists()
