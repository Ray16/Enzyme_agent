"""Atomic bulk writes and diversity scheduling, using synthetic candidates."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from enzyme_discovery_agent.models import candidate_registry as registry
from enzyme_discovery_agent.models.rolling import diversify_evaluation_batch


def test_batch_matches_single_lifecycle_rules_and_commits_once(tmp_path):
    updates = [
        ('SYNTH_A', 'discovered', {'protein_name': 'shared'}),
        ('SYNTH_B', 'discovered', {'protein_name': 'shared'}),
        ('SYNTH_A', 'screened', None),
        ('SYNTH_A', 'discovered', {'new_source': True}),
        ('SYNTH_A', 'rejected', {'reason': 'synthetic conflict'}),
        ('SYNTH_A', 'evaluated', None),
        ('SYNTH_A', 'reopened', {'reason': 'new evidence'}),
        ('SYNTH_A', 'evaluated', None),
    ]
    single, bulk = tmp_path/'single', tmp_path/'bulk'
    for update in updates:
        registry.update_candidate_registry(single, *update)
    with patch.object(registry, 'atomic_write_json', wraps=registry.atomic_write_json) as write:
        registry.update_candidate_registry_batch(bulk, updates)
        assert write.call_count == 1
    a = registry.load_candidate_registry(single)['candidates']
    b = registry.load_candidate_registry(bulk)['candidates']
    assert set(a) == set(b) == {'SYNTH_A', 'SYNTH_B'}
    for acc in a:
        assert a[acc]['metadata'] == b[acc]['metadata']
        assert [s['status'] for s in a[acc]['statuses']] == [s['status'] for s in b[acc]['statuses']]
    assert b['SYNTH_A']['metadata']['new_source']


def test_failed_bulk_commit_is_visible_and_prior_snapshot_survives(tmp_path):
    registry.update_candidate_registry(tmp_path, 'SYNTH_A', 'screened')
    with patch.object(registry, 'atomic_write_json', side_effect=OSError('synthetic storage failure')):
        with pytest.raises(OSError):
            registry.update_candidate_registry_batch(tmp_path, [('SYNTH_A', 'evaluated', None), ('SYNTH_B', 'discovered', None)])
    entries = registry.load_candidate_registry(tmp_path, strict=True)['candidates']
    assert set(entries) == {'SYNTH_A'}
    assert registry.discovery_status(entries['SYNTH_A']) == 'screened'


def test_parallel_batches_merge_without_losing_candidates(tmp_path):
    def save(batch):
        registry.update_candidate_registry_batch(tmp_path, [
            (f'SYNTH_{batch}_{i}', 'discovered', None) for i in range(10)])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(4)))
    assert len(registry.load_candidate_registry(tmp_path, strict=True)['candidates']) == 40


def test_family_diversity_precedes_different_names_in_same_family():
    def candidate(family, gene, status='screened'):
        return {'statuses': [{'status': status}], 'metadata': {
            'screening_family_id': family, 'gene_names': [gene]}}
    entries = {
        'a0': candidate('SYNTH_FOLD_A', 'old', 'evaluated'),
        'a1': candidate('SYNTH_FOLD_A', 'first'),
        'a2': candidate('SYNTH_FOLD_A', 'second'),
        'b1': candidate('SYNTH_FOLD_B', 'first'),
        'c1': candidate('SYNTH_FOLD_C', 'first'),
    }
    ordered = ['a1', 'a2', 'b1', 'c1']
    assert diversify_evaluation_batch(entries, ordered, 2) == ['b1', 'c1']
    assert set(diversify_evaluation_batch(entries, ordered, 10)) == set(ordered)


@pytest.mark.parametrize('status', ['evidence_pending', 'rejected'])
def test_unresolved_assessments_also_count_toward_family_effort(status):
    entries = {
        'old': {'statuses': [{'status': status}], 'metadata': {'screening_family_id': 'SYNTH_FOLD_A'}},
        'a': {'statuses': [{'status': 'screened'}], 'metadata': {'screening_family_id': 'SYNTH_FOLD_A'}},
        'b': {'statuses': [{'status': 'screened'}], 'metadata': {'screening_family_id': 'SYNTH_FOLD_B'}},
    }
    assert diversify_evaluation_batch(entries, ['a', 'b'], 1) == ['b']


def test_failed_survey_promotion_can_retry(tmp_path):
    from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
    gate = FamilySurveyGate()
    members = [{'accession': 'SYNTH_A', 'claim_validation': {'promotion_eligible': True}}]
    with patch.object(registry, 'atomic_write_json', side_effect=OSError('synthetic storage failure')):
        with pytest.raises(OSError):
            gate._promote_relevant(tmp_path, 'SYNTH_FOLD', members)
    assert gate._promoted == set()
    assert gate._promote_relevant(tmp_path, 'SYNTH_FOLD', members) == 1
    assert gate._promote_relevant(tmp_path, 'SYNTH_FOLD', members) == 0


def test_screening_passes_saved_member_fit_to_allocator(tmp_path, monkeypatch):
    from enzyme_discovery_agent.gates.ec_triage import auto_screen_candidates
    from enzyme_discovery_agent.gates import family_relevance
    from enzyme_discovery_agent.tools import reasoning
    registry.update_candidate_registry(tmp_path, 'SYNTH_A', 'discovered', {'substrate_fit': .8})
    monkeypatch.setattr(reasoning, '_fetch_uniprot_entry_light', lambda acc: {'accession': acc, 'seq_length': 200})
    observed = []
    def allocate(root, reps, **kwargs):
        observed.extend(reps)
        return set(), []
    monkeypatch.setattr(family_relevance, 'build_relevance_and_allocate', allocate)
    auto_screen_candidates(tmp_path)
    assert observed[0][1]['substrate_fit'] == .8


def test_equal_evidence_prefers_member_fit_over_length_without_dropping_unknowns(tmp_path, monkeypatch):
    from enzyme_discovery_agent.gates import family_relevance as relevance
    monkeypatch.setattr(relevance, '_target_profile_for_run', lambda root: ('formation', set()))
    monkeypatch.setattr(relevance, '_most_relevant_fold', lambda *args: 'SYNTH_FOLD')
    monkeypatch.setattr(relevance, '_candidate_direction', lambda *args: 'same')
    monkeypatch.setattr(relevance, '_fold_fg_max', lambda *args: .9)
    reps = [
        ('LONG_LOW', {'interpro_ids': ['SYNTH_FOLD'], 'seq_length': 900, 'substrate_fit': .1}),
        ('SHORT_HIGH', {'interpro_ids': ['SYNTH_FOLD'], 'seq_length': 200, 'substrate_fit': .9}),
        ('UNKNOWN', {'interpro_ids': ['SYNTH_FOLD'], 'seq_length': 1000}),
    ]
    selected, _ = relevance.build_relevance_and_allocate(tmp_path, reps, budget=1, floor=1, cap=3)
    assert selected == {'SHORT_HIGH'}
    selected, _ = relevance.build_relevance_and_allocate(tmp_path, reps, budget=3, floor=1, cap=3)
    assert selected == {'LONG_LOW', 'SHORT_HIGH', 'UNKNOWN'}


def test_survey_gate_seeds_fit_cache_for_later_allocation(tmp_path, monkeypatch):
    from enzyme_discovery_agent.gates import AgentState
    from enzyme_discovery_agent.gates import family_relevance as relevance
    from enzyme_discovery_agent.gates import family_survey
    (tmp_path / 'target_context.json').write_text('{}')
    (tmp_path / 'target_reasoning.json').write_text('{}')
    monkeypatch.setattr(relevance, 'collect_survey_folds', lambda *args: [{'fold_id': 'SYNTH_FOLD'}])
    monkeypatch.setattr(family_survey, 'execute_tool', lambda *args, **kwargs: {
        'status': 'success', 'data': {'fold_id': 'SYNTH_FOLD',
        'family_reaction_profile': {'functional_group_similarity': {'max': .7}}}})
    gate = family_survey.FamilySurveyGate()
    assert gate.execute(AgentState(tmp_path), []).intercept
    with patch('enzyme_discovery_agent.tools.family_reactions.survey_family_reactions') as repeated:
        assert relevance._fold_fg_max(tmp_path, 'SYNTH_FOLD', {}) == .7
        repeated.assert_not_called()


def test_allocation_reuses_member_survey_facts_without_repeating_family_survey(tmp_path, monkeypatch):
    from enzyme_discovery_agent.gates import family_relevance as relevance
    monkeypatch.setattr(relevance, '_target_profile_for_run', lambda root: ('formation', set()))
    monkeypatch.setattr(relevance, '_most_relevant_fold', lambda *args: 'SYNTH_FOLD')
    monkeypatch.setattr(relevance, '_candidate_direction', lambda *args: 'same')
    reps = [('SYNTH_A', {'interpro_ids': ['SYNTH_FOLD'], 'survey_functional_group_similarity': .8})]
    with patch.object(relevance, '_fold_fg_max') as repeated:
        selected, landscape = relevance.build_relevance_and_allocate(tmp_path, reps, budget=1, floor=1, cap=1)
        repeated.assert_not_called()
    assert selected == {'SYNTH_A'}
    assert landscape[0]['functional_group_similarity_max'] == .8
    assert landscape[0]['fit_basis'] == 'registered_member_surveys'


def test_survey_does_not_rearm_itself_from_its_own_promotions(tmp_path, monkeypatch):
    from enzyme_discovery_agent.gates import AgentState
    from enzyme_discovery_agent.gates import family_relevance as relevance
    from enzyme_discovery_agent.gates import family_survey
    monkeypatch.setattr(relevance, 'collect_survey_folds', lambda *args: [{'fold_id': 'SYNTH_FOLD'}])
    monkeypatch.setattr(family_survey, 'execute_tool', lambda *args, **kwargs: {
        'status': 'success', 'data': {'fold_id': 'SYNTH_FOLD', 'relevant_members': [
            {'accession': 'SYNTH_A', 'claim_validation': {'promotion_eligible': True}}]}})
    gate = family_survey.FamilySurveyGate()
    state = AgentState(tmp_path, mechanism_families_done=True)
    assert gate.should_fire(state)
    assert gate.execute(state, []).intercept
    assert not gate.should_fire(state)
    registry.update_candidate_registry(tmp_path, 'SYNTH_B', 'discovered')
    assert gate.should_fire(state)


def test_fit_method_refresh_preserves_candidate_evaluation_and_lifecycle(tmp_path):
    from enzyme_discovery_agent.gates.family_survey import FamilySurveyGate
    own_validation = {'promotion_eligible': True, 'scope': 'candidate_evaluation'}
    registry.update_candidate_registry(tmp_path, 'SYNTH_A', 'evaluated', {
        'substrate_fit': .9, 'claim_validation': own_validation})
    gate = FamilySurveyGate()
    gate._promoted.add('SYNTH_A')
    assert gate._promote_relevant(tmp_path, 'SYNTH_FOLD', [{
        'accession': 'SYNTH_A', 'fit_method': 'role_aware_accommodation_v1',
        'substrate_fit': .2, 'claim_validation': {'promotion_eligible': True}}]) == 0
    entry = registry.load_candidate_registry(tmp_path)['candidates']['SYNTH_A']
    assert registry.discovery_status(entry) == 'evaluated'
    assert entry['metadata']['substrate_fit'] == .2
    assert entry['metadata']['claim_validation'] == own_validation
    assert entry['metadata']['survey_claim_validation']['promotion_eligible']
