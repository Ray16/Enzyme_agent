"""Survey promotion and ranking must use the same required-substrate evidence."""
from enzyme_discovery_agent.tools.family_reactions import select_relevant_members, _substrate_facts
from enzyme_discovery_agent.output.report_generator import _accommodation_score
from enzyme_discovery_agent.substrate_accommodation import accommodation_facts


def test_missing_required_role_cannot_be_hidden_by_high_average_features():
    pair = {'functional_group_similarity': .9, 'mw_similarity': .9,
            'chain_length_similarity': .9}
    incomplete = {'accession': 'SYNTH_A', 'direction_vs_target': 'same',
                  'substrate': {'dimensions': pair, 'property_comparisons': [pair]},
                  'accommodation': {'unaccommodated_targets': ['SYNTH_SECOND_ROLE']}}
    moderate_pair = {key: .5 for key in pair}
    complete = {'accession': 'SYNTH_B', 'direction_vs_target': 'same',
                'substrate': {'dimensions': moderate_pair, 'property_comparisons': [moderate_pair, moderate_pair]},
                'accommodation': {'unaccommodated_targets': []}}
    selected = select_relevant_members([incomplete, complete])
    assert [row['accession'] for row in selected] == ['SYNTH_B', 'SYNTH_A']
    assert selected[1]['substrate_fit'] == 0
    for member, row in ((complete, selected[0]), (incomplete, selected[1])):
        assert row['fit_method'] == 'role_aware_accommodation_v1'
        assert row['substrate_fit'] == _accommodation_score({'substrate_similarity': {
            'property_comparisons': member['substrate']['property_comparisons'],
            'accommodation': member['accommodation']}})


def test_real_multi_input_facts_retain_missing_role_for_selection():
    targets, natives = ['CCO', 'CCN'], ['CCO']
    substrate = _substrate_facts(natives, targets)
    accommodation = accommodation_facts(targets, natives)
    assert substrate['n_target_molecules'] == 2
    assert substrate['n_matched_targets'] == len(substrate['property_comparisons']) == 1
    assert accommodation['unaccommodated_targets']
    rows = select_relevant_members([{'accession': 'SYNTH_A', 'direction_vs_target': 'same',
                                    'substrate': substrate, 'accommodation': accommodation}])
    assert rows[0]['substrate_fit'] == 0


def test_summary_only_callers_are_explicitly_labelled_as_legacy():
    row = select_relevant_members([{'accession': 'SYNTH_A', 'direction_vs_target': 'same',
                                   'substrate': {'functional_group_similarity': .6, 'size_ratio': .6}}])[0]
    assert row['fit_method'] == 'legacy_scalar_summary'
