"""Physical reasoning, explicit assay context and compatible plan declarations."""
from copy import deepcopy

import pytest

from enzyme_discovery_agent.structural_proxy_planning import (
    load_proxy_catalog, proxy_guidance, validate_assay_context,
    validate_proxy_plan, validate_search_design,
)
from enzyme_discovery_agent.tools.structural_proxy_planning import get_structural_proxy_guidance
from enzyme_discovery_agent.tool_registry import get_all_tools


def guidance(context=None, mechanism='declared enzyme mechanism', inputs=None):
    return proxy_guidance(mechanism, 'declared hydrolytic step',
                          'measured product formation', inputs or [], context)


def plan():
    return {
        'schema_version': '1.0', 'mechanism': 'declared mechanism',
        'reaction_step': 'declared chemical step', 'assay_endpoint': 'matched activity',
        'context': {}, 'hypotheses': [{
            'proxy_id': 'approach_distance', 'expected_direction': '-',
            'rationale': 'The declared reacting atom must approach its electrophile',
            'evidence_ids': ['inference:mechanism'],
        }],
    }


def process_map(data):
    return {p['id']: p for p in data['physical_process_map']}


def test_legacy_four_argument_guidance_and_plan_remain_supported():
    old = proxy_guidance('declared mechanism', 'declared step', 'matched endpoint', ['reactive_roles'])
    assert old['schema_version'] == '1.0'
    assert old['assay_context'] == {}
    assert validate_proxy_plan(plan()) == []
    descriptors = {d['id']: d for d in old['descriptors']}
    assert descriptors['approach_distance']['ready_from_declared_inputs']
    assert descriptors['approach_distance']['expected_direction_prior'] == '-'
    assert descriptors['near_attack_fraction']['expected_direction_prior'] == '+'
    assert descriptors['reactive_group_sasa']['expected_direction_prior'] is None
    assert descriptors['complex_plddt']['expected_direction_prior'] is None


def test_family_and_assay_keywords_do_not_infer_missing_context():
    baseline = proxy_guidance('generic enzyme', 'generic step', 'generic endpoint', [])
    names = proxy_guidance('nylon hydrolase NylC Ntn mature precursor',
                           'PA66 powder insoluble polymer processing',
                           'crude lysate at65C', [])
    assert names['assay_context'] == baseline['assay_context'] == {}
    assert names['physical_process_map'] == baseline['physical_process_map']
    assert names['modeled_state_gaps'] == baseline['modeled_state_gaps'] == []
    assert set(names['unresolved_context']) == {
        'substrate_state', 'measurement_basis', 'modeled_state', 'assembly_status',
    }
    assert all(p['priority'] == 'consider' for p in names['physical_process_map'])


def test_solid_substrate_crude_mature_assay_surfaces_distinct_physical_gaps():
    context = {
        'substrate_state': 'insoluble_polymer', 'measurement_basis': 'crude_lysate',
        'modeled_state': 'mature', 'requires_processing': True,
        'assembly_status': 'incomplete', 'temperature_C': 65., 'duration_hours': 24.,
        'substrate_is_assayed': False, 'substrate_representation': 'capped soluble oligomer',
    }
    original = deepcopy(context)
    data = guidance(context)
    processes = process_map(data)
    for key in ['substrate_access', 'surface_adsorption', 'maturation', 'assembly', 'stability']:
        assert processes[key]['priority'] == 'consider_first'
        assert processes[key]['priority_reasons']
    assert processes['chemistry']['priority'] == 'consider'
    assert all(p['evidence_status'] == 'physical_hypothesis' for p in processes.values())
    assert {g['process'] for g in data['modeled_state_gaps']} >= {
        'maturation', 'assembly', 'substrate_access', 'active_enzyme_amount',
    }
    assert context == original
    priors = {d['id']: d['expected_direction_prior'] for d in load_proxy_catalog()['descriptors']}
    assert {d['id']: d['expected_direction_prior'] for d in data['descriptors']} == priors


def test_soluble_substrate_can_still_have_an_explicit_access_hypothesis():
    soluble = guidance({'substrate_state': 'soluble'})
    assert process_map(soluble)['substrate_access']['priority'] == 'consider'
    explicit = guidance({'substrate_state': 'soluble', 'limiting_processes': ['substrate_access']})
    hypothesis = process_map(explicit)['substrate_access']
    assert hypothesis['priority'] == 'consider_first'
    assert hypothesis['evidence_status'] == 'physical_hypothesis'
    assert 'not an established cause' in ' '.join(hypothesis['priority_reasons'])


@pytest.mark.parametrize('basis', ['enzyme_normalized', 'crude_lysate', 'whole_cell'])
def test_total_preparation_normalization_does_not_establish_active_fraction(basis):
    data = guidance({'measurement_basis': basis, 'requires_processing': True, 'modeled_state': 'mature'})
    assert {'active_enzyme_amount', 'maturation'} <= {g['process'] for g in data['modeled_state_gaps']}


def test_active_enzyme_normalization_does_not_make_mature_model_cover_precursor():
    data = guidance({'measurement_basis': 'active_enzyme_normalized',
                     'requires_processing': True, 'modeled_state': 'mature'})
    gaps = {g['process'] for g in data['modeled_state_gaps']}
    assert 'active_enzyme_amount' not in gaps
    assert 'maturation' in gaps


def test_product_release_requires_its_own_molecular_state():
    missing = guidance({'limiting_processes': ['product_release'], 'modeled_state': 'substrate_bound'})
    represented = guidance({'limiting_processes': ['product_release'], 'modeled_state': 'product_bound'})
    assert 'product_release' in {g['process'] for g in missing['modeled_state_gaps']}
    assert 'product_release' not in {g['process'] for g in represented['modeled_state_gaps']}
    assert process_map(represented)['product_release']['evidence_status'] == 'physical_hypothesis'


def test_available_inputs_do_not_turn_targeted_methods_into_executed_measurements():
    catalog = load_proxy_catalog()
    inputs = sorted({req for d in catalog['descriptors'] for req in d['requirements']})
    data = guidance({'substrate_state': 'insoluble_polymer'}, inputs=inputs)
    for descriptor in data['descriptors']:
        assert not descriptor['missing_requirements']
        assert descriptor['ready_from_declared_inputs'] == (descriptor['execution'] == 'builtin')
        assert descriptor['evidence_status'] in {'candidate_descriptor', 'confidence_baseline'}
    assert 'not relevance or validation' in data['selection_note']
    known = {d['id'] for d in data['descriptors']}
    assert all(set(p['candidate_descriptors']) <= known for p in data['physical_process_map'])


def test_exploration_retains_physical_definitions_selection_controls_and_integrity_qc():
    protocol = guidance()['exploration_protocol']
    assert protocol['choose_process_before_descriptor'] is True
    assert 'failed hypothesis' in protocol['discovery_mode']
    assert 'training folds' in protocol['combinations']
    assert 'not biological replicates' in protocol['validation']
    assert 'no universal all-pose' in protocol['qc']
    assert 'source hashes' in protocol['artifacts']
    assert 'thread caps' in protocol['parallelism']


@pytest.mark.parametrize('context', [
    [], {'requires_processing': 'true'}, {'substrate_is_assayed': 1},
    {'substrate_state': 'powder'}, {'modeled_state': 'NylC'},
    {'limiting_processes': ['chemistry', 'chemistry']}, {'limiting_processes': [{}]},
    {'temperature_C': True}, {'temperature_C': float('nan')}, {'pH': 15}, {'duration_hours': 0},
])
def test_invalid_explicit_context_is_reported_without_guessing(context):
    assert validate_assay_context(context)
    candidate = plan()
    candidate['assay_context'] = context
    assert validate_proxy_plan(candidate)
    with pytest.raises(ValueError):
        guidance(context)


@pytest.mark.parametrize('design', [
    [], {'biological_unit': 'pose'}, {'budgets': [True]}, {'budgets': [0]},
    {'budgets': [10, 10]}, {'record_negative_results': False},
    {'combination_policy': 'physical_pairs'},
    {'qc_policy': {'minimum_valid_poses': True}},
    {'qc_policy': {'require_all_poses': True}},
    {'qc_policy': {'require_all_poses': True, 'sensitivity_analysis': True}},
    {'qc_policy': {'require_all_poses': True, 'sensitivity_analysis': ' '}},
])
def test_invalid_discovery_or_qc_declarations_are_rejected(design):
    assert validate_search_design(design)
    candidate = plan()
    candidate['search_design'] = design
    assert validate_proxy_plan(candidate)


def test_explicit_discovery_supports_small_physical_pairs_and_qc_sensitivity():
    candidate = plan()
    candidate['assay_context'] = {'substrate_state': 'insoluble_polymer'}
    candidate['search_design'] = {
        'mode': 'discovery', 'objective': 'high_producer_recovery',
        'biological_unit': 'enzyme_construct', 'budgets': [10, 15],
        'validation': 'nested_enzyme_heldout', 'selection_correction': 'repeat_full_search',
        'combination_policy': 'physical_pairs',
        'combination_rationale': 'The substrate must reach the pocket and adopt the reactive conformation',
        'record_negative_results': True,
        'qc_policy': {'minimum_valid_poses': 20, 'require_all_poses': True,
                      'sensitivity_analysis': 'Compare integrity-only and assembly-specific coverage'},
    }
    assert validate_proxy_plan(candidate) == []
    candidate['hypotheses'][0]['expected_direction'] = None
    assert any('declare a direction' in error for error in validate_proxy_plan(candidate))


def test_tool_context_is_optional_and_invalid_context_returns_tool_error():
    schema = next(s['function']['parameters'] for s in get_all_tools()
                  if s['function']['name'] == 'get_structural_proxy_guidance')
    assert 'assay_context' in schema['properties']
    assert 'assay_context' not in schema['required']
    legacy = get_structural_proxy_guidance('mechanism', 'step', 'endpoint')
    assert legacy['status'] == 'success'
    explicit = get_structural_proxy_guidance('mechanism', 'step', 'endpoint',
                                            assay_context={'substrate_state': 'insoluble_polymer'})
    assert explicit['status'] == 'success'
    assert process_map(explicit['data'])['surface_adsorption']['priority'] == 'consider_first'
    invalid = get_structural_proxy_guidance('mechanism', 'step', 'endpoint',
                                           assay_context={'requires_processing': 'true'})
    assert invalid['status'] == 'error'
    assert 'requires_processing' in invalid['summary']
