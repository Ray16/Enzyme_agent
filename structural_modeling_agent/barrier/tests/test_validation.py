from enzyme_barrier.validation import validate_profile,conditional_metal_comparison
import copy
import pytest


def test_static_gap_cannot_be_promoted_by_small_errorbar():
    result=validate_profile(dict(quantity='electronic_energy_gap',barrier_kcal=19.2,statistical_error_kcal=.01))
    assert not result['validated_activation_free_energy']
    assert 'quantity is not an activation free energy' in result['failures']


def test_conditional_metal_barriers_do_not_imply_populations():
    result=conditional_metal_comparison(dict(reactant_reference='2Mg R',barrier_kcal=20),dict(reactant_reference='3Mg R',barrier_kcal=18))
    assert result['conditional_barrier_difference_kcal']==-2
    assert not result['predicts_observable_rate']


def _complete_evidence():
    return dict(quantity='activation_free_energy',setup=dict(
        exact_sequence_verified=True,molecular_graph_verified=True,assembly_reviewed=True,
        protonation_reviewed=True,metal_shell_reviewed=True,reactant_reference='declared bound basin'),
        method=dict(energy_force_check_passed=True,embedding_radius_checked=True,qm_region_size_checked=True,
                    alternative_pathways_reviewed=True,target_level_ensemble_checked=True,reference_potential_only=False),
        replicas=[dict(starting_configuration_sha256=f'{i:064x}',sampling_passed=True,
                       statistical_error_kcal=.2,hysteresis_kcal=.5,target_ess_per_critical_region=100) for i in range(3)],
        transition_state=dict(connectivity_verified=True,committor_shots=50,committor_mean=.5))


@pytest.mark.parametrize('field',['statistical_error_kcal','hysteresis_kcal','target_ess_per_critical_region'])
def test_nan_cannot_pass_the_reporting_gate(field):
    evidence=_complete_evidence()
    assert validate_profile(evidence)['validated_activation_free_energy']
    evidence['replicas'][0][field]=float('nan')
    assert not validate_profile(evidence)['validated_activation_free_energy']


def test_missing_replica_hash_cannot_count_as_an_independent_configuration():
    evidence=copy.deepcopy(_complete_evidence());evidence['replicas'][0].pop('starting_configuration_sha256')
    assert not validate_profile(evidence)['validated_activation_free_energy']
