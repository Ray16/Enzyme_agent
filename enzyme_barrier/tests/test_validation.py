from enzyme_barrier.validation import validate_profile,conditional_metal_comparison


def test_static_gap_cannot_be_promoted_by_small_errorbar():
    result=validate_profile(dict(quantity='electronic_energy_gap',barrier_kcal=19.2,statistical_error_kcal=.01))
    assert not result['validated_activation_free_energy']
    assert 'quantity is not an activation free energy' in result['failures']


def test_conditional_metal_barriers_do_not_imply_populations():
    result=conditional_metal_comparison(dict(reactant_reference='2Mg R',barrier_kcal=20),dict(reactant_reference='3Mg R',barrier_kcal=18))
    assert result['conditional_barrier_difference_kcal']==-2
    assert not result['predicts_observable_rate']
