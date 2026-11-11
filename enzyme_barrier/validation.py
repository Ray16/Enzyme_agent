"""Evidence-based reporting gate; static energies never pass as enzyme free energies."""


def validate_profile(evidence):
    failures=[]
    if evidence.get('quantity')!='activation_free_energy':failures.append('quantity is not an activation free energy')
    setup=evidence.get('setup',{})
    for field in ['exact_sequence_verified','molecular_graph_verified','assembly_reviewed','protonation_reviewed','metal_shell_reviewed']:
        if setup.get(field) is not True:failures.append(f'setup: {field}')
    if not setup.get('reactant_reference'):failures.append('reactant reference is unspecified')
    method=evidence.get('method',{})
    for field in ['energy_force_check_passed','embedding_radius_checked','qm_region_size_checked','alternative_pathways_reviewed','target_level_ensemble_checked']:
        if method.get(field) is not True:failures.append(f'method: {field}')
    if method.get('reference_potential_only',True):failures.append('reference-potential result has no validated target-level correction')
    replicas=evidence.get('replicas',[])
    if len(replicas)<3:failures.append('fewer than three independent replicas')
    if len({r.get('starting_configuration_sha256') for r in replicas})!=len(replicas):failures.append('replicas lack distinct equilibrated starting configurations')
    for n,r in enumerate(replicas):
        if r.get('sampling_passed') is not True:failures.append(f'replica {n}: sampling checks failed')
        if r.get('statistical_error_kcal',float('inf'))>1:failures.append(f'replica {n}: statistical uncertainty exceeds 1 kcal/mol')
        if r.get('hysteresis_kcal',float('inf'))>2:failures.append(f'replica {n}: hysteresis exceeds 2 kcal/mol')
        if r.get('target_ess_per_critical_region',0)<50:failures.append(f'replica {n}: inadequate target-level overlap')
    ts=evidence.get('transition_state',{})
    if ts.get('connectivity_verified') is not True:failures.append('chemical connectivity not verified')
    if ts.get('committor_shots',0)<40 or not .35<=ts.get('committor_mean',-1)<=.65:
        failures.append('transition-state ensemble lacks adequate committor support')
    if evidence.get('step')=='aminolysis' and evidence.get('proton_transfer_reviewed') is not True:
        failures.append('amine activation/proton transfer not resolved')
    return dict(validation_status='validated' if not failures else 'provisional',
                validated_activation_free_energy=not failures,failures=failures)


def conditional_metal_comparison(first,second):
    if first.get('reactant_reference') is None or second.get('reactant_reference') is None:
        raise ValueError('Each composition needs its own reactant reference')
    return dict(conditional_barrier_difference_kcal=second['barrier_kcal']-first['barrier_kcal'],
                comparison='conditional on each specified metal occupancy',
                predicts_observable_rate=False,
                missing_for_observable=['metal binding thermodynamics','bulk Mg activity','occupancy-weighted kinetics'])
