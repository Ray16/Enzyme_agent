"""A hydrolyzed precursor's shape does not establish direct partner transfer."""
from enzyme_discovery_agent.tools.native_reactions import annotate_indirect_donor_roles
from enzyme_discovery_agent.substrate_fit import accommodation_score


def entry(description='Synthetic catalyst [precursor-hydrolyzing]',function='Uses precursor as a source of nitrogen.'):
    return {'proteinDescription':{'recommendedName':{'fullName':{'value':description}}},
        'comments':[{'commentType':'FUNCTION','texts':[{'value':function,
                     'evidences':[{'evidenceCode':'ECO:0000269','source':'PubMed','id':'SYNTHETIC_SOURCE'}]}]}]}


def test_explicit_indirect_role_removes_false_whole_molecule_fit_evidence():
    original={'per_target':[{'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_PRECURSOR'}],
              'unaccommodated_targets':[]}
    native=[{'name':'L-precursor zwitterion','smiles':'SYNTH_PRECURSOR'}]
    roles=annotate_indirect_donor_roles(entry(),native,original)
    assert roles['role_transfer_status']=='unresolved'
    assert roles['indirect_donor_role_mappings'][0]['source']['quote']=='Uses precursor as a source of nitrogen.'
    score=accommodation_score({'accommodation':roles,'property_comparisons':[{
        'functional_group_similarity':.9,'mw_similarity':.9,'chain_length_similarity':.9}]})
    assert score is None
    assert 'role_transfer_status' not in original


def test_exact_native_precursor_is_not_penalized_for_its_real_role():
    original={'per_target':[{'target_smiles':'SYNTH_PRECURSOR','native_smiles':'SYNTH_PRECURSOR'}]}
    roles=annotate_indirect_donor_roles(entry(),[{'name':'precursor','smiles':'SYNTH_PRECURSOR'}],original)
    assert roles==original


def test_generic_nitrogen_donor_phrase_alone_does_not_invent_hydrolysis():
    original={'per_target':[{'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_PRECURSOR'}]}
    assert annotate_indirect_donor_roles(entry(description='Synthetic direct catalyst'),
        [{'name':'precursor','smiles':'SYNTH_PRECURSOR'}],original)==original


def test_missing_functional_source_is_not_replaced_by_an_enzyme_name_guess():
    original={'per_target':[{'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_PRECURSOR'}]}
    assert annotate_indirect_donor_roles(entry(function='Unresolved native chemistry.'),
        [{'name':'precursor','smiles':'SYNTH_PRECURSOR'}],original)==original


def test_report_ranking_cannot_resurrect_a_role_invalid_fit_from_legacy_fallback():
    from enzyme_discovery_agent.output.report_generator import _deterministic_mechanistic_ranking
    unresolved={'data':{'accession':'SYNTH_ROLE_UNKNOWN','substrate_similarity':{
        'accommodation':{'role_transfer_status':'unresolved'}}}}
    measured={'data':{'accession':'SYNTH_MEASURED','substrate_similarity':{
        'property_comparisons':[{'functional_group_similarity':.2,'mw_similarity':.2}]}}}
    ranking=_deterministic_mechanistic_ranking([unresolved,measured],{},
        fallback_substrate_fit_by_acc={'SYNTH_ROLE_UNKNOWN':.99})
    assert ranking[0]['accession']=='SYNTH_MEASURED'
