"""The role contract spans transformations without enzyme-specific identity rules."""
import pytest
from enzyme_discovery_agent.models.reaction_roles import validate_roles,apply_roles
from enzyme_discovery_agent.substrate_fit import accommodation_score


def data():
    return {'native_substrates':[{'name':'synthetic participant','smiles':'SYNTH_NATIVE'}],
            'function_comments':['The synthetic participant supplies the declared reaction component at the stated site.'],
            'catalytic_activities':['Synthetic inputs are converted to synthetic products.']}


def role(kind,unit,site):
    return {'native_smiles':'SYNTH_NATIVE','role':kind,'transferred_unit':unit,'site_relation':site,
            'interpretation':'Synthetic role assertion; source integrity does not establish mechanistic entailment.',
            'source':{'locator':'function_comments/0','quote':data()['function_comments'][0]}}


@pytest.mark.parametrize('kind,unit,site',[
    ('precursor','atom','tunneled_intermediate'),('cofactor','electrons','same_site'),
    ('carrier','fragment','carrier_tethered'),('solvent','proton','same_site'),
    ('unresolved','unresolved','unresolved')])
def test_shape_cannot_prove_unresolved_atom_fragment_or_electron_role_transfer(kind,unit,site):
    source=data();record=validate_roles(source,[role(kind,unit,site)])
    accommodation={'per_target':[{'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_NATIVE'}]}
    result=apply_roles(source,record,accommodation)
    assert result['role_transfer_status']=='unresolved'
    assert accommodation_score({'accommodation':result,'property_comparisons':[{'mw_similarity':.99}]}) is None


def test_direct_whole_partner_keeps_its_structural_comparison_separate_from_activity():
    source=data();record=validate_roles(source,[role('direct_reactant','whole_molecule','same_site')])
    result=apply_roles(source,record,{'per_target':[{'target_smiles':'SYNTH_TARGET','native_smiles':'SYNTH_NATIVE'}]})
    assert 'participant_role_mismatches' not in result
    assert accommodation_score({'accommodation':result,'property_comparisons':[{'mw_similarity':.6}]})==.6
    assert 'mechanistic interpretation remains an agent assertion' in record['validation_scope']


def test_unknown_participant_and_fabricated_quotes_do_not_enter_the_record():
    with pytest.raises(ValueError,match='resolved native'):
        validate_roles(data(),[{**role('precursor','atom','separate_site'),'native_smiles':'UNSEEN'}])
    bad=role('precursor','atom','separate_site');bad['source']={'locator':'function_comments/0','quote':'A source passage that was never retrieved.'}
    with pytest.raises(ValueError,match='not present'):
        validate_roles(data(),[bad])


def test_role_interpretation_is_invalidated_when_its_scientific_source_changes():
    source=data();record=validate_roles(source,[role('precursor','atom','separate_site')])
    changed={**source,'function_comments':['New and different native-function evidence.']}
    result=apply_roles(changed,record,{'per_target':[]})
    assert result['role_record_status'].startswith('stale')


def test_protein_carrier_or_cofactor_does_not_need_an_invented_small_molecule_structure():
    source=data();source['function_comments']=['Synthetic carrier protein transfers a fragment between the declared active sites.']
    item=role('carrier','fragment','carrier_tethered');item.pop('native_smiles');item['participant_name']='carrier protein'
    item['source']={'locator':'function_comments/0','quote':source['function_comments'][0]}
    record=validate_roles(source,[item])
    assert record['participants'][0]['participant_name']=='carrier protein'
    assert record['participants'][0]['native_smiles'] is None
    item['participant_name']='unmentioned carrier'
    with pytest.raises(ValueError,match='named in'):
        validate_roles(source,[item])
