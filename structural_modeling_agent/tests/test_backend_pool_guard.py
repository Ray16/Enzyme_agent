"""The provider consults current shared records rather than a launch-time alias list."""
from structural_modeling_agent.lambda_boltz_backend import check_prediction_pool
import hashlib
import yaml


def test_new_sequence_is_checked_against_the_supplied_pool(tmp_path, monkeypatch):
    calls = []
    document={'sequences':[
        {'protein':{'id':'X','sequence':'AGGA'}},
        {'protein':{'id':'Y','sequence':'GAGA'}},
        {'ligand':{'id':'Z','smiles':'CCO'}}]}
    path=tmp_path/'input.yaml';path.write_text(yaml.safe_dump(document))
    def lookup(pool, **kwargs):
        calls.append((pool, kwargs))
        return {'matches': [{'execution': {'output_presence': 'complete'},'setup':{'input_source':{
            'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}] if kwargs['sequence'] == 'AGGA' else []}
    monkeypatch.setattr('campaign_catalog.lookup.lookup_predictions', lookup)
    result = check_prediction_pool({'sequences': [
        {'protein': {'id': 'X', 'sequence': 'AGGA'}},
        {'protein': {'id': 'Y', 'sequence': 'GAGA'}},
        {'ligand': {'id': 'Z', 'smiles': 'CCO'}}]}, tmp_path)
    assert result['existing_predictions']
    assert [item[1]['sequence'] for item in calls] == ['AGGA', 'GAGA']
    assert all(item[0] == tmp_path for item in calls)


def test_native_or_other_target_ligand_does_not_block_the_requested_new_system(tmp_path,monkeypatch):
    old={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':'CCO'}}]}
    path=tmp_path/'old.yaml';path.write_text(yaml.safe_dump(old))
    match={'setup':{'input_source':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
    monkeypatch.setattr('campaign_catalog.lookup.lookup_predictions',lambda *a,**k:{'matches':[match]})
    target={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':'CCC'}}]}
    assert not check_prediction_pool(target,tmp_path)['existing_predictions']
    old['sequences'][0]['protein']['id']=['A','C']
    assert not check_prediction_pool(old,tmp_path)['existing_predictions']


def test_charge_and_missing_input_provenance_prevent_same_system_reuse(tmp_path,monkeypatch):
    old={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':'CC(=O)O'}}]}
    path=tmp_path/'old.yaml';path.write_text(yaml.safe_dump(old))
    match={'setup':{'input_source':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
    monkeypatch.setattr('campaign_catalog.lookup.lookup_predictions',lambda *a,**k:{'matches':[match]})
    target={'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':'CC(=O)[O-]'}}]}
    assert not check_prediction_pool(target,tmp_path)['existing_predictions']
    match['setup']['input_source']['sha256']='changed'
    assert not check_prediction_pool(old,tmp_path)['existing_predictions']


def test_quality_minimum_cannot_be_reduced_by_a_model_proposal():
    import pytest
    from structural_modeling_agent.lambda_boltz_backend import validate_sample_count
    policy={'min_predictions_per_job':50,'max_predictions_per_job':50}
    validate_sample_count(50,policy)
    for count in [1,49,51,True]:
        with pytest.raises(ValueError):validate_sample_count(count,policy)
