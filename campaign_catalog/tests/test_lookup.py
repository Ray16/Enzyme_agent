"""A reuse lookup checks index integrity and respects the complete saved setup."""
import json

import pytest

from campaign_catalog.catalog import build
from campaign_catalog.lookup import lookup_predictions
import test_catalog


def test_accession_lookup_preserves_legacy_uncertainty(tmp_path):
    campaign = test_catalog.CatalogTest().fixture(tmp_path)
    build(campaign)
    result = lookup_predictions(campaign, accession='P12345')
    assert result['lookup_tables_verified']
    match, = result['matches']
    assert match['identity_match'] == 'accession'
    assert match['setup']['prediction_input_binding'] == 'not_verified_from_legacy_manifest'
    assert match['input_integrity'] == 'unavailable'
    assert not lookup_predictions(campaign, accession='UNSEEN')['matches']


def test_exact_sequence_lookup_checks_the_original_input_hash(tmp_path):
    test = test_catalog.CatalogTest()
    campaign = test.fixture(tmp_path)
    path = tmp_path / 'models/inputs/step1_P12345.yaml'
    path.parent.mkdir()
    path.write_text('sequences:\n- protein:\n    id: A\n    sequence: AGGA\n')
    build(campaign)
    match, = lookup_predictions(campaign, sequence='AGGA')['matches']
    assert match['identity_match'] == 'exact_input_sequence'
    assert match['input_integrity'] == 'verified'
    path.write_text('sequences:\n- protein:\n    id: A\n    sequence: GAGA\n')
    assert not lookup_predictions(campaign, sequence='GAGA')['matches']
    assert lookup_predictions(campaign, accession='P12345')['matches'][0]['input_integrity'] == 'changed'


def test_new_target_does_not_treat_unknown_historical_inputs_as_current_models(tmp_path):
    campaign = test_catalog.CatalogTest().fixture(tmp_path)
    build(campaign)
    result = lookup_predictions(campaign, accession='P12345', target_id='synthetic-other-target',
                                required_ligand_smiles=['CCO'])
    assert not result['matches']
    assert len(result['other_target_matches']) == 1
    assert result['other_target_matches'][0]['target_applicability'] == 'other_target_unverified_molecular_inputs'


def test_verified_ligand_identity_allows_cross_target_reuse_and_rejects_a_different_ligand(tmp_path):
    campaign = test_catalog.CatalogTest().fixture(tmp_path)
    path = tmp_path / 'models/inputs/step1_P12345.yaml'
    path.parent.mkdir()
    path.write_text('sequences:\n- protein:\n    id: A\n    sequence: AGGA\n- ligand:\n    id: B\n    smiles: CC(=O)[O-]\n')
    build(campaign)
    result = lookup_predictions(campaign, accession='P12345', target_id='synthetic-other-target',
                                required_ligand_smiles=['CC(=O)O'])
    assert len(result['matches']) == 1
    assert not result['other_target_matches']
    result = lookup_predictions(campaign, accession='P12345', required_ligand_smiles=['CCO'])
    assert not result['matches']
    assert result['other_target_matches'][0]['target_applicability'] == 'required_molecular_input_absent'


@pytest.mark.parametrize('filename', ['proteins.jsonl', 'modeling_setups.jsonl', 'modeling_executions.jsonl'])
def test_changed_pool_metadata_cannot_authorize_a_new_job(tmp_path, filename):
    campaign = test_catalog.CatalogTest().fixture(tmp_path)
    build(campaign)
    pointer = json.loads((campaign / 'catalog/current.json').read_text())
    (type(campaign)(pointer['manifest']).parent / filename).write_text('{}\n')
    with pytest.raises(ValueError, match='table changed'):
        lookup_predictions(campaign, accession='P12345')
