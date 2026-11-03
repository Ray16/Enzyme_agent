"""An estimated volume must not masquerade as a 3D comparison."""
from unittest.mock import patch

from enzyme_discovery_agent.chemistry_logic import compare_substrate_properties
from enzyme_discovery_agent.tools import substrate


def test_dummy_atom_volume_is_marked_approximate_and_not_scored():
    substrate._analyzer_cache.clear()
    target = substrate.compute_substrate_properties('CCO')
    native = substrate.compute_substrate_properties('*CC')
    assert target.molecular_volume_method == '3d_conformer'
    assert native.molecular_volume_method == 'heavy_atom_estimate'
    result = compare_substrate_properties('CCO', '*CC', substrate.compute_substrate_properties)
    assert result['volume_similarity'] is None
    assert result['volume_method_native'] == 'heavy_atom_estimate'


def test_failed_embedding_never_counts_as_3d():
    substrate._analyzer_cache.clear()
    with patch.object(substrate.AllChem, 'EmbedMolecule', return_value=-1), \
         patch.object(substrate.AllChem, 'MMFFOptimizeMolecule') as optimize:
        value = substrate.compute_substrate_properties('CCN')
    optimize.assert_not_called()
    assert value.molecular_volume_method == 'heavy_atom_estimate'
    assert value.to_dict()['molecular_volume_method'] == 'heavy_atom_estimate'


def test_invalid_structure_is_parsed_only_once_per_process():
    import pytest
    substrate._invalid_smiles_cache.clear()
    with patch.object(substrate.Chem, 'MolFromSmiles', return_value=None) as parser:
        for _ in range(3):
            with pytest.raises(ValueError, match='Invalid SMILES'):
                substrate.compute_substrate_properties('synthetically-invalid')
    assert parser.call_count == 1
    substrate._invalid_smiles_cache.clear()
