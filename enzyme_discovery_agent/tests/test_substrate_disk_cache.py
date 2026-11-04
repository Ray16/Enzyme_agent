"""Persistent descriptors preserve results without reusing stale implementations."""
import json
from unittest.mock import patch

from enzyme_discovery_agent.tools import substrate


def test_disk_hit_preserves_properties_and_skips_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(substrate, '_DISK_CACHE_DIR', tmp_path)
    monkeypatch.setattr(substrate, '_analyzer_cache', {})
    first=substrate.compute_substrate_properties('CCO', 'first label')
    substrate._analyzer_cache.clear()
    with patch.object(substrate.AllChem, 'EmbedMolecule') as embed:
        second=substrate.compute_substrate_properties('CCO', 'second label')
    embed.assert_not_called()
    assert second.name=='second label'
    assert second.molecular_volume==first.molecular_volume
    assert second.ecfp4.ToBase64()==first.ecfp4.ToBase64()


def test_changed_descriptor_context_cannot_reuse_disk_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(substrate, '_DISK_CACHE_DIR', tmp_path)
    monkeypatch.setattr(substrate, '_analyzer_cache', {})
    substrate.compute_substrate_properties('CCO')
    old=substrate._disk_cache_path('CCO')
    saved=json.loads(old.read_text())
    saved['context']['rdkit_version']='different implementation'
    old.write_text(json.dumps(saved))
    assert substrate._load_from_disk('CCO', '') is None
    monkeypatch.setattr(substrate, '_CACHE_CONTEXT', {'rdkit_version':'changed'})
    assert substrate._disk_cache_path('CCO')!=old
