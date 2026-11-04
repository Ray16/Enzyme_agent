"""An automatic attachment preserves owner identity and explicit compute boundaries."""
import json
from enzyme_discovery_agent import rolling_modeling


def test_repeated_attach_keeps_a_live_coordinator_and_does_not_submit_again(tmp_path,monkeypatch):
    folder=tmp_path/'rolling_modeling';folder.mkdir()
    owner={'pid':123,'host':'synthetic','process_start_ticks':'42'}
    (folder/'launch_manifest.json').write_text(json.dumps(owner))
    monkeypatch.setattr(rolling_modeling,'live',lambda record:True)
    monkeypatch.setattr(rolling_modeling.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('duplicate submission')))
    assert rolling_modeling.attach(tmp_path)==owner


def test_dead_coordinator_requires_reconciliation_not_duplicate_gpu_jobs(tmp_path,monkeypatch):
    import pytest
    folder=tmp_path/'rolling_modeling';folder.mkdir()
    (folder/'launch_manifest.json').write_text(json.dumps({'pid':123}))
    monkeypatch.setattr(rolling_modeling,'live',lambda record:False)
    with pytest.raises(RuntimeError,match='reconcile'):rolling_modeling.attach(tmp_path)


def test_frozen_runtime_excludes_generated_campaigns_and_keeps_static_contracts(tmp_path):
    hashes=rolling_modeling._snapshot(tmp_path)
    assert hashes
    assert not any('/cofold/' in name or '/integrated_campaigns/' in name or '/runs/' in name for name in hashes)
    assert 'structural_modeling_agent/validation_contract.json' in hashes
    assert 'structural_modeling_agent/knowledge/attack_angle_metric.json' in hashes
    assert 'enzyme_discovery_agent/rolling_modeling.py' in hashes
