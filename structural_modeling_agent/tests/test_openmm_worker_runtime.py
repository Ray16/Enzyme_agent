"""Run in processivity-md; the lightweight controller environment is separate."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("westpa")
import openmm
from westpa.core.segment import Segment
from structural_modeling_agent.sampling.openmm_worker import PersistentOpenMMPropagator, claim_seed
from structural_modeling_agent.sampling.pilot import digest


def test_worker_seed_claims_retry_collisions(tmp_path, monkeypatch):
    values = iter([0, 0, 1])
    monkeypatch.setattr("secrets.randbelow", lambda _: next(values))
    assert claim_seed(tmp_path, "integrator") == 1
    assert claim_seed(tmp_path, "barostat") == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize("changed", [False, True])
def test_completed_segment_cache_rechecks_artifacts_without_gpu(tmp_path, changed):
    prop = PersistentOpenMMPropagator.__new__(PersistentOpenMMPropagator)
    prop.profile_path = tmp_path / "profile.json"; prop.profile_path.write_text("{}")
    prop.profile_hash = digest(prop.profile_path)
    from structural_modeling_agent.sampling import openmm_worker
    prop.implementation_hash = digest(openmm_worker.__file__)
    parent = tmp_path / "parent.xml"; parent.write_text("no context is initialized for a cache hit")
    out = tmp_path / "segment"; out.mkdir()
    np.save(out / "pcoord.npy", np.array([[1.], [2.], [3.]], dtype=np.float32))
    (out / "seg.xml").write_text("cached endpoint")
    (out / "complete.json").write_text(json.dumps(dict(profile_sha256=prop.profile_hash,
        parent_sha256=digest(parent), implementation_sha256=prop.implementation_hash,
        openmm_version=openmm.__version__, artifacts={n:digest(out/n) for n in ["pcoord.npy","seg.xml"]})))
    prop.update_args_env_iter = lambda args, env, n: None
    def paths(args, env, segment):
        env[prop.ENV_CURRENT_SEG_DATA_REF] = str(out)
        env[prop.ENV_PARENT_DATA_REF] = str(parent)
    prop.update_args_env_segment = paths
    prop._initialize = lambda: pytest.fail("Cache hit must not initialize a GPU context")
    if changed: (out / "seg.xml").write_text("altered endpoint")
    segment = Segment(n_iter=2, seg_id=0, parent_id=0)
    if changed:
        with pytest.raises(ValueError, match="artifact changed"): prop.propagate([segment])
    else:
        assert prop.propagate([segment])[0].status == Segment.SEG_STATUS_COMPLETE
        np.testing.assert_array_equal(segment.pcoord[:,0], [1,2,3])
