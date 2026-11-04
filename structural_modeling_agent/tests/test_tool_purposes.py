"""Default cofolding and evidence-justified follow-ups; sampling is mocked."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from structural_modeling_agent.validation import fingerprint, file_digest
from structural_modeling_agent.scientific_reasoning import assess_reasoning
from test_validation import modeling_record, write_pose
from test_modeling_agent import make_agent, propose, msa, respond

ACC = {"accession": "SYNTHETIC1"}


def baseline(tmp_path, displacement=0):
    agent = make_agent(tmp_path, objective="model_and_validate")
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    directory=tmp_path/"cofold"; directory.mkdir()
    path=write_pose(directory/"model_0.pdb",displacement=displacement)
    revision=agent._current("SYNTHETIC1")
    producer=tmp_path/"producer.json"
    producer.write_text(json.dumps({"input_sha256":revision["input_sha256"],"execution_sha256":fingerprint(revision["execution"]),
        "predictions":[{"path":str(path),"sha256":file_digest(path)}]}))
    agent.dispatch("validate_predictions",{**ACC,"directory":str(directory),"prediction_manifest":str(producer)})
    respond(agent,"investigate" if displacement else "retain")
    return agent


def followup(agent, purpose="dynamics"):
    revision=agent._current("SYNTHETIC1")
    plan={"purpose":purpose,"question":"Could pocket conformational variability explain the observed pose?",
        "why_cofold_insufficient":"A static cofold cannot distinguish stable exclusion from alternative accessible conformations.",
        "expected_information":"Distinguish accessibility hypotheses from a persistent geometric mismatch.",
        "decision_if_supported":"Investigate ensemble-supported accessibility with independent measurements.",
        "decision_if_refuted":"Revise the accessibility hypothesis or choose another candidate.",
        "assessment_sha256":assess_reasoning(revision)["assessment_sha256"],"hypothesis_ids":["H1","H2"],"num_samples":4}
    return agent.dispatch("plan_followup",{**ACC,"plan":plan})


def test_default_is_cofold_and_followups_cannot_replace_it(tmp_path):
    agent=make_agent(tmp_path,objective="model_and_validate")
    policy=agent.dispatch("inspect_workflow_policy",ACC)
    assert policy["default"]=="cofolding" and not policy["followups_available"]
    options=agent.dispatch("inspect_mlmm_engine_options",{})
    assert "not a separate checkpoint" in options["uma"]["nitro_is"]
    assert {option["id"] for option in options["embedding_options"]}=={"mechanical","xtb_oniom_ee","dft_electrostatic","emle"}
    agent.dispatch("propose_model",propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs",msa(tmp_path))
    with pytest.raises(ValueError,match="cofolding is the default prerequisite"):
        followup(agent)
    with pytest.raises(ValueError,match="saved dynamics"):
        agent.dispatch("prepare_bioemu2",{**ACC,"followup_id":"missing","checkpoint_path":"unused","seed":1})
    agent.backend={"provider":"lambda","engine":"bioemu2"}
    with pytest.raises(ValueError,match="initial submission is cofolding"):
        agent.dispatch("submit_cofold",ACC)


def test_unproductive_but_traceable_cofold_can_motivate_dynamics(tmp_path):
    agent=baseline(tmp_path,displacement=30)
    assert agent._current("SYNTHETIC1")["pose_validation"]["status"]=="fail"
    assert agent.dispatch("inspect_workflow_policy",ACC)["followups_available"]
    dynamics=followup(agent)
    assert dynamics["tool"]=="bioemu2"
    design=followup(agent,"de_novo_design")
    assert design["tool"]=="rfd4-proteina"
    handoff=json.loads(Path(design["handoff"]).read_text())
    assert handoff["recipient"]=="enzyme_design_agent" and handoff["purpose"]=="de_novo_enzyme_design"
    assert followup(agent)["artifact"]==dynamics["artifact"]


def test_bioemu2_preparation_preserves_entities_and_msas_and_is_optional(tmp_path):
    agent=baseline(tmp_path)
    checkpoint=tmp_path/"weights.ckpt"; checkpoint.write_text("fake local weights; never loaded")
    (tmp_path/"config.yaml").write_text("fake config")
    plan=followup(agent)
    prepared=agent.dispatch("prepare_bioemu2",{**ACC,"followup_id":plan["id"],"checkpoint_path":str(checkpoint),"seed":7})
    job=json.loads(Path(prepared["job_path"]).read_text())
    assert job["purpose"]=="dynamics" and job["mode"]=="equilibrium" and job["num_samples"]==4
    system=json.loads(Path(job["system"]).read_text())
    assert [e["entity_type"] for e in system["entities"]]==["protein","ligand"]
    assert system["entities"][1]["smiles"]=="CC=O"
    assert Path(job["msa_by_chain"]["A"]).read_text().startswith(">alignment_0\nAG\n")
    assert "bioemu2" not in sys.modules
    assert agent.completion_errors()==[]  # Optional study does not rewrite primary evidence.


def test_mocked_bioemu2_run_and_dynamics_inspection_preserve_rejected_frames(tmp_path,monkeypatch):
    from structural_modeling_agent.bioemu2_adapter import run
    agent=baseline(tmp_path)
    original=fingerprint(agent._current("SYNTHETIC1")["pose_validation"])
    checkpoint=tmp_path/"weights.ckpt"; checkpoint.write_text("fake")
    (tmp_path/"config.yaml").write_text("fake")
    plan=followup(agent)
    prepared=agent.dispatch("prepare_bioemu2",{**ACC,"followup_id":plan["id"],"checkpoint_path":str(checkpoint),"seed":7})
    class Samples:
        def __init__(self,indices): self.indices=indices
        def __len__(self): return len(self.indices)
        def __getitem__(self,key): return Samples(self.indices[key])
    def save(samples,directory,physicality=None):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        write_pose(directory/"topology.cif",displacement=samples.indices[0]*.1,cif=True)
        (directory/"samples.xtc").write_bytes(bytes(samples.indices))
        if physicality is not None:
            (directory/"physicality.json").write_text(json.dumps([{"frame":i,"status":s,"reason":None} for i,s in enumerate(["passed","rejected","not_checked","passed"])]))
    class Session:
        def __init__(self,*args,**kwargs): assert kwargs["mode"]=="equilibrium" and kwargs["msa"]=="local msa"
        def sample(self,**kwargs):
            assert kwargs=={"num_samples":4,"seed":7}; return Samples(list(range(4)))
    def get_msa(system,source):
        assert set(source)=={"A"} and all(Path(p).is_file() for p in source.values()); return "local msa"
    fake=SimpleNamespace(System=SimpleNamespace(from_yaml=lambda path:json.loads(Path(path).read_text())),
        get_msa=get_msa,load_model=lambda **kwargs:"local checkpoint",EmuSession=Session,
        check_physicality=lambda *args,**kwargs:object(),save_frames=save)
    monkeypatch.setitem(sys.modules,"bioemu2",fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","mock-only")
    result=run(prepared["job_path"],tmp_path/"results")
    inspected=agent.dispatch("inspect_dynamics",{**ACC,"followup_id":plan["id"],"prediction_manifest":result["prediction_manifest"]})
    rows=inspected["cofold_geometry_checks_for_comparison"]["poses"]
    assert len(rows)==4 and any(c["status"]=="fail" for c in rows[1]["checks"] if c["check"].startswith("bioemu2"))
    assert any(c["status"]=="abstain" for c in rows[2]["checks"] if c["check"].startswith("bioemu2"))
    assert fingerprint(agent._current("SYNTHETIC1")["pose_validation"])==original
    (tmp_path/"other").mkdir()
    other=make_agent(tmp_path/"other",objective="model_and_validate")
    # Direct primary ingestion is forbidden even though BioEmu2 samples complexes.
    other_dir=tmp_path/"other"; other.dispatch("propose_model",propose(modeling_record(other_dir))); other.dispatch("render_inputs",msa(other_dir))
    with pytest.raises(ValueError,match="cannot satisfy primary cofolding"):
        other.dispatch("validate_predictions",{**ACC,"directory":result["predictions"],"prediction_manifest":result["prediction_manifest"]})
