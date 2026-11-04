"""New-system output isolation and discoverability without GPU/model calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from structural_modeling_agent.agent import ModelingAgent
from structural_modeling_agent.result_layout import new_layout, system_identity, read_run_index
from test_modeling_agent import make_agent, propose, msa
from test_validation import modeling_record, write_pose
from test_tool_purposes import baseline, followup

ACC = {"accession": "SYNTHETIC1"}


def test_system_and_run_identity_do_not_mix_targets_or_overwrite_runs(tmp_path):
    a={"target":{"id":"enzyme-system","reaction":"reaction A"},"candidates":[]}
    b=copy.deepcopy(a); b["target"]["reaction"]="reaction B"
    first=new_layout(a,results_root=tmp_path)
    again=new_layout(a,results_root=tmp_path)
    other=new_layout(b,results_root=tmp_path)
    assert first["system"]==again["system"]
    assert first["run_dir"]!=again["run_dir"]
    assert first["system"]["key"]!=other["system"]["key"]
    assert not list(tmp_path.iterdir())  # Naming is a pure operation.
    with pytest.raises(ValueError): system_identity(a,"../outside")
    with pytest.raises(ValueError): new_layout(a,run_dir=tmp_path/"run",results_root=tmp_path)


def test_default_root_and_index_are_explicit_and_resume_keeps_them(tmp_path,monkeypatch):
    from structural_modeling_agent import result_layout
    original_default=result_layout.default_results_root()
    task={"schema_version":"1.0","target":{"id":"nylon-ligation"},"objective":"prepare_inputs",
          "candidates":[{"accession":"SYNTHETIC1","sequence":"AG","evidence":{}}]}
    path=tmp_path/"task.json"; path.write_text(json.dumps(task))
    monkeypatch.delenv("ENZYME_AGENT_RESULTS_ROOT",raising=False)
    monkeypatch.chdir(tmp_path)
    assert result_layout.default_results_root()==original_default
    monkeypatch.setattr(result_layout,"default_results_root",lambda:tmp_path/"results")
    agent=ModelingAgent(task_path=path,system_id="new_enzyme")
    assert agent.run_dir.is_relative_to(tmp_path/"results")
    assert "new_enzyme-" in str(agent.run_dir)
    assert (agent.run_dir/"run_manifest.json").is_file()
    assert json.loads((agent.run_dir/"task_snapshot.json").read_text())==task
    index=agent.dispatch("inspect_result_locations",{})
    assert index["run_status"]=="running"
    assert index["candidates"]["SYNTHETIC1"]["current_revision"] is None
    elsewhere=tmp_path/"elsewhere"; elsewhere.mkdir(); monkeypatch.chdir(elsewhere)
    resumed=ModelingAgent(run_dir=agent.run_dir,resume=True)
    assert resumed.run_dir==agent.run_dir
    assert resumed.dispatch("inspect_result_locations",{})["layout"]==index["layout"]
    with pytest.raises(ValueError,match="cannot replace"):
        ModelingAgent(run_dir=agent.run_dir,resume=True,results_root=elsewhere)


def test_revision_paths_are_separate_and_index_does_not_infer_success(tmp_path):
    agent=make_agent(tmp_path)
    args=propose(modeling_record(tmp_path))
    agent.dispatch("propose_model",args)
    rendered=agent.dispatch("render_inputs",msa(tmp_path))
    first=Path(rendered["input_yaml"])
    assert first.parent==agent.run_dir/"models/SYNTHETIC1/revision_1/cofold"
    assert "sequences" in agent.dispatch("read_evidence",{"path":str(first)})["text"]
    agent.dispatch("propose_model",args)
    agent.dispatch("render_inputs",msa(tmp_path))
    agent._save()
    index=read_run_index(agent.run_dir)
    revisions=index["candidates"]["SYNTHETIC1"]["revisions"]
    assert len(revisions)==2 and first.is_file()
    assert revisions[0]["cofold"]["input_yaml"]!=revisions[1]["cofold"]["input_yaml"]
    assert all(r["cofold"]["pose_validation_status"]=="not_inspected" for r in revisions)
    # A stale compatibility index cannot replace the authoritative checkpoint.
    (agent.run_dir/"results_index.json").write_text('{"run_status":"complete"}')
    assert read_run_index(agent.run_dir)["run_status"]=="running"


def test_dynamics_and_design_destinations_share_the_system_but_not_the_stage(tmp_path):
    agent=baseline(tmp_path)
    dynamics=followup(agent)
    design=followup(agent,"de_novo_design")
    assert Path(dynamics["paths"]["root"]).is_relative_to(agent.run_dir)
    assert Path(design["paths"]["root"]).is_relative_to(agent.run_dir)
    assert dynamics["paths"]["root"]!=design["paths"]["root"]
    assert "/followups/dynamics/" in dynamics["paths"]["root"]
    assert "/followups/de_novo_design/" in design["paths"]["root"]
    assert design["review_command"][-2:]==["--output-dir",design["paths"]["design_runs_root"]]
    assert design["design_agent_arguments"]["output_dir"]==design["paths"]["design_runs_root"]
    checkpoint=tmp_path/"weights.ckpt"; checkpoint.write_text("mock weights")
    (tmp_path/"config.yaml").write_text("mock config")
    job=agent.dispatch("prepare_bioemu2",{**ACC,"followup_id":dynamics["id"],"checkpoint_path":str(checkpoint),"seed":7})
    assert Path(job["job_path"]).parent==Path(dynamics["paths"]["preparation_dir"])
    assert job["run_command"][-2:]==["--output",dynamics["paths"]["results_dir"]]


def test_rfd4_receives_the_allocated_design_directory_and_link_is_not_validation(tmp_path):
    from enzyme_design_agent.tools import rfd4_design as rfd
    agent=baseline(tmp_path)
    design=followup(agent,"de_novo_design")
    child=Path(design["paths"]["design_runs_root"])/"native_design_run"
    child.mkdir(parents=True)
    source=write_pose(tmp_path/"annotated.cif",cif=True)
    with patch.object(rfd,"_preflight",return_value={"ready":True,"free_gpus":["mock"]}), \
         patch.object(rfd.subprocess,"Popen",return_value=type("Launcher",(),{"pid":os.getpid()})()):
        launched=rfd.design_enzyme_with_rfd4(str(source),run_dir=str(child))
    assert launched["status"]=="running"
    assert Path(launched["data"]["job_dir"]).is_relative_to(child/"rfd4")
    (child/"manifest.json").write_text(json.dumps({"agent":"enzyme_design_agent","status":"complete","designs":0}))
    (child/"DESIGN_REPORT.md").write_text("No generated enzyme has been validated.")
    linked=agent.dispatch("register_design_run",{**ACC,"followup_id":design["id"],"design_run_dir":str(child)})
    assert linked["generation_validation"]=="not_established_by_agent_report"
    agent._save()
    index=read_run_index(agent.run_dir)
    study=index["candidates"]["SYNTHETIC1"]["revisions"][0]["followups"][0]
    assert study["registered_design_run"]["run_dir"]==str(child)
    with pytest.raises(ValueError,match="allocated output root"):
        agent.dispatch("register_design_run",{**ACC,"followup_id":design["id"],"design_run_dir":str(tmp_path)})


def test_legacy_runs_are_inspected_and_resumed_without_moving_outputs(tmp_path,monkeypatch):
    agent=make_agent(tmp_path)
    agent.state.pop("result_layout")  # Emulate a pre-layout checkpoint.
    agent.dispatch("propose_model",propose(modeling_record(tmp_path)))
    old=Path(agent.dispatch("render_inputs",msa(tmp_path))["input_yaml"])
    assert old.parent.name=="revision_1"
    agent._save()
    other=tmp_path/"other"; other.mkdir(); monkeypatch.chdir(other)
    resumed=ModelingAgent(run_dir=agent.run_dir,resume=True)
    index=read_run_index(agent.run_dir)
    assert index["layout"]["schema_version"]=="legacy"
    assert index["candidates"]["SYNTHETIC1"]["revisions"][0]["cofold"]["input_yaml"]==str(old)
    assert old.is_file() and not (old.parent/"cofold").exists()
    assert resumed._current("SYNTHETIC1")["input_yaml"]==str(old)


def test_generated_directory_symlinks_cannot_grant_access_outside_this_run(tmp_path):
    agent=make_agent(tmp_path)
    secret=tmp_path/"outside.txt"; secret.write_text("not supplied")
    (agent.run_dir/"models").mkdir()
    link=agent.run_dir/"models/foreign.txt"; link.symlink_to(secret)
    with pytest.raises(PermissionError):agent.dispatch("read_evidence",{"path":str(link)})


def test_offline_run_inspection_needs_no_model_provider(tmp_path):
    agent=make_agent(tmp_path)
    result=subprocess.run([sys.executable,"-m","structural_modeling_agent","--inspect-run",str(agent.run_dir)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["layout"]["run_dir"]==str(agent.run_dir)


def test_cli_can_initialize_a_new_system_without_starting_models(tmp_path):
    task=tmp_path/"task.json"
    task.write_text(json.dumps({"schema_version":"1.0","target":{"id":"new-system"},"candidates":[
        {"accession":"SYNTHETIC1","sequence":"AG","evidence":{}}]}))
    result=subprocess.run([sys.executable,"-m","structural_modeling_agent","--task",str(task),
        "--results-root",str(tmp_path/"results"),"--system-id","new-system","--initialize-only"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    data=json.loads(result.stdout)
    assert not data["model_called"] and not data["compute_submitted"]
    index=read_run_index(data["run_dir"])
    assert index["run_status"]=="initialized"
    assert index["candidates"]["SYNTHETIC1"]["planned_next_cofold_dir"].endswith("revision_1/cofold")
