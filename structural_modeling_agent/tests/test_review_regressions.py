"""Offline reproductions from the structural review. No live models/providers."""
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from structural_modeling_agent.validation import validate_ensemble, validate_pose_plan, file_digest, fingerprint
from test_validation import modeling_record, execution_for, write_pose
from test_modeling_agent import make_agent, propose, msa, respond


def test_incomplete_protein_cannot_complete_or_export_validated(tmp_path):
    agent = make_agent(tmp_path, objective="model_and_validate")
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    directory = tmp_path / "poses"; directory.mkdir()
    path = write_pose(directory / "s_model_0.pdb")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line for line in lines if not line.startswith("ATOM  ") or
        (line[22:26].strip(), line[12:16].strip()) in {("1", "N"), ("2", "CA")}) + "\n")
    revision = agent._current("SYNTHETIC1")
    manifest = tmp_path / "producer.json"
    manifest.write_text(json.dumps({"input_sha256": revision["input_sha256"], "execution_sha256": fingerprint(revision["execution"]),
                                    "predictions": [{"path": str(path), "sha256": file_digest(path)}]}))
    result = agent.dispatch("validate_predictions", {"accession": "SYNTHETIC1", "directory": str(directory), "prediction_manifest": str(manifest)})
    assert result["status"] != "pass"
    assert any(c["check"].startswith("protein_completeness") and c["status"] == "abstain" for c in result["poses"][0]["checks"])
    respond(agent)
    assert agent.dispatch("finish_task", {"outcome": "complete", "rationale": "Cannot ignore missing protein atoms"})["status"] == "blocked"
    agent._write_report()
    exported = json.loads((agent.run_dir / "structural_modeling_evidence.json").read_text())
    assert exported["candidates"][0]["status"] == "incomplete"


def test_declared_omission_cannot_make_required_protein_atoms_complete(tmp_path):
    record=modeling_record(tmp_path)
    path=write_pose(tmp_path/"model.pdb")
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not (line.startswith("ATOM  ") and line[12:16].strip()=="CB"))+"\n")
    record["validation_plan"]["poses"]["protein_atom_policy"]={"allowed_missing":[{
        "chain":"A","residue":"1","atom":"CB","rationale":"Record this source-backed omission without pretending the environment is complete",
        "evidence":copy.deepcopy(record["validation_plan"]["poses"]["evidence"])}]}
    result=validate_ensemble([path],execution_for(record))
    assert result["status"]!="pass"
    check=next(c for c in result["poses"][0]["checks"] if c["check"]=="protein_completeness.A.1")
    assert check["missing_atoms"]==["CB"] and check["declared_missing_atoms"]==["CB"]


def test_duplicate_physical_site_and_absent_acceptance_plan_block_before_compute(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, objective="model_and_validate")
    record = modeling_record(tmp_path)
    record["validation_plan"]["setup"]["assembly"]["expected"]["active_sites"] = 2
    duplicate = copy.deepcopy(record["validation_plan"]["poses"]["sites"][0]); duplicate["id"] = "another_id"
    record["validation_plan"]["poses"]["sites"].append(duplicate)
    assert "duplicate physical" in str(validate_pose_plan(record))
    agent.dispatch("propose_model", propose(record))
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"
    assert validate_ensemble([write_pose(tmp_path / "p.pdb")], execution_for(record))["status"] != "pass"
    # A separate incomplete plan also cannot reach a provider.
    record = modeling_record(tmp_path); record["validation_plan"]["poses"] = {"expected_predictions": 1}
    respond(agent, "revise")
    agent.dispatch("propose_model", propose(record))
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"
    called = []
    monkeypatch.setattr(agent, "_backend_call", lambda *args: called.append(args))
    with pytest.raises(ValueError):
        agent.dispatch("submit_cofold", {"accession": "SYNTHETIC1"})
    assert called == []


def test_malformed_sample_keeps_fingerprint_and_never_crashes_completion(tmp_path):
    agent = make_agent(tmp_path, objective="model_and_validate")
    record = modeling_record(tmp_path)
    record["validation_plan"]["poses"].update(expected_predictions=2, minimum_passing_predictions=1)
    agent.dispatch("propose_model", propose(record)); agent.dispatch("render_inputs", msa(tmp_path))
    directory = tmp_path / "poses"; directory.mkdir()
    good = write_pose(directory / "s_model_0.pdb")
    bad = directory / "s_model_1.pdb"; bad.write_text("invalid structure")
    result = agent.dispatch("validate_predictions", {"accession": "SYNTHETIC1", "directory": str(directory)})
    row = next(row for row in result["poses"] if row["path"] == str(bad))
    assert row["sha256"] == file_digest(bad) and row["status"] == "fail"
    respond(agent)
    assert isinstance(agent.completion_errors(), list)
    agent._write_report()


def test_changed_prediction_bytes_are_not_rebound_to_old_measurements(tmp_path,monkeypatch):
    from structural_modeling_agent import validation
    record=modeling_record(tmp_path); path=write_pose(tmp_path/"model.pdb")
    original=validation._read_structure
    def changed(file):
        result=original(file)
        Path(file).write_text(Path(file).read_text()+"REMARK changed while measuring\n")
        return result
    monkeypatch.setattr(validation,"_read_structure",changed)
    result=validation.validate_ensemble([path],execution_for(record))
    assert result["status"]!="pass"
    assert result["poses"][0]["sha256"]!=file_digest(path)


def test_twenty_candidate_context_is_bounded_and_detailed_reasoning_remains_readable(tmp_path):
    record = modeling_record(tmp_path)
    task = {"schema_version": "1.0", "target": {}, "objective": "prepare_inputs",
            "evidence_paths": [str(tmp_path / "reference.txt")],
            "candidates": [{"accession": f"S{i}", "sequence": "AG", "evidence": {}} for i in range(20)]}
    path = tmp_path / "task.json"; path.write_text(json.dumps(task))
    from structural_modeling_agent.agent import ModelingAgent
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / "run")
    for i in range(20):
        args = propose(record); args["accession"] = f"S{i}"
        agent.dispatch("propose_model", args)
        assert len(json.dumps(agent._context())) <= 48000
    agent._save()
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True)
    assert len(json.dumps(resumed._context())) <= 48000
    page = resumed.dispatch("read_reasoning_section", {"accession": "S19", "section": "scientific_reasoning"})
    assert "H1" in page["text"] and page["characters"] > 0


def test_directional_dispersion_does_not_call_opposite_directions_aligned():
    from structural_modeling_agent.proxies.library import BY_ID, Ensemble
    from structural_modeling_agent.proxies.roles import ReactiveSite
    def measure(vectors):
        return BY_ID["approach_consistency"](Ensemble([ReactiveSite(np.zeros(3), 3 * v) for v in vectors], []))
    axes = np.vstack([np.eye(3), -np.eye(3)])
    assert measure(axes) == pytest.approx(90)
    assert measure(np.array([[1., 0, 0]] * 6)) == pytest.approx(0)
    assert BY_ID["approach_consistency"].version == "2.0"


@pytest.mark.parametrize("value,threshold", [(float("nan"), .5), (float("inf"), .5), (.8, float("nan"))])
def test_malformed_developability_abstains(value, threshold):
    from structural_modeling_agent.developability.predictors import SolubilityPrediction, parse_netsolp_csv
    from structural_modeling_agent.developability.assess import assess_developability
    from structural_modeling_agent.developability.evidence import LiabilityKnowledge
    class Predictor:
        def predict(self, sequence):
            return SolubilityPrediction(value, "synthetic", threshold=threshold)
    result = assess_developability(sequence="AGAGA", knowledge=LiabilityKnowledge([]), predictor=Predictor())
    assert result.verdict == "abstain" and result.score is None
    assert parse_netsolp_csv("sid,predicted_solubility\nx,nan\n")[0]["predicted_solubility"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_queue_report_never_ranks_deleted_or_legacy_unbound_predictions(tmp_path):
    from structural_modeling_agent.cofold.queue_report import publish
    (tmp_path / "completed").mkdir()
    manifest = {"ranking": "inverse_distance_attack_angle_v1", "limits": [], "items": [{"id": "item", "input_sha256": "old",
        "expected_predictions": 1, "name": "synthetic", "priority": "test", "geometry": {"phosphate": "PG"}}]}
    source = tmp_path / "manifest.json"; source.write_text(json.dumps(manifest))
    (tmp_path / "completed/item.json").write_text(json.dumps({"input_sha256": "old", "score": {"n_poses": 1}, "output": "/missing"}))
    result = publish(source)
    assert result["status"] == "unresolved" and result["ranked_candidates"] == []


def test_queue_reuse_binds_seed_geometry_msa_and_actual_confidence_files(tmp_path):
    import gemmi
    from structural_modeling_agent.cofold import queue_worker as worker
    output = tmp_path / "predictions"; output.mkdir()
    path = write_pose(output / "item_model_0.pdb")
    structure = gemmi.read_structure(str(path))
    oxygen = gemmi.Atom(); oxygen.name="O2"; oxygen.element=gemmi.Element("O"); oxygen.pos=gemmi.Position(3,0,1.2)
    structure[0]["B"][0].add_atom(oxygen)
    for chain_id, name, fields in [("C", "ATP", [("PA","P",(6,0,0)),("O3A","O",(7.5,0,0))]),
                                    ("D","MG",[("MG","Mg",(8,3,0))])]:
        chain=gemmi.Chain(chain_id); residue=gemmi.Residue()
        residue.name=name; residue.seqid=gemmi.SeqId(1," "); residue.het_flag="H"
        for atom_name, element, xyz in fields:
            atom=gemmi.Atom(); atom.name=atom_name; atom.element=gemmi.Element(element); atom.pos=gemmi.Position(*xyz)
            residue.add_atom(atom)
        chain.add_residue(residue); structure[0].add_chain(chain)
    structure.write_pdb(str(path))
    sidecar=output / "confidence_item_model_0.json"; sidecar.write_text('{"complex_plddt":0.9}')
    source=tmp_path / "input.yaml"; source.write_text("input")
    alignment=tmp_path / "input.csv"; alignment.write_text("key,sequence\n0,AG\n")
    import hashlib
    item={"id":"item","input_yaml":str(source),"input_sha256":file_digest(source),
        "msa_path":str(alignment),"msa_sha256":file_digest(alignment),"seed":1,"expected_predictions":1,
        "sequence_sha256":hashlib.sha256(b"AG").hexdigest(),"protein_chain":"A","expected_chains":["A","B","C","D"],
        "metal_chain":"D","metal_element":"MG", "geometry":{"substrate_chain":"B","nucleotide_chain":"C","phosphate":"PA","bridge":"O3A"}}
    manifest={"producer_argv":["mock-only"],"ranking":"inverse_distance_attack_angle_v1"}
    spec, identity=worker.execution_spec(manifest,item)
    saved={"schema_version":"queue-completion-2.0","execution_spec":spec,"execution_sha256":identity,
           "output":str(output),**worker.verify_outputs(output,item)}
    assert worker.verify_completion(manifest,item,saved)["score"]["n_poses"]==1
    for change in [{"seed":2},{"geometry":{**item["geometry"],"phosphate":"PG"}}]:
        with pytest.raises(ValueError,match="identity"):
            worker.verify_completion(manifest,{**item,**change},saved)
    alignment.write_text("changed MSA")
    with pytest.raises(ValueError,match="fingerprint"):
        worker.verify_completion(manifest,item,saved)
    alignment.write_text("key,sequence\n0,AG\n")
    sidecar.write_text('{"complex_plddt":0.8}')
    with pytest.raises(ValueError,match="confidence_files"):
        worker.verify_completion(manifest,item,saved)


def test_orphan_output_does_not_block_a_new_journaled_attempt(tmp_path, monkeypatch):
    from structural_modeling_agent.cofold import queue_worker as worker
    source, msa_path = tmp_path / "input.yaml", tmp_path / "msa.csv"
    source.write_text("input"); msa_path.write_text("msa")
    orphan = tmp_path / "predictions/item/attempt_1"; orphan.mkdir(parents=True)
    manifest = {"cpu_slots_per_worker": 1, "maximum_attempts": 1, "failure_circuit_breaker": 2, "producer_argv": ["mock-only"],
        "items": [{"id": "item", "input_yaml": str(source), "input_sha256": worker.digest(source),
                   "msa_path": str(msa_path), "msa_sha256": worker.digest(msa_path), "seed": 1, "expected_predictions": 1}]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["queue_worker", "--manifest", str(path), "--worker-id", "test"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "mock-only")
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0}); monkeypatch.setattr(os, "getloadavg", lambda: (0,0,0))
    calls = []
    def producer(*args, **kwargs):
        journal = json.loads((tmp_path / "attempts/item/attempt_2.json").read_text())
        assert journal["status"] == "running"
        calls.append(args)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(worker.subprocess, "run", producer)
    worker.main()
    assert len(calls) == 1 and orphan.exists()
    assert json.loads((tmp_path / "attempts/item/attempt_2.json").read_text())["status"] == "failed"


def test_standalone_builder_rejects_wrong_msa_and_fingerprints_valid_query(tmp_path):
    import subprocess
    record=modeling_record(tmp_path)
    handoff=tmp_path/"handoff.json"; handoff.write_text(json.dumps({"schema_version":"2.0","target":{},
        "candidates":[{"accession":"SYNTHETIC1","evidence":{},"selection":{},"modeling_input":record}]}))
    fasta=tmp_path/"input.fa"; fasta.write_text(">SYNTHETIC1\nAG\n")
    alignment=tmp_path/"query.csv"; alignment.write_text("key,sequence\n0,WW\n")
    index=tmp_path/"index.tsv"; index.write_text(f"SYNTHETIC1\tunused\t{alignment}\n")
    def build(name):
        out=tmp_path/name
        subprocess.run([sys.executable,"-m","structural_modeling_agent.scripts.build_step1_inputs",
            "--handoff",str(handoff),"--fasta",str(fasta),"--msa-index",str(index),"--out",str(out),
            "--require-validated-setup"],check=True,capture_output=True)
        return json.loads((out/"campaign_manifest.json").read_text())
    assert build("wrong")["executions"]==[]
    alignment.write_text("key,sequence\n0,AG\n")
    result=build("valid")["executions"][0]
    assert result["msa_evidence"][0]["sha256"]==file_digest(alignment)


def test_frozen_bundle_rejects_copied_prediction_contents(tmp_path):
    from test_integration import bundle, candidate
    from structural_modeling_agent.integration import apply_bundle
    directory=tmp_path/"poses"; directory.mkdir()
    for i in range(3):
        (directory/f"item_model_{i}.pdb").write_text("identical prediction bytes")
    b=bundle(); c=candidate(); c.update(prediction_dir=str(directory))
    result=apply_bundle(b,[c],tmp_path)["candidates"][0]
    assert result["status"]!="scored"
    assert "byte-identical" in str(result)


def test_producer_sample_ids_are_required_when_the_bundle_declares_that_requirement(tmp_path):
    from test_integration import bundle, candidate
    from structural_modeling_agent.integration import apply_bundle, definition_digest
    b=bundle(); b["definition"]["require_producer_sample_ids"]=True
    b["definition_sha256"]=definition_digest(b["definition"])
    paths=[]
    for i in range(10):
        path=tmp_path/f"sample_model_{i}.pdb"; path.write_text(f"distinct identity-only test fixture {i}")
        paths.append(path)
    c=candidate(); c["prediction_files"]=[str(p) for p in paths]; c["input_sha256"]="a"*64
    assert apply_bundle(b,[c],tmp_path)["candidates"][0]["status"]=="unavailable"
    producer={"input_sha256":c["input_sha256"],"predictions":[{"path":str(p),"sha256":file_digest(p),"sample_id":str(i)} for i,p in enumerate(paths)]}
    manifest=tmp_path/"producer.json"; manifest.write_text(json.dumps(producer)); c["prediction_manifest"]=str(manifest)
    assert apply_bundle(b,[c],tmp_path)["candidates"][0]["status"]=="scored"
    producer["predictions"][1]["sample_id"]="0"; manifest.write_text(json.dumps(producer))
    result=apply_bundle(b,[c],tmp_path)["candidates"][0]
    assert result["status"]=="unavailable" and "duplicated" in str(result)


def test_old_directional_calibrations_require_version_review():
    from test_integration import bundle
    from structural_modeling_agent.integration import validate_bundle, definition_digest
    b=bundle(); b["definition"]["features"][0]="approach_consistency"
    b["definition_sha256"]=definition_digest(b["definition"])
    with pytest.raises(ValueError,match="recalibrate"):
        validate_bundle(b)


def test_exposure_audit_is_read_only_and_does_not_flag_unscored_candidates(tmp_path):
    from structural_modeling_agent.audit_exports import audit
    source=tmp_path/"candidate_proxy_evidence.json"
    source.write_text(json.dumps({"schema_version":"2.0","candidates":[
        {"accession":"lead","status":"not_applicable","source_files":[]}]}))
    before=file_digest(source)
    result=audit([tmp_path])
    assert file_digest(source)==before
    assert result["artifacts"][0]["findings"][0]["status"]=="not_scored_no_sampling_claim"


def test_legacy_metal_conflict_is_blocked_without_false_applied_provenance():
    from structural_modeling_agent.modeling_reasoning.resolve_handoff import resolve_handoff
    from structural_modeling_agent.modeling_reasoning.literature import LiteratureKnowledge
    import importlib.util
    path=Path(__file__).parents[1]/"modeling_reasoning/tests/test_resolve_handoff.py"
    spec=importlib.util.spec_from_file_location("resolver_review_fixture",path)
    fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    manifest=copy.deepcopy(fixture.GENERIC_MANIFEST)
    manifest["resolutions"]["catalytic_metal"].update(strategy="explicit",per_site_metal_count=2,uniprot_cross_check=False)
    candidate=fixture._candidate("synthetic")
    candidate["modeling_input"]["cofactors"].append({"ccd":"MG","count":1,"role":"catalytic_metal"})
    result=resolve_handoff({"schema_version":"2.0","candidates":[candidate]},manifest,LiteratureKnowledge.load(),fetch=lambda *args:None)
    assert result["synthetic"]["status"]!="ready"
    provenance=candidate["modeling_input"]["resolution_provenance"]["catalytic_metal"]
    assert "resolved_cofactor" not in provenance and provenance["proposed_cofactor"]["count"]==2
