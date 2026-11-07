"""Standalone agent decisions, validation-driven revision, recovery and budgets."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from structural_modeling_agent.agent import ModelingAgent, _controller_lock
from structural_modeling_agent.validation import file_digest, fingerprint
from test_validation import modeling_record, write_pose


class Message:
    def __init__(self, calls):
        self.content = "Investigate the current evidence and choose the next modeling action."
        self.tool_calls = [{"id": f"call_{i}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]

    def model_dump(self):
        return {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}


class Client:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.contexts = []

    def call(self, **kwargs):
        self.contexts.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=Message(next(self.turns)))],
                               usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10))


def make_agent(tmp_path, client=None, objective="prepare_inputs"):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"schema_version": "1.0", "objective": objective,
                               "target": {"reaction": "synthetic target transformation"},
                               "evidence_paths": [str(tmp_path / "reference.txt")],
                               "candidates": [{"accession": "SYNTHETIC1", "sequence": "AG", "evidence": {}}]}))
    return ModelingAgent(task_path=task, run_dir=tmp_path / "agent", llm_client=client)


def test_large_source_history_is_bounded_with_complete_tool_transactions(tmp_path):
    agent=make_agent(tmp_path)
    for index in range(8):
        agent.messages.extend([{'role':'assistant','content':None,'tool_calls':[{
            'id':f'read_{index}','type':'function','function':{'name':'read_evidence','arguments':'{"path":"synthetic"}'}}]},
            {'role':'tool','tool_call_id':f'read_{index}','content':json.dumps({
                'artifact':f'/synthetic/source_{index}.json','summary':'S'*20000})}])
    context=agent._context()
    assert len(json.dumps(context))<=48000
    requested={call['id'] for item in context for call in item.get('tool_calls',[])}
    returned={item['tool_call_id'] for item in context if item['role']=='tool'}
    assert requested==returned
    assert 'source_7.json' in json.dumps(context)
    assert 'preview_only' in json.dumps(context)
    assert len(agent.messages[-1]['content'])>20000


def propose(record):
    record = copy.deepcopy(record)
    record["modeled_step"]["reaction_center"] = "other"
    return {"accession": "SYNTHETIC1", "modeling_input": record,
            "scientific_reasoning": scientific_reasoning(record),
            "rationale": "Compare the explicit construct and participant choices with the synthetic source.",
            "alternatives": [{"description": "A larger model", "reason_not_selected": "The reference describes a complete single-chain site."}]}


def scientific_reasoning(record):
    evidence = copy.deepcopy(record["validation_plan"]["setup"]["reaction_step"]["evidence"])
    for source in evidence:
        source["effect"] = "supports"
    hypothesis = {"id": "H1", "statement": "The declared construct can produce physically consistent test poses.",
        "mechanism": record["modeled_step"]["mechanism"], "step_index": record["modeled_step"]["index"],
        "step_description": "synthetic transfer", "reaction_center": "other",
        "center_rationale": "This synthetic test uses an explicitly mapped carbon center.",
        "assumptions": ["The synthetic tolerances describe this test fixture."],
        "uncertainties": ["Physical geometry does not establish activity."], "evidence": evidence,
        "predictions": [{"id": "P1", "metric": "pose_pass_fraction", "scope": "structural",
            "unit": "fraction", "endpoint": "declared_pose_checks", "conditions": {"basis": "declared_ensemble"},
            "expected_interval": [1, 1], "decision_if_refuted": "Investigate the assumed construct and reactive mapping."}],
        "discriminating_test": {"measurement": "Compare an independent pose with the declared geometry.",
            "expected_if_selected": "An independently measured geometry meets the declared interval.",
            "expected_if_alternative": "Independent geometry violates the declared interval.",
            "controls": ["Verify molecular identity before comparing geometry."]}}
    alternative = copy.deepcopy(hypothesis)
    alternative.update(id="H2", statement="The construct or molecular mapping yields nonproductive test poses.", evidence=[])
    alternative["predictions"][0].update(id="P2", expected_interval=[0, 0])
    return {"schema_version": "1.0", "selected_hypothesis_id": "H1", "hypotheses": [hypothesis, alternative],
            "selection_rationale": "The supplied synthetic source supports H1; H2 remains a testable modeling alternative."}


def response_args(agent, action="retain"):
    assessment = agent.dispatch("inspect_reasoning", {"accession": "SYNTHETIC1"})["assessment"]
    return {"accession": "SYNTHETIC1", "assessment_sha256": assessment["assessment_sha256"], "action": action,
        "rationale": "Retain all failed observations and test construct error against a geometry-proxy error.",
        "addressed_hypotheses": ["H1", "H2"],
        "next_test": {"measurement": "Compare independent geometry and a normalized assay.",
            "expected_if_selected": "Geometry and assay agree after confirming construct identity.",
            "expected_if_alternative": "Confirmed constructs retain a geometry/assay discrepancy.",
            "controls": ["Normalize active enzyme amount and retain failed poses."]}}


def respond(agent, action="retain"):
    return agent.dispatch("respond_to_reasoning_feedback", response_args(agent, action))


def msa(tmp_path):
    path = tmp_path / "query.csv"
    path.write_text("key,sequence\n0,AG\n")
    return {"accession": "SYNTHETIC1", "msa_index": {"SYNTHETIC1": str(path)}}


def test_agent_revisits_failed_setup_then_completes_independently(tmp_path):
    record = modeling_record(tmp_path)
    wrong = copy.deepcopy(record)
    wrong["enzyme_state"]["chains"] *= 2
    client = Client([
        [("propose_model", propose(wrong))],
        [("render_inputs", msa(tmp_path))],
        [("finish_task", {"outcome": "complete", "rationale": "Attempt completion before repairing setup."})],
        # Use an adaptive response below: the fingerprint is generated by checks.
        [("propose_model", propose(record))],
        [("render_inputs", msa(tmp_path))],
        [("finish_task", {"outcome": "complete", "rationale": "Current construct and rendered inputs agree with the evidence."})],
    ])
    agent = make_agent(tmp_path, client)
    original_call = client.call
    def decision(**kwargs):
        if agent.state["turn"] == 4:
            respond(agent, "revise")
        return original_call(**kwargs)
    client.call = decision
    report = agent.run(max_turns=8)
    assert report["status"] == "complete", report
    revisions = agent.state["candidates"]["SYNTHETIC1"]["revisions"]
    assert len(revisions) == 2
    assert revisions[0]["setup_validation"]["status"] == "fail"
    assert not revisions[0].get("execution")
    assert revisions[1]["setup_validation"]["status"] == "pass"
    assert any("setup is not validated" in str(context) for context in client.contexts)
    assert (agent.run_dir / "transcript.json").exists()
    assert json.loads((agent.run_dir / "report.json").read_text())["completion_gaps"] == []


def test_plain_completion_and_turn_allowance_leave_resumable_work(tmp_path):
    agent = make_agent(tmp_path, Client([[]]))
    report = agent.run(max_turns=1)
    assert report["status"] == "budget_exhausted"
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True,
                           llm_client=Client([[("finish_task", {"outcome": "complete", "rationale": "unsupported"})]]))
    report = resumed.run(max_turns=1)
    assert report["status"] == "budget_exhausted"
    assert resumed.state["turn"] == 2
    assert len(resumed.messages) > len(agent.messages)


def test_changed_source_cannot_be_completed_under_an_old_pass(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    assert agent.completion_errors() == []
    (tmp_path / "reference.txt").write_text("changed evidence")
    assert agent.dispatch("finish_task", {"outcome": "complete", "rationale": "old pass"})["status"] == "blocked"


def test_new_revision_cannot_inherit_previous_ensemble(tmp_path):
    agent = make_agent(tmp_path, objective="model_and_validate")
    record = modeling_record(tmp_path)
    agent.dispatch("propose_model", propose(record))
    agent.dispatch("render_inputs", msa(tmp_path))
    folder = tmp_path / "predictions"
    folder.mkdir()
    path = write_pose(folder / "model.pdb")
    revision = agent._current("SYNTHETIC1")
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps({"input_sha256": revision["input_sha256"],
                                    "execution_sha256": fingerprint(revision["execution"]),
                                    "predictions": [{"path": str(path), "sha256": file_digest(path)}]}))
    result = agent.dispatch("validate_predictions", {"accession": "SYNTHETIC1", "directory": str(folder),
                                                      "prediction_manifest": str(producer)})
    assert result["status"] == "pass"
    respond(agent)
    assert agent.completion_errors() == []
    agent.dispatch("propose_model", propose(record))
    assert agent.completion_errors()
    assert not agent._current("SYNTHETIC1").get("pose_validation")


def test_submission_requires_config_and_never_blindly_retries(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    with pytest.raises(ValueError, match="no provider"):
        agent.dispatch("submit_cofold", {"accession": "SYNTHETIC1"})
    agent.backend = {"provider": "synthetic", "argv": ["synthetic-adapter"],
                     "resource_policy": {"reservation_managed": True, "max_jobs": 1, "max_predictions_per_job": 2}}
    calls = []
    def adapter(*args, **kwargs):
        request = json.loads(kwargs["input"])
        calls.append(request)
        return SimpleNamespace(stdout=json.dumps({"idempotency_key": request["idempotency_key"], "status": "running"}))
    with patch("structural_modeling_agent.agent.subprocess.run", side_effect=adapter):
        agent.dispatch("submit_cofold", {"accession": "SYNTHETIC1"})
        result = agent.dispatch("submit_cofold", {"accession": "SYNTHETIC1"})
        assert result["status"] == "already_requested"
        agent.dispatch("check_cofold", {"accession": "SYNTHETIC1"})
    assert [r["operation"] for r in calls] == ["submit", "status"]
    with pytest.raises(ValueError, match="reconcile"):
        agent.dispatch("propose_model", propose(modeling_record(tmp_path)))


def test_single_controller_lock_and_no_cross_task_resume(tmp_path):
    agent = make_agent(tmp_path)
    with _controller_lock(agent.run_dir), pytest.raises(BlockingIOError):
        with _controller_lock(agent.run_dir):
            pass
    with pytest.raises(ValueError, match="replace its target"):
        ModelingAgent(task_path=tmp_path / "task.json", run_dir=agent.run_dir, resume=True)


def test_evidence_access_requires_explicit_task_path_and_rejects_symlink_escape(tmp_path):
    agent = make_agent(tmp_path)
    allowed = tmp_path / "reference.txt"
    allowed.write_text("permitted evidence")
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    assert agent.dispatch("read_evidence", {"path": str(allowed)})["text"] == "permitted evidence"
    with pytest.raises(PermissionError):
        agent.dispatch("read_evidence", {"path": str(secret)})
    with pytest.raises(PermissionError):
        agent.dispatch("inspect_experimental_proxy_bundle", {"bundle_path": str(secret)})
    allowed.unlink()
    allowed.symlink_to(secret)
    with pytest.raises(PermissionError):
        agent.dispatch("read_evidence", {"path": str(allowed)})


def test_checkpoint_remains_coherent_when_compatibility_write_is_interrupted(tmp_path):
    agent = make_agent(tmp_path)
    agent.state["turn"] = 42
    agent.messages.append({"role": "user", "content": "checkpoint marker"})
    from enzyme_discovery_agent.state_store import atomic_write_json

    def interrupted(path, value, **kwargs):
        if path.name == "transcript.json":
            raise OSError("simulated interruption")
        return atomic_write_json(path, value, **kwargs)

    with patch("structural_modeling_agent.agent.atomic_write_json", side_effect=interrupted):
        with pytest.raises(OSError, match="interruption"):
            agent._save()
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True)
    assert resumed.state["turn"] == 42
    assert resumed.messages[-1]["content"] == "checkpoint marker"


def test_model_cannot_introduce_an_undeclared_source_file(tmp_path):
    agent = make_agent(tmp_path)
    record = modeling_record(tmp_path)
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    record["validation_plan"]["setup"]["assembly"]["evidence"][0]["path"] = str(secret)
    with pytest.raises(PermissionError):
        agent.dispatch("propose_model", propose(record))


def test_cli_help_does_not_require_a_live_provider():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "structural_modeling_agent", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--task" in result.stdout and "--resume" in result.stdout


def test_environment_diagnostics_needs_no_provider():
    from structural_modeling_agent.__main__ import environment_diagnostics
    result = environment_diagnostics()
    assert result["available"]["numpy"] is True
    assert set(result["missing"]) == {name for name, found in result["available"].items() if not found}
    assert "credentials" in result["note"]


def test_validation_report_returns_to_discovery_without_changing_rank(tmp_path):
    from enzyme_discovery_agent.import_structural_evidence import import_structural_evidence
    from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry, discovery_status
    agent = make_agent(tmp_path, objective="model_and_validate")
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    folder = tmp_path / "predictions"
    folder.mkdir()
    path = write_pose(folder / "synthetic_model_0.pdb")
    revision = agent._current("SYNTHETIC1")
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps({"input_sha256": revision["input_sha256"],
        "execution_sha256": fingerprint(revision["execution"]),
        "predictions": [{"path": str(path), "sha256": file_digest(path)}]}))
    agent.dispatch("validate_predictions", {"accession": "SYNTHETIC1", "directory": str(folder), "prediction_manifest": str(producer)})
    respond(agent)
    agent.llm = Client([[("finish_task", {"outcome": "complete", "rationale": "All synthetic input and pose checks passed."})]])
    assert agent.run(max_turns=1)["status"] == "complete"
    exported = agent.run_dir / "structural_modeling_evidence.json"
    assert json.loads(exported.read_text())["candidates"][0]["status"] == "validated"
    discovery = tmp_path / "discovery"
    update_candidate_registry(discovery, "SYNTHETIC1", "ranked", metadata={"rank": 7})
    review = import_structural_evidence(discovery, exported)
    entry = load_candidate_registry(discovery)["candidates"]["SYNTHETIC1"]
    assert discovery_status(entry) == "ranked"
    assert entry["metadata"]["rank"] == 7
    evidence = entry["metadata"]["modeling_validation_evidence"]
    assert evidence["geometry_metrics"]["site.site1"]["weighted_attack_angle_deg"] == 180
    assert evidence["geometry_metrics"]["site.site1"]["harmonic_distance_A"] == 3
    assert review.name == "structural_modeling_review.json"
    # Reimport does not inherit an obsolete pass after a source changes.
    (tmp_path / "reference.txt").write_text("changed source")
    import_structural_evidence(discovery, exported)
    entry = load_candidate_registry(discovery)["candidates"]["SYNTHETIC1"]
    assert entry["metadata"]["modeling_review_status"] == "incomplete_modeling_evidence"


def test_proxy_tools_are_available_and_plans_are_immutable(tmp_path):
    from structural_modeling_agent.agent import tool_schemas
    names = {t["function"]["name"] for t in tool_schemas()}
    assert {"get_structural_proxy_guidance", "validate_structural_proxy_plan", "declare_proxy_plan", "evaluate_proxy_plan"} <= names
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    plan = {"schema_version": "1.0", "mechanism": "synthetic_transfer", "reaction_step": "synthetic transfer",
        "assay_endpoint": "synthetic measurement", "context": {}, "hypotheses": [{"proxy_id": "complex_plddt",
        "rationale": "Prediction confidence baseline", "expected_direction": None, "evidence_ids": ["synthetic:reference"]}]}
    assert agent.dispatch("validate_structural_proxy_plan", {"plan": plan})["status"] == "success"
    assert agent.dispatch("declare_proxy_plan", {"accession": "SYNTHETIC1", "plan": plan})["status"] == "declared"
    with pytest.raises(ValueError, match="inspect"):
        agent.dispatch("evaluate_proxy_plan", {"accession": "SYNTHETIC1"})
    with pytest.raises(ValueError, match="immutable"):
        agent.dispatch("declare_proxy_plan", {"accession": "SYNTHETIC1", "plan": {**plan, "assay_endpoint": "different"}})


def test_proxy_evaluation_preserves_reviewed_inputs_and_checks_the_file_set(tmp_path):
    from structural_modeling_agent.modeling_reasoning.reaction import load_atom_selectors
    agent = make_agent(tmp_path)
    record = modeling_record(tmp_path)
    selector = next(iter(load_atom_selectors()["ligand_smarts"]))
    record["geometry"]["nucleophile"] = {"kind": "ligand", "component_index": 1, "atom_selector": selector}
    agent.dispatch("propose_model", propose(record))
    agent.dispatch("render_inputs", msa(tmp_path))
    folder = tmp_path / "predictions"
    folder.mkdir()
    write_pose(folder / "synthetic_model_0.pdb")
    (folder / "confidence_synthetic_model_0.json").write_text('{"complex_plddt": 0.8}')
    agent.dispatch("validate_predictions", {"accession": "SYNTHETIC1", "directory": str(folder)})
    plan = {"schema_version": "1.0", "mechanism": "synthetic_transfer", "reaction_step": "synthetic transfer",
        "assay_endpoint": "synthetic endpoint", "context": {}, "hypotheses": [{"proxy_id": "complex_plddt",
        "rationale": "Prediction confidence baseline", "expected_direction": None, "evidence_ids": ["synthetic:reference"]}]}
    agent.dispatch("declare_proxy_plan", {"accession": "SYNTHETIC1", "plan": plan})
    before = fingerprint(agent._current("SYNTHETIC1")["modeling_input"])
    result = agent.dispatch("evaluate_proxy_plan", {"accession": "SYNTHETIC1"})
    assert result["measurements"]["complex_plddt"]["value"] == .8
    assert fingerprint(agent._current("SYNTHETIC1")["modeling_input"]) == before
    respond(agent)
    assert agent.completion_errors() == []
    (folder / "confidence_synthetic_model_0.json").write_text('{"complex_plddt": 0.9}')
    assert any("proxy evaluation integrity" in error for error in agent.completion_errors())
    write_pose(folder / "extra_model_1.pdb")
    with pytest.raises(ValueError, match="exact inspected"):
        agent.dispatch("evaluate_proxy_plan", {"accession": "SYNTHETIC1"})


def test_resume_repairs_unanswered_tools_without_resubmitting(tmp_path):
    agent = make_agent(tmp_path)
    agent.messages.append(Message([("inspect_validation_contract", {})]).model_dump())
    agent._save()
    client = Client([[("finish_task", {"outcome": "inconclusive", "rationale": "Required source evidence is unavailable."})]])
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True, llm_client=client)
    assert resumed.run(max_turns=1)["status"] == "inconclusive"
    assert any(m.get("role") == "tool" and "interrupted" in m["content"] for m in client.contexts[0])


def test_resume_reconciles_pending_jobs_before_model_decision(tmp_path):
    agent = make_agent(tmp_path)
    agent.backend = {"provider": "test"}
    agent.state["backend"] = agent.backend
    agent.state["candidates"]["SYNTHETIC1"]["revisions"].append(
        {"revision": 1, "modeling_input": {}, "job": {"idempotency_key": "test-key", "status": "submission_pending"}})
    agent._save()
    client = Client([[("finish_task", {"outcome": "inconclusive", "rationale": "Review remains incomplete."})]])
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True, llm_client=client)
    calls = []

    def status(operation, revision):
        calls.append(operation)
        revision["job"] = {"idempotency_key": "test-key", "status": "failed"}
        return revision["job"]

    with patch.object(resumed, "_backend_call", side_effect=status):
        assert resumed.run(max_turns=1)["status"] == "inconclusive"
    assert calls == ["status"]
    assert resumed.state["candidates"]["SYNTHETIC1"]["revisions"][0]["job"]["status"] == "failed"


def test_failed_resume_reconciliation_keeps_job_pending(tmp_path):
    agent = make_agent(tmp_path)
    agent.backend = {"provider": "test"}
    agent.state["backend"] = agent.backend
    agent.state["candidates"]["SYNTHETIC1"]["revisions"].append(
        {"revision": 1, "modeling_input": {}, "job": {"idempotency_key": "test-key", "status": "running"}})
    agent._save()
    client = Client([[("finish_task", {"outcome": "inconclusive", "rationale": "Status remains unknown."})]])
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True, llm_client=client)
    with patch.object(resumed, "_backend_call", side_effect=TimeoutError("status unavailable")):
        assert resumed.run(max_turns=1)["status"] == "waiting_for_backend"
    assert resumed.state["candidates"]["SYNTHETIC1"]["revisions"][0]["job"]["status"] == "running"


def test_job_status_is_reconcilable_when_evidence_changes(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    agent.backend = {"provider": "synthetic", "argv": ["synthetic-adapter"],
                     "resource_policy": {"reservation_managed": True, "max_jobs": 1, "max_predictions_per_job": 2}}
    def adapter(*args, **kwargs):
        request = json.loads(kwargs["input"])
        return SimpleNamespace(stdout=json.dumps({"idempotency_key": request["idempotency_key"], "status": "running"}))
    with patch("structural_modeling_agent.agent.subprocess.run", side_effect=adapter):
        agent.dispatch("submit_cofold", {"accession": "SYNTHETIC1"})
        (tmp_path / "reference.txt").write_text("changed source")
        assert agent.dispatch("check_cofold", {"accession": "SYNTHETIC1"})["status"] == "running"
        with pytest.raises(ValueError, match="evidence"):
            agent.dispatch("submit_cofold", {"accession": "SYNTHETIC1"})


def test_unbound_predictions_do_not_complete_the_task(tmp_path):
    agent = make_agent(tmp_path, objective="model_and_validate")
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    directory = tmp_path / "predictions"
    directory.mkdir()
    write_pose(directory / "model.pdb")
    report = agent.dispatch("validate_predictions", {"accession": "SYNTHETIC1", "directory": str(directory)})
    assert report["status"] == "pass"
    assert report["prediction_provenance"]["status"] == "abstain"
    assert any("provenance" in error for error in agent.completion_errors())


def test_the_last_setup_turn_forces_a_provisional_proposal_instead_of_a_stall(tmp_path):
    """Left free, the setup stage reads sources until the limit and saves nothing at all."""
    (tmp_path / "reference.txt").write_text("Synthetic source evidence for the candidate.")
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"schema_version": "1.0", "objective": "model_and_validate",
        "workflow_mode": "production_cofold", "setup_turn_limit": 2,
        "target": {"reaction": "synthetic target transformation"},
        "evidence_paths": [str(tmp_path / "reference.txt")],
        "candidates": [{"accession": "SYNTHETIC1", "sequence": "AG", "evidence": {}}]}))
    seen = []

    class Recorder(Client):
        def call(self, **kwargs):
            seen.append({tool["function"]["name"] for tool in kwargs["tools"]})
            return super().call(**kwargs)

    client = Recorder([
        [("read_evidence", {"path": str(tmp_path / "reference.txt")})],
        [("finish_task", {"outcome": "inconclusive",
                          "rationale": "No defensible proposal from the evidence held."})],
    ])
    agent = ModelingAgent(task_path=task, run_dir=tmp_path / "agent", llm_client=client)
    agent.run(max_turns=3)
    assert "read_evidence" in seen[0], "the stage must be free to gather evidence first"
    # On the closing turn the only ways out are a revisable proposal or an explicit
    # inconclusive outcome -- never another read that ends in a bare stall.
    assert seen[1] == {"propose_model", "finish_task"}
    assert any("last setup turn" in str(m.get("content") or "") for m in agent.messages)
    assert agent.state["status"] != "setup_stalled"


def test_sole_candidate_accession_defaults_and_unknown_is_clear(tmp_path):
    agent = make_agent(tmp_path)
    # A single-candidate run resolves the accession when the tool call omits it.
    assert agent.dispatch("inspect_candidate", {})["candidate"]["accession"] == "SYNTHETIC1"
    # An explicit wrong accession names the valid choices instead of a bare KeyError.
    try:
        agent.dispatch("inspect_candidate", {"accession": "NOPE"})
        assert False, "expected a clear error for an unknown accession"
    except ValueError as exc:
        assert "SYNTHETIC1" in str(exc)
