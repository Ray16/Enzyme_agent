"""Decision-level tests for the reasoning requirements in the supplied essay.

Synthetic evidence checks software behavior; none of these fixtures is an enzyme
activity result or evidence that an LLM autonomously discovers a mechanism.
"""
import copy
import json

import pytest

from structural_modeling_agent.agent import ModelingAgent, tool_schemas
from structural_modeling_agent.scientific_reasoning import validate_reasoning
from structural_modeling_agent.validation import file_digest
from test_modeling_agent import (Message, make_agent, modeling_record, msa,
                                 propose, respond, response_args, write_pose)


ACC = {"accession": "SYNTHETIC1"}


def assessment(agent):
    return agent.dispatch("inspect_reasoning", ACC)["assessment"]


def experimental_proposal(tmp_path):
    args = propose(modeling_record(tmp_path))
    for i, hypothesis in enumerate(args["scientific_reasoning"]["hypotheses"]):
        hypothesis["predictions"] = [{"id": f"activity_{i}", "metric": "L2_concentration", "scope": "experimental",
            "unit": "mM", "endpoint": "L2_accumulation", "conditions": {"minutes": 60, "normalization": "active_enzyme"},
            "expected_interval": [1, 10] if i == 0 else [0, .5],
            "decision_if_refuted": "Compare model error, proxy error and assay/active-enzyme confounding."}]
    return args


def observation(agent, value=.1, identifier="assay_1", **changes):
    # Sources created under this run are authorized without broadening task access.
    source = agent.run_dir / "sources" / f"{identifier}.txt"
    source.parent.mkdir(exist_ok=True)
    source.write_text(f"Synthetic controlled measurement: L2={value} mM at 60 minutes, active enzyme normalized.")
    result = {"id": identifier, "metric": "L2_concentration", "scope": "experimental", "unit": "mM",
        "endpoint": "L2_accumulation", "conditions": {"minutes": 60, "normalization": "active_enzyme"},
        "value": value, "interpretation": "Manually interpreted synthetic measurement; endpoint is not an elementary rate.",
        "source": {"kind": "solution_measurement", "source_id": f"synthetic:{identifier}", "path": str(source),
            "sha256": file_digest(source), "quote": source.read_text(), "reviewed": True,
            "rationale": "Synthetic source for exercising a prospective prediction comparison."}}
    result.update(changes)
    return result


def record_observation(agent, **kwargs):
    return agent.dispatch("record_reasoning_observation", {**ACC, "observation": observation(agent, **kwargs)})


def test_prose_only_proposals_and_inconsistent_mechanisms_cannot_execute(tmp_path):
    agent = make_agent(tmp_path)
    args = propose(modeling_record(tmp_path))
    missing = copy.deepcopy(args)
    del missing["scientific_reasoning"]
    with pytest.raises(ValueError, match="scientific_reasoning"):
        agent.dispatch("propose_model", missing)
    wrong = copy.deepcopy(args)
    wrong["scientific_reasoning"]["hypotheses"][0]["mechanism"] = "another mechanism"
    with pytest.raises(ValueError, match="must match modeled_step"):
        agent.dispatch("propose_model", wrong)
    assert agent.state["candidates"]["SYNTHETIC1"]["revisions"] == []


def test_focused_inspection_pages_keep_offsets_and_exact_content_in_model_context(tmp_path):
    agent = make_agent(tmp_path)
    schema = agent.dispatch("inspect_validation_contract", {"section": "scientific_reasoning"})
    assert "hypotheses" in schema and "execution_rules" not in schema
    assert len(json.dumps(schema)) < 4800
    source = tmp_path / "reference.txt"
    original = 'Quoted Unicode text α with escapes "\\\n' * 600
    source.write_text(original)
    offset, reconstructed = 0, ""
    while True:
        page = agent.dispatch("read_evidence", {"path": str(source), "offset": offset})
        assert len(json.dumps(page)) <= 4800
        reconstructed += page["text"]
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert reconstructed == original


def test_scientific_schema_and_read_source_passages_survive_history_compaction_and_resume(tmp_path):
    agent = make_agent(tmp_path)
    source = tmp_path / "reference.txt"
    source.write_text("A short, source-grounded observation already inspected by this run.")
    agent.dispatch("read_evidence", {"path": str(source)})
    for index in range(8):
        agent.messages.extend([Message([("inspect_validation_contract", {})]).model_dump(),
            {"role": "tool", "tool_call_id": "call_0", "content": json.dumps({"summary": "x" * 12000})}])
    context = agent._context()
    assert "expected_interval" in context[0]["content"]
    assert "A short, source-grounded observation" in context[-1]["content"]
    assert len(json.dumps(context)) <= 48000
    agent._save()
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True)
    assert "A short, source-grounded observation" in resumed._context()[-1]["content"]
    source.write_text("Changed after inspection")
    assert '"integrity": "changed"' in resumed._context()[-1]["content"]


def test_model_reference_proposals_apply_explicit_changes_without_bypassing_validation(tmp_path):
    from structural_modeling_agent.validation import fingerprint
    initial = propose(modeling_record(tmp_path))
    base = initial["modeling_input"]
    supplied_msa = msa(tmp_path)["msa_index"]["SYNTHETIC1"]
    task = {"schema_version": "1.0", "objective": "prepare_inputs", "target": {},
            "evidence_paths": [str(tmp_path / "reference.txt")],
            "candidates": [{"accession": "SYNTHETIC1", "sequence": "AG", "modeling_input": base,
                            "evidence": {"msa_path": supplied_msa}}]}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    agent = ModelingAgent(task_path=path, run_dir=tmp_path / "agent")
    args = {k: v for k, v in initial.items() if k != "modeling_input"}
    args.update(modeling_input_from="candidate", modeling_input_changes={"geometry": {"ideal_angle": 180}})
    first = agent.dispatch("propose_model", args)
    assert first["input_derivation"]["base_sha256"] == fingerprint(base)
    assert first["modeling_input"]["geometry"]["ideal_angle"] == 180
    assert "ideal_angle" not in agent.task["candidates"][0]["modeling_input"]["geometry"]
    with pytest.raises(ValueError, match="accession key 'SYNTHETIC1'"):
        agent.dispatch("render_inputs", {**ACC, "msa_index": {"A": supplied_msa}})
    assert agent.dispatch("render_inputs", ACC)["status"] == "rendered"
    focused = agent.dispatch("read_candidate_section", {**ACC, "section": "modeling_input", "pointer": "/modeled_step/mechanism"})
    assert json.loads(focused["text"]) == "synthetic_transfer"
    # The same base-copy API cannot carry old validation into a changed model.
    args.update(modeling_input_from="current", modeling_input_changes={"enzyme_state": {"chains": []}})
    second = agent.dispatch("propose_model", args)
    assert second["setup_validation"]["status"] != "pass"
    assert not second.get("execution")
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"
    with pytest.raises(ValueError, match="not both"):
        agent.dispatch("propose_model", {**args, "modeling_input": base})


@pytest.mark.parametrize("center,phosphate,bridge", [("atp_alpha", "PA", "O3A"), ("atp_gamma", "PG", "O3B")])
def test_atp_center_changes_the_allowed_atoms_and_proxy_plan(tmp_path, center, phosphate, bridge):
    agent = make_agent(tmp_path)
    args = propose(modeling_record(tmp_path))
    model, reason = args["modeling_input"], args["scientific_reasoning"]
    model["modeled_step"].update(mechanism=center, reaction_center=center)
    model["cofactors"] = [{"ccd": "ATP", "count": 1, "role": "activation"}]
    chosen = reason["hypotheses"][0]
    chosen.update(mechanism=center, reaction_center=center)
    site = model["validation_plan"]["poses"]["sites"][0]
    site["electrophile"].update(chain="B", atom=phosphate)
    site["angle_reference"].update(chain="B", atom=bridge)
    validate_reasoning(reason, model)
    swapped = copy.deepcopy(model)
    swapped["validation_plan"]["poses"]["sites"][0]["electrophile"]["atom"] = "PG" if phosphate == "PA" else "PA"
    with pytest.raises(ValueError, match="requires"):
        validate_reasoning(reason, swapped)
    extra_site = copy.deepcopy(site)
    extra_site["angle_reference"]["atom"] = "WRONG"
    model["validation_plan"]["poses"]["sites"].append(extra_site)
    with pytest.raises(ValueError, match="every site"):
        validate_reasoning(reason, model)
    model["validation_plan"]["poses"]["sites"].pop()
    agent.dispatch("propose_model", args)  # Invalid setup is inspectable but cannot render.
    plan = {"schema_version": "1.0", "mechanism": center, "reaction_step": "synthetic transfer",
        "assay_endpoint": "test", "context": {"reaction_center": center}, "hypotheses": [{
            "proxy_id": "complex_plddt", "rationale": "Confidence control", "expected_direction": None,
            "evidence_ids": ["synthetic:reference"]}]}
    assert agent.dispatch("declare_proxy_plan", {**ACC, "plan": plan})["status"] == "declared"
    plan["mechanism"] = "unrelated_step"
    with pytest.raises(ValueError, match="proxy mechanism"):
        agent.dispatch("declare_proxy_plan", {**ACC, "plan": plan})
    plan["mechanism"] = center
    plan["context"]["reaction_center"] = "atp_gamma" if center == "atp_alpha" else "atp_alpha"
    with pytest.raises(ValueError, match="context.reaction_center"):
        agent.dispatch("declare_proxy_plan", {**ACC, "plan": plan})


def test_selected_hypothesis_without_evidence_remains_an_investigation(tmp_path):
    agent = make_agent(tmp_path)
    args = propose(modeling_record(tmp_path))
    args["scientific_reasoning"]["hypotheses"][0]["evidence"] = []
    agent.dispatch("propose_model", args)
    rendered = agent.dispatch("render_inputs", msa(tmp_path))
    assert rendered["status"] == "blocked"
    assert "supporting source evidence" in str(rendered)
    assert not agent._current("SYNTHETIC1").get("execution")


def test_hypothesis_comparison_prioritizes_discriminating_measurements_and_controls(tmp_path):
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    for index, hypothesis in enumerate(args["scientific_reasoning"]["hypotheses"]):
        uninformative = copy.deepcopy(hypothesis["predictions"][0])
        uninformative.update(id=f"binding_{index}", metric="binding_fraction", unit="fraction", expected_interval=[0, 1])
        hypothesis["predictions"].append(uninformative)
    agent.dispatch("propose_model", args)
    comparison = agent.dispatch("compare_hypothesis_tests", ACC)
    chosen = comparison["tests"][0]
    assert chosen["readout"]["metric"] == "L2_concentration"
    assert chosen["unresolved_pairs"] == [["H1", "H2"]]
    assert chosen["controls"]
    assert comparison["tests"][1]["distinguishable_pairs"] == []
    assert assessment(agent)["test_priorities"][0]["readout"]["metric"] == "L2_concentration"
    record_observation(agent)
    comparison = agent.dispatch("compare_hypothesis_tests", ACC)
    assert comparison["contradicted_hypotheses"] == ["H1"]
    assert comparison["tests"][0]["unresolved_pairs"] == []
    assert assessment(agent)["test_priorities"] == []


@pytest.mark.parametrize("changes", [
    {"expected_interval": [.5, 2]},  # Touching prediction intervals do not separate the hypotheses.
    {"unit": "uM"}, {"endpoint": "initial_rate"}, {"conditions": {"minutes": 1}}])
def test_overlapping_or_incommensurate_predictions_require_a_better_test(tmp_path, changes):
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    args["scientific_reasoning"]["hypotheses"][0]["predictions"][0].update(changes)
    agent.dispatch("propose_model", args)
    result = agent.dispatch("compare_hypothesis_tests", ACC)
    assert result["pairs_needing_new_predictions"] == [["H1", "H2"]]
    assert not any(test["distinguishable_pairs"] for test in result["tests"])


@pytest.mark.parametrize("bounds,status", [([.05, .2], "contradicted"), ([.05, 2], "unresolved"), ([1, 2], "consistent")])
def test_measurement_uncertainty_changes_the_strength_of_a_contradiction(tmp_path, bounds, status):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", experimental_proposal(tmp_path))
    value = sum(bounds) / 2
    record_observation(agent, value=value, measurement_interval=bounds)
    assert assessment(agent)["predictions"][0]["status"] == status


def test_structural_evidence_does_not_silently_transfer_to_a_different_model(tmp_path):
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    for hypothesis in args["scientific_reasoning"]["hypotheses"]:
        hypothesis["predictions"][0]["scope"] = "structural"
    agent.dispatch("propose_model", args)
    record_observation(agent, scope="structural")
    assert assessment(agent)["predictions"][0]["status"] == "contradicted"
    respond(agent, "revise")
    revised = copy.deepcopy(args)
    revised["modeling_input"]["validation_plan"]["poses"]["sites"][0]["distance_range_A"] = [2, 5]
    agent.dispatch("propose_model", revised)
    result = assessment(agent)
    assert result["inapplicable_observations"] == {"assay_1": "bound to another model or missing its original model context"}
    assert {p["status"] for p in result["predictions"]} == {"unresolved"}
    assert result["observations"][0]["value"] == .1  # Preserved in the record, not erased.


def test_independent_structural_evidence_can_transfer_with_an_explicit_rationale(tmp_path):
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    for hypothesis in args["scientific_reasoning"]["hypotheses"]:
        hypothesis["predictions"][0]["scope"] = "structural"
    agent.dispatch("propose_model", args)
    with pytest.raises(ValueError, match="transfer_rationale"):
        record_observation(agent, scope="structural", applicability="candidate")
    record_observation(agent, scope="structural", applicability="candidate",
                       transfer_rationale="Independent structural measurement of the same enzyme and molecular conditions.")
    respond(agent, "revise")
    revised = copy.deepcopy(args)
    revised["modeling_input"]["validation_plan"]["poses"]["sites"][0]["distance_range_A"] = [2, 5]
    agent.dispatch("propose_model", revised)
    assert assessment(agent)["predictions"][0]["status"] == "contradicted"
    assert assessment(agent)["inapplicable_observations"] == {}


def test_registered_record_and_execution_binding_prevent_rewriting_predictions(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    current = agent._current("SYNTHETIC1")
    assert current["execution"]["scientific_reasoning_sha256"] == current["reasoning_registration"]["reasoning_sha256"]
    assert agent.completion_errors() == []
    current["scientific_reasoning"]["hypotheses"][0]["predictions"][0]["expected_interval"] = [0, 1]
    errors = agent.completion_errors()
    assert any("registered scientific reasoning" in error for error in errors)
    assert any("execution is not bound" in error for error in errors)


def test_legacy_pose_feedback_can_be_acknowledged_but_needs_a_new_scientific_record(tmp_path):
    agent = make_agent(tmp_path)
    args = propose(modeling_record(tmp_path))
    agent.dispatch("propose_model", args)
    agent.dispatch("render_inputs", msa(tmp_path))
    directory = tmp_path / "legacy_predictions"
    directory.mkdir()
    write_pose(directory / "model.pdb")
    agent.dispatch("validate_predictions", {**ACC, "directory": str(directory)})
    current = agent._current("SYNTHETIC1")
    del current["scientific_reasoning"]
    del current["reasoning_registration"]
    agent._save()
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True)
    result = assessment(resumed)
    assert result["response_hypothesis_ids"] == ["legacy_unrecorded"]
    response = response_args(resumed, "revise")
    response["addressed_hypotheses"] = result["response_hypothesis_ids"]
    resumed.dispatch("respond_to_reasoning_feedback", response)
    assert resumed.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"
    resumed.dispatch("propose_model", args)
    assert resumed.dispatch("render_inputs", msa(tmp_path))["status"] == "rendered"
    assert len(resumed.state["candidates"]["SYNTHETIC1"]["revisions"]) == 2


def test_new_observation_changes_actions_and_cannot_be_overwritten_or_prose_completed(tmp_path):
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    agent.dispatch("propose_model", args)
    agent.dispatch("render_inputs", msa(tmp_path))
    assert agent.completion_errors() == []
    result = record_observation(agent)
    assert [p["status"] for p in result["predictions"]] == ["contradicted", "consistent"]
    assert result["next_actions"]
    assert not result["response_current"]
    assert agent.dispatch("finish_task", {"outcome": "complete", "rationale": "Ignore discrepancy"})["status"] == "blocked"
    with pytest.raises(ValueError, match="before revising"):
        agent.dispatch("propose_model", args)
    with pytest.raises(ValueError, match="immutable"):
        record_observation(agent)
    respond(agent, "investigate")
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"
    assert agent.completion_errors()
    with pytest.raises(ValueError, match="reasoning evidence"):
        agent.dispatch("submit_cofold", ACC)
    respond(agent, "retain")
    assert agent.completion_errors() == []
    assert assessment(agent)["predictions"][0]["status"] == "contradicted"
    # A second, favorable measurement does not erase the first contradiction.
    record_observation(agent, value=2, identifier="assay_2")
    assert not assessment(agent)["response_current"]
    assert assessment(agent)["predictions"][0]["status"] == "contradicted"
    assert agent.completion_errors()


@pytest.mark.parametrize("changes", [{"unit": "uM"}, {"endpoint": "initial_rate"},
    {"scope": "structural"}, {"conditions": {"minutes": 10, "normalization": "total_protein"}}])
def test_assay_endpoint_units_scope_and_confounders_must_match(tmp_path, changes):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", experimental_proposal(tmp_path))
    agent.dispatch("record_reasoning_observation", {**ACC, "observation": observation(agent, **changes)})
    result = assessment(agent)
    assert {p["status"] for p in result["predictions"]} == {"unresolved"}
    assert result["unmatched_observation_ids"] == ["assay_1"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_nonfinite_measurements_are_not_zero(tmp_path, value):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", experimental_proposal(tmp_path))
    with pytest.raises(ValueError, match="finite"):
        record_observation(agent, value=value)
    assert assessment(agent)["observations"] == []


def test_source_and_feedback_changes_invalidate_response_including_unmatched_evidence(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", experimental_proposal(tmp_path))
    record_observation(agent, endpoint="another_endpoint")
    old_args = response_args(agent)
    respond(agent)
    source = agent.run_dir / "sources" / "assay_1.txt"
    source.write_text("Changed source after the interpretation")
    result = assessment(agent)
    assert result["observation_errors"]
    assert not result["response_current"]
    with pytest.raises(ValueError, match="assessment changed"):
        agent.dispatch("respond_to_reasoning_feedback", old_args)


def test_contradictory_sources_are_authorized_and_rechecked_too(tmp_path):
    agent = make_agent(tmp_path)
    args = propose(modeling_record(tmp_path))
    negative = copy.deepcopy(args["scientific_reasoning"]["hypotheses"][0]["evidence"][0])
    negative["effect"] = "contradicts"
    private = tmp_path / "undeclared.txt"
    private.write_text("This is not authorized evidence.")
    negative.update(path=str(private), sha256=file_digest(private), quote=private.read_text())
    args["scientific_reasoning"]["hypotheses"][1]["evidence"] = [negative]
    with pytest.raises(PermissionError):
        agent.dispatch("propose_model", args)
    negative.update(path=str(tmp_path / "reference.txt"), sha256=file_digest(tmp_path / "reference.txt"), quote="ABSENT")
    agent.dispatch("propose_model", args)
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"


def test_failed_pose_is_a_contradiction_but_missing_atoms_are_unresolved(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", propose(modeling_record(tmp_path)))
    agent.dispatch("render_inputs", msa(tmp_path))
    directory = tmp_path / "poses"
    directory.mkdir()
    path = write_pose(directory / "model.pdb", displacement=30)
    agent.dispatch("validate_predictions", {**ACC, "directory": str(directory)})
    assert assessment(agent)["predictions"][0]["status"] == "contradicted"
    assert assessment(agent)["observations"][0]["value"] == 0
    respond(agent, "revise")
    args = propose(modeling_record(tmp_path))
    args["modeling_input"]["validation_plan"]["poses"]["sites"][0]["nucleophile"]["atom"] = "MISSING"
    agent.dispatch("propose_model", args)
    agent.dispatch("render_inputs", msa(tmp_path))
    agent.dispatch("validate_predictions", {**ACC, "directory": str(directory)})
    assert assessment(agent)["observations"][0]["value"] is None
    assert {p["status"] for p in assessment(agent)["predictions"]} == {"unresolved"}
    respond(agent)
    path.write_text("changed prediction")
    assert not assessment(agent)["response_current"]


def test_proxy_values_feed_predictions_and_changed_sidecars_invalidate_feedback(tmp_path):
    from structural_modeling_agent.modeling_reasoning.reaction import load_atom_selectors
    agent = make_agent(tmp_path)
    args = propose(modeling_record(tmp_path))
    args["modeling_input"]["geometry"]["nucleophile"] = {"kind": "ligand", "component_index": 1,
        "atom_selector": next(iter(load_atom_selectors()["ligand_smarts"]))}
    for i, hypothesis in enumerate(args["scientific_reasoning"]["hypotheses"]):
        hypothesis["predictions"][0].update(metric="proxy:complex_plddt", unit="0..1", endpoint="declared_proxy_plan",
            conditions={"parameters": {}}, expected_interval=[.9, 1] if i == 0 else [0, .85])
    agent.dispatch("propose_model", args)
    agent.dispatch("render_inputs", msa(tmp_path))
    directory = tmp_path / "proxy_predictions"
    directory.mkdir()
    write_pose(directory / "synthetic_model_0.pdb")
    confidence = directory / "confidence_synthetic_model_0.json"
    confidence.write_text('{"complex_plddt": 0.8}')
    agent.dispatch("validate_predictions", {**ACC, "directory": str(directory)})
    plan = {"schema_version": "1.0", "mechanism": "synthetic_transfer", "reaction_step": "synthetic transfer",
        "assay_endpoint": "test", "context": {}, "hypotheses": [{"proxy_id": "complex_plddt",
            "rationale": "Prediction confidence baseline", "expected_direction": None, "evidence_ids": ["synthetic:reference"]}]}
    agent.dispatch("declare_proxy_plan", {**ACC, "plan": plan})
    agent.dispatch("evaluate_proxy_plan", ACC)
    result = assessment(agent)
    assert [p["status"] for p in result["predictions"]] == ["contradicted", "consistent"]
    respond(agent)
    assert assessment(agent)["response_current"]
    # A self-consistent saved artifact using the old geometry map still cannot
    # authorize a score under the current validated atom identities.
    revision = agent._current("SYNTHETIC1")
    revision["proxy_evaluation"]["role_resolution"] = {"basis": "handoff_geometry"}
    from pathlib import Path
    Path(revision["proxy_evaluation_artifact"]).write_text(json.dumps(revision["proxy_evaluation"]))
    assert {p["status"] for p in assessment(agent)["predictions"]} == {"unresolved"}
    assert any("validated atoms" in gap for gap in agent.completion_errors())
    agent.dispatch("evaluate_proxy_plan", ACC)
    respond(agent)
    confidence.write_text('{"complex_plddt": 1.0}')
    result = assessment(agent)
    assert not result["response_current"]
    assert {p["status"] for p in result["predictions"]} == {"unresolved"}


def test_proxy_coordinates_follow_the_validated_phosphate_not_a_conflicting_geometry_block(tmp_path):
    import gemmi
    from structural_modeling_agent.proxies.workflow import run_plan
    record = modeling_record(tmp_path)
    # Different P atoms have deliberately different approach distances. This is
    # a wiring fixture, not a physically complete nucleotide model.
    pose_path = write_pose(tmp_path / "phosphate_model_0.pdb")
    structure = gemmi.read_structure(str(pose_path))
    nucleotide = gemmi.Chain("C")
    residue = gemmi.Residue()
    residue.name, residue.seqid, residue.het_flag = "ATP", gemmi.SeqId(1, " "), "H"
    for name, element, x in [("PA", "P", 3), ("O3A", "O", 4), ("PG", "P", 8), ("O3B", "O", 9)]:
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = name, gemmi.Element(element), gemmi.Position(x, 0, 0)
        residue.add_atom(atom)
    nucleotide.add_residue(residue)
    structure[0].add_chain(nucleotide)
    structure.write_pdb(str(pose_path))
    site = record["validation_plan"]["poses"]["sites"][0]
    site["electrophile"].update(chain="C", atom="PG")
    site["angle_reference"].update(chain="C", atom="O3B")
    record["geometry"].update(ideal_angle=180,
        nucleophile={"kind": "protein", "chain": "A", "residue": "1", "atom": "N"},
        electrophile={"kind": "cofactor", "chain": "C", "atom": "PA"},
        angle_reference={"kind": "cofactor", "chain": "C", "atom": "O3A"})
    plan = {"schema_version": "1.0", "mechanism": "gamma_transfer", "reaction_step": "synthetic gamma transfer",
        "assay_endpoint": "test", "context": {}, "hypotheses": [{"proxy_id": "approach_distance",
            "rationale": "Distance to the selected phosphate", "expected_direction": "-", "evidence_ids": ["synthetic:reference"]}]}
    old_mapping = run_plan(tmp_path, copy.deepcopy(record), plan)
    bound = run_plan(tmp_path, record, plan, use_validated_sites=True)
    assert old_mapping["measurements"]["approach_distance"]["value"] == pytest.approx(3)
    assert bound["measurements"]["approach_distance"]["value"] == pytest.approx(8)
    assert bound["role_resolution"]["roles"]["electrophile"]["atom"] == "PG"


def test_multisite_proxy_extraction_requires_an_explicit_site_and_preserves_missingness(tmp_path):
    from structural_modeling_agent.modeling_reasoning.reactive_site import resolver_from_validation
    from structural_modeling_agent.proxies.roles import parse_pose
    record = modeling_record(tmp_path)
    sites = record["validation_plan"]["poses"]["sites"]
    other = copy.deepcopy(sites[0])
    other["id"] = "site2"
    other["nucleophile"]["atom"] = "MISSING"
    sites.append(other)
    with pytest.raises(ValueError, match="context.site_id"):
        resolver_from_validation(record)
    pose = parse_pose(str(write_pose(tmp_path / "test_model_0.pdb")))
    resolver, declaration = resolver_from_validation(record, "site1")
    assert resolver(pose).attack_distance == pytest.approx(3)
    assert declaration["site_id"] == "site1"
    missing, _ = resolver_from_validation(record, "site2")
    assert missing(pose) is None
    with pytest.raises(ValueError, match="one validated site"):
        resolver_from_validation(record, "unknown_site")


def test_revision_resume_and_export_preserve_negative_evidence_and_prior_predictions(tmp_path):
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    agent.dispatch("propose_model", args)
    record_observation(agent)
    respond(agent, "revise")
    with pytest.raises(ValueError, match="changed model"):
        agent.dispatch("propose_model", args)
    revised = copy.deepcopy(args)
    revised["scientific_reasoning"]["selected_hypothesis_id"] = "H2"
    supporting_source = copy.deepcopy(agent._current("SYNTHETIC1")["reasoning_observations"][0]["source"])
    supporting_source["effect"] = "supports"
    revised["scientific_reasoning"]["hypotheses"][1]["evidence"] = [supporting_source]
    revised["scientific_reasoning"]["selection_rationale"] = "The observed endpoint contradicts H1; investigate H2 and assay confounding."
    agent.dispatch("propose_model", revised)
    current = agent._current("SYNTHETIC1")
    assert current["reasoning_observations"][0]["value"] == .1
    assert current["reasoning_registration"]["known_observation_ids"] == ["assay_1"]
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "blocked"
    respond(agent)
    assert agent.dispatch("render_inputs", msa(tmp_path))["status"] == "rendered"
    agent._save()
    resumed = ModelingAgent(run_dir=agent.run_dir, resume=True)
    assert assessment(resumed)["response_current"]
    assert "contradicted" in json.dumps(resumed._context())
    report = resumed._write_report()
    assert len(report["candidates"][0]["reasoning_history"]) == 2
    assert report["candidates"][0]["reasoning_history"][0]["assessment"]["predictions"][0]["status"] == "contradicted"
    evidence = json.loads((resumed.run_dir / "structural_modeling_evidence.json").read_text())
    assert evidence["candidates"][0]["reasoning_assessment"]["predictions"][0]["status"] == "contradicted"
    assert evidence["candidates"][0]["reasoning_review"]["status"] == "consistent_record"


def test_response_covers_all_contradictions_and_requires_a_discriminating_control(tmp_path):
    agent = make_agent(tmp_path)
    agent.dispatch("propose_model", experimental_proposal(tmp_path))
    record_observation(agent, value=20)  # Both hypotheses contradicted.
    args = response_args(agent)
    args["addressed_hypotheses"] = ["H1"]
    with pytest.raises(ValueError, match="all contradicted"):
        agent.dispatch("respond_to_reasoning_feedback", args)
    args["addressed_hypotheses"] = ["H1", "H2"]
    args["next_test"]["expected_if_alternative"] = args["next_test"]["expected_if_selected"]
    with pytest.raises(ValueError, match="different expected outcomes"):
        agent.dispatch("respond_to_reasoning_feedback", args)


def test_controller_tool_loop_observes_feedback_before_completion(tmp_path):
    from types import SimpleNamespace
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    measured = observation(agent)
    class Decisions:
        requested_model = "scripted-offline-eval"
        def call(self, **kwargs):
            turn = agent.state["turn"]
            calls = {
                1: [("propose_model", args)], 2: [("render_inputs", msa(tmp_path))],
                3: [("record_reasoning_observation", {**ACC, "observation": measured})],
                4: [("finish_task", {"outcome": "complete", "rationale": "Attempt premature completion"})],
                6: [("finish_task", {"outcome": "complete", "rationale": "Prepared inputs; contradiction retained with a controlled next test."})]}
            chosen = [("respond_to_reasoning_feedback", response_args(agent))] if turn == 5 else calls[turn]
            return SimpleNamespace(choices=[SimpleNamespace(message=Message(chosen))], usage=None)
    agent.llm = Decisions()
    report = agent.run(max_turns=6)
    assert report["status"] == "complete"
    assert report["candidates"][0]["reasoning_assessment"]["predictions"][0]["status"] == "contradicted"
    results = [json.loads(m["content"]) for m in agent.messages if m["role"] == "tool"]
    assert any("blocked" in json.dumps(result) and "reasoning" in json.dumps(result) for result in results)
    names = {t["function"]["name"] for t in tool_schemas()}
    assert {"inspect_reasoning", "record_reasoning_observation", "respond_to_reasoning_feedback"} <= names


def test_tool_loop_selects_a_test_and_revises_the_selected_hypothesis_from_its_result(tmp_path):
    from types import SimpleNamespace
    agent = make_agent(tmp_path)
    args = experimental_proposal(tmp_path)
    measured = observation(agent)
    class Decisions:
        requested_model = "scripted-adaptive-eval"
        def call(self, **kwargs):
            turn = agent.state["turn"]
            if turn == 1:
                calls = [("propose_model", args)]
            elif turn == 2:
                calls = [("compare_hypothesis_tests", ACC)]
            elif turn == 3:
                assert assessment(agent)["test_priorities"][0]["readout"]["metric"] == measured["metric"]
                calls = [("record_reasoning_observation", {**ACC, "observation": measured})]
            elif turn == 4:
                assert assessment(agent)["predictions"][0]["status"] == "contradicted"
                calls = [("respond_to_reasoning_feedback", response_args(agent, "revise"))]
            elif turn == 5:
                result = assessment(agent)
                chosen = next(p["hypothesis_id"] for p in result["predictions"] if p["status"] == "consistent")
                revised = copy.deepcopy(args)
                reasoning = revised["scientific_reasoning"]
                reasoning["selected_hypothesis_id"] = chosen
                reasoning["selection_rationale"] = "The measured endpoint contradicts H1 and is consistent with H2; causal attribution still needs controls."
                source = {**copy.deepcopy(measured["source"]), "effect": "supports"}
                next(h for h in reasoning["hypotheses"] if h["id"] == chosen)["evidence"] = [source]
                calls = [("propose_model", revised)]
            elif turn == 6:
                calls = [("respond_to_reasoning_feedback", response_args(agent, "retain"))]
            elif turn == 7:
                calls = [("render_inputs", msa(tmp_path))]
            else:
                calls = [("finish_task", {"outcome": "complete", "rationale": "Prepare inputs under the revised hypothesis and retain the conflicting prior prediction and required controls."})]
            return SimpleNamespace(choices=[SimpleNamespace(message=Message(calls))], usage=None)
    agent.llm = Decisions()
    report = agent.run(max_turns=8)
    assert report["status"] == "complete"
    current = report["candidates"][0]
    assert current["scientific_reasoning"]["selected_hypothesis_id"] == "H2"
    assert len(current["reasoning_history"]) == 2
    assert current["reasoning_history"][0]["responses"][0]["action"] == "revise"
    assert current["reasoning_assessment"]["predictions"][0]["status"] == "contradicted"
    assert ModelingAgent(run_dir=agent.run_dir, resume=True).completion_errors() == []
