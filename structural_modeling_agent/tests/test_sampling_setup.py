"""The agent chooses protocols; tools preserve decisions and expose gaps."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from structural_modeling_agent.sampling.setup import DECISIONS, assess, contract, validate_execution_binding
from structural_modeling_agent.validation import file_digest, fingerprint
from test_modeling_agent import make_agent

TARGET = {"reaction": "synthetic target transformation"}


def declaration(tmp_path, sequence="AG"):
    path = tmp_path / "reference.txt"
    path.write_text("Synthetic reference for testing decision traceability; not enzyme evidence.")
    source = {"id": "synthetic", "category": "primary_literature", "path": str(path),
              "sha256": file_digest(path), "quote": path.read_text(),
              "interpretation": "Synthetic fixture, not a scientific conclusion", "reviewed": True}
    return {"schema_version": "sampling-setup-1.0", "question": "Which protocol tests this synthetic hypothesis?",
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(), "target_sha256": fingerprint(TARGET), "sources": [source],
            "decisions": {key: {"status": "resolved", "choice": {"agent_selected": key},
                                 "rationale": "Synthetic explicit decision, without a recipe default",
                                 "evidence_ids": ["synthetic"], "alternatives": [],
                                 "assumptions": [], "uncertainties": []} for key in DECISIONS}}


def test_contract_has_questions_without_enzyme_or_protocol_defaults():
    data = json.dumps(contract())
    assert set(contract()["decisions"]) == set(DECISIONS)
    assert not any(x in data for x in ["AS9", "His209", "pH 8", "uma-s-1p2p1"])


def test_agent_selected_values_are_preserved_without_a_fixed_recipe(tmp_path):
    setup = declaration(tmp_path)
    setup["decisions"]["conditions"]["choice"] = {"pH": 6.2, "temperature_K": 310}
    setup["decisions"]["metals"]["choice"] = {"count": 2, "identity": "synthetic test metal"}
    setup["decisions"]["sampling"]["choice"] = {"method": "agent_selected_method_not_implemented_here"}
    original = copy.deepcopy(setup)
    result = assess(setup, "AG", TARGET)
    assert result["status"] == "ready_for_preparation"
    assert result["execution_authorized"] is False
    assert setup == original


def test_missing_and_pending_choices_are_not_filled_in(tmp_path):
    setup = declaration(tmp_path)
    del setup["decisions"]["protonation"]
    setup["decisions"]["metals"].update(status="unresolved", choice=None,
                                        rationale="Need independent metal stoichiometry evidence")
    result = assess(setup, "AG", TARGET)
    assert result["status"] == "needs_investigation"
    assert {g["decision"] for g in result["gaps"]} == {"protonation", "metals"}
    assert "protonation" not in setup["decisions"]
    assert setup["decisions"]["metals"]["choice"] is None


def test_conditional_states_require_uncertainty_and_a_discriminating_test(tmp_path):
    setup = declaration(tmp_path)
    d = setup["decisions"]["protonation"]
    d.update(status="conditional", uncertainties=["Synthetic tautomer ambiguity"])
    with pytest.raises(ValueError, match="discriminating_test"):
        assess(setup, "AG", TARGET)
    d["discriminating_test"] = "Compare the explicitly justified alternatives against independent references"
    assert assess(setup, "AG", TARGET)["status"] == "ready_for_preparation"


def test_changed_source_and_different_sequence_require_new_investigation(tmp_path):
    setup = declaration(tmp_path)
    with pytest.raises(ValueError, match="exact candidate sequence"):
        assess(setup, "AD", TARGET)
    with pytest.raises(ValueError, match="exact requested target"):
        assess(setup, "AG", {"reaction": "different substrate or reaction"})
    Path(setup["sources"][0]["path"]).write_text("changed source")
    assert assess(setup, "AG", TARGET)["status"] == "needs_investigation"


def test_agent_tools_save_resumable_decisions_without_submitting_or_loading_models(tmp_path):
    agent = make_agent(tmp_path)
    setup = declaration(tmp_path)
    assert agent.dispatch("inspect_sampling_setup_contract", {})["schema_version"] == "sampling-setup-1.0"
    saved = agent.dispatch("declare_sampling_setup", {"accession": "SYNTHETIC1", "setup": setup})
    assert saved["status"] == "ready_for_preparation"
    assert not agent.state["candidates"]["SYNTHETIC1"]["revisions"]
    original = Path(saved["artifact"]).read_bytes()
    Path(setup["sources"][0]["path"]).write_text("changed source")
    updated = agent.dispatch("declare_sampling_setup", {"accession": "SYNTHETIC1", "setup": setup})
    assert updated["status"] == "needs_investigation"
    assert Path(saved["artifact"]).read_bytes() == original
    page = agent.dispatch("inspect_sampling_setup", {"accession": "SYNTHETIC1", "setup_id": saved["id"]})
    assert page["status"] == "needs_investigation"
    agent._save()
    from structural_modeling_agent.result_layout import read_run_index
    index = read_run_index(agent.run_dir)
    assert index["candidates"]["SYNTHETIC1"]["sampling_setups"][0]["artifact"] == saved["artifact"]


def test_execution_cannot_replace_agent_selected_conditions_or_method(tmp_path):
    setup = declaration(tmp_path)
    config = {"temperature_K": 310, "models": [{"id": "selected-profile"}],
              "umbrella": {"enabled": False, "centers_A": [], "spring_eV_per_A2": 0},
              "timestep_fs": 0.25, "friction_per_fs": 0.005, "equilibration_steps": 10,
              "production_steps": 20, "sample_interval": 2, "checkpoint_interval": 10, "seeds": [9],
              "max_wall_seconds": 60, "resource_budget": {"max_workers": 2, "max_wall_seconds_per_task": 100}}
    setup["decisions"]["conditions"]["choice"] = {"pH": 6.2, "temperature_K": 310}
    setup["decisions"]["model_backend"]["choice"] = {"profile_ids": ["selected-profile"]}
    parameters = {k: config[k] for k in ("umbrella", "timestep_fs", "friction_per_fs", "equilibration_steps",
                                         "production_steps", "sample_interval", "checkpoint_interval", "seeds")}
    setup["decisions"]["sampling"]["choice"] = {"method": "unbiased_md", "parameters": parameters}
    setup["decisions"]["resources"]["choice"] = config["resource_budget"]
    declaration_path = tmp_path / "decision.json"
    declaration_path.write_text(json.dumps({"setup": setup}))
    config.update(setup_declaration=str(declaration_path), setup_declaration_sha256=file_digest(declaration_path))
    preparation = {"sequence": "AG", "sequence_sha256": setup["sequence_sha256"], "target": TARGET,
                   "agent_setup_sha256": fingerprint(setup), "assay_conditions": {"pH": 6.2}}
    assert validate_execution_binding(config, preparation, tmp_path / "config.json")["status"] == "ready_for_preparation"
    config["temperature_K"] = 300
    with pytest.raises(ValueError, match="Run conditions differ"):
        validate_execution_binding(config, preparation, tmp_path / "config.json")
    config["temperature_K"] = 310
    setup["decisions"]["sampling"]["choice"]["method"] = "another_agent_selected_method"
    declaration_path.write_text(json.dumps({"setup": setup}))
    config["setup_declaration_sha256"] = file_digest(declaration_path)
    preparation["agent_setup_sha256"] = fingerprint(setup)
    with pytest.raises(ValueError, match="no automatic substitution"):
        validate_execution_binding(config, preparation, tmp_path / "config.json")


def test_category_synonyms_are_accepted_and_unknown_is_clear(tmp_path):
    # A computed pKa diagnostic labeled 'diagnostic' maps to method_output rather than failing.
    setup = declaration(tmp_path)
    setup["sources"][0]["category"] = "diagnostic"
    report = assess(setup, "AG", TARGET)
    assert any(s["category"] == "method_output" for s in report["sources"])
    # The contract now documents each category.
    assert isinstance(contract()["source_categories"], dict)
    assert "method_output" in contract()["source_categories"]
    # A genuinely unknown category still fails, naming the method_output hint.
    setup2 = declaration(tmp_path)
    setup2["sources"][0]["category"] = "nonsense"
    with pytest.raises(ValueError, match="method_output"):
        assess(setup2, "AG", TARGET)
