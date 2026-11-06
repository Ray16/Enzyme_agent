"""Synthetic solver tests are not examples of molecular processivity."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from structural_modeling_agent.sampling.kinetics import evaluate
from structural_modeling_agent.validation import file_digest, fingerprint


def network(tmp_path):
    source = tmp_path / "synthetic_rates.txt"
    source.write_text("Synthetic rates for numerical validation only; no enzyme experiment or simulation.")
    data = {"schema_version": "processivity-network-1.0", "question": "Test competing growth and release in a synthetic network",
            "conditions": {"scope": "synthetic mathematical test"}, "start": "bound_0", "minimum_extensions": 2,
            "extension_definition": "One synthetic chain-extension cycle, not an ATP or other auxiliary bond event",
            "association_definition": "Committed escape ends the original association; rebinding starts a new one",
            "states": [{"id": f"bound_{n}", "associated": True, "extensions": n, "terminal": None,
                        "description": f"Synthetic associated state after {n} extensions"} for n in range(3)] +
                       [{"id": f"released_{n}", "associated": False, "extensions": n, "terminal": "release",
                         "description": "Synthetic committed escape"} for n in range(2)], "transitions": []}
    for left, right, event, rate in [("bound_0", "bound_1", "extension", 4),
                                      ("bound_0", "released_0", "release", 1),
                                      ("bound_1", "bound_2", "extension", 2),
                                      ("bound_1", "released_1", "release", 2)]:
        data["transitions"].append({"from": left, "to": right, "event": event, "rate_s_inv": rate,
                                    "conditions_sha256": fingerprint(data["conditions"]), "assumptions": [],
                                    "source": {"kind": "synthetic", "path": str(source), "sha256": file_digest(source),
                                               "quote": source.read_text(), "interpretation": "Software fixture only", "reviewed": True}})
    return data


def test_two_growth_steps_compete_with_first_release(tmp_path):
    result = evaluate(network(tmp_path))
    assert result["probability_repeated_growth_before_release"] == pytest.approx((4/5)*(2/4))
    assert result["mean_time_to_first_outcome_s"] == pytest.approx(1/5 + (4/5)*(1/4))
    assert result["status"] == "software_demonstration_only"
    assert result["processivity_demonstrated"] is False


def test_fast_rebinding_cannot_be_counted_as_processive_continuation(tmp_path):
    data = network(tmp_path)
    edge = copy.deepcopy(data["transitions"][0])
    edge.update({"from": "released_0", "to": "bound_0", "event": "binding", "rate_s_inv": 1e9})
    data["transitions"].append(edge)
    assert evaluate(data)["probability_repeated_growth_before_release"] == pytest.approx(.4)


def test_missing_rate_or_changed_source_cannot_produce_a_probability(tmp_path):
    data = network(tmp_path)
    data["transitions"][0]["rate_s_inv"] = None
    result = evaluate(data)
    assert result["status"] == "needs_rate_investigation"
    assert result["probability_repeated_growth_before_release"] is None
    data = network(tmp_path)
    Path(data["transitions"][0]["source"]["path"]).write_text("changed source")
    assert evaluate(data)["probability_repeated_growth_before_release"] is None


def test_forced_retention_and_auxiliary_bonds_cannot_establish_processivity(tmp_path):
    data = network(tmp_path)
    data["transitions"] = [e for e in data["transitions"] if e["event"] == "extension"]
    with pytest.raises(ValueError, match="competing committed-release"):
        evaluate(data)
    data = network(tmp_path)
    data["transitions"][0]["event"] = "chemistry"
    with pytest.raises(ValueError, match="Only a declared extension"):
        evaluate(data)


def test_incompatible_rate_conditions_are_not_pooled(tmp_path):
    data = network(tmp_path)
    data["transitions"][0]["conditions_sha256"] = fingerprint({"different": "protonation or concentrations"})
    assert evaluate(data)["status"] == "needs_rate_investigation"


def test_agent_binds_network_to_exact_target_and_preserves_synthetic_scope(tmp_path):
    from test_modeling_agent import make_agent
    agent = make_agent(tmp_path)
    sources = agent.run_dir / "sources"
    sources.mkdir(exist_ok=True)
    data = network(sources)
    data.update(sequence_sha256=hashlib.sha256(b"AG").hexdigest(), target_sha256=fingerprint(agent.task["target"]))
    assert agent.dispatch("inspect_processivity_contract", {})["schema_version"] == "processivity-network-1.0"
    result = agent.dispatch("assess_processivity_network", {"accession": "SYNTHETIC1", "network": data})
    assert result["status"] == "software_demonstration_only"
    assert result["processivity_demonstrated"] is False
    assert Path(result["artifact"]).is_file()
    data["sequence_sha256"] = hashlib.sha256(b"different genotype").hexdigest()
    with pytest.raises(ValueError, match="exact candidate sequence"):
        agent.dispatch("assess_processivity_network", {"accession": "SYNTHETIC1", "network": data})
