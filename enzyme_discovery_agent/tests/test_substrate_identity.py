"""Reject valid-but-wrong structures and unresolved target mixtures."""
import json
from types import SimpleNamespace
from unittest.mock import patch

from enzyme_discovery_agent.chemistry_logic import reason_target_substrate_smiles


def client(substrates):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=json.dumps({"substrates": substrates})))])
    return SimpleNamespace(call=lambda **kwargs: response)


def test_valid_but_wrong_molecule_is_corrected_against_named_identity():
    model = client([{"name": "input compound", "smiles": "CCO"}])
    reference = {"name": "input compound", "smiles": "CCCO", "molecular_formula": "C3H8O"}
    audit = []
    with patch("enzyme_discovery_agent.chemistry_logic.resolve_named_substrate_identity", return_value=reference):
        actual = reason_target_substrate_smiles("convert input compound to product", model, audit.append)
    assert actual == "CCCO"
    assert audit[0]["status"] == "model_proposal_corrected"
    assert audit[0]["proposal_matches_reference"] is False


def test_unresolved_named_compound_does_not_fall_back_to_model_smiles():
    model = client([{"name": "input compound", "smiles": "CCO"}])
    with patch("enzyme_discovery_agent.chemistry_logic.resolve_named_substrate_identity", return_value=None):
        assert reason_target_substrate_smiles("convert input compound", model) is None


def test_partial_mixture_and_ungrounded_names_remain_unresolved():
    model = client([{"name": "compound A"}, {"name": "compound B"}])
    with patch("enzyme_discovery_agent.chemistry_logic.resolve_named_substrate_identity",
               side_effect=[{"smiles": "CCO"}, None]):
        assert reason_target_substrate_smiles("react compound A with compound B", model) is None
    hallucinated = client([{"name": "invented compound", "smiles": "CCO"}])
    with patch("enzyme_discovery_agent.chemistry_logic.resolve_named_substrate_identity") as resolver:
        assert reason_target_substrate_smiles("convert input compound", hallucinated) is None
        resolver.assert_not_called()
