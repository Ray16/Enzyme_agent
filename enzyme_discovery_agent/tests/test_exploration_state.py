"""Decision-state visibility and observation-only exploration diagnostics."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from enzyme_discovery_agent.context import build_model_context, build_research_state
from enzyme_discovery_agent.models.candidate_registry import update_candidate_registry, load_candidate_registry
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.models.exploration_state import exploration_state, exploration_page
from enzyme_discovery_agent.models.progress import research_signature
from enzyme_discovery_agent.tools.reasoning import get_exploration_state


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def candidate(root, accession="SYNTHETIC1", family="IPR000001", **metadata):
    update_candidate_registry(root, accession, "evaluated", metadata={
        "protein_name": "Synthetic catalyst", "interpro_family_id": family,
        "gene_names": ["synthetic_gene"], **metadata})


def model_report():
    return {"status": "validated", "revision": 1, "review_errors": [],
        "setup_validation": {"status": "pass", "record_sha256": "synthetic-input"},
        "pose_validation": {"status": "pass", "n_predictions": 50, "n_passing": 8,
                            "updated_at": "first", "note": "saved interpretation"},
        "geometry_metrics": {"site.synthetic": {"weighted_attack_angle_deg": 160.,
            "harmonic_distance_A": 3.7, "n_poses": 50, "complete_observed_ensemble": True}}}


def test_imported_modeling_changes_progress_and_default_working_state(tmp_path):
    candidate(tmp_path)
    before = research_signature(tmp_path, [])
    update_candidate_registry(tmp_path, "SYNTHETIC1", "structurally_evaluated", metadata={
        "modeling_validation_evidence": model_report(), "modeling_review_status": "ready_for_modeling_review"})
    assert research_signature(tmp_path, []) != before
    state = build_research_state(tmp_path)
    observation = state["modeling_feedback"][0]
    assert observation["site_metrics"][0]["harmonic_distance_A"] == 3.7
    assert observation["reported_status"] == "validated"
    assert "activity is unestablished" in observation["interpretation"]
    assert state["exploration_diagnostics"]["overview"]["modeling_records"] == 1


def test_repeat_import_wrapper_and_prose_changes_do_not_create_progress(tmp_path):
    candidate(tmp_path)
    report = model_report()
    update_candidate_registry(tmp_path, "SYNTHETIC1", None, metadata={"modeling_validation_evidence": report})
    before = research_signature(tmp_path, [])
    report["pose_validation"].update(updated_at="second", note="rewritten interpretation")
    update_candidate_registry(tmp_path, "SYNTHETIC1", None, metadata={
        "modeling_validation_evidence": report, "modeling_validation_source": {"path": "other-report", "sha256": "new-wrapper"}})
    assert research_signature(tmp_path, []) == before
    report["geometry_metrics"]["site.synthetic"]["harmonic_distance_A"] = 10.
    update_candidate_registry(tmp_path, "SYNTHETIC1", None, metadata={"modeling_validation_evidence": report})
    assert research_signature(tmp_path, []) != before


def test_proxy_units_and_transfer_scope_survive_without_normalization(tmp_path):
    candidate(tmp_path)
    before = research_signature(tmp_path, [])
    update_candidate_registry(tmp_path, "SYNTHETIC1", None, metadata={"structural_proxy_evidence": {
        "status": "scored", "score": -2.3, "score_units": "declared assay units", "transfer_status": "family_transfer"}})
    assert research_signature(tmp_path, []) != before
    observation = build_research_state(tmp_path)["modeling_feedback"][0]
    assert observation["score"] == -2.3
    assert observation["transfer_status"] == "family_transfer"


def test_frontier_reads_do_not_mutate_registry_decisions_or_search_ledgers(tmp_path):
    candidate(tmp_path, sequence="ACDEFG")
    put(tmp_path, "family_decisions.json", {"decisions": {"IPR000001": {"decision": "stop", "reason": "synthetic"}}})
    put(tmp_path, "agent_state/expansion_searches.json", {"foldseek": {"SYNTHETIC1": {"status": "error", "attempts": 1}}})
    files = [tmp_path / name for name in ("candidate_registry.json", "family_decisions.json", "agent_state/expansion_searches.json")]
    before = [path.read_bytes() for path in files]
    with patch("enzyme_discovery_agent.tools.reasoning._fetch_uniprot_entry") as network:
        first = get_exploration_state("families", run_dir=str(tmp_path))
        second = get_exploration_state("families", run_dir=str(tmp_path))
    assert first == second
    assert [path.read_bytes() for path in files] == before
    network.assert_not_called()
    assert first["data"]["observational_only"] is True


def test_exploration_sources_resolve_name_keys_and_disambiguated_names(tmp_path):
    name = 'Synthetic / catalyst ~ label'
    for accession in ('SYNTH_A', 'SYNTH_B'):
        update_candidate_registry(tmp_path, accession, 'evaluated', {'protein_name': name})
    path = tmp_path/'candidate_registry.json'
    before = path.read_bytes()
    stored = json.loads(before)['candidates']
    assert len(stored) == 2
    questions = exploration_state(tmp_path)['questions']
    sources = {row['accession']: row['source'] for row in questions if row.get('accession')}
    assert set(sources) == {'SYNTH_A', 'SYNTH_B'}
    for accession, source in sources.items():
        filename, fragment = source.split('#', 1)
        value = json.loads((tmp_path/filename).read_text())
        for token in fragment.lstrip('/').split('/'):
            value = value[token.replace('~1', '/').replace('~0', '~')]
        assert value['accession'] == accession
    assert path.read_bytes() == before


def test_duplicate_accessions_do_not_imply_biological_diversity(tmp_path):
    for i in range(8):
        candidate(tmp_path, f"SYNTHETIC{i}", sequence="ACDEFG")
    overview = exploration_state(tmp_path)["overview"]
    assert overview["candidate_records"] == 8
    assert overview["canonical_family_labels"] == 1
    assert overview["annotation_groups"] == 1
    assert overview["unique_known_sequences"] == 1
    assert overview["exact_sequence_duplicate_records"] == 7


def test_missing_sequences_and_generic_names_cannot_establish_sequence_saturation(tmp_path):
    for i in range(5):
        candidate(tmp_path, f"SYNTHETIC{i}")
    state = exploration_state(tmp_path)
    assert state["overview"]["records_with_sequence"] == 0
    assert state["families"][0]["sequence_clustering"] == "unassessed"
    assert state["overview"]["global_exhaustion_established"] is False


def test_legacy_null_annotations_and_attempt_counts_remain_inspectable(tmp_path):
    put(tmp_path, "candidate_registry.json", {"candidates": {"SYNTHETIC1": {
        "statuses": [{"status": "discovered"}],
        "metadata": {"protein_name": None, "gene_names": None, "ec_numbers": None}}}})
    put(tmp_path, "agent_state/expansion_searches.json", {"foldseek": {
        "SYNTHETIC1": {"status": "error", "attempts": None}}})
    state = exploration_state(tmp_path)
    assert state["overview"]["canonical_family_labels"] == 0
    assert state["methods"][0]["recorded_attempts"] == 1


def test_errors_and_partial_results_are_not_empty_completed_searches(tmp_path):
    for i in range(4):
        candidate(tmp_path, f"SYNTHETIC{i}")
        append_event(tmp_path, "tool_call", tool="search_pubmed", args={"query": "same query"}, status="error")
    put(tmp_path, "agent_state/expansion_searches.json", {"foldseek": {
        "SYNTHETIC0": {"status": "error", "new_accessions": []},
        "SYNTHETIC1": {"status": "partial", "new_accessions": []}}})
    state = exploration_state(tmp_path)
    method = state["methods"][0]
    assert method["successful_seed_records"] == 0
    assert method["observation"] == "insufficient_evidence_for_local_plateau"
    searches = state["searches"][0]
    assert searches["outcomes"] == {"error": 4}
    assert searches["lexically_repeated_requests"] == 3
    assert "inspect_search_failures" in {q["kind"] for q in state["questions"]}


def test_local_method_yield_and_pending_returns_are_distinguished(tmp_path):
    candidate(tmp_path, "SYNTHETIC1")
    candidate(tmp_path, "SYNTHETIC2")
    put(tmp_path, "agent_state/expansion_searches.json", {"foldseek": {
        "SYNTHETIC1": {"status": "success", "new_accessions": []},
        "SYNTHETIC2": {"status": "success", "new_accessions": []}}})
    assert exploration_state(tmp_path)["methods"][0]["observation"] == "review_method_yield"
    put(tmp_path, "agent_state/expansion_searches.json", {"foldseek": {
        "SYNTHETIC1": {"status": "success", "new_accessions": ["UNRESOLVED1"]}}})
    method = exploration_state(tmp_path)["methods"][0]
    assert method["observation"] == "assess_returned_evidence"
    assert method["pending_returned_groups"] == 1


def test_unknown_expansion_returns_cannot_establish_local_plateau(tmp_path):
    candidate(tmp_path, "SYNTHETIC1")
    candidate(tmp_path, "SYNTHETIC2")
    put(tmp_path, "agent_state/expansion_searches.json", {"foldseek": {
        "SYNTHETIC1": {"status": "success", "count": 0},
        "SYNTHETIC2": {"status": "success", "count": 0}}})
    assert exploration_state(tmp_path)["methods"][0]["observation"] == "insufficient_evidence_for_local_plateau"


def test_old_global_cue_remains_a_cue_and_unmapped_hypotheses_remain_visible(tmp_path):
    for _ in range(4):
        append_event(tmp_path, "candidates_registered", new_families=0, new_accessions=3)
    put(tmp_path, "mechanism_families.json", [{"name": "Synthetic proposed route"}])
    state = exploration_state(tmp_path)
    assert state["overview"]["registration_plateau_cue"] is True
    assert state["overview"]["global_exhaustion_established"] is False
    assert {q["kind"] for q in state["questions"]} >= {"review_breadth", "investigate_hypothesis"}


def test_pagination_preserves_records_beyond_context_examples_and_run_scope(tmp_path):
    for i in range(30):
        candidate(tmp_path, f"SYNTHETIC{i:03}", modeling_validation_evidence=model_report())
    state = build_research_state(tmp_path)
    assert len(state["modeling_feedback"]) == 6 and state["omitted_modeling_records"] == 24
    found, offset = [], 0
    while True:
        page = exploration_page(tmp_path, "modeling", offset=offset, limit=10)
        assert len(json.dumps(page)) < 9000
        found.extend(row["accession"] for row in page["records"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert len(set(found)) == len(found) == 30
    assert exploration_page(tmp_path / "other", "modeling")["total"] == 0
    assert exploration_page(tmp_path, "modeling", query="SYNTHETIC029")["total"] == 1


def test_model_feedback_survives_context_compaction_and_keeps_audit_untouched(tmp_path):
    candidate(tmp_path, modeling_validation_evidence=model_report())
    messages = [{"role": "system", "content": "synthetic constraints"}, {"role": "user", "content": "synthetic target"}]
    for _ in range(20):
        messages.extend([{"role": "assistant", "content": "old narrative " * 1000}, {"role": "user", "content": "continue"}])
    original = copy.deepcopy(messages)
    context = build_model_context(messages, tmp_path)
    assert messages == original
    assert "harmonic_distance_A" in context[2]["content"]
    assert len(json.dumps(context)) < 90000


@pytest.mark.parametrize("options", [{"offset": -1}, {"limit": 0}, {"limit": 21}, {"section": "unknown"}])
def test_invalid_inspection_requests_are_errors(tmp_path, options):
    assert get_exploration_state(run_dir=str(tmp_path), **options)["status"] == "error"
