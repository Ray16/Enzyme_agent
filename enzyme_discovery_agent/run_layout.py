"""Organized run files, with explicit compatibility for older flat directories."""

import hashlib
import json
from pathlib import Path

_STATE = {
    "candidate_registry.json", "claim_validation.json", "discovery_coverage.json",
    "family_budget.json", "family_landscape.json", "family_learning.json", "family_decisions.json",
    "finalization_status.json", "mechanism_families.json", "notebook.json", "pass1_ranking.json",
    "reasoning_trace.json", "substrate_identity_verification.json", "target_context.json",
    "target_reasoning.json", "token_usage.json", "candidate_map.json", "embedding_store",
}
# Secondary output files are folded into artifacts/ subfolders so the run's TOP
# level presents only the primary deliverable (ranked_candidates.csv) plus the
# run descriptor (manifest.json) and the structural subdirs. ranked_candidates.csv
# and manifest.json are deliberately NOT routed -- they stay at the top level
# (run_index and the production controller read them there).
_CANDIDATE_OUTPUTS = (
    "research_portfolio.csv", "discovery_candidates.csv", "discovery_candidates.fasta",
    "discovery_candidates.xlsx", "primary_candidates.csv", "exploratory_candidates.csv",
    "discovery_alternatives.csv", "candidate_coverage.csv", "candidate_assessment_queue.csv",
    "discovery_panel_summary.json", "experimental_panel.csv", "experimental_panel.fasta",
    "experimental_panel.xlsx", "experimental_panel_summary.json",
)
_REPORT_OUTPUTS = ("report.md", "report_confidence.md", "report_explorative.md", "RUN_SUMMARY.md")
# NOTE: structural_modeling_handoff.{json,md} are deliberately NOT routed -- the
# handoff writer requires them at the run root (its handoff_bundle uses paths
# relative to root), enforced by a hard check in write_structural_modeling_handoff.
_HANDOFF_OUTPUTS = (
    "discovery_modeling_handoff.json", "primary_modeling_handoff.json", "modeling_portfolio.json",
)
_FIGURE_OUTPUTS = ("family_substrate_fit.png", "candidate_map.png")
_LITERATURE_OUTPUTS = ("literature_mentions.csv", "unresolved_literature_leads.csv")
_DESTINATIONS = {**{name: f"agent_state/{name}" for name in _STATE},
                 **{name: f"artifacts/candidates/{name}" for name in _CANDIDATE_OUTPUTS},
                 **{name: f"artifacts/reports/{name}" for name in _REPORT_OUTPUTS},
                 **{name: f"artifacts/handoffs/{name}" for name in _HANDOFF_OUTPUTS},
                 **{name: f"artifacts/figures/{name}" for name in _FIGURE_OUTPUTS},
                 **{name: f"artifacts/literature/{name}" for name in _LITERATURE_OUTPUTS},
                 "reference_enzymes.json": "artifacts/analysis/reference_enzymes.json",
                 "events.jsonl": "logs/events.jsonl", "console_log.txt": "logs/console_log.txt",
                 "exploration_log.md": "logs/exploration_log.md",
                 "citation_verification.json": "artifacts/evidence/citation_verification.json",
                 "exploration_summary.json": "artifacts/analysis/exploration_summary.json",
                 "code_changes.patch": "provenance/code_changes.patch", "scripts": "provenance/scripts",
                 "report_interim.md": "checkpoints/report_interim.md"}


def run_file(run_dir, name):
    """Canonical in organized runs; retain legacy files until explicit migration."""
    root = Path(run_dir)
    name = str(name)
    destination = _DESTINATIONS.get(name)
    if destination is None:
        return root / name
    canonical = root / destination
    if canonical.exists():
        return canonical
    legacy = root / name
    if legacy.exists():
        return legacy
    if (root / "agent_state" / "run_layout.json").exists():
        return canonical
    return legacy


def initialize_layout(run_dir):
    from enzyme_discovery_agent.state_store import atomic_write_json
    root = Path(run_dir)
    for destination in _DESTINATIONS.values():
        (root / destination).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(run_dir) / "agent_state" / "run_layout.json", {
        "version": 1, "files": _DESTINATIONS,
    })


def organize_run(run_dir):
    """Move working files in an inactive run and save hashes and a reversible map."""
    from enzyme_discovery_agent.state_store import atomic_write_json
    root = Path(run_dir).resolve()
    moved = []
    # Callers must establish that no discovery producer is active before moving
    # state or its lock files. A migration never removes or replaces data.
    for name, relative in _DESTINATIONS.items():
        source, destination = root / name, root / relative
        if not source.exists():
            continue
        if destination.exists():
            raise FileExistsError(f"Migration destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        source.rename(destination)
        lock = source.with_name(source.name + ".lock")
        if lock.exists():
            lock.rename(destination.with_name(destination.name + ".lock"))
        moved.append({"from": name, "to": relative, "sha256": digest})
        atomic_write_json(root / "provenance" / "run_layout_migration.json", {"moves": moved})
    initialize_layout(root)
    return moved
