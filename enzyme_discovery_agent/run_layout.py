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
_DESTINATIONS = {**{name: f"agent_state/{name}" for name in _STATE},
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
