"""Agent expand/stop decisions over the family landscape.

The framework surfaces the grounded family landscape (folds + their direction /
substrate resemblance / saturation); the AGENT reasons over it and decides which
families are worth expanding and which to stop.  This module persists those
decisions so the deterministic evaluation budget can honour them: a *stopped*
fold is waived from the breadth floor instead of the framework forcing coverage
of an off-target family.

Decisions are REVISITABLE by design — a later call overwrites an earlier one for
the same key, and "stop" is a current judgment, not a ban, so the agent can
resume a family when new context warrants it.  Keyed by the landscape's
``fold_id`` (InterPro id, or ``unresolved:<acc>``) so it matches
``allocate_evaluation_budget`` directly.

Mirrors the other model modules: module-level, thread-safe, best-effort, never
raises into the agent loop.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

_decisions_lock = threading.Lock()

_VALID = {"expand", "stop"}


def load_family_decisions(run_dir: str | Path) -> dict:
    """Return ``{fold_id: {"decision", "reason", "ts"}}`` (empty if none)."""
    path = Path(run_dir) / "family_decisions.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            d = data.get("decisions")
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def stopped_folds(run_dir: str | Path) -> set[str]:
    """Fold ids the agent has currently marked ``stop`` (honoured by the budget)."""
    return {
        k for k, v in load_family_decisions(run_dir).items()
        if isinstance(v, dict) and v.get("decision") == "stop"
    }


def record_family_decisions(
    run_dir: str | Path,
    expand: list[str] | None = None,
    stop: list[str] | None = None,
    reasons: dict[str, str] | None = None,
    names: dict[str, str] | None = None,
) -> dict:
    """Record expand/stop decisions; latest call wins per fold (revisitable).

    Returns the merged decisions dict.  Emits one ``family_decision`` event per
    fold so the trajectory records the agent's family-level choices and the
    reason behind each.  ``names`` maps fold id -> human-readable InterPro family
    name, stored alongside the id so the decision trail is auditable without an id
    lookup.  Best-effort: a write failure is logged, not raised.
    """
    run_dir = Path(run_dir)
    reasons = reasons or {}
    names = names or {}
    now = datetime.now().astimezone().isoformat()
    updates: list[tuple[str, str, str, str]] = []
    for key in (expand or []):
        updates.append((key, "expand", str(reasons.get(key, "")), str(names.get(key, ""))))
    for key in (stop or []):
        updates.append((key, "stop", str(reasons.get(key, "")), str(names.get(key, ""))))

    with _decisions_lock:
        decisions = load_family_decisions(run_dir)
        for key, decision, reason, name in updates:
            if decision not in _VALID or not key:
                continue
            entry = {"decision": decision, "reason": reason, "ts": now}
            if name:
                entry["name"] = name
            decisions[key] = entry
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "family_decisions.json").write_text(
                json.dumps({"decisions": decisions}, indent=2) + "\n"
            )
        except OSError as e:
            logger.warning("Failed to write family_decisions: %s", e)

    # Emit outside the lock: telemetry must never hold the write lock.
    from enzyme_discovery_agent.models.event_log import append_event
    for key, decision, reason in updates:
        if decision in _VALID and key:
            append_event(run_dir, "family_decision",
                         fold=key, decision=decision, reason=reason[:200])
    return decisions
