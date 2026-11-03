"""Durable discovery state for mechanism-family hypotheses.

The candidate registry answers "which proteins did we find?". This ledger answers
the complementary recall question: "which plausible catalytic routes did we
consider, resolve, survey, and either support or rule out?" Missing evidence is a
first-class unresolved state, never an implicit negative result.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

from enzyme_discovery_agent.models.family_hypothesis import normalize_family_hypothesis


from enzyme_discovery_agent.state_store import atomic_write_json, state_lock

SCHEMA_VERSION = "1.0"
TERMINAL_STATES = frozenset({"supported", "contradicted"})
_lock = threading.Lock()


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def family_hypothesis_id(family: dict) -> str:
    """Return a stable, readable id for one model-proposed route."""
    name = str(family.get("name") or "family").casefold()
    return (re.sub(r"[^a-z0-9]+", "-", name).strip("-") or "family")[:80]


def load_family_ledger(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "family_hypothesis_ledger.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"schema_version": SCHEMA_VERSION, "families": {}}
    if not isinstance(data, dict) or not isinstance(data.get("families"), dict):
        return {"schema_version": SCHEMA_VERSION, "families": {}}
    data["schema_version"] = SCHEMA_VERSION
    return data


def _write(run_dir: Path, ledger: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "family_hypothesis_ledger.json", ledger)


def initialize_family_ledger(run_dir: str | Path, families: list[dict]) -> dict:
    """Merge newly proposed hypotheses without erasing accumulated evidence."""
    run_dir = Path(run_dir)
    with _lock, state_lock(run_dir / "family_hypothesis_ledger.json"):
        ledger = load_family_ledger(run_dir)
        records = ledger["families"]
        now = _now()
        for family in families:
            fid = family_hypothesis_id(family)
            rec = records.setdefault(fid, {
                "hypothesis_id": fid,
                "state": "proposed",
                "proposed_at": now,
                "resolution_candidates": [],
                "survey_attempts": [],
                "resolved_folds": [],
                "surveyed_folds": [],
            })
            rec["hypothesis"] = normalize_family_hypothesis(family)
            rec["route_type"] = family.get("route_type", "direct_native_catalyst")
            rec["uncertainty"] = family.get("uncertainty", "medium")
            rec["updated_at"] = now
        _write(run_dir, ledger)
    return ledger


def record_family_resolution(
    run_dir: str | Path, family: dict, candidates: list[dict],
) -> dict:
    """Persist all plausible InterPro mappings, including an empty resolution."""
    run_dir = Path(run_dir)
    with _lock, state_lock(run_dir / "family_hypothesis_ledger.json"):
        ledger = load_family_ledger(run_dir)
        fid = family_hypothesis_id(family)
        rec = ledger["families"].setdefault(fid, {
            "hypothesis_id": fid,
            "hypothesis": normalize_family_hypothesis(family),
            "proposed_at": _now(),
            "survey_attempts": [],
            "surveyed_folds": [],
        })
        rec["resolution_candidates"] = candidates
        rec["resolved_folds"] = list(dict.fromkeys(
            c.get("fold_id") for c in candidates if c.get("fold_id")
        ))
        rec["state"] = "resolved" if rec["resolved_folds"] else "unresolved"
        rec["updated_at"] = _now()
        _write(run_dir, ledger)
    return rec


def record_survey_attempt(
    run_dir: str | Path,
    fold_id: str,
    *,
    success: bool,
    family: dict | None = None,
    verdict: str | None = None,
    member_count: int | None = None,
    error: str | None = None,
) -> None:
    """Attach one retryable survey result to every matching family hypothesis."""
    run_dir = Path(run_dir)
    with _lock, state_lock(run_dir / "family_hypothesis_ledger.json"):
        ledger = load_family_ledger(run_dir)
        now = _now()
        family_id = family_hypothesis_id(family) if family else None
        for fid, rec in ledger["families"].items():
            if fid != family_id and fold_id not in rec.get("resolved_folds", []):
                continue
            rec.setdefault("survey_attempts", []).append({
                "fold_id": fold_id,
                "success": bool(success),
                "verdict": verdict,
                "member_count": member_count,
                "error": error,
                "ts": now,
            })
            if success:
                surveyed = rec.setdefault("surveyed_folds", [])
                if fold_id not in surveyed:
                    surveyed.append(fold_id)
                if verdict == "supported_family_hypothesis":
                    rec["state"] = "supported"
                elif verdict == "contradicted_family_hypothesis":
                    rec["state"] = "contradicted"
                else:
                    rec["state"] = "surveyed_insufficient"
            elif rec.get("state") not in TERMINAL_STATES:
                rec["state"] = "survey_failed"
            rec["updated_at"] = now
        _write(run_dir, ledger)


def family_coverage_summary(run_dir: str | Path) -> dict:
    ledger = load_family_ledger(run_dir)
    records = ledger.get("families", {})
    by_state: dict[str, int] = {}
    for rec in records.values():
        state = rec.get("state", "proposed")
        by_state[state] = by_state.get(state, 0) + 1
    unresolved = sorted(
        rec.get("hypothesis", {}).get("name") or fid
        for fid, rec in records.items()
        if rec.get("state", "proposed") not in TERMINAL_STATES
    )
    return {
        "total": len(records),
        "by_state": by_state,
        "resolved": sum(bool(rec.get("resolved_folds")) for rec in records.values()),
        "surveyed": sum(bool(rec.get("surveyed_folds")) for rec in records.values()),
        "terminal": sum(rec.get("state") in TERMINAL_STATES for rec in records.values()),
        "unresolved_hypotheses": unresolved,
    }
