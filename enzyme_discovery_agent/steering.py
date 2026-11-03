"""Durable user steering for a live or explicitly resumed discovery campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from enzyme_discovery_agent.state_store import atomic_write_json, state_lock

ACTIONS = ("instruction", "clear_instructions", "pause", "resume", "checkpoint")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _paths(run_dir):
    root = Path(run_dir) / "agent_state" / "steering"
    return root / "commands", root / "state.json"


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def read_state(run_dir):
    _, path = _paths(run_dir)
    return _read(path, {"paused": False, "revision": 0, "instructions": [],
                        "receipts": {}, "snapshot_requested": 0, "snapshot_completed": 0})


def submit(run_dir, action, message=None):
    """Append a command without changing the running controller's state directly."""
    if action not in ACTIONS:
        raise ValueError("Unknown steering action")
    if not (Path(run_dir) / "manifest.json").is_file():
        raise ValueError("Expected a run directory containing manifest.json")
    if action == "instruction" and (not isinstance(message, str) or not message.strip()):
        raise ValueError("An instruction needs nonempty message text")
    if message and len(message) > 8000:
        raise ValueError("Instruction exceeds 8000 characters; supply evidence paths instead")
    commands, _ = _paths(run_dir)
    command_id = f"{time.time_ns():020d}_{uuid.uuid4().hex}"
    command = {"id": command_id, "action": action, "message": message,
               "submitted_at": _now(), "source": "user"}
    atomic_write_json(commands / f"{command_id}.json", command)
    return {**command, "status": "queued"}


def has_pending(run_dir):
    commands, _ = _paths(run_dir)
    receipts = read_state(run_dir)["receipts"]
    return any(p.stem not in receipts for p in commands.glob("*.json"))


def consume(run_dir):
    """Apply commands once, in submission order, at a controller boundary."""
    commands, path = _paths(run_dir)
    with state_lock(path):
        state = read_state(run_dir)
        accepted = []
        for item in sorted(commands.glob("*.json")):
            if item.stem in state["receipts"]:
                continue
            try:
                command = _read(item, {})
                action = command["action"]
                if command.get("id") != item.stem or action not in ACTIONS:
                    raise ValueError("Invalid command identity or action")
                if action == "instruction":
                    text = command.get("message")
                    if not isinstance(text, str) or not text.strip() or len(text) > 8000:
                        raise ValueError("Invalid instruction text")
                    state["instructions"].append({"id": item.stem, "message": text,
                                                   "submitted_at": command.get("submitted_at")})
                    state["revision"] += 1
                elif action == "clear_instructions":
                    state["instructions"] = []
                    state["revision"] += 1
                elif action == "pause":
                    state["paused"] = True
                    state["snapshot_requested"] += 1
                elif action == "resume":
                    state["paused"] = False
                elif action == "checkpoint":
                    state["snapshot_requested"] += 1
                state["receipts"][item.stem] = {"action": action, "status": "accepted",
                                                 "accepted_at": _now()}
                accepted.append(command)
            except (ValueError, KeyError, TypeError) as exc:
                state["receipts"][item.stem] = {"status": "rejected", "reason": str(exc)}
        if accepted or state != read_state(run_dir):
            atomic_write_json(path, state)
    return state, accepted


def update_state(run_dir, **changes):
    _, path = _paths(run_dir)
    with state_lock(path):
        state = read_state(run_dir)
        state.update(changes)
        state["boundary_at"] = _now()
        atomic_write_json(path, state)
    return state


def mark_delivered(run_dir, command_ids, iteration):
    _, path = _paths(run_dir)
    with state_lock(path):
        state = read_state(run_dir)
        for command_id in command_ids:
            receipt = state["receipts"].get(command_id)
            if receipt and receipt["status"] == "accepted":
                receipt.update(status="delivered_to_model", iteration=iteration, delivered_at=_now())
        atomic_write_json(path, state)


def mark_controls_applied(run_dir, iteration):
    _, path = _paths(run_dir)
    with state_lock(path):
        state = read_state(run_dir)
        for receipt in state["receipts"].values():
            if receipt.get("status") == "accepted" and receipt.get("action") != "instruction":
                receipt.update(status="applied", iteration=iteration, applied_at=_now())
        atomic_write_json(path, state)


def context_message(run_dir):
    instructions = read_state(run_dir)["instructions"]
    if not instructions:
        return None
    return {"role": "user", "content":
            "CURRENT USER STEERING (persists until explicitly cleared)\n" +
            json.dumps(instructions, ensure_ascii=False) +
            "\nApply these priorities in your next actions and explain the resulting change. "
            "Treat supplied names and claims as user leads, not verified activity evidence. "
            "Preserve the target contract; ask for clarification if instructions conflict with it."}


def status(run_dir):
    state = read_state(run_dir)
    commands, _ = _paths(run_dir)
    return {**state, "queued": [_read(p, {}) for p in sorted(commands.glob("*.json"))
                               if p.stem not in state["receipts"]],
            "boundary_semantics": "Controls take effect between work batches; in-flight work finishes first."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("action", choices=(*ACTIONS, "status"))
    parser.add_argument("--message")
    args = parser.parse_args()
    try:
        result = status(args.run_dir) if args.action == "status" else submit(
            args.run_dir, args.action, args.message)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
