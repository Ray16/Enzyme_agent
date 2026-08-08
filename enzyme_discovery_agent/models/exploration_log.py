"""Exploration log: append-only markdown audit trail of agent actions."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

_exploration_log_lock = threading.Lock()


def append_exploration_log(
    run_dir: Path,
    entry_type: str,
    title: str,
    details: dict | str,
) -> None:
    """Append a timestamped markdown entry to run_dir/exploration_log.md.

    Entry types: SEARCH, CANDIDATE_DISCOVERED, SCREEN, EVALUATION,
    DECISION, NOTEBOOK_ARCHIVE
    """
    log_path = run_dir / "exploration_log.md"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    lines = [f"### [{now}] {entry_type} — {title}\n"]
    if isinstance(details, dict):
        for k, v in details.items():
            lines.append(f"- **{k}**: {v}\n")
    else:
        lines.append(f"{details}\n")
    lines.append("\n")

    entry = "".join(lines)

    with _exploration_log_lock:
        try:
            with open(log_path, "a") as f:
                f.write(entry)
        except Exception as e:
            logger.warning("Failed to write exploration log: %s", e)
