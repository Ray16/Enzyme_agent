"""Reasoning trace: structured JSON event log for analytics."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

_trace_lock = threading.Lock()


def record_trace_event(
    run_dir: Path,
    stage: str,
    event_type: str,
    data: dict,
) -> None:
    """Append a structured event to run_dir/reasoning_trace.json."""
    trace_path = run_dir / "reasoning_trace.json"

    with _trace_lock:
        if trace_path.exists():
            try:
                with open(trace_path) as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError):
                trace = {"events": []}
        else:
            trace = {"events": []}

        trace["events"].append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "event_type": event_type,
            "data": data,
        })

        try:
            with open(trace_path, "w") as f:
                json.dump(trace, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to write reasoning trace: %s", e)
