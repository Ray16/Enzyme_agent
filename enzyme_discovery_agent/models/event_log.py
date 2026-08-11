"""Canonical run event stream: the single append-only, machine-readable record
of what the agent did, *why* it did it, and at what cost.

Every other log surface -- ``console_log.txt`` (human timeline) and
``exploration_log.md`` (curated narrative) -- is a *projection*.  This JSONL
stream is the source of truth a finished run can be replayed and audited from,
so it must capture decisions and outcomes, not only actions.

Design mirrors :func:`models.exploration_log.append_exploration_log`: a
module-level, thread-safe, best-effort writer that gates, the registry, and the
agent loop all call with only a ``run_dir`` -- no client object threaded through
the call graph.  It never raises into the agent loop (telemetry must not be able
to fail a campaign) and depends only on the standard library.

Event envelope (every event): ``{seq, ts, type, ...fields}``.
  * ``seq`` -- monotonic per-run ordinal, seeded from the existing file so it
    stays stable across a resume / process restart.
  * ``ts``  -- ISO-8601 local timestamp (matches the pre-existing events).
  * ``type`` -- the event kind; downstream readers filter on it.

Adding new event *types* or new *fields* is always safe: consumers
(``eval/metrics.py``, ``rerun_report.py``, ``render_events.py``) filter by
``type`` and ignore unknown fields.  Never rename or drop a field on an existing
event type without updating those readers.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("enzyme_agent")

_event_lock = threading.Lock()
# Per-run monotonic ordinal.  Keyed by the resolved run directory so two runs in
# one process (or a resumed run) never share or reset each other's counter.
_seq_counters: dict[str, int] = {}


def _next_seq(run_dir: Path, run_key: str) -> int:
    """Return the next per-run sequence number, seeding from disk on first use.

    Seeding from the existing line count keeps ``seq`` continuous when a run is
    resumed in a fresh process, so the stream never restarts its ordinal mid-run.
    Caller holds ``_event_lock``.
    """
    if run_key not in _seq_counters:
        existing = 0
        path = run_dir / "events.jsonl"
        if path.exists():
            try:
                with path.open() as f:
                    existing = sum(1 for _ in f)
            except OSError:
                existing = 0
        _seq_counters[run_key] = existing
    _seq_counters[run_key] += 1
    return _seq_counters[run_key]


def append_event(run_dir: str | Path, event_type: str, **fields) -> None:
    """Append one event to ``run_dir/events.jsonl``.

    Stamps a monotonic ``seq`` and an ISO ``ts``; passes ``fields`` through
    verbatim so a caller attaches whatever context the event needs (iteration,
    tool, novelty counts, token usage, ...).  Best-effort: any failure is logged
    and swallowed so a telemetry problem can never abort the run.
    """
    run_dir = Path(run_dir)
    run_key = str(run_dir.resolve())
    path = run_dir / "events.jsonl"
    try:
        with _event_lock:
            event = {
                "seq": _next_seq(run_dir, run_key),
                "ts": datetime.now().isoformat(),
                "type": event_type,
                **fields,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:  # never break the loop on a logging failure
        logger.warning("Failed to append event %s: %s", event_type, e)


def read_events(run_dir: str | Path) -> list[dict]:
    """Read ``run_dir/events.jsonl`` into a list of event dicts (order preserved).

    Malformed lines are skipped rather than raising, so a partially-written tail
    (e.g. from an interrupted run) never breaks a reader.
    """
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
