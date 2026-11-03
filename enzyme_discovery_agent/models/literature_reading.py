"""Durable download and extraction work; a batch size is not a campaign cap."""

from pathlib import Path
from enzyme_discovery_agent.models.family_learning import _read
from enzyme_discovery_agent.state_store import atomic_write_json

TERMINAL_EXTRACTIONS = {"done", "irrelevant"}


def reading_state(run_dir):
    return _read(Path(run_dir) / "agent_state" / "literature_reading.json", {"downloads": {}, "extractions": {}})


def save_reading_state(run_dir, state):
    atomic_write_json(Path(run_dir) / "agent_state" / "literature_reading.json", state)


def pending_reading(run_dir, papers):
    from enzyme_discovery_agent.tools.fulltext import _fulltext_key
    from enzyme_discovery_agent.tools.literature_extraction import _is_non_article
    root = Path(run_dir)
    state = reading_state(root)
    acquired = _read(root / "artifacts" / "fulltext" / "manifest.json", [])
    done = {m.get("key") for m in acquired}
    pending = []
    for paper in papers:
        key = _fulltext_key(paper)
        if key and key not in done and not _is_non_article(paper.get("title", "")):
            if state["downloads"].get(key, {}).get("attempts", 0) < 2:
                pending.append(key)
    for row in acquired:
        key = row.get("key")
        extraction = state["extractions"].get(key, {})
        if row.get("text_file") and extraction.get("status") not in TERMINAL_EXTRACTIONS and extraction.get("attempts", 0) < 2:
            pending.append(key)
    return sorted(set(pending))
