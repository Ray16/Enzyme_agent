"""Score a completed agent run's LITERATURE EXPLORATION against a benchmark.

Reads only the run's on-disk artifacts (events.jsonl + artifacts/), so it works
on any finished run without re-running the agent. The metrics answer: did the
agent search broadly and in parallel, avoid redundancy, consolidate, fetch full
text — and, crucially, did the answer-family survive into what it retrieved.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

_LIT_TOOLS = {
    "search_pubmed", "search_openalex", "search_europepmc",
    "search_semantic_scholar", "search_crossref", "deep_literature_search",
}


def _load_events(run: Path) -> list[dict]:
    path = run / "events.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _recover_literature_papers(run: Path) -> list[dict]:
    papers: list[dict] = []
    for p in sorted(glob.glob(str(run / "artifacts" / "literature" / "*.json"))):
        try:
            data = json.loads(Path(p).read_text()).get("result", {}).get("data", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if isinstance(data, list):
            papers.extend(x for x in data if isinstance(x, dict))
    return papers


def _dedup(papers: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in papers:
        key = ((p.get("doi") or "").lower(), str(p.get("pmid") or ""),
               (p.get("title") or "").lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _all_artifact_text(run: Path) -> str:
    """Lowercased concatenation of every artifact JSON (recall search space)."""
    chunks = []
    for p in glob.glob(str(run / "artifacts" / "**" / "*.json"), recursive=True):
        try:
            chunks.append(Path(p).read_text())
        except OSError:
            continue
    nb = run / "artifacts" / "notebook.json"
    if nb.exists():
        try:
            chunks.append(nb.read_text())
        except OSError:
            pass
    return " ".join(chunks).lower()


def _fulltext_manifest(run: Path) -> list[dict]:
    path = run / "artifacts" / "fulltext" / "manifest.json"
    if not path.exists():
        return []
    try:
        m = json.loads(path.read_text())
        return m if isinstance(m, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _tokens(run: Path) -> int | None:
    """Best-effort total-token read from the provenance manifest."""
    path = run / "manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("total_tokens", "tokens"):
        if isinstance(data.get(key), int):
            return data[key]
    usage = data.get("token_usage") or data.get("usage") or {}
    if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
        return usage["total_tokens"]
    return None


def score_run(run_dir: str | Path, benchmark: dict) -> dict:
    """Return literature-exploration metrics for a finished run."""
    run = Path(run_dir)
    events = _load_events(run)
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    by_iter = Counter(e.get("iteration") for e in tool_calls)
    tools = Counter(e.get("tool") for e in tool_calls)

    papers = _recover_literature_papers(run)
    unique = _dedup(papers)

    markers = [m.lower() for m in benchmark.get("answer_markers", [])]
    accessions = [a.lower() for a in benchmark.get("answer_accessions", [])]

    def _text(p):
        return ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()

    marker_hits = sorted({m for p in unique for m in markers if m in _text(p)})
    artifact_text = _all_artifact_text(run)
    acc_hits = sorted({a for a in accessions if a in artifact_text})

    fulltext = _fulltext_manifest(run)
    sources_used = sorted(t for t in tools if t in _LIT_TOOLS)

    return {
        "iterations": max([e.get("iteration", 0) for e in tool_calls] or [0]),
        "total_tool_calls": len(tool_calls),
        "max_fanout": max(by_iter.values() or [0]),
        "literature_sources_used": sources_used,
        "n_sources_used": len(sources_used),
        "consolidate_calls": tools.get("consolidate_literature", 0),
        "fetch_full_text_calls": tools.get("fetch_full_text", 0),
        "deep_search_calls": tools.get("deep_literature_search", 0),
        "papers_raw": len(papers),
        "papers_unique": len(unique),
        "redundancy": round(1 - len(unique) / len(papers), 3) if papers else 0.0,
        "fulltext_downloaded": len(fulltext),
        "answer_markers_found": marker_hits,
        "answer_marker_recall": (round(len(marker_hits) / len(markers), 3)
                                 if markers else None),
        "answer_accessions_found": acc_hits,
        "answer_accession_recall": (round(len(acc_hits) / len(accessions), 3)
                                    if accessions else None),
        "found_answer": bool(marker_hits or acc_hits),
        "tokens": _tokens(run),
    }
