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
    # Use the SAME identity key the agent's consolidation uses (DOI -> PMID ->
    # normalized title), so papers_unique matches real pipeline behavior.
    from enzyme_discovery_agent.tools.literature import _paper_key
    seen, out = set(), []
    for p in papers:
        key = _paper_key(p)
        if key is None:
            out.append(p)  # unkeyable: keep (can't dedup safely)
            continue
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
    """Total tokens for the run, from token_usage.json (the agent writes it)."""
    path = run / "token_usage.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("total_tokens") if isinstance(data.get("total_tokens"), int) else None


def score_run(run_dir: str | Path, benchmark: dict) -> dict:
    """Return literature-exploration metrics for a finished run."""
    run = Path(run_dir)
    events = _load_events(run)
    tool_calls = [e for e in events if e.get("type") == "tool_call"]
    by_iter = Counter(e.get("iteration") for e in tool_calls)
    tools = Counter(e.get("tool") for e in tool_calls)

    papers = _recover_literature_papers(run)
    unique = _dedup(papers)

    def _text(p):
        return ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()

    papers_text = " ".join(_text(p) for p in unique)
    artifacts_text = _all_artifact_text(run)

    # Data-driven recall: score every declared target generically, so the goal
    # can change by editing the benchmark, not this scorer.
    targets = _resolve_targets(benchmark)
    recall: dict[str, dict] = {}
    for name, tgt in targets.items():
        hay = papers_text if tgt.get("scope") == "papers" else artifacts_text
        vals = [v.lower() for v in tgt.get("values", [])]
        found = sorted(v for v in vals if v in hay)
        recall[name] = {
            "found": found,
            "n": len(vals),
            "recall": round(len(found) / len(vals), 3) if vals else None,
            "hit": bool(found),
        }
    primary = benchmark.get("primary_targets") or list(targets.keys())
    found_answer = any(recall[n]["hit"] for n in primary if n in recall)

    # Family coverage: how many distinct amide-forming families the run spanned.
    fam_spec = benchmark.get("family_coverage", {})
    families = {
        fam: {"hit": any(v.lower() in artifacts_text for v in vals),
              "found": sorted(v for v in vals if v.lower() in artifacts_text)}
        for fam, vals in fam_spec.items()
    }
    n_hit = sum(1 for f in families.values() if f["hit"])
    family_coverage = {
        "covered": sorted(f for f, d in families.items() if d["hit"]),
        "missed": sorted(f for f, d in families.items() if not d["hit"]),
        "fraction": round(n_hit / len(families), 3) if families else None,
        "detail": families,
    }

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
        "recall": recall,
        "found_answer": found_answer,
        "tokens": _tokens(run),
    }


def _resolve_targets(benchmark: dict) -> dict:
    """Return recall targets, supporting the legacy answer_* keys too."""
    if benchmark.get("recall_targets"):
        return benchmark["recall_targets"]
    targets = {}
    if benchmark.get("answer_accessions"):
        targets["specific_accession"] = {"scope": "artifacts",
                                         "values": benchmark["answer_accessions"]}
    if benchmark.get("answer_markers"):
        targets["specific_member"] = {"scope": "papers",
                                      "values": benchmark["answer_markers"]}
    return targets
