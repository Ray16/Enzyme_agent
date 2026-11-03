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
from enzyme_discovery_agent.run_layout import run_file

_LIT_TOOLS = {
    "search_pubmed", "search_openalex", "search_europepmc",
    "search_semantic_scholar", "search_crossref", "deep_literature_search",
}


def _load_events(run: Path) -> list[dict]:
    path = run_file(run, "events.jsonl")
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
            result = json.loads(Path(p).read_text()).get("result", {})
            if result.get("status") in {"error", "skipped"}:
                continue
            data = result.get("data", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if isinstance(data, list):
            papers.extend(x for x in data if isinstance(x, dict))
        elif isinstance(data, dict):
            for name in ("seed_papers", "reference_papers"):
                papers.extend(x for x in data.get(name, []) if isinstance(x, dict))
    return papers


def _dedup(papers: list[dict]) -> list[dict]:
    # Use the SAME identity key the agent's consolidation uses (DOI -> PMID ->
    # normalized title), so papers_unique matches real pipeline behavior.
    from enzyme_discovery_agent.tools.literature import dedupe_and_merge
    return dedupe_and_merge(papers)[0]


def _all_artifact_text(run: Path) -> str:
    """Search only data returned by nonfailed tools, never arguments or errors."""
    excluded = {"args", "query", "queries", "search_query", "rerank_query",
                "target_query", "error", "errors", "suggestions", "summary"}

    def evidence_only(value):
        if isinstance(value, dict):
            return {key: evidence_only(item) for key, item in value.items()
                    if str(key).lower() not in excluded}
        if isinstance(value, list):
            return [evidence_only(item) for item in value]
        return value

    chunks = []
    for p in glob.glob(str(run / "artifacts" / "**" / "*.json"), recursive=True):
        try:
            payload = json.loads(Path(p).read_text())
            result = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(result, dict) and result.get("status") not in {"error", "skipped"}:
                chunks.append(json.dumps(evidence_only(result.get("data")), default=str))
        except (OSError, ValueError):
            continue
    return " ".join(chunks).lower()


def _candidate_text(run: Path) -> str:
    """Lowercased metadata of candidates currently eligible after evaluation."""
    from enzyme_discovery_agent.models.candidate_registry import discovery_status, load_candidate_registry
    try:
        candidates = load_candidate_registry(run, strict=True).get("candidates", {})
    except (OSError, ValueError):
        return ""
    eligible = []
    for accession, entry in candidates.items():
        metadata = entry.get("metadata") or {}
        validation = metadata.get("claim_validation") or {}
        if (discovery_status(entry) in {"evaluated", "ranked"}
                and validation.get("promotion_eligible") is True
                and metadata.get("is_enzyme") is not False):
            eligible.append({"accession": accession, "metadata": metadata})
    return json.dumps(eligible, default=str).lower()


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
    path = run_file(run, "token_usage.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("total_tokens") if isinstance(data.get("total_tokens"), int) else None


def _stage_metrics(run: Path, benchmark: dict) -> dict:
    """Measure candidate retention and known-positive rank at each durable stage."""
    from enzyme_discovery_agent.models.candidate_registry import discovery_status
    try:
        registry = json.loads((run_file(run, "candidate_registry.json")).read_text())
    except (OSError, json.JSONDecodeError):
        registry = {}
    entries = list((registry.get("candidates") or {}).values())
    stages = ("discovered", "screened", "evaluated", "eligible", "ranked")
    stage_sets = {stage: set() for stage in stages}
    for entry in entries:
        accession = str(entry.get("accession") or "").casefold()
        seen = {s.get("status") for s in entry.get("statuses") or []}
        for stage in ("discovered", "screened", "evaluated"):
            if stage in seen:
                stage_sets[stage].add(accession)
        metadata = entry.get("metadata") or {}
        validation = metadata.get("claim_validation") or {}
        current = discovery_status(entry)
        if (current in {"evaluated", "ranked"} and
                validation.get("promotion_eligible") is True and
                metadata.get("is_enzyme") is not False):
            stage_sets["eligible"].add(accession)
            if current == "ranked":
                stage_sets["ranked"].add(accession)

    positives = {
        str(value).casefold()
        for name in benchmark.get("primary_targets") or []
        for value in (benchmark.get("recall_targets", {}).get(name, {}).get("values") or [])
        if re_full_accession(str(value))
    }
    retention = {}
    previous = None
    for stage in stages:
        count = len(stage_sets[stage])
        retention[stage] = {
            "count": count,
            "from_previous": (
                round(count / previous, 3) if previous else None
            ),
            "known_positives": sorted(stage_sets[stage] & positives),
        }
        previous = count

    ranks = {}
    try:
        ranking = json.loads((run_file(run, "pass1_ranking.json")).read_text())
    except (OSError, json.JSONDecodeError):
        ranking = []
    for position, item in enumerate(ranking, 1):
        accession = str(item.get("accession") or "").casefold()
        if accession in positives and accession in stage_sets["eligible"]:
            ranks[accession] = int(item.get("rank") or position)
    return {
        "retention": retention,
        "known_positive_ranks": ranks,
        "best_known_positive_rank": min(ranks.values()) if ranks else None,
    }


def re_full_accession(value: str) -> bool:
    """Recognize accession-like benchmark labels without importing agent state."""
    import re
    return bool(re.fullmatch(
        r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})",
        value.upper(),
    ))


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

    # Family coverage, measured at TWO stages so the drop-off is visible:
    #   searched   - family appears anywhere the agent retrieved (breadth of search)
    #   candidates - family survives into the ranked candidate list (what's kept)
    # families "dropped" (searched but not in candidates) localize the leak to the
    # search->candidate SELECTION step.
    fam_spec = benchmark.get("family_coverage", {})
    cand_text = _candidate_text(run)

    def _cov(hay):
        hit = {f: any(v.lower() in hay for v in vals) for f, vals in fam_spec.items()}
        n = sum(hit.values())
        return {
            "covered": sorted(f for f, h in hit.items() if h),
            "missed": sorted(f for f, h in hit.items() if not h),
            "fraction": round(n / len(fam_spec), 3) if fam_spec else None,
        }

    searched, candidates = _cov(artifacts_text), _cov(cand_text)
    family_coverage = {
        "searched": searched,
        "candidates": candidates,
        # searched but did NOT survive into candidates = selection drop-off
        "dropped": sorted(set(searched["covered"]) - set(candidates["covered"])),
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
        "family_coverage": family_coverage,
        "candidate_stages": _stage_metrics(run, benchmark),
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
