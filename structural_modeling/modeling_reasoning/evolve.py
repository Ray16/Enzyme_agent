#!/usr/bin/env python3
"""Self-evolution of the structural-modeling reasoning knowledge.

The compiler's decisions are only as good as the vendored literature rules. This
tool closes the improvement loop: as the reasoning module extracts more
computational-enzyme-design PDFs and re-synthesizes its candidate rules, ``evolve``
pulls the richer synthesis into ``knowledge/cofolding_rules.json``, appends a
measurable version record, and reports how the reasoning changed. Because
``resolve_handoff`` always reads whatever is in ``knowledge/`` and matches rules by
decision point and feature guard, more (and better-corroborated) papers
automatically yield richer, better-grounded handoff resolutions with no code edit.

Two modes:

* default (pull): re-vendor from the reasoning repo's current synthesis, version
  it, and diff against the previous snapshot. Offline, deterministic.
* ``--run-pipeline``: first (re)run the reasoning extraction so the synthesis
  reflects newly queried PDFs, then pull. The extraction command is supplied via
  ``--pipeline-cmd`` (it needs the LLM endpoint), keeping this module decoupled
  from the reasoning repo's evolving CLI.

The version history in ``knowledge/VERSIONS.jsonl`` is the audit trail: each line
records the corpus size, rule/decision-point coverage, and checksum, so any past
resolved handoff stays reconstructable and reasoning growth is quantifiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(HERE, "knowledge")
VENDORED = os.path.join(KNOWLEDGE_DIR, "cofolding_rules.json")
VERSIONS = os.path.join(KNOWLEDGE_DIR, "VERSIONS.jsonl")
DEFAULT_SYNTHESIS = (
    "/homes/rzhu/kg-memory-system/reasoning/results/v2/cofolding_batch/reasoning_synthesis.json"
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(document: dict) -> dict:
    rules = document.get("candidate_rules") or []
    papers = {
        e.get("paper_id")
        for rule in rules
        for e in (rule.get("evidence") or [])
        if e.get("paper_id")
    }
    return {
        "rule_count": len(rules),
        "decision_points": sorted({r.get("decision_point") for r in rules if r.get("decision_point")}),
        "papers": sorted(p for p in papers if p),
        "corpus": document.get("corpus", {}),
        "schema_version": document.get("schema_version"),
    }


def _rule_index(document: dict) -> dict:
    """decision_point -> {guidance -> set(papers)} for structural diffing."""
    index: dict = {}
    for rule in document.get("candidate_rules") or []:
        point = rule.get("decision_point")
        guidance = rule.get("guidance")
        papers = {e.get("paper_id") for e in (rule.get("evidence") or []) if e.get("paper_id")}
        index.setdefault(point, {})[guidance] = papers
    return index


def diff(old: dict | None, new: dict) -> dict:
    """What improved from the previous snapshot to the new one."""
    old_stats = stats(old) if old else {"decision_points": [], "papers": [], "rule_count": 0}
    new_stats = stats(new)
    old_index = _rule_index(old) if old else {}
    new_index = _rule_index(new)

    strengthened = []
    new_rules = []
    for point, guidances in new_index.items():
        for guidance, papers in guidances.items():
            before = (old_index.get(point) or {}).get(guidance)
            if before is None:
                new_rules.append({"decision_point": point, "guidance": guidance, "papers": sorted(papers)})
            elif len(papers) > len(before):
                strengthened.append(
                    {
                        "decision_point": point,
                        "guidance": guidance,
                        "papers_before": len(before),
                        "papers_after": len(papers),
                        "crossed_to_multi_paper": len(before) < 2 <= len(papers),
                    }
                )
    return {
        "new_decision_points": sorted(set(new_stats["decision_points"]) - set(old_stats["decision_points"])),
        "new_papers": sorted(set(new_stats["papers"]) - set(old_stats["papers"])),
        "rule_count_delta": new_stats["rule_count"] - old_stats["rule_count"],
        "new_rules": new_rules,
        "strengthened_rules": strengthened,
    }


def _load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def evolve(
    synthesis_path: str = DEFAULT_SYNTHESIS,
    *,
    run_pipeline_cmd: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Pull the latest synthesis into the vendored knowledge and version it."""
    if run_pipeline_cmd:
        # (Re)extract from newly queried PDFs. Requires the reasoning repo + LLM endpoint.
        subprocess.run(run_pipeline_cmd, check=True)

    if not os.path.exists(synthesis_path):
        raise FileNotFoundError(f"reasoning synthesis not found: {synthesis_path}")
    new_doc = _load(synthesis_path)
    old_doc = _load(VENDORED) if os.path.exists(VENDORED) else None
    changes = diff(old_doc, new_doc)
    new_stats = stats(new_doc)

    if dry_run:
        return {"changes": changes, "stats": new_stats, "written": False}

    shutil.copyfile(synthesis_path, VENDORED)
    sha = _sha256(VENDORED)
    with open(os.path.join(KNOWLEDGE_DIR, "SOURCE.sha256"), "w") as handle:
        handle.write(f"{sha}  reasoning_synthesis.json\n")
    record = {
        "evolved_at": datetime.now(timezone.utc).isoformat(),
        "source": os.path.abspath(synthesis_path),
        "sha256": sha,
        "rule_count": new_stats["rule_count"],
        "decision_points": new_stats["decision_points"],
        "paper_count": len(new_stats["papers"]),
        "corpus": new_stats["corpus"],
        "changes": changes,
    }
    with open(VERSIONS, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return {"changes": changes, "stats": new_stats, "written": True, "sha256": sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis", default=DEFAULT_SYNTHESIS, help="reasoning synthesis JSON to pull")
    parser.add_argument("--dry-run", action="store_true", help="report the diff; do not write")
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="first run --pipeline-cmd to (re)extract PDFs, then pull",
    )
    parser.add_argument(
        "--pipeline-cmd",
        help="shell command that regenerates the synthesis (needs the LLM endpoint)",
    )
    args = parser.parse_args(argv)

    cmd = None
    if args.run_pipeline:
        if not args.pipeline_cmd:
            parser.error("--run-pipeline requires --pipeline-cmd")
        cmd = args.pipeline_cmd.split()

    result = evolve(args.synthesis, run_pipeline_cmd=cmd, dry_run=args.dry_run)
    changes = result["changes"]
    stats_ = result["stats"]
    print(f"knowledge: {stats_['rule_count']} rules over {len(stats_['papers'])} papers")
    print(f"decision points: {', '.join(stats_['decision_points'])}")
    print(f"rule_count_delta: {changes['rule_count_delta']:+d}")
    if changes["new_decision_points"]:
        print(f"NEW decision points: {', '.join(changes['new_decision_points'])}")
    if changes["new_papers"]:
        print(f"NEW papers: {', '.join(changes['new_papers'])}")
    for item in changes["strengthened_rules"]:
        marker = " (now cross-paper)" if item["crossed_to_multi_paper"] else ""
        print(f"STRENGTHENED [{item['decision_point']}] {item['papers_before']}->{item['papers_after']} papers{marker}")
    for item in changes["new_rules"]:
        print(f"NEW RULE [{item['decision_point']}] {len(item['papers'])} paper(s): {item['guidance'][:70]}")
    print("written" if result.get("written") else "dry-run (not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
