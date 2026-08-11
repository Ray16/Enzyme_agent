"""Render the canonical event stream (events.jsonl) into a human timeline.

This is the "human view derived from the source of truth": rather than the agent
writing a separate prose log that can drift from the machine record, this reader
*projects* the one canonical stream into something a person reads -- a per-turn
timeline, a candidate funnel, a search-novelty (ortholog) summary, and a cost
breakdown.

The novelty/ortholog summary also falls back to the candidate registry, so it is
informative even on runs recorded before novelty events existed: it collapses the
registry by :func:`family_key` and shows how many distinct families the raw
candidate count actually represents.

Usage:
    python -m enzyme_discovery_agent.render_events <run_dir>
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry,
    load_families,
)
from enzyme_discovery_agent.models.event_log import read_events


def _hms(ts: str) -> str:
    """HH:MM:SS from an ISO timestamp, best-effort."""
    return ts[11:19] if len(ts) >= 19 else ts


def _fmt_tokens(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def render(run_dir: Path) -> str:
    events = read_events(run_dir)
    out: list[str] = []
    w = out.append

    start = next((e for e in events if e.get("type") == "run_start"), {})
    w("=" * 78)
    w(f"RUN  {run_dir.name}")
    if start:
        w(f"  query: {start.get('query', '')}")
        w(f"  model: {start.get('model', '')}   substrate: {start.get('substrate', '')}")
    w("=" * 78)

    # ---- Per-iteration timeline -------------------------------------------
    by_iter: dict[object, list[dict]] = defaultdict(list)
    for e in events:
        by_iter[e.get("iteration", "-")].append(e)

    w("\nTIMELINE")
    for it in sorted((k for k in by_iter if isinstance(k, int))):
        evs = by_iter[it]
        decision = next((e for e in evs if e["type"] == "assistant_decision"), None)
        istart = next((e for e in evs if e["type"] == "iteration_start"), None)
        llm = next((e for e in evs if e["type"] == "llm_call"), None)
        phase = istart.get("phase", "?") if istart else "?"
        header = f"\n  iter {it:>3}  [{phase}]"
        if llm:
            header += f"  ({_fmt_tokens(llm.get('total_tokens', 0))} tok)"
        w(header)
        if decision and decision.get("reasoning"):
            reason = " ".join(decision["reasoning"].split())
            w(f"     think: {reason[:140]}")
        for e in evs:
            t = e["type"]
            if t == "gate":
                flag = "intercept" if e.get("intercept") else "noted"
                w(f"     GATE  {e.get('gate')} [{flag}] — {e.get('message') or ''}")
            elif t == "tool_call":
                nov = e.get("novelty")
                tail = ""
                if isinstance(nov, dict):
                    tail = (f"  →  {nov['registered']} reg / "
                            f"{nov['distinct_families']} fam / "
                            f"{nov['new_families']} new-fam / "
                            f"{nov['redundant_orthologs']} orthologs")
                elif e.get("count"):
                    tail = f"  [{e.get('count')} hits]"
                q = (e.get("args", {}) or {}).get("query") or \
                    (e.get("args", {}) or {}).get("gene_name") or ""
                w(f"     tool  {e.get('tool')}({str(q)[:44]}) {e.get('status')}{tail}")
            elif t == "candidates_screened":
                w(f"     SCREEN  {e.get('newly_screened')} newly screened "
                  f"({e.get('distinct_genes')} genes, {e.get('fold_families')} folds, "
                  f"{e.get('folds_held_at_floor')} folds held at floor)")
            elif t == "candidates_evaluated":
                w(f"     EVAL  {e.get('evaluated')}/{e.get('screened_pending')} evaluated")

    # ---- Candidate funnel --------------------------------------------------
    registry = load_candidate_registry(run_dir).get("candidates", {})
    status_counts: Counter = Counter()
    for info in registry.values():
        seen = {s.get("status") for s in info.get("statuses", [])}
        for st in ("discovered", "screened", "evaluated", "ranked"):
            if st in seen:
                status_counts[st] += 1
    w("\n" + "-" * 78)
    w("CANDIDATE FUNNEL")
    w(f"  discovered {status_counts['discovered']:>4}  →  "
      f"screened {status_counts['screened']:>4}  →  "
      f"evaluated {status_counts['evaluated']:>4}  →  "
      f"ranked {status_counts['ranked']:>4}")

    # ---- Ortholog / family collapse (works on any run) --------------------
    # Uses the registry's family index: InterPro family for evaluated candidates,
    # gene/name fallback otherwise.  load_families computes it for older runs.
    families = load_families(run_dir)
    if families:
        n_cand = sum(f.get("n_members", 0) for f in families.values())
        n_fam = len(families)
        redundancy = 1 - n_fam / n_cand if n_cand else 0.0
        n_ipr = sum(1 for f in families.values() if f.get("interpro_id"))
        w("\nSEARCH NOVELTY  (registry family collapse)")
        w(f"  {n_cand} candidates collapse to {n_fam} distinct families "
          f"— {redundancy*100:.0f}% are redundant orthologs "
          f"({n_ipr} families resolved to an InterPro id)")
        w("  largest families (candidate = 1 enzyme, N = ortholog copies):")
        for f in list(families.values())[:8]:
            if f.get("n_members", 0) > 1:
                w(f"     {f['n_members']:>3}×  {f.get('family')}")

    # From novelty events, if present, an honest per-search breakdown.
    reg_events = [e for e in events if e.get("type") == "candidates_registered"]
    if reg_events:
        tot_reg = sum(e.get("registered", 0) for e in reg_events)
        tot_ortho = sum(e.get("redundant_orthologs", 0) for e in reg_events)
        tot_newfam = sum(e.get("new_families", 0) for e in reg_events)
        w(f"  across {len(reg_events)} searches: {tot_reg} registered, "
          f"{tot_newfam} new families, {tot_ortho} redundant orthologs")

    # ---- Cost --------------------------------------------------------------
    llm_calls = [e for e in events if e.get("type") == "llm_call"]
    done = next((e for e in events if e.get("type") == "run_complete"), None)
    if llm_calls or done:
        w("\n" + "-" * 78)
        w("COST")
        if done:
            w(f"  total {_fmt_tokens(done.get('total_tokens', 0))} tokens "
              f"({_fmt_tokens(done.get('prompt_tokens', 0))} prompt / "
              f"{_fmt_tokens(done.get('completion_tokens', 0))} completion) "
              f"over {done.get('iterations_used', '?')} iterations")
        elif llm_calls:
            w(f"  {_fmt_tokens(sum(e.get('total_tokens', 0) for e in llm_calls))} tokens "
              f"over {len(llm_calls)} LLM calls (run not finalized)")

    w("=" * 78)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    run_dir = Path(argv[0])
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}")
        return 1
    print(render(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
