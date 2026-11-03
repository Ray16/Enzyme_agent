"""Live candidate tables and lifecycle plots from saved campaign records only."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import html
import io
import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import time

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, discovery_status
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.models.expansion import seed_evidence
from enzyme_discovery_agent.models.evaluations import current_evaluations,priority_concerns
from enzyme_discovery_agent.output.report_generator import (
    _accommodation_score, _deterministic_mechanistic_ranking, _evidence_level,
    _family_relevance_partition, _family_rank_map,
)
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def snapshot(run_dir):
    root = Path(run_dir)
    candidates = load_candidate_registry(root, strict=True).get("candidates", {})
    from enzyme_discovery_agent.models.exploration_state import exploration_state
    from enzyme_discovery_agent.tools.literature import _recover_literature_papers, dedupe_and_merge
    frontier = exploration_state(root, registry=candidates)
    data = {key: seed_evidence(key, entry) for key, entry in candidates.items()}
    evaluations = current_evaluations(root,[{"data": data[key]} for key, entry in candidates.items()
                                          if discovery_status(entry) in {'evaluated','ranked'}])
    stopped, supported, reasons = _family_relevance_partition(root, evaluations)
    ranking = _deterministic_mechanistic_ranking(
        [e for e in evaluations if e["data"]["accession"] not in stopped], {},
        family_rank_by_acc=_family_rank_map(stopped, supported))
    ranks = {row["accession"]: row for row in ranking}
    rows, transitions = [], []
    for key, entry in candidates.items():
        d, rank = data[key], ranks.get(key, {})
        accession = entry.get("accession") or key
        status = discovery_status(entry) or "unknown"
        blocking = (d.get("claim_validation") or {}).get("blocking_claims") or []
        reason = d.get("rejection_reason") or reasons.get(key) or "; ".join(d.get("critical_warnings") or [])
        reason = reason or ("Missing claims: " + ", ".join(map(str, blocking)) if blocking else "")
        fit = _accommodation_score(d)
        assessment=d.get('priority_assessment') or {}
        if assessment:
            reason=f"Shortlist {assessment.get('decision')}: {assessment.get('reason')}"
        elif priority_concerns(d):
            reason='Known priority concern: '+priority_concerns(d)[0]
        rows.append({"rank": rank.get("rank", ""), "status": status, "accession": accession,
                     "gene": (d.get("gene_names") or [""])[0], "enzyme": d.get("literature_name") or entry.get("enzyme_name") or d.get("protein_name") or key,
                     "database_name": "" if d.get("is_literature_lead") else d.get("protein_name") or "", "identity_status": d.get("identity_resolution_status") or "sequence_record",
                     "family": d.get("interpro_family") or d.get("interpro_family_id") or "",
                     "family_id": d.get("interpro_family_id") or d.get("screening_family_id") or "",
                     "substrate_fit": round(fit, 3) if fit is not None else "",
                     "native_evidence": _evidence_level(d),
                     "reason": reason or rank.get("rationale") or "Awaiting evidence or evaluation",
                     "source": d.get("source_tool") or "", "updated_at": entry.get("last_updated") or ""})
        for transition in entry.get("statuses") or []:
            if transition.get("timestamp"):
                transitions.append({"timestamp": transition["timestamp"], "status": transition.get("status"),
                                    "accession": accession, "label": f"{(d.get('gene_names') or [key])[0]} · {accession}"})
    rows.sort(key=lambda r: (0 if r["rank"] else 2 if r["status"] == "rejected" else 1,
                            r["rank"] or 0, r["accession"]))
    transitions.sort(key=lambda r: (datetime.fromisoformat(r["timestamp"]).timestamp(), r["accession"]))
    events = read_events(root)
    research = [e for e in events if e["type"] in {"tool_call", "gate", "candidates_registered", "candidates_evaluated", "candidates_screened"}]
    searches = [e for e in events if e.get("type") == "tool_call" and
                str(e.get("tool") or "").startswith(("search_", "follow_", "deep_literature"))]
    recent_searches = [{"time": e.get("ts"), "tool": e.get("tool"),
                        "query": (e.get("args") or {}).get("query") or (e.get("args") or {}).get("doi") or "",
                        "status": e.get("status"), "count": e.get("count"), "seconds": e.get("elapsed_s")}
                       for e in searches[-30:]]
    decisions = [{"time": e.get("ts"), "type": e.get("type"),
                  "detail": e.get("reasoning") or e.get("message") or e.get("reason") or "",
                  "tools": e.get("tools") or []}
                 for e in events if e.get("type") in {"assistant_decision", "gate", "discovery_recovery", "llm_model_switch",
                                                      "research_proposal_rejected", "research_review_deferred"}][-20:]
    reading = _read(root / "agent_state" / "literature_reading.json", {})
    extractions = (reading.get("extractions") or {})
    papers = _recover_literature_papers(str(root))
    literature_linked = 0
    supported_literature_claims = 0
    for entry in candidates.values():
        metadata = entry.get("metadata") or {}
        literature_linked += bool(metadata.get("literature_evidence") or metadata.get("literature_name") or metadata.get("is_literature_lead"))
        validation = metadata.get("claim_validation") or {}
        supported_literature_claims += any(isinstance(c, dict) and c.get("verdict") == "supported" and
            (c.get("evidence") or {}).get("source") == "primary_literature" for c in validation.get("claims") or [])
    literature = {"paper_records": len(papers), "unique_papers": len(dedupe_and_merge(papers)[0]),
                  "full_texts": len(_read(root / "artifacts/fulltext/manifest.json", [])),
                  "extractions": dict(Counter((row or {}).get("status", "unknown") for row in extractions.values())),
                  "literature_linked_candidates_or_leads": literature_linked,
                  "candidates_with_supported_literature_claims": supported_literature_claims}
    context = _read(run_file(root, "target_context.json"), {})
    manifest = _read(root / "manifest.json", {})
    usage = _read(run_file(root, "token_usage.json"), {})
    invocation = _read(root / "agent_state/invocation_resources.json", {})
    experiment_budget = _read(root / "agent_state/experiment_budget.json", {})
    prior_tokens = experiment_budget.get("prior_attempt_tokens")
    experiment_limit = experiment_budget.get("total_token_allowance")
    if not (isinstance(prior_tokens, int) and prior_tokens >= 0 and
            isinstance(experiment_limit, int) and experiment_limit > 0):
        prior_tokens, experiment_limit = None, None
    handoff = _read(root / "structural_modeling_handoff.json", {})
    intake = _read(root / "structural_intake_check/intake_audit.json", {})
    handoff_hash = (handoff.get("integrity") or {}).get("snapshot_sha256")
    modeling = {"handoff_schema": handoff.get("schema_version"),
                "handoff_discovery_status": (handoff.get("producer") or {}).get("discovery_status"),
                "handoff_may_be_stale": manifest.get("status") == "running" and bool(handoff),
                "admitted": len(handoff.get("candidates") or []),
                "unresolved_proposals": sum((c.get("modeling_input") or {}).get("status") != "ready"
                                            for c in handoff.get("candidates") or []),
                "intake_status": ("current" if handoff_hash and intake.get("handoff_snapshot_sha256") == handoff_hash
                                  else "stale_or_unavailable"),
                "intake_candidates": intake.get("candidates") or [],
                "gpu_jobs_at_intake": intake.get("gpu_jobs")}
    latest_model = next((e for e in reversed(events) if e.get("type") in {"llm_call", "llm_model_switch"}), {})
    active_model = (latest_model.get("to_model") if latest_model.get("type") == "llm_model_switch" else
                    ":".join(str(latest_model.get(k)) for k in ("backend", "model") if latest_model.get(k)))
    warnings_path = root / "agent_state" / "warnings.log"
    volume_fallbacks = None
    if warnings_path.exists():
        with warnings_path.open(errors="replace") as handle:
            volume_fallbacks = sum("3D embedding failed" in line for line in handle)
    steering = _read(root / "agent_state/steering/state.json", {})
    owner = _read(root / "agent_state/discovery_owner.json", {})
    from enzyme_discovery_agent.models.research_decisions import decision_summary
    alive = None
    if owner.get("hostname") == socket.gethostname() and owner.get("pid"):
        proc = Path("/proc") / str(owner["pid"])
        if proc.exists():
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            alive = fields[0] != "Z" and fields[19] == str(owner.get("process_start_ticks"))
        else:
            alive = False
    return {"run": root.name, "query": context.get("query") or manifest.get("query") or "",
            "status": "paused" if steering.get("paused") else manifest.get("status", "unknown"),
            "worker_alive": alive, "candidate_count": len(rows), "ranked_count": len(ranking),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "model": active_model or usage.get("model") or manifest.get("model"),
            "tokens": usage.get("total_tokens"), "token_accounting_complete": usage.get("accounting_complete"),
            "experiment_tokens": (usage.get("total_tokens") + prior_tokens
                                  if isinstance(usage.get("total_tokens"), int) and prior_tokens is not None else None),
            "experiment_token_budget": experiment_limit,
            "invocation_tokens": max(0, usage.get("total_tokens", 0) - invocation.get("starting_tokens", 0)) if usage.get("total_tokens") is not None else None,
            "token_budget": invocation.get("token_budget", (manifest.get("args") or {}).get("token_budget")),
            "quality_warnings": {"volume_estimate_fallbacks": volume_fallbacks,
                "note": "Fallback estimates are not equivalent to a successful 3D volume calculation; inspect affected comparisons."},
            "literature": literature, "searches": recent_searches, "decisions": decisions,
            "reasoning": decision_summary(root),
            "modeling": modeling,
            "frontier": frontier,
            "last_research_event": research[-1].get("ts") if research else None,
            "generated_at": datetime.now(timezone.utc).isoformat(), "rows": rows, "transitions": transitions}


def trajectory_figure(state):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.3, .7],
                        vertical_spacing=.08, subplot_titles=["Candidate coverage", "Candidate lifecycle"])
    transitions = state["transitions"]
    seen, evaluated, rejected = set(), set(), set()
    curves = {"Registered": [], "Evaluated": [], "Rejected": []}
    colors = {"discovered": "#4477AA", "screened": "#66CCEE", "evidence_pending": "#AA3377",
              "evaluated": "#228833", "ranked": "#228833", "rejected": "#CC3311", "reopened": "#EE7733"}
    for step, transition in enumerate(transitions, 1):
        accession, status = transition["accession"], transition["status"]
        seen.add(accession)
        if status in {"evaluated", "ranked"}: evaluated.add(accession)
        if status == "rejected": rejected.add(accession)
        for name, values in zip(curves, (seen, evaluated, rejected)):
            curves[name].append(len(values))
    for name, y in curves.items():
        fig.add_trace(go.Scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines", name=name,
                                line_shape="hv"), row=1, col=1)
    groups = {}
    for step, transition in enumerate(transitions, 1):
        groups.setdefault(transition["status"], []).append((step, transition))
    for status, points in groups.items():
        fig.add_trace(go.Scatter(x=[step for step, _ in points], y=[t["label"] for _, t in points],
                                customdata=[[t["timestamp"], t["accession"]] for _, t in points],
                                mode="markers", name=status.replace("_", " "),
                                marker={"size": 9, "color": colors.get(status, "#777777")},
                                hovertemplate="%{customdata[1]}<br>" + html.escape(status) +
                                              "<br>%{customdata[0]}<extra></extra>"), row=2, col=1)
    fig.update_layout(template="plotly_white", height=min(1800, max(650, state["candidate_count"] * 19 + 300)),
                      margin={"l": 180, "r": 30, "t": 65, "b": 70}, legend={"orientation": "h"})
    fig.update_xaxes(title_text="Recorded lifecycle transition · hover for timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Distinct candidates · cumulative", row=1, col=1)
    return fig


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def publish(root):
    state = snapshot(root)
    columns = list(state["rows"][0]) if state["rows"] else ["rank", "status", "accession", "enzyme"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader(); writer.writerows(state["rows"])
    _atomic_text(root / "candidate_ranking.csv", output.getvalue())
    display = ["rank", "status", "identity_status", "gene", "accession", "enzyme", "database_name", "family", "substrate_fit", "native_evidence", "reason"]
    def table(rows):
        head = "".join(f"<th>{html.escape(k.replace('_', ' '))}</th>" for k in display)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in display) + "</tr>" for row in rows)
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    def records_table(rows, columns):
        head = "".join(f"<th>{html.escape(name.replace('_', ' '))}</th>" for name in columns)
        body = "".join("<tr>" + "".join(
            f"<td>{html.escape(str(row.get(name) if row.get(name) is not None else ''))}</td>"
            for name in columns) + "</tr>" for row in rows)
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    ranked = [r for r in state["rows"] if r["rank"]]
    pending = [r for r in state["rows"] if not r["rank"] and r["status"] != "rejected"]
    leads = [r for r in pending if r["identity_status"] == "pending"]
    other_pending = [r for r in pending if r["identity_status"] != "pending"]
    rejected = [r for r in state["rows"] if r["status"] == "rejected"]
    chart = trajectory_figure(state).to_html(full_html=False, include_plotlyjs=True,
                                           config={"displaylogo": False}, div_id="trajectory")
    literature = state["literature"]
    frontier = state["frontier"]
    text = ("<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
            "<title>Discovery exploration</title><style>body{font:15px system-ui;margin:32px;color:#1d2939}"
            "h1,h2{font-weight:650}.muted{color:#667085}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}"
            "td,th{padding:9px;text-align:left;border-bottom:1px solid #e4e7ec;vertical-align:top}"
            "th{background:#f2f4f7}details{margin:24px 0}summary{cursor:pointer;font-size:18px}</style></head><body>"
            f"<h1>Discovery exploration</h1><p>{html.escape(state['query'])}</p>"
            f"<p><strong>{state['candidate_count']} explored · {state['ranked_count']} ranked · {len(pending)} pending · {len(rejected)} rejected</strong></p>"
            f"<p>Status: {html.escape(state['status'])} · Worker alive: {state['worker_alive']} · "
            f"Last research event: {html.escape(str(state['last_research_event']))}</p>"
            f"<p>Serving model: {html.escape(str(state['model'] or 'unknown'))} · "
            f"Invocation tokens: {html.escape(str(state['invocation_tokens'] if state['invocation_tokens'] is not None else 'unknown'))} / "
            f"{html.escape(str(state['token_budget'] if state['token_budget'] is not None else 'unbounded'))} · "
            f"Cumulative tokens: {html.escape(str(state['tokens'] if state['tokens'] is not None else 'unknown'))} · "
            f"Accounting complete: {html.escape(str(state['token_accounting_complete']))}</p>"
            f"<p>Volume estimate fallbacks: {html.escape(str(state['quality_warnings']['volume_estimate_fallbacks']))}. "
            f"{html.escape(state['quality_warnings']['note'])}</p>"
            f"<p class='muted'>Snapshot: {state['generated_at']}. This page refreshes every 30 seconds. "
            "Ranks are provisional and use the existing evidence-ranking method. Pending candidates have no rank. "
            "Native-function evidence does not establish target activity.</p>"
            "<p><a href='candidate_ranking.csv'>Download candidate list</a> · <a href='logs/console_log.txt'>Live log</a> · "
            "<a href='structural_modeling_handoff.md'>Modeling handoff when available</a></p>"
            "<h2>Literature: retrieval to evidence</h2>"
            f"<p>{literature['paper_records']} retrieved records · {literature['unique_papers']} unique papers · "
            f"{literature['full_texts']} full texts · {literature['literature_linked_candidates_or_leads']} linked candidates or leads · "
            f"{literature['candidates_with_supported_literature_claims']} candidates with a supported literature-sourced claim</p>"
            f"<p>Extraction outcomes: {html.escape(json.dumps(literature['extractions'], sort_keys=True))}</p>"
            "<p class='muted'>A retrieved paper or extracted name is not proof of relevance or activity. "
            "A supported claim is an automated verdict, not independent expert verification. "
            "Counts alone do not establish that reading changed decisions or improved discovery.</p>"
            "<h2>Recent search activity</h2>" + records_table(list(reversed(state['searches'])),
                ['time', 'tool', 'query', 'status', 'count', 'seconds']) +
            "<h2>Family hypotheses</h2>" + records_table(frontier['hypotheses'],
                ['name', 'mapped_to_assessed_family', 'evidence_status']) +
            "<h2>Family exploration</h2>" + records_table(frontier['families'],
                ['family', 'records', 'annotation_groups', 'statuses', 'saved_native_feedback', 'saved_family_decision']) +
            "<h2>Open questions</h2>" + records_table(frontier['questions'][:40],
                ['kind', 'family', 'accession', 'question', 'source']) +
            "<h2>Recent decisions</h2>" + records_table(list(reversed(state['decisions'])),
                ['time', 'type', 'detail', 'tools']) +
            "<h2>Structural-modeling handoff</h2>" + records_table([state['modeling']],
                ['handoff_schema', 'handoff_discovery_status', 'handoff_may_be_stale', 'admitted',
                 'unresolved_proposals', 'intake_status', 'gpu_jobs_at_intake']) +
            "<h2>Current ranked candidates</h2>" + table(ranked) +
            f"<details open><summary>Registered literature candidates awaiting identity ({len(leads)})</summary>" + table(leads) + "</details>" +
            f"<details><summary>Pending and unresolved candidates ({len(other_pending)})</summary>" + table(other_pending) + "</details>" +
            f"<details><summary>Rejected candidates ({len(rejected)})</summary>" + table(rejected) + "</details>" +
            "<h2>Exploration trajectory</h2><p class='muted'>Recorded lifecycle transitions, including the saved history before resume. "
            "The horizontal axis omits inactive wall-clock gaps. These are status transitions, not reconstructed historical ranks.</p>" + chart + "</body></html>")
    _atomic_text(root / "candidate_progress.html", text)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (root / "logs/live_progress.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, failures = None, 0
        while True:
            try:
                paths = [run_file(root, "candidate_registry.json"), run_file(root, "events.jsonl"),
                         root / "manifest.json", root / "agent_state/steering/state.json",
                         run_file(root, "family_decisions.json"), root / "agent_state/discovery_owner.json",
                         run_file(root, "token_usage.json"), run_file(root, "family_learning.json"),
                         run_file(root, "mechanism_families.json"), root / "agent_state/literature_reading.json",
                         root / "artifacts/fulltext/manifest.json", root / "agent_state/expansion_searches.json",
                         root / "agent_state/warnings.log", root / "agent_state/invocation_resources.json",
                         root / "structural_modeling_handoff.json", root / "structural_intake_check/intake_audit.json"]
                identity = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists())
                owner = _read(root / "agent_state/discovery_owner.json", {})
                if owner.get("hostname") == socket.gethostname() and owner.get("pid"):
                    proc = Path("/proc") / str(owner["pid"])
                    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split() if proc.exists() else []
                    alive = bool(fields and fields[0] != "Z" and fields[19] == str(owner.get("process_start_ticks")))
                    identity += (("worker_alive", alive),)
                if identity != previous:
                    state = publish(root)
                    print(json.dumps({k: state[k] for k in ("generated_at", "candidate_count", "ranked_count", "status_counts")}), flush=True)
                    previous = identity
                status = {"pid": os.getpid(), "hostname": socket.gethostname(), "checked_at": datetime.now(timezone.utc).isoformat(),
                          "cpu_affinity": sorted(os.sched_getaffinity(0)), "status": "running"}
                atomic_write_json(root / "logs/live_progress_status.json", status)
                failures = 0
                if not args.watch or _read(root / "manifest.json", {}).get("status") != "running":
                    status["status"] = "finished"
                    atomic_write_json(root / "logs/live_progress_status.json", status)
                    return
            except (OSError, ValueError, KeyError) as exc:
                failures += 1
                print(f"Refresh failure {failures}: {type(exc).__name__}: {exc}", flush=True)
                if failures >= 3: raise
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
