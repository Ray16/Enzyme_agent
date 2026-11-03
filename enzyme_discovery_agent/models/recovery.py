"""Durable evidence-gathering actions when the model stops doing research."""

import hashlib
import json
from pathlib import Path

from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.run_layout import run_file


def _key(tool, args):
    if tool == "resolve_literature_lead":
        return hashlib.sha256(json.dumps([tool, args], sort_keys=True).encode()).hexdigest()
    if tool == "lookup_genbank_to_uniprot":
        return hashlib.sha256(f"{tool}:{args.get('genbank_accession', '')}".encode()).hexdigest()
    if tool in {"follow_paper_citations", "fetch_paper_metadata"}:
        identifier = str(args.get("doi") or args.get("pmid") or "").lower().strip()
        return hashlib.sha256(f"{tool}:{identifier}".encode()).hexdigest()
    query = " ".join(str(args.get("query") or "").lower().split())
    axis = ":recent" if args.get("sort_by_citations") is False else ""
    return hashlib.sha256(f"{tool}:{query}{axis}".encode()).hexdigest()


def _ledger(run_dir):
    try:
        return json.loads((Path(run_dir) / "agent_state" / "discovery_recovery.json").read_text())
    except FileNotFoundError:
        return {}


def next_recovery_action(run_dir, tool_names):
    """Search saved hypotheses and evaluated seeds, never invent candidates.

    Every distinct query has at most two attempts. Successful negative searches
    close that action; errors get one retry. New hypotheses or evaluated seeds
    add work across resumes without imposing a campaign turn limit.
    """
    from enzyme_discovery_agent.models.literature_strategy import SOURCES, complementary_actions, citation_actions
    sources = [name for name in SOURCES if name in tool_names]
    if not sources and not {"follow_paper_citations", "fetch_paper_metadata", "lookup_genbank_to_uniprot", "resolve_literature_lead"} & tool_names:
        return None
    root = Path(run_dir)
    ledger = _ledger(root)
    completed = {_key(e.get("tool"), e.get("args") or {})
                 for e in read_events(root) if e.get("type") == "tool_call"
                 and e.get("status") in {"success", "partial", "empty"}}
    try:
        families = json.loads((run_file(root, "mechanism_families.json")).read_text())
    except FileNotFoundError:
        families = []
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry
    from enzyme_discovery_agent.models.expansion import select_expansion_seeds, seed_evidence
    from enzyme_discovery_agent.models.family_learning import refresh_family_learning
    registry = load_candidate_registry(root, strict=True).get("candidates", {})
    policy = refresh_family_learning(root, registry)
    exploitation, exploration = [], []
    role_followups = []
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    from enzyme_discovery_agent.models.candidate_registry import discovery_status
    from enzyme_discovery_agent.models.ortholog import tentative_label_key
    seen_roles = set()
    for accession, entry in registry.items():
        if discovery_status(entry) not in {'evaluated', 'ranked'}:
            continue
        evidence = seed_evidence(accession, entry)
        roles = role_transfer_summary(evidence)
        group = tentative_label_key(evidence) or accession
        name = evidence.get('protein_name')
        family = policy['candidates'].get(accession, {}).get('family')
        if (not roles['review_required'] or group in seen_roles or not isinstance(name, str) or not name.strip()
                or policy['families'].get(family, {}).get('agent_decision') in {'stop', 'defer'}):
            continue
        seen_roles.add(group)
        if sources:
            role_followups.append({'tool':sources[0],
                'args':{'query':name.strip()+' reaction mechanism substrate specificity', 'max_results':5},
                'reason':roles['next_question'], 'accession':accession,
                'selection':'evidence_followup', 'lane':'participant_role'})
    seen_families = set()
    # One strongest evaluated seed per family. Homologs cannot consume the
    # whole queue merely by multiplying an already explored enzyme.
    for accession in select_expansion_seeds(registry, len(registry), root):
        value = policy["candidates"].get(accession, {})
        family = value.get("family")
        if policy["families"].get(family, {}).get("agent_decision") == "stop":
            continue
        if family in seen_families:
            continue
        seen_families.add(family)
        evidence = seed_evidence(accession, registry[accession])
        name = evidence.get("protein_name")
        if isinstance(name, str) and name.strip():
            destination = exploitation if value.get("state") == "supported" else exploration
            destination.append((name.strip(), f"Find characterization evidence for evaluated seed {accession}", family))

    def hypothesis_priority(family):
        states = [row["state"] for row in policy["families"].values()
                  if row.get("mechanism_hypothesis") == family.get("name")]
        return min(({"supported": 0, "unresolved": 1, "deprioritized": 2}[state] for state in states), default=1)
    for family in sorted(families, key=hypothesis_priority):
        if str(family.get("promise", "")).lower() not in {"high", "medium"}:
            continue
        # Unresolved hypotheses get breadth, even when a known family is strong.
        # They do not displace characterization of supported candidates on every
        # recovery turn, and a descriptor is still only a search hypothesis.
        for query in (family.get("literature_terms") or []) + (family.get("descriptors") or []):
            if isinstance(query, str) and query.strip():
                exploration.append((query.strip(), f"Resolve mechanism hypothesis: {family.get('name', '')}", None))

    # Reserve one in five recovery actions for breadth while productive known
    # families have outstanding work. The counter is durable across resumes.
    finished = sum(row.get("status") in {"success", "partial", "empty", "deduplicated"} for row in ledger.values())
    breadth_turn = finished % 5 == 4
    tasks = exploration + exploitation if breadth_turn else exploitation + exploration
    actions = role_followups[:]
    for accession, entry in registry.items():
        metadata = entry.get("metadata") or {}
        if not metadata.get("is_literature_lead") or metadata.get("identity_resolution_status") != "pending":
            continue
        name = metadata.get("literature_name")
        organism = metadata.get("organism") or ""
        identifier = (metadata.get("literature_lead") or {}).get("deposited_identifier")
        if "resolve_literature_lead" in tool_names:
            extra = [s["path"] for s in (metadata.get("identity_resolution") or {}).get("sources", [])
                     if s.get("path") != (metadata.get("literature_lead") or {}).get("text_path")]
            actions.append({"tool": "resolve_literature_lead", "args": {"lead_id": accession, "source_paths": extra},
                            "identity_evidence": {"lead": metadata.get("literature_lead"), "sources": [
                                {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None}
                                for path in extra], "implementation": hashlib.sha256(
                                    (Path(__file__).parents[1] / "tools/identity_resolution.py").read_bytes()).hexdigest()},
                            "reason": f"Resolve and link the registered sequence identity for {accession}",
                            "selection": "identity_followup", "lane": "registered_literature_lead"})
        if identifier and "lookup_genbank_to_uniprot" in tool_names:
            actions.append({"tool": "lookup_genbank_to_uniprot", "args": {"genbank_accession": identifier},
                            "reason": f"Resolve the deposited sequence for registered literature lead {accession}",
                            "selection": "identity_followup", "lane": "registered_literature_lead"})
        elif name and sources:
            actions.append({"tool": sources[0], "args": {"query": f"{name} {organism} enzyme sequence characterization".strip(), "max_results": 5},
                            "reason": f"Resolve sequence identity for registered literature lead {accession}",
                            "selection": "identity_followup", "lane": "registered_literature_lead"})
    for query, reason, family in tasks:
        args = {"query": query, "max_results": 5}
        if sources:
            actions.append({"tool": sources[0], "args": args, "reason": reason,
                            "family": family, "selection": "exploration" if (query, reason, family) in exploration else "evidence_followup"})
    complementary = complementary_actions(root, tool_names, families)
    baseline = [a for a in complementary if a["selection"] == "baseline"]
    other = [a for a in complementary if a["selection"] != "baseline"]
    citations = citation_actions(root, tool_names)
    # Alternate evidence follow-up with breadth and citation exploration instead
    # of exhausting one model-generated axis before touching the others.
    combined = baseline[:]
    for i in range(max(len(actions), len(other), len(citations))):
        for lane in (actions, other, citations):
            if i < len(lane):
                combined.append(lane[i])
    for action in combined:
        key = _key(action["tool"], action["args"])
        if action.get("identity_evidence"):
            key = hashlib.sha256(json.dumps([key, action["identity_evidence"]], sort_keys=True).encode()).hexdigest()
        saved = ledger.get(key) or {}
        if key not in completed and saved.get("status") not in {"success", "partial", "empty", "deduplicated"} and saved.get("attempts", 0) < 2:
            return {"key": key, **action}
    return None


def record_recovery_attempt(run_dir, action, status, result=None):
    ledger = _ledger(run_dir)
    previous = ledger.get(action["key"]) or {}
    ledger[action["key"]] = {**action, "status": status,
                            "attempts": previous.get("attempts", 0) + (status == "started")}
    if result is not None:
        ledger[action["key"]]["result_count"] = result.get("count", 0)
    atomic_write_json(Path(run_dir) / "agent_state" / "discovery_recovery.json", ledger)


def resume_failed_routes(run_dir):
    """Retry failed public services after resume while retaining attempt history."""
    ledger = _ledger(run_dir)
    for row in ledger.values():
        if row.get("status") in {"error", "started"}:
            row["previous_attempts"] = row.get("previous_attempts", 0) + row.get("attempts", 0)
            row.update(attempts=0, status="pending")
    if ledger:
        atomic_write_json(Path(run_dir) / "agent_state" / "discovery_recovery.json", ledger)
    from enzyme_discovery_agent.models.literature_reading import reading_state, save_reading_state
    reading = reading_state(run_dir)
    for stages in reading.values():
        for row in stages.values():
            if row.get("status") in {"unavailable", "started", "error", "empty"}:
                row["previous_attempts"] = row.get("previous_attempts", 0) + row.get("attempts", 0)
                row["attempts"] = 0
    save_reading_state(run_dir, reading)
