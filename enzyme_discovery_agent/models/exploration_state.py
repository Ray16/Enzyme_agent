"""Read-only, campaign-local exploration diagnostics and unresolved questions.

These observations do not rank candidates, declare global exhaustion, modify
family decisions, or authorize modeling. Missing novelty data remains unknown.
"""

from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import discovery_status, family_label, load_candidate_registry
from enzyme_discovery_agent.models.event_log import read_events
from enzyme_discovery_agent.models.modeling_feedback import modeling_feedback_summary
from enzyme_discovery_agent.models.ortholog import tentative_label_key as ortholog_key
from enzyme_discovery_agent.run_layout import run_file


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _metadata(entry):
    metadata = entry.get("metadata") or {}
    return {**(metadata.get("evaluation_data") or {}), **metadata}


def _grouping_fields(data):
    # Older incomplete records may explicitly store null annotation fields.
    view = dict(data)
    for key in ("gene_names", "ec_numbers"):
        value = view.get(key)
        view[key] = [value] if isinstance(value, str) and value else [x for x in (value or []) if isinstance(x, str) and x]
    view["protein_name"] = view.get("protein_name") if isinstance(view.get("protein_name"), str) else ""
    return view


def _source(root, accession):
    pointer = str(accession).replace("~", "~0").replace("/", "~1")
    return str(run_file(root, "candidate_registry.json").relative_to(root)) + "#/candidates/" + pointer


def exploration_state(run_dir, *, registry=None, learning=None):
    """Inspect saved records only; annotation groups are not sequence clusters."""
    root = Path(run_dir)
    if registry is None:
        registry = load_candidate_registry(root, strict=True).get("candidates", {})
    if learning is None:
        learning = _read(run_file(root, "family_learning.json"), {})
    feedback = learning.get("candidates") or {}
    decisions = _read(run_file(root, "family_decisions.json"), {}).get("decisions", {})
    families, groups, sequences, models, questions = {}, {}, defaultdict(list), [], []

    def question(kind, text, *, accession=None, family=None, source=None):
        questions.append({"kind": kind, "question": text, "accession": accession,
                          "family": family, "source": source, "authorizes_compute": False})

    for accession, entry in sorted(registry.items()):
        data = _metadata(entry)
        grouping = _grouping_fields(data)
        family = family_label(grouping, accession)
        group = ortholog_key(grouping) or f"accession:{accession}"
        groups[accession] = (family, group)
        row = families.setdefault(family, {"family": family,
            "canonical_assignment": bool(re.fullmatch(r"ipr:IPR\d{6}", family)), "records": 0,
            "statuses": Counter(), "_groups": set(), "_sequences": set(), "sequence_records": 0,
            "modeling_records": 0, "expression_unknown": 0})
        status = discovery_status(entry)
        row["records"] += 1
        row["statuses"][status or "unknown"] += 1
        row["_groups"].add(group)
        sequence = data.get("sequence")
        if isinstance(sequence, str) and sequence.strip():
            digest = hashlib.sha256("".join(sequence.split()).upper().encode()).hexdigest()
            sequences[digest].append(accession)
            row["_sequences"].add(digest)
            row["sequence_records"] += 1
        host = ((data.get("expression_data") or {}).get("preferred_host_assessment") or {}).get("level") or "unknown"
        row["expression_unknown"] += host == "unknown"
        source = _source(root, accession)
        observation = modeling_feedback_summary(accession, data, source)
        if observation:
            row["modeling_records"] += 1
            models.append(observation)
            question("review_modeling_evidence", "What do the saved setup, joint geometry, sampling and transfer limitations support?",
                     accession=accession, family=family, source=source)
        if data.get("identity_resolution_status") == "pending" or data.get("is_literature_lead"):
            question("resolve_identity", "Which exact sequence and construct does the retrieved evidence establish?",
                     accession=accession, family=family, source=source)
        elif status in {"discovered", "screened", "reopened", "evidence_pending"}:
            question("resolve_functional_evidence", "Which required functional claims remain unsupported or contradictory?",
                     accession=accession, family=family, source=source)
        validation = data.get("claim_validation") or {}
        if not observation and status in {"evaluated", "ranked"} and validation.get("promotion_eligible") is True:
            question("assess_modeling_readiness", "Is there a specific structural question, a reviewed setup and an applicable metric?",
                     accession=accession, family=family, source=source)
        if host == "unknown" and status in {"evaluated", "ranked"}:
            question("review_expression", "What host- and construct-specific expression evidence is available?",
                     accession=accession, family=family, source=source)

    for family, row in families.items():
        row["annotation_groups"] = len(row.pop("_groups"))
        row["unique_known_sequences"] = len(row.pop("_sequences"))
        row["statuses"] = dict(row["statuses"])
        policy = (learning.get("families") or {}).get(family, {})
        row["saved_native_feedback"] = policy.get("state", "unresolved")
        row["saved_family_decision"] = decisions.get(family.removeprefix("ipr:"), {}).get("decision")
        row["sequence_clustering"] = "unassessed"
        if row["saved_family_decision"] == "stop" and policy.get("supported", 0):
            question("reconsider_family", "Does the supported member evidence warrant revisiting the saved family stop?",
                     family=family, source=str(run_file(root, "family_decisions.json").relative_to(root)))

    methods = {}
    searches = _read(root / "agent_state/expansion_searches.json", {})
    for method, seeds in sorted(searches.items()):
        for accession, result in sorted(seeds.items()):
            family, seed_group = groups.get(accession, ("unassigned", f"accession:{accession}"))
            key = (method, family)
            row = methods.setdefault(key, {"method": method, "family": family,
                "seed_records": 0, "recorded_attempts": 0, "statuses": Counter(),
                "_returned": set(), "_pending": set(), "_supported": set(),
                "successful_seed_records": 0, "successful_records_with_known_returns": 0})
            row["seed_records"] += 1
            attempts = result.get("attempts")
            row["recorded_attempts"] += attempts if type(attempts) is int and attempts >= 0 else 1
            status = result.get("status", "unknown")
            row["statuses"][status] += 1
            # Partial responses and failed requests cannot establish depletion.
            if status not in {"success", "empty"}:
                continue
            row["successful_seed_records"] += 1
            returned = result.get("new_accessions")
            if not isinstance(returned, list):
                continue
            row["successful_records_with_known_returns"] += 1
            for child in returned:
                if child not in groups:
                    row["_pending"].add(("unassigned", child))
                    continue
                child_family, child_group = groups[child]
                identity = (child_family, child_group)
                if identity == (family, seed_group):
                    continue
                row["_returned"].add(identity)
                state = feedback.get(child, {}).get("state")
                if state == "supported":
                    row["_supported"].add(identity)
                elif state != "deprioritized":
                    row["_pending"].add(identity)
    for row in methods.values():
        row["returned_annotation_groups_other_than_seed"] = len(row.pop("_returned"))
        row["pending_returned_groups"] = len(row.pop("_pending"))
        row["supported_returned_groups"] = len(row.pop("_supported"))
        row["statuses"] = dict(row["statuses"])
        known = row["successful_records_with_known_returns"] == row["successful_seed_records"]
        row["observation"] = ("assess_returned_evidence" if row["pending_returned_groups"] else
            "review_method_yield" if known and row["successful_seed_records"] >= 2 and not row["returned_annotation_groups_other_than_seed"] else
            "insufficient_evidence_for_local_plateau")
        row["global_exhaustion_established"] = False
        if row["observation"] != "insufficient_evidence_for_local_plateau":
            question(row["observation"], "Should returned evidence be resolved or this family/method allocation be reconsidered?",
                     family=row["family"], source="agent_state/expansion_searches.json")

    events = read_events(root)
    requests, outcomes, tools = Counter(), Counter(), Counter()
    for event in events:
        name = str(event.get("tool") or "")
        if event.get("type") != "tool_call" or not name.startswith(("search_", "fetch_", "follow_", "deep_literature")):
            continue
        args = event.get("args") or {}
        normalized = {k: " ".join(v.casefold().split()) if isinstance(v, str) else v for k, v in args.items()}
        requests[json.dumps([name, normalized], sort_keys=True, default=str)] += 1
        outcomes[event.get("status") or "unknown"] += 1
        tools[name] += 1
    registrations = [e for e in events if e.get("type") == "candidates_registered"][-4:]
    registration_cue = len(registrations) == 4 and sum(int(e.get("new_families", 0) or 0) for e in registrations) == 0
    if outcomes["error"]:
        question("inspect_search_failures", "Which failed retrievals need an alternate route or retry rather than a scientific negative?", source=str(run_file(root, "events.jsonl").relative_to(root)))
    if registration_cue:
        question("review_breadth", "Is recent family-label repetition a local retrieval plateau, missing annotation, or a reason to investigate another hypothesis?", source=str(run_file(root, "events.jsonl").relative_to(root)))

    hypotheses = _read(run_file(root, "mechanism_families.json"), [])
    mapped = {row.get("mechanism_hypothesis") for row in (learning.get("families") or {}).values()}
    hypothesis_rows = []
    for hypothesis in hypotheses:
        name = hypothesis.get("name")
        row = {"name": name, "mapped_to_assessed_family": name in mapped,
               "evidence_status": "proposed search hypothesis; not established candidate function"}
        hypothesis_rows.append(row)
        if name not in mapped:
            question("investigate_hypothesis", "Which retrieved sources could establish or challenge this unmapped hypothesis?",
                     family=name, source=str(run_file(root, "mechanism_families.json").relative_to(root)))
    order = {"review_modeling_evidence": 0, "reconsider_family": 1, "resolve_identity": 2}
    questions.sort(key=lambda row: (order.get(row["kind"], 3), row["kind"], row["family"] or "", row["accession"] or ""))
    return {"schema_version": "1.0", "observational_only": True,
        "overview": {"candidate_records": len(registry), "canonical_family_labels": sum(row["canonical_assignment"] for row in families.values()),
            "annotation_groups": len(set(groups.values())), "records_with_sequence": sum(len(v) for v in sequences.values()),
            "unique_known_sequences": len(sequences), "exact_sequence_duplicate_records": sum(len(v)-1 for v in sequences.values()),
            "modeling_records": len(models), "unresolved_questions": len(questions),
            "question_counts": dict(Counter(q["kind"] for q in questions)), "registration_plateau_cue": registration_cue,
            "global_exhaustion_established": False},
        "families": sorted(families.values(), key=lambda row: (row["records"], row["family"])),
        "hypotheses": hypothesis_rows, "methods": list(methods.values()), "modeling": models, "questions": questions,
        "searches": [{"requests": sum(requests.values()), "unique_lexical_requests": len(requests),
            "lexically_repeated_requests": sum(n-1 for n in requests.values()), "outcomes": dict(outcomes), "tools": dict(tools)}],
        "limitations": ["Annotation groups are heuristic, not verified functional independence or sequence clusters.",
            "Known sequences may cover only part of the pool; exact duplicates do not measure near-homology.",
            "Search counters are lexical and depend on recorded events, not semantic query equivalence.",
            "Expansion rows summarize saved seed records; missing return identities and partial responses cannot establish exhaustion.",
            "Questions are advisory; no ranking, family decision, metric, provider routing or compute permission changes here."]}


def exploration_page(run_dir, section="questions", *, offset=0, limit=10, query=""):
    """Bounded pagination over the full frontier, including rows outside context."""
    if section not in {"overview", "families", "hypotheses", "methods", "modeling", "questions", "searches"}:
        raise ValueError("unknown exploration-state section")
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("offset must be nonnegative and limit between 1 and 20")
    state = exploration_state(run_dir)
    rows = [state[section]] if section == "overview" else state[section]
    if query.strip():
        rows = [row for row in rows if query.casefold() in json.dumps(row, ensure_ascii=False).casefold()]
    page, used = [], 0
    for row in rows[offset:offset+limit]:
        text = json.dumps(row, ensure_ascii=False)
        if len(text) > 4500:
            row = {"source": row.get("source"), "preview": text[:3000], "truncated": True}
        size = len(json.dumps(row, ensure_ascii=False))
        if page and used + size > 6500:
            break
        page.append(row)
        used += size
    end = offset + len(page)
    return {"section": section, "records": page, "total": len(rows), "next_offset": end if end < len(rows) else None,
            "observational_only": True, "limitations": state["limitations"]}
