"""Update search priorities from saved evidence, without another model call.

These are revisable native-chemistry hypotheses, not probabilities of activity
on the target. Independent enzyme groups, rather than homolog counts or model
confidence, determine the feedback. Unknown evidence retains exploration depth.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from enzyme_discovery_agent.models.candidate_registry import discovery_status, family_label, load_candidate_registry
from enzyme_discovery_agent.models.claim_validation import reaction_topology
from enzyme_discovery_agent.models.ortholog import tentative_label_key as ortholog_key
from enzyme_discovery_agent.models.target_reasoning import target_claims
from enzyme_discovery_agent.state_store import atomic_write_json, state_lock
from enzyme_discovery_agent.run_layout import run_file


def _read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def candidate_feedback(data, target):
    """Use recorded native chemistry and topology; absence is unresolved."""
    direction, bonds, topology = target
    claims = {c.get("claim_type"): c for c in (data.get("claim_validation") or {}).get("claims", [])}
    reasons = []
    if data.get("is_enzyme") is False:
        reasons.append("record is not an enzyme")
    for kind in ("native_reaction", "reaction_direction", "bond_type"):
        if claims.get(kind, {}).get("verdict") == "contradicted":
            reasons.append(f"native {kind} contradicts the target")
    chemistry = data.get("reaction_chemistry") or {}
    observed_direction = chemistry.get("reaction_direction")
    observed_bonds = set(chemistry.get("bond_types") or [])
    if direction in {"formation", "cleavage"} and observed_direction in {"formation", "cleavage"} and direction != observed_direction:
        reasons.append("native reaction runs in the opposite direction")
    if bonds and observed_bonds and not set(bonds) & observed_bonds:
        reasons.append("native reaction acts on a different bond type")
    observed_topology = reaction_topology(data.get("catalytic_activities") or [])
    observed_topology = observed_topology or claims.get("substrate_topology", {}).get("evidence", {}).get("observed_value")
    if topology == "free_small_molecule" and observed_topology in {"macromolecule", "polymer", "carrier_protein_tethered"}:
        reasons.append(f"native reaction requires {observed_topology} substrates; free-substrate activity is unestablished")
    elif topology == "carrier_protein_tethered" and observed_topology and observed_topology != topology:
        reasons.append("native reaction does not establish the target carrier requirement")
    if data.get("requires_polymer_substrate") and topology == "free_small_molecule":
        reasons.append("record explicitly requires a polymer substrate")
    from enzyme_discovery_agent.output.report_generator import _accommodation_score
    from enzyme_discovery_agent.models.reaction_roles import role_transfer_summary
    roles = role_transfer_summary(data)
    fit = _accommodation_score(data)
    if fit is None and not roles['review_required']:
        try:
            value = float(data.get("substrate_fit"))
            if math.isfinite(value) and 0 <= value <= 1:
                fit = value
        except (ValueError, TypeError):
            pass
    if reasons:
        state = "deprioritized"
    else:
        native_supported = claims.get("native_reaction", {}).get("verdict") == "supported" or bool(data.get("catalytic_activities"))
        direction_supported = not direction or claims.get("reaction_direction", {}).get("verdict") == "supported" or direction == observed_direction
        bond_supported = not bonds or claims.get("bond_type", {}).get("verdict") == "supported" or bool(set(bonds) & observed_bonds)
        topology_supported = not topology or observed_topology == topology
        validation = data.get("claim_validation") or {}
        claims_ready = validation.get("promotion_eligible") is not False and not data.get("critical_warnings")
        state = "supported" if native_supported and direction_supported and bond_supported and topology_supported and claims_ready and not roles['review_required'] else "unresolved"
        reasons = ["recorded native chemistry supports a target-relevant starting point" if state == "supported"
                   else "native chemistry or substrate topology is not fully established"]
        if roles['review_required']:
            reasons = ['participant-role transfer requires evidence before structural resemblance can support expansion']
    return {"state": state, "fit": fit, "native_topology": observed_topology, "reasons": reasons,
            "role_review_required": roles['review_required'], "next_question": roles['next_question']}


def build_family_learning(candidates, landscape, decisions, target, searches=None):
    """Pure aggregation. A supported exception can reopen a weak family."""
    groups = defaultdict(dict)
    feedback = {}
    names = {}
    for acc, entry in candidates.items():
        meta = entry.get("metadata") or {}
        data = {**(meta.get("evaluation_data") or {}), **meta, "accession": acc}
        family = family_label(data, acc)
        names[family] = data.get("interpro_family") or data.get("protein_name") or family
        if discovery_status(entry) not in {"evaluated", "ranked", "rejected", "evidence_pending"}:
            continue
        value = candidate_feedback(data, target)
        value["family"] = family
        feedback[acc] = value
        # Missing evidence and provider errors cannot turn a rejection into a
        # scientific negative. Count only actual observed contradictions.
        group = ortholog_key(data) or f"accession:{acc}"
        previous = groups[family].get(group)
        order = {"supported": 0, "unresolved": 1, "deprioritized": 2}
        if previous is None or (order[value["state"]], -(value["fit"] or 0)) < (order[previous["state"]], -(previous["fit"] or 0)):
            groups[family][group] = value

    surveys = {f"ipr:{f['fold_id']}": f for f in landscape if f.get("fold_id")}
    families = {}
    for fid in set(groups) | set(surveys):
        rows = list(groups[fid].values())
        counts = Counter(row["state"] for row in rows)
        survey = surveys.get(fid, {})
        verdict = (survey.get("family_claim_validation") or {}).get("verdict")
        fits = [row["fit"] for row in rows if row["state"] == "supported" and row["fit"] is not None]
        decision = decisions.get(fid.removeprefix("ipr:"), {}).get("decision")
        if counts["supported"]:
            state = "supported"
        elif (counts["deprioritized"] and not counts["unresolved"]) or verdict == "contradicted_family_hypothesis":
            state = "deprioritized"
        else:
            state = "unresolved"
        families[fid] = {"family": fid, "name": survey.get("fold_name") or names.get(fid, fid),
                         "state": state, "independent_enzymes_assessed": len(rows),
                         "label_groups_assessed": len(rows),
                         "grouping_basis": "tentative label; biological independence unverified",
                         "supported": counts["supported"], "unresolved": counts["unresolved"],
                         "deprioritized": counts["deprioritized"],
                         "median_supported_fit": median(fits) if fits else None,
                         "best_supported_fit": max(fits) if fits else None,
                         "supported_fraction": counts["supported"] / len(rows) if rows else None,
                         "survey_verdict": verdict, "direction_counts": survey.get("direction_vs_target_counts"),
                         "survey_fit": survey.get("holistic_fit"), "agent_decision": decision,
                         "mechanism_hypothesis": (survey.get("derived_family_hypothesis") or {}).get("name"),
                         "expansion_attempts": 0, "new_supported_groups": 0}

    for method, attempted in (searches or {}).items():
        for acc in attempted:
            family = feedback.get(acc, {}).get("family")
            if family in families:
                families[family]["expansion_attempts"] += 1
    returned = defaultdict(set)
    for acc, entry in candidates.items():
        data = entry.get("metadata") or {}
        seed = (data.get("structural_homology") or {}).get("seed_accession")
        family = feedback.get(seed, {}).get("family")
        group = ortholog_key(data) or f"accession:{acc}"
        seed_group = ortholog_key((candidates.get(seed) or {}).get("metadata") or {}) or f"accession:{seed}"
        if group != seed_group and family in families and feedback.get(acc, {}).get("state") == "supported":
            returned[family].add(group)
    for fid, rows in returned.items():
        families[fid]["new_supported_groups"] = len(rows)

    def priority(fid):
        row = families[fid]
        state = 3 if row["agent_decision"] in {"stop", "defer"} else {"supported": 0, "unresolved": 1, "deprioritized": 2}[row["state"]]
        return (state, -(row["best_supported_fit"] if row["best_supported_fit"] is not None else -1),
                -(row["supported_fraction"] or 0),
                -(row["new_supported_groups"] / row["expansion_attempts"] if row["expansion_attempts"] else 0), fid)
    return {"schema_version": 1, "basis": "native-chemistry feedback; target activity remains unproven; label-based group counts are tentative and do not establish independence",
            "work_deferrals": [key for key, value in (decisions or {}).items() if value.get('decision') in {'stop', 'defer'}],
            "families": families, "priority_order": sorted(families, key=priority), "candidates": feedback}


def refresh_family_learning(run_dir, candidates=None):
    """Persist only evidence changes; repeated reads never manufacture progress."""
    root = Path(run_dir)
    if candidates is None:
        candidates = load_candidate_registry(root, strict=True).get("candidates", {})
    policy = build_family_learning(candidates, _read(run_file(root, "family_landscape.json"), []),
        _read(run_file(root, "family_decisions.json"), {}).get("decisions", {}), target_claims(root),
        _read(root / "agent_state" / "expansion_searches.json", {}))
    signature = hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()
    path = run_file(root, "family_learning.json")
    with state_lock(path):
        previous = _read(path, {})
        if previous.get("signature") == signature:
            return previous
        policy.update(signature=signature, revision=previous.get("revision", 0) + 1)
        atomic_write_json(path, policy)
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(root, "family_learning_update", revision=policy["revision"], signature=signature,
                 priorities=policy["priority_order"][:12],
                 states={fid: row["state"] for fid, row in policy["families"].items()})
    return policy


def prioritize_seeds(ordered, candidates, policy):
    """Respect scientific assessments and the shared holistic order, not fit alone."""
    old_order = {acc: i for i, acc in enumerate(ordered)}
    def key(acc):
        value = policy["candidates"].get(acc, {})
        state = 2 if value.get('state')=='deprioritized' else 0
        family = value.get("family") or family_label(candidates[acc].get("metadata") or {}, acc)
        row = policy["families"].get(family, {})
        if row.get("agent_decision") in {"stop", "defer"}:
            state = 3
        # Unknown families retain the prior evidence ordering. They have not
        # earned a depth preference simply by appearing earlier in the registry.
        assessment=(candidates[acc].get('metadata') or {}).get('priority_assessment') or {}
        focus=0 if assessment.get('decision')=='prioritize' else 1
        family_focus=0 if row.get('agent_decision')=='expand' else 1
        return (focus, state, family_focus, old_order[acc])
    # Keep unresolved evaluated candidates available for exploration. Known
    # native-substrate mismatches get focused evidence work instead of spending
    # an automatic structural/embedding expansion on unsupported machinery.
    from enzyme_discovery_agent.models.rolling import work_deferred
    allowed = [acc for acc in ordered if not work_deferred(candidates[acc].get('metadata') or {}, policy.get('work_deferrals', []))
               and (policy["candidates"].get(acc, {}).get("state") != "deprioritized"
                    or (candidates[acc].get('metadata') or {}).get('priority_assessment',{}).get('decision')=='prioritize')
               and policy["families"].get(policy["candidates"].get(acc, {}).get("family"), {}).get("agent_decision") not in {"stop", "defer"}]
    return sorted(allowed, key=key)
