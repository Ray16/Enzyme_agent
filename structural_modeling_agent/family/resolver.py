"""Once-per-family bounded LLM protocol resolver with fail-closed RDKit validation.

Used by ``family-cofold`` ONLY when:
1. A new target reaction has no cached shared protocol yet, or
2. A newly discovered enzyme family lands in ``pending_families`` with no
   matching mechanism protocol.

Every LLM proposal is validated deterministically through ``compile_protocol``
(including RDKit stereochemistry preservation, pH/role-aware protonation and
charge checks, and reaction-coordinate atom-selector checks) before any GPU
cofolding job is created. Proposals that fail validation after at most one
repair turn are rejected fail-closed and cached so they are never retried in
an unbounded loop.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from enzyme_discovery_agent.state_store import atomic_write_json

from ..modeling_reasoning.reaction import ReactionCoordinates

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw or markdown-fenced LLM output."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty resolver response")
    match = _JSON_BLOCK_RE.search(raw)
    candidate = match.group(1).strip() if match else raw
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Resolver response must be a JSON object")
    return data


def _default_llm_caller(model: str) -> Callable[[str, str], tuple[str, dict]]:
    """Create a minimal single-turn caller backed by the shared LLMClient."""
    from enzyme_discovery_agent.llm.client import LLMClient

    client = LLMClient(model=model)

    def _call(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        resp = client.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=0.0,
            usage_label="family_protocol_resolver",
        )
        content = ""
        if hasattr(resp, "choices") and resp.choices:
            content = getattr(resp.choices[0].message, "content", "") or ""
        elif isinstance(resp, dict):
            content = resp.get("content") or ""
        usage = {}
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }
        return content, usage

    return _call


def _available_activation_modes(specification: dict | None = None) -> list[dict[str, Any]]:
    """Summarize supported reaction coordinates without hardcoding any enzyme."""
    coords_path = (specification or {}).get("reaction_coordinates_path")
    coordinates = ReactionCoordinates.load(coords_path) if coords_path else ReactionCoordinates.load()
    summary = []
    for key, coord in (coordinates.mechanisms or {}).items():
        steps = [
            {
                "index": st.get("index"),
                "name": st.get("name"),
                "participants": sorted((st.get("participants") or {}).keys()),
            }
            for st in (coord.get("steps") or [])
        ]
        summary.append({
            "coordinate_key": key,
            "activation_modes": (coord.get("keys") or {}).get("activation_mode") or [],
            "steps": steps,
        })
    return summary


_SYSTEM_PROMPT = (
    "You are a rigorous structural-enzymology protocol resolver. "
    "Given a target reaction contract, supported reaction coordinates, and representative "
    "members of an unresolved enzyme family, decide whether a justified, chemically valid "
    "shared structural-modeling protocol can be specified once for that family.\n\n"
    "Rules:\n"
    "1. Return ONLY a valid JSON object (no prose outside JSON).\n"
    "2. Distinguish recorded native activity from family transfer. If the family is "
    "mechanistically incompatible with the target or lacks a supported reaction coordinate "
    "(e.g., requires an unsupported co-enzyme or different bond chemistry), return "
    '{"decision": "defer_unsupported", "rationale": "..."} rather than guessing.\n'
    "3. Preserve exact stereochemistry from source_smiles (never drop or invert chiral centers).\n"
    "4. Set protonation states for the experimental/physiological assay pH: free carboxylic acids "
    "must be deprotonated (-COO-) at pH >= 6.0; an attacking amine nucleophile must have one "
    "neutral unprotonated -NH2/-NH-, while spectator aliphatic amines on diamines/polyamines "
    "must be protonated (-NH3+) at pH <= 9.5.\n"
    "5. Set protein_copies=2 and active_sites=1 ONLY when structural/subunit evidence shows an "
    "obligate inter-subunit (interfacial) active site; otherwise use protein_copies=1.\n"
)


def _build_family_prompt(
    *,
    target: dict[str, Any],
    family_id: str,
    members: list[dict[str, Any]],
    available_coordinates: list[dict[str, Any]],
    existing_spec: dict[str, Any] | None,
) -> str:
    payload = {
        "task": "resolve_family_protocol" if existing_spec else "resolve_target_protocol",
        "target_contract": {
            k: target.get(k)
            for k in ("query", "substrate_smiles", "product_smiles", "reaction_type", "assay_pH")
            if target.get(k) is not None
        },
        "family_id": family_id,
        "representative_members": [
            {
                k: m.get(k)
                for k in (
                    "accession", "enzyme_name", "rank", "is_swissprot",
                    "raw_family_id", "recorded_activities",
                    "architecture_and_tractability", "target_assessment_summary",
                )
                if m.get(k) is not None
            }
            for m in members[:3]
        ],
        "available_reaction_coordinates": available_coordinates,
        "existing_mechanism_modes": sorted(((existing_spec or {}).get("mechanism_templates") or {}).keys()),
        "required_json_schema": {
            "decision": "resolve_protocol | defer_unsupported",
            "rationale": "concise scientific basis",
            "mode": "activation_mode string matching one of available_reaction_coordinates",
            "assay_pH": "float (default 7.4 if unspecified)",
            "protein_copies": "int (1 for intra-subunit site, 2 for obligate interfacial active site)",
            "active_sites": "int (optional; 1 when modeling 1 interfacial site across 2 copies)",
            "assembly_basis": "short justification of oligomeric unit",
            "mechanism_template": {
                "nucleotide_ccd": "optional CCD e.g. ATP or null",
                "cofactors": [{"ccd": "CCD", "count": 1, "role": "cofactor role"}],
                "metal": {"metal": "MG", "per_site_metal_count": 1, "rationale": "..."},
            },
            "components": [
                {
                    "name": "substrate name",
                    "source_smiles": "exact source SMILES",
                    "states": {
                        "<mode>": {
                            "1": {"include_in_model": True, "modeling_smiles": "...", "role": "..."}
                        }
                    },
                }
            ],
            "step_rationale": {"1": "..."},
            "assignment_rule": {"mode": "<mode>", "required": [], "excluded": [], "family_regex": ""},
        },
    }
    return json.dumps(payload, indent=2)


def _merge_proposal_into_spec(
    base_spec: dict[str, Any] | None,
    proposal: dict[str, Any],
    *,
    family_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return a validated-candidate copy of specification with proposal merged in."""
    spec = copy.deepcopy(base_spec) if isinstance(base_spec, dict) else {
        "schema_version": "shared-two-step-protocol-1",
        "components": [],
        "mechanism_templates": {},
        "assignment_rules": [],
        "step_rationale": {},
        "protein_copies": 1,
        "assembly_basis": proposal.get("assembly_basis") or "Resolved shared family protocol.",
    }
    mode = str(proposal.get("mode") or "").strip()
    if not mode:
        raise ValueError("Proposal missing 'mode'")

    if proposal.get("assay_pH") is not None and spec.get("assay_pH") is None:
        spec["assay_pH"] = float(proposal["assay_pH"])

    # Merge mechanism template if supplied and not already present
    tmpl = proposal.get("mechanism_template")
    if isinstance(tmpl, dict) and mode not in spec.setdefault("mechanism_templates", {}):
        spec["mechanism_templates"][mode] = tmpl
    elif mode not in spec.get("mechanism_templates", {}):
        raise ValueError(f"Mode {mode!r} is not in existing mechanism_templates and no mechanism_template was supplied")

    # Merge components / per-mode states
    proposed_components = proposal.get("components") or []
    if not spec.get("components") and proposed_components:
        spec["components"] = copy.deepcopy(proposed_components)
    elif proposed_components:
        by_source = {c.get("source_smiles"): c for c in spec["components"] if isinstance(c, dict)}
        for comp in proposed_components:
            if not isinstance(comp, dict):
                continue
            existing = by_source.get(comp.get("source_smiles"))
            if existing is not None:
                existing.setdefault("states", {})
                mode_state = (comp.get("states") or {}).get(mode)
                if isinstance(mode_state, dict):
                    existing["states"][mode] = mode_state
            else:
                spec["components"].append(copy.deepcopy(comp))

    # Merge step rationale
    for step_key, text in (proposal.get("step_rationale") or {}).items():
        spec.setdefault("step_rationale", {}).setdefault(str(step_key), str(text))

    # Merge family-specific assembly/metal override if resolving a family
    copies = proposal.get("protein_copies")
    active_sites = proposal.get("active_sites")
    if family_id and (copies is not None or active_sites is not None or proposal.get("assembly_basis")):
        ov = spec.setdefault("family_overrides", {}).setdefault(family_id, {})
        if copies is not None:
            ov["protein_copies"] = int(copies)
        if active_sites is not None:
            ov["active_sites"] = int(active_sites)
        if proposal.get("assembly_basis"):
            ov["assembly_basis"] = str(proposal["assembly_basis"])
        if proposal.get("assay_pH") is not None:
            ov["assay_pH"] = float(proposal["assay_pH"])
    elif copies is not None:
        spec["protein_copies"] = int(copies)
        if active_sites is not None:
            spec["active_sites"] = int(active_sites)
        if proposal.get("assembly_basis"):
            spec["assembly_basis"] = str(proposal["assembly_basis"])

    # Merge assignment rule so future members of this family match with 0 LLM calls
    rule = proposal.get("assignment_rule")
    if isinstance(rule, dict) and rule.get("mode"):
        rules = spec.setdefault("assignment_rules", [])
        if rule not in rules:
            rules.append(rule)
    elif family_id:
        # Ensure the resolved family itself is mapped deterministically on next pass
        escaped = re.escape(family_id)
        auto_rule = {"mode": mode, "family_regex": f"^{escaped}$"}
        rules = spec.setdefault("assignment_rules", [])
        if auto_rule not in rules:
            rules.append(auto_rule)

    return spec, mode


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.is_file():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
    return {"schema_version": "family-resolver-cache-1", "entries": {}}


def resolve_one_family(
    *,
    target: dict[str, Any],
    family_id: str,
    members: list[dict[str, Any]],
    base_spec: dict[str, Any] | None,
    llm_caller: Callable[[str, str], tuple[str, dict]],
    compile_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    max_repair_turns: int = 1,
) -> dict[str, Any]:
    """Run one bounded LLM resolution (plus at most one deterministic repair turn)."""
    coords = _available_activation_modes(base_spec)
    user_prompt = _build_family_prompt(
        target=target,
        family_id=family_id,
        members=members,
        available_coordinates=coords,
        existing_spec=base_spec,
    )
    calls = 0
    total_tokens = 0
    last_error = None
    prompt = user_prompt

    for attempt in range(max_repair_turns + 1):
        calls += 1
        raw_text, usage = llm_caller(_SYSTEM_PROMPT, prompt)
        total_tokens += int((usage or {}).get("total_tokens") or 0)
        try:
            proposal = _extract_json_object(raw_text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"JSON parse error: {exc}"
            prompt = (
                user_prompt
                + f"\n\nPrevious response failed JSON parsing ({last_error}). "
                "Return ONLY a single valid JSON object matching required_json_schema."
            )
            continue

        decision = str(proposal.get("decision") or "").strip()
        if decision == "defer_unsupported":
            return {
                "status": "deferred",
                "family_id": family_id,
                "rationale": str(proposal.get("rationale") or "Marked unsupported by family resolver."),
                "llm_calls": calls,
                "total_tokens": total_tokens,
            }
        if decision != "resolve_protocol":
            last_error = f"Unsupported decision {decision!r}; expected 'resolve_protocol' or 'defer_unsupported'"
            prompt = user_prompt + f"\n\nPrevious response error: {last_error}. Please fix and return ONLY JSON."
            continue

        try:
            candidate_spec, mode = _merge_proposal_into_spec(base_spec, proposal, family_id=family_id)
            override = (candidate_spec.get("family_overrides") or {}).get(family_id) or {}
            compile_spec = {**candidate_spec, **override}
            compiled = compile_fn(mode, compile_spec)
            return {
                "status": "resolved",
                "family_id": family_id,
                "mode": mode,
                "rationale": str(proposal.get("rationale") or ""),
                "specification": candidate_spec,
                "compiled_protocol": compiled,
                "llm_calls": calls,
                "total_tokens": total_tokens,
            }
        except (KeyError, ValueError, TypeError) as exc:
            last_error = str(exc)
            prompt = (
                user_prompt
                + f"\n\nDeterministic chemistry/protocol validation FAILED on your previous proposal:\n{last_error}\n"
                "Fix the stereochemistry, protonation/charge, cofactor/metal, or step schema error "
                "(or return decision='defer_unsupported' if not supported) and return ONLY JSON."
            )

    return {
        "status": "failed_validation",
        "family_id": family_id,
        "error": last_error or "Unknown validation failure",
        "llm_calls": calls,
        "total_tokens": total_tokens,
    }


def resolve_pending_families(
    run_dir: Path,
    handoff: dict[str, Any],
    specification: dict[str, Any],
    spec_path: Path,
    out_dir: Path,
    *,
    model: str | None = None,
    llm_caller: Callable[[str, str], tuple[str, dict]] | None = None,
    max_families: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve each unique pending family at most once; update spec_path if any succeed.

    Returns ``(updated_specification, resolution_report)``.
    """
    from enzyme_discovery_agent.models.target_review import target_contract
    from .cofold import compile_protocol

    pending = handoff.get("pending_families") or []
    if not pending:
        return specification, {"resolved": [], "deferred": [], "failed": [], "llm_calls": 0, "total_tokens": 0}
    if llm_caller is None:
        if not model:
            return specification, {"resolved": [], "deferred": [], "failed": [], "llm_calls": 0, "total_tokens": 0}
        llm_caller = _default_llm_caller(model)

    cache_path = Path(out_dir) / "family_resolver_cache.json"
    cache = _load_cache(cache_path)
    entries = cache.setdefault("entries", {})
    target = target_contract(run_dir) or {}

    by_family: dict[str, list[dict[str, Any]]] = {}
    for member in pending:
        fid = str(member.get("family_id") or "unresolved")
        by_family.setdefault(fid, []).append(member)

    current_spec = copy.deepcopy(specification)
    report = {"resolved": [], "deferred": [], "failed": [], "llm_calls": 0, "total_tokens": 0}
    changed = False

    for family_id, members in list(by_family.items())[:max_families]:
        cache_key = f"family::{family_id}"
        if cache_key in entries:
            continue
        outcome = resolve_one_family(
            target=target,
            family_id=family_id,
            members=members,
            base_spec=current_spec,
            llm_caller=llm_caller,
            compile_fn=compile_protocol,
        )
        report["llm_calls"] += outcome.get("llm_calls", 0)
        report["total_tokens"] += outcome.get("total_tokens", 0)
        entries[cache_key] = {
            "status": outcome["status"],
            "family_id": family_id,
            "mode": outcome.get("mode"),
            "rationale": outcome.get("rationale"),
            "error": outcome.get("error"),
            "llm_calls": outcome.get("llm_calls", 0),
            "total_tokens": outcome.get("total_tokens", 0),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
        if outcome["status"] == "resolved":
            current_spec = outcome["specification"]
            report["resolved"].append(family_id)
            changed = True
        elif outcome["status"] == "deferred":
            report["deferred"].append({"family_id": family_id, "rationale": outcome.get("rationale")})
        else:
            report["failed"].append({"family_id": family_id, "error": outcome.get("error")})

    atomic_write_json(cache_path, cache)
    if changed:
        atomic_write_json(spec_path, current_spec)
    return current_spec, report


def resolve_target_protocol(
    run_dir: Path,
    spec_path: Path,
    out_dir: Path,
    *,
    model: str | None = None,
    llm_caller: Callable[[str, str], tuple[str, dict]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bootstrap a shared protocol once for a brand-new target reaction with no cached spec."""
    from enzyme_discovery_agent.models.target_review import target_contract
    from .cofold import compile_protocol, register_protocol

    if llm_caller is None:
        if not model:
            return None, {"status": "skipped", "llm_calls": 0, "total_tokens": 0}
        llm_caller = _default_llm_caller(model)

    cache_path = Path(out_dir) / "family_resolver_cache.json"
    cache = _load_cache(cache_path)
    entries = cache.setdefault("entries", {})
    cache_key = "target::initial"
    if cache_key in entries:
        return None, entries[cache_key]

    target = target_contract(run_dir) or {}
    outcome = resolve_one_family(
        target=target,
        family_id="target_bootstrap",
        members=[],
        base_spec=None,
        llm_caller=llm_caller,
        compile_fn=compile_protocol,
    )
    entries[cache_key] = {
        "status": outcome["status"],
        "mode": outcome.get("mode"),
        "rationale": outcome.get("rationale"),
        "error": outcome.get("error"),
        "llm_calls": outcome.get("llm_calls", 0),
        "total_tokens": outcome.get("total_tokens", 0),
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(cache_path, cache)
    if outcome["status"] == "resolved":
        spec = outcome["specification"]
        atomic_write_json(spec_path, spec)
        registry_path = Path(run_dir).parent / ".family_protocols" / "index.json"
        try:
            register_protocol(spec_path, registry_path)
        except (OSError, ValueError, KeyError):
            pass
        return spec, entries[cache_key]
    return None, entries[cache_key]
