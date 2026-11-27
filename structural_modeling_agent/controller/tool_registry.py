"""Unified declarative tool registry for ModelingAgent.

Single source of truth for tool schemas, dispatch handlers, scope/accession
handling, objective/stage availability, and repeated-inspection guards.
Domain modules register ToolSpec objects so adding or renaming a tool never
requires manually syncing multiple files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from . import dispatch_handlers as dh
from .domain_tools import register_domain_tools
from .tool_schemas import build_core_schemas

VALID_SCOPES = frozenset({"pre_accession", "general", "candidate", "revision"})


@dataclass(frozen=True)
class ToolSpec:
    """Declarative specification of one agent-callable tool."""

    name: str
    schema: dict[str, Any]
    handler: Callable[..., Any]
    scope: str = "general"
    domain: str = "core"
    stages: frozenset[str] = frozenset()
    repeatable_reader: bool = False
    deprecated: bool = False


class ToolRegistry:
    """Registry holding ToolSpec entries with cached schema and stage lookups."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._schema_cache: tuple[dict[str, Any], ...] | None = None
        self._stage_cache: dict[str, frozenset[str]] = {}
        self._readers_cache: frozenset[str] | None = None

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool registration: {spec.name!r}")
        if spec.scope not in VALID_SCOPES:
            raise ValueError(f"invalid tool scope {spec.scope!r} for {spec.name!r}")
        schema_name = ((spec.schema or {}).get("function") or {}).get("name")
        if schema_name != spec.name:
            raise ValueError(f"schema function name {schema_name!r} != ToolSpec.name {spec.name!r}")
        self._specs[spec.name] = spec
        self._schema_cache = None
        self._stage_cache.clear()
        self._readers_cache = None
        return spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise ValueError(f"unknown tool {name}")
        return self._specs[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def schemas(self, allowed: Iterable[str] | None = None) -> list[dict[str, Any]]:
        if self._schema_cache is None:
            self._schema_cache = tuple(spec.schema for spec in self._specs.values())
        if allowed is None:
            return list(self._schema_cache)
        allowed_set = allowed if isinstance(allowed, (set, frozenset)) else set(allowed)
        return [s for s in self._schema_cache if s["function"]["name"] in allowed_set]

    def names_for_stage(self, stage: str) -> frozenset[str]:
        if stage not in self._stage_cache:
            self._stage_cache[stage] = frozenset(
                spec.name for spec in self._specs.values() if stage in spec.stages
            )
        return self._stage_cache[stage]

    def names_for_domain(self, domain: str, *, include_deprecated: bool = False) -> frozenset[str]:
        return frozenset(
            spec.name
            for spec in self._specs.values()
            if spec.domain == domain and (include_deprecated or not spec.deprecated)
        )

    def active_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs.values() if not spec.deprecated)

    def deprecated_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs.values() if spec.deprecated)

    def repeatable_readers(self) -> frozenset[str]:
        if self._readers_cache is None:
            self._readers_cache = frozenset(
                spec.name for spec in self._specs.values() if spec.repeatable_reader
            )
        return self._readers_cache

    def dispatch(self, agent, name: str, args: dict[str, Any]):
        spec = self.get(name)
        if spec.scope == "pre_accession":
            return spec.handler(agent, args)

        # Single-candidate runs default omitted accession and reject unknown ones clearly.
        if isinstance(args, dict):
            if "accession" not in args and len(agent.state["candidates"]) == 1:
                args = {**args, "accession": next(iter(agent.state["candidates"]))}
            elif args.get("accession") is not None and args["accession"] not in agent.state["candidates"]:
                raise ValueError(
                    f"unknown accession {args['accession']!r}; choose one of {list(agent.state['candidates'])}"
                )

        if spec.scope == "general":
            return spec.handler(agent, args)

        acc = args.get("accession")
        if acc is None:
            ids = list(agent.state["candidates"])
            if len(ids) != 1:
                raise ValueError(f"accession is required; choose one of {ids}")
            acc = ids[0]
        elif acc not in agent.state["candidates"]:
            raise ValueError(f"unknown accession {acc!r}; choose one of {list(agent.state['candidates'])}")

        candidate = agent._candidate(acc)
        if spec.scope == "candidate":
            return spec.handler(agent, args, acc, candidate)

        revision = agent._current(acc)
        return spec.handler(agent, args, acc, candidate, revision)


# ---------------------------------------------------------------------------
# Declarative stage, domain, and repeatable-reader metadata
# ---------------------------------------------------------------------------

_REPEATABLE_READER_NAMES = frozenset({
    "inspect_candidate",
    "inspect_result_locations",
    "inspect_investigation_lead",
    "read_candidate_section",
    "read_evidence",
    "read_reasoning_section",
    "inspect_processivity_toolchain",
    "check_cofold",
    "check_sampling",
    "check_barrier_calculation",
    "inspect_barrier_result",
})

_DEPRECATED_TOOLS: frozenset[str] = frozenset()

_STAGE_MEMBERSHIPS: dict[str, frozenset[str]] = {
    "prepare_sampling": frozenset({
        "inspect_candidate", "read_candidate_section", "read_evidence",
        "retrieve_reference", "search_literature", "assess_assembly", "inspect_molecule",
        "inspect_result_locations",
        "inspect_preparation_structure", "declare_preparation_study",
        "run_preparation_operator", "inspect_processivity_toolchain",
        "declare_sampling_setup", "record_preparation_findings", "finish_task",
        "submit_sampling", "check_sampling",
        "prepare_barrier_inputs", "plan_barrier_calculation",
        "submit_barrier_calculation", "check_barrier_calculation", "inspect_barrier_result",
        "record_barrier_assessment",
    }),
    "prepare_decision": frozenset({
        "record_preparation_findings", "declare_sampling_setup", "declare_preparation_study",
        "run_preparation_operator", "submit_sampling", "check_sampling", "finish_task",
        "prepare_barrier_inputs", "plan_barrier_calculation", "submit_barrier_calculation",
        "check_barrier_calculation", "record_barrier_assessment",
    }),
    "production_setup": frozenset({
        "inspect_candidate", "read_candidate_section", "inspect_investigation_lead",
        "inspect_molecule", "read_evidence", "retrieve_reference", "search_literature",
        "assess_assembly", "propose_model", "finish_task", "inspect_result_locations",
        "prepare_barrier_inputs", "plan_barrier_calculation",
        "submit_barrier_calculation", "check_barrier_calculation", "inspect_barrier_result",
        "record_barrier_assessment",
    }),
    "production_post_proposal": frozenset({
        "validate_setup", "render_inputs", "inspect_reasoning", "read_reasoning_section",
        "respond_to_reasoning_feedback", "record_reasoning_observation", "compare_hypothesis_tests",
        "analyze_pocket_envelope", "analyze_substrate_contacts", "analyze_substrate_channels",
        "analyze_side_reactions", "analyze_preorganization", "assess_developability",
        "build_catalytic_profile",
    }),
    "production_cached": frozenset({
        "inspect_candidate", "read_candidate_section", "read_evidence", "retrieve_reference",
        "review_existing_predictions", "finish_task", "inspect_result_locations",
        "check_barrier_calculation", "inspect_barrier_result",
    }),
    "evaluate_barriers": frozenset({
        "prepare_barrier_inputs", "plan_barrier_calculation",
        "submit_barrier_calculation", "check_barrier_calculation", "inspect_barrier_result",
        "record_barrier_assessment", "assess_mlmm_profile",
        "inspect_candidate", "read_candidate_section", "read_evidence",
        "retrieve_reference", "search_literature", "assess_assembly", "inspect_molecule",
        "inspect_result_locations", "run_preparation_operator", "inspect_preparation_structure",
        "record_preparation_findings", "finish_task",
    }),
}

_DOMAIN_BY_TOOL: dict[str, str] = {
    "prepare_barrier_inputs": "barrier",
    "inspect_barrier_result": "barrier",
    "plan_barrier_calculation": "barrier",
    "submit_barrier_calculation": "barrier",
    "check_barrier_calculation": "barrier",
    "record_barrier_assessment": "barrier",
    "inspect_experimental_proxy_bundle": "proxies",
    "declare_proxy_plan": "proxies",
    "evaluate_proxy_plan": "proxies",
    "inspect_reasoning": "reasoning",
    "read_reasoning_section": "reasoning",
    "compare_hypothesis_tests": "reasoning",
    "record_reasoning_observation": "reasoning",
    "respond_to_reasoning_feedback": "reasoning",
    "plan_followup": "followups",
    "prepare_bioemu2": "followups",
    "inspect_dynamics": "followups",
    "register_design_run": "followups",
    "inspect_preparation_structure": "sampling",
    "declare_preparation_study": "sampling",
    "record_preparation_findings": "sampling",
    "run_preparation_operator": "sampling",
    "declare_sampling_setup": "sampling",
    "inspect_processivity_toolchain": "sampling",
    "assess_processivity_network": "sampling",
    "propose_model": "cofold",
    "validate_setup": "cofold",
    "render_inputs": "cofold",
    "submit_cofold": "cofold",
    "check_cofold": "cofold",
    "validate_predictions": "cofold",
    "review_existing_predictions": "cofold",
}


def _stages_for(name: str) -> frozenset[str]:
    return frozenset(stage for stage, members in _STAGE_MEMBERSHIPS.items() if name in members)


def build_default_registry() -> ToolRegistry:
    """Build the unified registry from core, barrier, proxy, analysis, and ML/MM tools."""
    registry = ToolRegistry()

    scope_tables = (
        ("pre_accession", dh.PRE_ACCESSION_HANDLERS),
        ("general", dh.GENERAL_HANDLERS),
        ("candidate", dh.CANDIDATE_HANDLERS),
        ("revision", dh.REVISION_HANDLERS),
    )
    handler_and_scope: dict[str, tuple[str, Callable[..., Any]]] = {}
    for scope, table in scope_tables:
        for name, fn in table.items():
            handler_and_scope[name] = (scope, fn)

    for schema in build_core_schemas():
        name = schema["function"]["name"]
        scope, fn = handler_and_scope[name]
        registry.register(
            ToolSpec(
                name=name,
                schema=schema,
                handler=fn,
                scope=scope,
                domain=_DOMAIN_BY_TOOL.get(name, "core"),
                stages=_stages_for(name),
                repeatable_reader=name in _REPEATABLE_READER_NAMES,
                deprecated=name in _DEPRECATED_TOOLS,
            )
        )

    register_domain_tools(registry, ToolSpec, _stages_for, frozenset(_STAGE_MEMBERSHIPS))
    return registry


_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Return the process-wide cached ToolRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY


def select_offered_tools(agent) -> list[dict[str, Any]] | None:
    """Select the cached tool schemas offered to the LLM for the current turn.

    Derives stage filtering, preparation-progress guards, production setup
    bounds, and repeated-inspection suppression from the unified ToolRegistry.
    Returns ``None`` only when ``production_cofold`` hits ``setup_stalled``.
    """
    registry = get_registry()
    objective = agent.task.get("objective")
    workflow_mode = agent.task.get("workflow_mode")
    allowed: set[str] | None = None

    if objective == "prepare_sampling":
        if agent.state.get("preparation_read_turns", 0) >= 2:
            allowed = set(registry.names_for_stage("prepare_decision"))
        else:
            allowed = set(registry.names_for_stage("prepare_sampling"))

    if workflow_mode == "production_cofold" and objective not in {"evaluate_barriers", "prepare_sampling"}:
        before_proposal = not any(c["revisions"] for c in agent.state["candidates"].values())
        cached_only = all(
            (candidate.get("evidence") or {}).get("do_not_recofold")
            and (candidate.get("evidence") or {}).get("existing_cofolds")
            for candidate in agent.task["candidates"]
        )
        setup_limit = agent.task.get("setup_turn_limit", 8)
        if before_proposal and not cached_only and agent.state["turn"] > setup_limit:
            agent.state.update(
                status="setup_stalled",
                checkpoint_reason="No source-linked proposal within the bounded setup stage; preserve missing evidence and advance the research queue.",
            )
            return None
        if cached_only:
            setup_tools = set(registry.names_for_stage("production_cached"))
        elif before_proposal and agent.state["turn"] >= setup_limit:
            setup_tools = {"propose_model", "finish_task"}
            if not agent.state.get("setup_closing_notice"):
                agent.state["setup_closing_notice"] = True
                agent.messages.append({
                    "role": "user",
                    "content": (
                        "This is the last setup turn. Save a provisional proposal now from the evidence "
                        "you already hold, recording unresolved choices and their missing evidence in the "
                        "proposal itself; it is revisable afterwards. If the evidence cannot support any "
                        "defensible proposal, finish the task as inconclusive and state what is missing. "
                        "Do not retrieve or read further."
                    ),
                })
                agent._event("setup_closing_turn", turn=agent.state["turn"], setup_turn_limit=setup_limit)
        else:
            setup_tools = set(registry.names_for_stage("production_setup"))
            if not before_proposal:
                setup_tools |= registry.names_for_stage("production_post_proposal")
                if any((candidate.get("evidence") or {}).get("existing_cofolds") for candidate in agent.task["candidates"]):
                    setup_tools.add("validate_predictions")
                if agent.state.get("reuse_review_required") and all(
                    c["revisions"] and c["revisions"][-1].get("execution")
                    for c in agent.state["candidates"].values()
                ):
                    setup_tools.discard("render_inputs")
            if any(
                (c["revisions"][-1].get("pose_validation") or {}).get("poses")
                for c in agent.state["candidates"].values()
                if c["revisions"]
            ):
                setup_tools.discard("render_inputs")
        allowed = setup_tools if allowed is None else (allowed & setup_tools)

    if objective == "evaluate_barriers":
        barrier_allowed = set(registry.names_for_stage("evaluate_barriers"))
        allowed = barrier_allowed if allowed is None else (allowed & barrier_allowed)

    # Progressive domain disclosure (default ON for standard runs; pass
    # progressive_tool_disclosure=False in task to expose all non-deprecated tools).
    if agent.task.get("progressive_tool_disclosure", True) and allowed is None:
        from .skills import get_skill

        active_domains = {"core", "cofold", "reasoning"}
        for skill_name in agent.state.get("active_skills", ()):
            try:
                active_domains |= set(get_skill(skill_name).domains)
            except ValueError:
                continue
        allowed = {
            name
            for domain in active_domains
            for name in registry.names_for_domain(domain, include_deprecated=False)
        }
    elif allowed is None:
        allowed = set(registry.active_names())

    if agent.state.get("repeated_inspections", 0) >= 3:
        readers = registry.repeatable_readers()
        allowed = allowed - readers

    return registry.schemas(allowed)
