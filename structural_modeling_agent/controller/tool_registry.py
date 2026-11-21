"""Unified declarative tool registry for ModelingAgent.

Single source of truth for tool schemas, dispatch handlers, scope/accession
handling, objective/stage availability, and repeated-inspection guards.
Domain modules register ToolSpec objects so adding or renaming a tool never
requires manually syncing multiple files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import dispatch_handlers as dh
from .tool_schemas import _schema, _string, build_core_schemas

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

    def names_for_domain(self, domain: str) -> frozenset[str]:
        return frozenset(spec.name for spec in self._specs.values() if spec.domain == domain)

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
# Newly registered domain handlers (analysis & ML/MM gating)
# ---------------------------------------------------------------------------

def _h_analyze_substrate_contacts(agent, args):
    from ..analysis import substrate_contacts as sc
    if sc.mda is None:
        raise RuntimeError(f"MDAnalysis is required for substrate contact analysis: {sc._MDA_ERR}")
    mode = args["mode"]
    mech = args.get("mech")
    rc_sel = args.get("reaction_center_sel")
    subr = set(args["substrate_resnames"]) if args.get("substrate_resnames") else None
    cofr = set(args["cofactor_resnames"]) if args.get("cofactor_resnames") else None
    hsel = args.get("hetero_sel")
    if mode == "pose":
        path = str(agent._authorized_evidence(args["pose_path"]))
        result = sc.analyze_structure(sc.mda.Universe(path), mech, rc_sel, subr, cofr, hsel)
    elif mode == "ensemble":
        paths = [str(agent._authorized_evidence(p)) for p in args.get("pose_paths", [])]
        if not paths:
            raise ValueError("ensemble mode requires non-empty pose_paths")
        result = sc.analyze_ensemble(paths, mech, rc_sel, subr, cofr)
    elif mode == "traj":
        top = str(agent._authorized_evidence(args["top_path"]))
        trajs = [str(agent._authorized_evidence(p)) for p in args.get("traj_paths", [])]
        if not trajs:
            raise ValueError("traj mode requires non-empty traj_paths")
        result = sc.analyze_trajectory(top, trajs, mech, rc_sel, subr, cofr, stride=args.get("stride", 1), hetero_sel=hsel)
    else:
        raise ValueError(f"unknown substrate contact mode: {mode!r}")
    artifact = agent._artifact(f"substrate_contacts_{mode}", result)
    return {"artifact": artifact, **result}


def _h_analyze_substrate_channels(agent, args):
    from ..analysis import substrate_channels as ch
    mode = args.get("mode", "parse")
    if mode == "parse":
        caver_dir = str(agent._authorized_evidence(args["caver_out_dir"]))
        n_poses = args["n_poses"]
        clusters = ch.classify_clusters(ch.parse_caver_output(caver_dir, n_poses))
        result = {"descriptive_only": True, "n_poses": n_poses, "n_clusters": len(clusters), "clusters": clusters}
    elif mode == "run":
        paths = [str(agent._authorized_evidence(p)) for p in args.get("pose_paths", [])]
        if not paths:
            raise ValueError("run mode requires non-empty pose_paths")
        out_dir = agent.run_dir / "artifacts" / f"channels_{args.get('label', 'run')}"
        result = ch.analyze_channels(
            paths,
            str(out_dir),
            label=args.get("label"),
            n_starts=args.get("n_starts", 1),
            probe_radius=args.get("probe_radius", ch.DEFAULT_PROBE_RADIUS),
            backend=args.get("backend", "fast"),
            skip_open=args.get("skip_open", False),
        )
    else:
        raise ValueError(f"unknown substrate channel mode: {mode!r}")
    artifact = agent._artifact(f"substrate_channels_{mode}", result)
    return {"artifact": artifact, **result}


def _h_load_skill(agent, args):
    from .skills import get_skill

    skill = get_skill(args["name"])
    active = agent.state.setdefault("active_skills", [])
    if skill.name not in active:
        active.append(skill.name)
    registry = get_registry()
    unlocked = sorted(
        name
        for domain in skill.domains
        for name in registry.names_for_domain(domain)
    )
    return {
        "name": skill.name,
        "description": skill.description,
        "domains": sorted(skill.domains),
        "tools_unlocked": unlocked,
        "instructions": skill.body,
    }


def _h_render_channel_visualization(agent, args):
    run_dir = str(agent._authorized_evidence(args["run_dir"]))
    fmt = args.get("format", "html")
    label = args.get("label") or Path(run_dir).name
    title = args.get("title") or label
    out_root = agent.run_dir / "artifacts" / "visualizations"
    out_root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    if fmt == "html":
        from ..analysis import channels_html
        out_path = str(out_root / f"{safe_label}_channels.html")
        channels_html.build(
            [(label, [(args.get("panel_label") or label, run_dir)])],
            out_path,
            title=title,
            cdn=bool(args.get("cdn", False)),
        )
    elif fmt == "png":
        from ..analysis import render_channels
        out_path = str(out_root / f"{safe_label}_channels.png")
        render_channels.render(
            run_dir,
            out_path,
            title=title,
            all_clusters=bool(args.get("all_clusters", False)),
            max_residues=int(args.get("max_residues", 0)),
            cutaway=bool(args.get("cutaway", False)),
            publication=bool(args.get("publication", False)),
        )
    else:
        raise ValueError(f"unknown channel visualization format: {fmt!r}")
    return {"format": fmt, "output_path": out_path, "run_dir": run_dir}


def _h_render_contact_figure(agent, args):
    from ..analysis import render_contacts

    pose_path = str(agent._authorized_evidence(args["pose_path"]))
    freq_path = str(agent._authorized_evidence(args["freq_json_path"]))
    md_freq_path = (
        str(agent._authorized_evidence(args["md_freq_json_path"]))
        if args.get("md_freq_json_path")
        else None
    )
    label = args.get("label") or Path(pose_path).stem
    title = args.get("title") or label
    out_root = agent.run_dir / "artifacts" / "visualizations"
    out_root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    out_path = str(out_root / f"{safe_label}_contact_frequency.png")
    render_contacts.build_figure(pose_path, freq_path, title, out_path, md_freq_json=md_freq_path)
    return {"output_path": out_path, "pose_path": pose_path, "freq_json_path": freq_path}


def _h_assess_mlmm_profile(agent, args):
    from ..mlmm import evaluate_profile, profile_from_freq_dirs

    results_dir = str(agent._authorized_evidence(args["results_dir"]))
    profile = profile_from_freq_dirs(
        results_dir,
        system_id=args["system_id"],
        step_id=args["step_id"],
        net_charge=args["net_charge"],
    )
    verdict = evaluate_profile(profile, strict=args.get("strict", True))
    payload = verdict.to_dict() if hasattr(verdict, "to_dict") else {"ok": bool(verdict.ok), "verdict": str(verdict)}
    artifact = agent._artifact(f"mlmm_verdict_{args['system_id']}", payload)
    return {"artifact": artifact, **payload}


# ---------------------------------------------------------------------------
# Declarative stage, domain, and repeatable-reader metadata
# ---------------------------------------------------------------------------

_REPEATABLE_READER_NAMES = frozenset({
    "inspect_candidate",
    "inspect_validation_contract",
    "inspect_workflow_policy",
    "read_candidate_section",
    "read_evidence",
    "read_reasoning_section",
    "inspect_preparation_contract",
    "inspect_sampling_setup_contract",
    "inspect_processivity_contract",
    "inspect_processivity_toolchain",
    "inspect_mlmm_engine_options",
})

_STAGE_MEMBERSHIPS: dict[str, frozenset[str]] = {
    "prepare_sampling": frozenset({
        "inspect_candidate", "read_candidate_section", "read_evidence",
        "retrieve_reference", "search_literature", "assess_assembly", "inspect_molecule",
        "inspect_result_locations", "inspect_mlmm_engine_options", "inspect_preparation_contract",
        "inspect_preparation_structure", "declare_preparation_study", "inspect_preparation_study",
        "run_preparation_operator", "inspect_sampling_setup_contract", "inspect_processivity_toolchain",
        "declare_sampling_setup", "inspect_sampling_setup", "record_preparation_findings", "finish_task",
        "inspect_barrier_contract", "prepare_barrier_inputs", "plan_barrier_calculation",
        "submit_barrier_calculation", "check_barrier_calculation", "inspect_barrier_result",
        "record_barrier_assessment",
    }),
    "prepare_decision": frozenset({
        "record_preparation_findings", "declare_sampling_setup", "declare_preparation_study",
        "run_preparation_operator", "finish_task",
        "prepare_barrier_inputs", "plan_barrier_calculation", "submit_barrier_calculation",
        "check_barrier_calculation", "record_barrier_assessment",
    }),
    "production_setup": frozenset({
        "inspect_candidate", "read_candidate_section", "inspect_investigation_lead",
        "inspect_molecule", "read_evidence", "retrieve_reference", "search_literature",
        "assess_assembly", "propose_model", "finish_task", "inspect_result_locations",
        "inspect_barrier_contract", "prepare_barrier_inputs", "plan_barrier_calculation",
        "submit_barrier_calculation", "check_barrier_calculation", "inspect_barrier_result",
        "record_barrier_assessment",
    }),
    "production_post_proposal": frozenset({
        "validate_setup", "render_inputs", "inspect_reasoning", "read_reasoning_section",
        "respond_to_reasoning_feedback", "record_reasoning_observation", "compare_hypothesis_tests",
    }),
    "production_cached": frozenset({
        "inspect_candidate", "read_candidate_section", "read_evidence", "retrieve_reference",
        "review_existing_predictions", "finish_task", "inspect_result_locations",
        "inspect_barrier_contract", "check_barrier_calculation", "inspect_barrier_result",
    }),
    "evaluate_barriers": frozenset({
        "inspect_barrier_contract", "prepare_barrier_inputs", "plan_barrier_calculation",
        "submit_barrier_calculation", "check_barrier_calculation", "inspect_barrier_result",
        "record_barrier_assessment",
        "inspect_candidate", "read_candidate_section", "read_evidence",
        "retrieve_reference", "search_literature", "assess_assembly", "inspect_molecule",
        "inspect_result_locations", "run_preparation_operator", "inspect_preparation_structure",
        "inspect_preparation_contract", "record_preparation_findings", "finish_task",
    }),
}

_DOMAIN_BY_TOOL: dict[str, str] = {
    "inspect_barrier_contract": "barrier",
    "prepare_barrier_inputs": "barrier",
    "inspect_barrier_result": "barrier",
    "plan_barrier_calculation": "barrier",
    "submit_barrier_calculation": "barrier",
    "check_barrier_calculation": "barrier",
    "record_barrier_assessment": "barrier",
    "get_structural_proxy_guidance": "proxies",
    "validate_structural_proxy_plan": "proxies",
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
    "inspect_preparation_contract": "sampling",
    "inspect_preparation_structure": "sampling",
    "declare_preparation_study": "sampling",
    "inspect_preparation_study": "sampling",
    "record_preparation_findings": "sampling",
    "run_preparation_operator": "sampling",
    "inspect_sampling_setup_contract": "sampling",
    "declare_sampling_setup": "sampling",
    "inspect_sampling_setup": "sampling",
    "inspect_processivity_toolchain": "sampling",
    "inspect_processivity_contract": "sampling",
    "assess_processivity_network": "sampling",
    "propose_model": "cofold",
    "validate_setup": "cofold",
    "render_inputs": "cofold",
    "submit_cofold": "cofold",
    "check_cofold": "cofold",
    "validate_predictions": "cofold",
    "review_existing_predictions": "cofold",
    "inspect_mlmm_engine_options": "mlmm",
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
            )
        )

    # Register domain-specific analysis & ML/MM tools cleanly via ToolSpec
    registry.register(
        ToolSpec(
            name="analyze_substrate_contacts",
            schema=_schema(
                "analyze_substrate_contacts",
                "Detect and aggregate substrate-accommodation contacts (salt bridges, H-bonds, hydrophobic packing, metal coordination) for a pose, Boltz ensemble, or MD trajectory; facts only, no hold/drift verdict",
                {
                    "mode": {"type": "string", "enum": ["pose", "ensemble", "traj"]},
                    "pose_path": _string("Authorized PDB path (mode=pose)"),
                    "pose_paths": {"type": "array", "items": {"type": "string"}, "description": "Authorized PDB paths (mode=ensemble)"},
                    "top_path": _string("Authorized topology path (mode=traj)"),
                    "traj_paths": {"type": "array", "items": {"type": "string"}, "description": "Authorized trajectory paths (mode=traj)"},
                    "stride": {"type": "integer", "minimum": 1},
                    "mech": _string("Optional mechanism hint"),
                    "reaction_center_sel": _string("Optional MDAnalysis selection for reaction center"),
                    "substrate_resnames": {"type": "array", "items": {"type": "string"}},
                    "cofactor_resnames": {"type": "array", "items": {"type": "string"}},
                    "hetero_sel": _string("Optional MDAnalysis ligand selection (excludes solvent/ions in MD)"),
                },
                ("mode",),
            ),
            handler=_h_analyze_substrate_contacts,
            scope="general",
            domain="analysis",
        )
    )

    registry.register(
        ToolSpec(
            name="analyze_substrate_channels",
            schema=_schema(
                "analyze_substrate_channels",
                "Parse existing CAVER 3.0 output or run descriptive full-assembly transport-channel analysis across an authorized pose ensemble; descriptive only, no ranking verdict",
                {
                    "mode": {"type": "string", "enum": ["parse", "run"]},
                    "caver_out_dir": _string("Authorized CAVER output directory (mode=parse)"),
                    "n_poses": {"type": "integer", "minimum": 1, "description": "Number of poses in CAVER run (mode=parse)"},
                    "pose_paths": {"type": "array", "items": {"type": "string"}, "description": "Authorized full-assembly pose paths (mode=run)"},
                    "label": _string("Label for output folder (mode=run)"),
                    "n_starts": {"type": "integer", "minimum": 1, "maximum": 5},
                    "probe_radius": {"type": "number", "minimum": 0.5},
                    "backend": {"type": "string", "enum": ["fast", "caver"], "description": "fast (default native scipy Voronoi/Dijkstra) or caver (CAVER 3.0 JVM)"},
                    "skip_open": {"type": "boolean", "description": "Skip tunnel tracing when ray-cast burial indicates a solvent-open site"},
                },
                ("mode",),
            ),
            handler=_h_analyze_substrate_channels,
            scope="general",
            domain="analysis",
        )
    )

    registry.register(
        ToolSpec(
            name="assess_mlmm_profile",
            schema=_schema(
                "assess_mlmm_profile",
                "Ingest an authorized ML/MM frequency/thermochemistry directory and run the fail-closed barrier reporting gate (imaginary-mode recount, geometry-based endpoint labeling, mode-count checks)",
                {
                    "results_dir": _string("Authorized ML/MM frequency output directory"),
                    "system_id": _string("System identifier"),
                    "step_id": _string("Reaction step identifier"),
                    "net_charge": {"type": "integer", "description": "Integer net charge of the ML/MM region"},
                    "strict": {"type": "boolean", "description": "Fail-closed strict reporting gate (default true)"},
                },
                ("results_dir", "system_id", "step_id", "net_charge"),
            ),
            handler=_h_assess_mlmm_profile,
            scope="general",
            domain="mlmm",
        )
    )

    registry.register(
        ToolSpec(
            name="load_skill",
            schema=_schema(
                "load_skill",
                "Load a domain playbook (scientific checklist, caveats, and tool workflow) and unlock that domain's tools when progressive disclosure is active",
                {"name": _string("Skill name from the available skills index (e.g. cofold-setup, md-preparation, reaction-barriers, contacts-and-channels, processivity-kinetics, dynamics-and-design, visualization)")},
                ("name",),
            ),
            handler=_h_load_skill,
            scope="general",
            domain="core",
            stages=frozenset(_STAGE_MEMBERSHIPS),
        )
    )

    registry.register(
        ToolSpec(
            name="render_channel_visualization",
            schema=_schema(
                "render_channel_visualization",
                "Render an interactive 3Dmol.js HTML viewer (fast, ~1-2 s, default) or a house-style PyMOL+matplotlib PNG figure from a substrate_channels output directory",
                {
                    "run_dir": _string("Authorized substrate_channels output directory containing channels.json"),
                    "format": {"type": "string", "enum": ["html", "png"], "description": "html (fast 3Dmol.js interactive, default) or png (PyMOL ray-traced)"},
                    "label": _string("Short enzyme label for output filename and dropdown"),
                    "panel_label": _string("Panel subtitle in HTML viewer"),
                    "title": _string("Figure or page title"),
                    "publication": {"type": "boolean", "description": "Full antialias PyMOL render (png only; default false = fast draft)"},
                    "all_clusters": {"type": "boolean", "description": "Draw all clusters, not only is_channel (png only)"},
                    "max_residues": {"type": "integer", "minimum": 0, "description": "Number of bottleneck-lining residues to show as sticks (png only)"},
                    "cutaway": {"type": "boolean", "description": "Solid cutaway surface instead of translucent cartoon (png only)"},
                    "cdn": {"type": "boolean", "description": "Load 3Dmol.js from CDN instead of inlining (html only)"},
                },
                ("run_dir",),
            ),
            handler=_h_render_channel_visualization,
            scope="general",
            domain="visualization",
        )
    )

    registry.register(
        ToolSpec(
            name="render_contact_figure",
            schema=_schema(
                "render_contact_figure",
                "Render a paired house-style PyMOL active-site structure + contact-frequency bar chart from an authorized pose PDB and substrate_contacts frequency JSON",
                {
                    "pose_path": _string("Authorized representative holo pose PDB path"),
                    "freq_json_path": _string("Authorized substrate_contacts frequency JSON path"),
                    "md_freq_json_path": _string("Optional authorized MD frequency JSON for hatched overlay bars"),
                    "label": _string("Short label for output filename"),
                    "title": _string("Figure title"),
                },
                ("pose_path", "freq_json_path"),
            ),
            handler=_h_render_contact_figure,
            scope="general",
            domain="visualization",
        )
    )

    return registry


_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Return the process-wide cached ToolRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY


def select_offered_tools(agent) -> list[dict[str, Any]]:
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

    # Opt-in progressive domain disclosure: start with core + cofold + reasoning tools
    # and unlock domain tools as skills are activated via load_skill(name).
    if agent.task.get("progressive_tool_disclosure") and allowed is None:
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
            for name in registry.names_for_domain(domain)
        }

    if agent.state.get("repeated_inspections", 0) >= 3:
        readers = registry.repeatable_readers()
        if allowed is None:
            allowed = set(registry.names()) - readers
        else:
            allowed = allowed - readers

    return registry.schemas(allowed)
