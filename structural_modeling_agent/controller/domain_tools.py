"""Domain-specific tool handlers and ToolSpec registrations for ModelingAgent.

Covers analysis (contacts, channels, pocket envelope, side reactions,
preorganization, catalytic profile), visualization, developability, ML/MM
gating, lazy skill loading, and reservation-aware sampling submission.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .tool_schemas import _schema, _string


def _cache_analysis(agent, args: dict[str, Any], key: str, result: dict[str, Any]) -> None:
    acc = args.get("accession")
    if acc and acc in agent.state.get("candidates", {}):
        agent.state["candidates"][acc].setdefault("analyses", {})[key] = result


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
        caver_dir = str(agent._authorized_evidence(args["caver_out_dir"], allow_dir=True))
        n_poses = args["n_poses"]
        clusters = ch.classify_clusters(ch.parse_caver_output(caver_dir, n_poses))
        result = {"descriptive_only": True, "n_poses": n_poses, "n_clusters": len(clusters), "clusters": clusters}
    elif mode == "run":
        paths = [str(agent._authorized_evidence(p)) for p in args.get("pose_paths", [])]
        if not paths:
            raise ValueError("run mode requires non-empty pose_paths")
        raw_label = args.get("label") or "run"
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(raw_label)) or "run"
        out_dir = agent.run_dir / "artifacts" / f"channels_{safe_label}"
        kwargs: dict[str, Any] = {
            "label": safe_label,
            "n_starts": args.get("n_starts", 1),
            "probe_radius": args.get("probe_radius", ch.DEFAULT_PROBE_RADIUS),
            "backend": args.get("backend", "fast"),
            "skip_open": args.get("skip_open", False),
        }
        for opt in ("start_xyz", "start_sel", "start_group", "reaction_center_sel", "hetero_sel", "keep_cofactor", "keep_metals"):
            if args.get(opt) is not None:
                kwargs[opt] = args[opt]
        if args.get("substrate_resnames"):
            kwargs["substrate_resnames"] = set(args["substrate_resnames"])
        if args.get("cofactor_resnames"):
            kwargs["cofactor_resnames"] = set(args["cofactor_resnames"])
        result = ch.analyze_channels(paths, str(out_dir), **kwargs)
    else:
        raise ValueError(f"unknown substrate channel mode: {mode!r}")
    artifact = agent._artifact(f"substrate_channels_{mode}", result)
    _cache_analysis(agent, args, "access", result)
    return {"artifact": artifact, **result}


def _h_load_skill(agent, args):
    from .skills import get_skill
    from .tool_registry import get_registry

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
    run_dir = str(agent._authorized_evidence(args["run_dir"], allow_dir=True))
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


def _h_assess_developability(agent, args):
    from ..developability.assess import assess_developability

    acc = args["accession"]
    candidate = agent._candidate(acc)
    fams = args.get("family_ids")
    if fams is None and args.get("family_id"):
        fams = [args["family_id"]]
    result = assess_developability(
        uniprot=acc,
        sequence=candidate.get("sequence"),
        family_ids=tuple(fams or ()),
        predictor_name=args.get("predictor_name", "netsolp"),
    )
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    artifact = agent._artifact(f"developability_{acc}", payload)
    agent.state["candidates"][acc]["developability"] = payload
    return {"artifact": artifact, **payload}


def _h_analyze_pocket_envelope(agent, args):
    from ..analysis.pocket_envelope import compare_native_vs_target

    target_poses = [str(agent._authorized_evidence(p)) for p in args["target_poses"]]
    native_poses = (
        [str(agent._authorized_evidence(p)) for p in args["native_poses"]]
        if args.get("native_poses")
        else None
    )
    result = compare_native_vs_target(
        target_poses,
        native_poses,
        target_substrate_chains=args.get("target_substrate_chains"),
        target_substrate_resnames=args.get("target_substrate_resnames"),
        native_substrate_chains=args.get("native_substrate_chains"),
        native_substrate_resnames=args.get("native_substrate_resnames"),
        reaction_center_xyz=args.get("reaction_center_xyz"),
        clash_cutoff_A=float(args.get("clash_cutoff_A", 2.2)),
    )
    artifact = agent._artifact("pocket_envelope", result)
    _cache_analysis(agent, args, "pocket_envelope", result)
    return {"artifact": artifact, **result}


def _h_analyze_side_reactions(agent, args):
    from ..analysis import side_reactions as sr

    mode = args["mode"]
    pathways = args["pathways"]
    if mode == "poses":
        paths = [str(agent._authorized_evidence(p)) for p in args.get("pose_paths", [])]
        if not paths:
            raise ValueError("poses mode requires non-empty pose_paths")
        result = sr.analyze_side_reactions_poses(paths, pathways)
    elif mode == "traj":
        top = str(agent._authorized_evidence(args["top_path"]))
        trajs = [str(agent._authorized_evidence(p)) for p in args.get("traj_paths", [])]
        if not trajs:
            raise ValueError("traj mode requires non-empty traj_paths")
        result = sr.analyze_side_reactions_trajectory(top, trajs, pathways, stride=int(args.get("stride", 1)))
    else:
        raise ValueError(f"unknown side-reaction analysis mode: {mode!r}")
    artifact = agent._artifact(f"side_reactions_{mode}", result)
    _cache_analysis(agent, args, "side_reactions", result)
    return {"artifact": artifact, **result}


def _h_analyze_preorganization(agent, args):
    from ..analysis.preorganization import analyze_preorganization

    apo = str(agent._authorized_evidence(args["apo_path"])) if args.get("apo_path") else None
    holo = str(agent._authorized_evidence(args["holo_path"])) if args.get("holo_path") else None
    poses = [str(agent._authorized_evidence(p)) for p in args["pose_paths"]] if args.get("pose_paths") else None
    prmtop = str(agent._authorized_evidence(args["prmtop_path"])) if args.get("prmtop_path") else None
    trajs = [str(agent._authorized_evidence(p)) for p in args["traj_paths"]] if args.get("traj_paths") else None

    result = analyze_preorganization(
        apo_path=apo,
        holo_path=holo,
        catalytic_residues=args.get("catalytic_residues"),
        pose_paths=poses,
        charge_center=args.get("charge_center"),
        prmtop_path=prmtop,
        traj_paths=trajs,
        axis_a_sel=args.get("axis_a_sel"),
        axis_b_sel=args.get("axis_b_sel"),
        environment_sel=args.get("environment_sel", "protein"),
        expected_favorable_sign=args.get("expected_favorable_sign"),
        cutoff_A=float(args.get("cutoff_A", 15.0)),
        stride=int(args.get("stride", 1)),
    )
    artifact = agent._artifact("preorganization", result)
    _cache_analysis(agent, args, "ts_and_preorganization", result)
    return {"artifact": artifact, **result}


def _h_build_catalytic_profile(agent, args):
    from ..analysis.catalytic_profile import build_catalytic_profile

    acc = args["accession"]
    cand = agent.state["candidates"][acc]
    cached = cand.get("analyses") or {}
    profile = build_catalytic_profile(
        acc,
        mechanism_gate=args.get("mechanism_gate"),
        pocket_envelope=args.get("pocket_envelope") or cached.get("pocket_envelope"),
        access=args.get("access") or cached.get("access"),
        reactive_persistence=args.get("reactive_persistence") or cached.get("reactive_persistence"),
        ts_and_preorganization=args.get("ts_and_preorganization") or cached.get("ts_and_preorganization"),
        side_reactions=args.get("side_reactions") or cached.get("side_reactions"),
        barrier_status=args.get("barrier_status") or cached.get("barrier_status"),
        developability=args.get("developability") or cand.get("developability"),
    )
    artifact = agent._artifact(f"catalytic_profile_{acc}", profile)
    cand["catalytic_profile"] = profile
    return {"artifact": artifact, **profile}


def _h_assess_mlmm_profile(agent, args):
    from ..mlmm import NotReportable, diagram_values, profile_from_freq_dirs

    results_dir = str(agent._authorized_evidence(args["results_dir"], allow_dir=True))
    profile = profile_from_freq_dirs(
        results_dir,
        system_id=args["system_id"],
        step_id=args["step_id"],
        net_charge=args["net_charge"],
    )
    strict = bool(args.get("strict", True))
    try:
        values = diagram_values(profile)
        payload = {"ok": True, "reportable": True, "diagram": values, "system_id": args["system_id"], "step_id": args["step_id"]}
    except NotReportable as exc:
        if strict:
            payload = {"ok": False, "reportable": False, "reason": str(exc), "system_id": args["system_id"], "step_id": args["step_id"]}
        else:
            payload = {"ok": False, "reportable": False, "warning": str(exc), "system_id": args["system_id"], "step_id": args["step_id"]}
    artifact = agent._artifact(f"mlmm_verdict_{args['system_id']}", payload)
    _cache_analysis(agent, args, "barrier_status", payload)
    return {"artifact": artifact, **payload}


def _h_submit_sampling(agent, args):
    from ..sampling.workflow import submit_sampling

    return submit_sampling(agent, args)


def _h_check_sampling(agent, args):
    from ..sampling.workflow import check_sampling

    return check_sampling(agent, args)


def register_domain_tools(registry, ToolSpec, stages_for: Callable[[str], frozenset[str]], all_stages: frozenset[str]) -> None:
    """Register domain-specific analysis, visualization, ML/MM, skill, and sampling tools."""
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
            stages=stages_for("analyze_substrate_contacts"),
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
                    "accession": _string("Optional candidate identifier for caching results into build_catalytic_profile"),
                    "caver_out_dir": _string("Authorized CAVER output directory (mode=parse)"),
                    "n_poses": {"type": "integer", "minimum": 1, "description": "Number of poses in CAVER run (mode=parse)"},
                    "pose_paths": {"type": "array", "items": {"type": "string"}, "description": "Authorized full-assembly pose paths (mode=run)"},
                    "label": _string("Label for output folder (mode=run)"),
                    "n_starts": {"type": "integer", "minimum": 1, "maximum": 5},
                    "probe_radius": {"type": "number", "minimum": 0.5},
                    "backend": {"type": "string", "enum": ["fast", "caver"], "description": "fast (default native scipy Voronoi/Dijkstra) or caver (CAVER 3.0 JVM)"},
                    "skip_open": {"type": "boolean", "description": "Skip tunnel tracing when ray-cast burial indicates a solvent-open site"},
                    "start_xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "Optional explicit starting point [x, y, z] for apo or custom-site channel tracing"},
                    "start_sel": _string("Optional MDAnalysis selection for starting point"),
                    "start_group": _string("Optional ligand functional-group hint for starting point"),
                    "substrate_resnames": {"type": "array", "items": {"type": "string"}},
                    "cofactor_resnames": {"type": "array", "items": {"type": "string"}},
                    "hetero_sel": _string("Optional MDAnalysis ligand selection"),
                    "keep_cofactor": {"type": "boolean"},
                    "keep_metals": {"type": "boolean"},
                },
                ("mode",),
            ),
            handler=_h_analyze_substrate_channels,
            scope="general",
            domain="analysis",
            stages=stages_for("analyze_substrate_channels"),
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
            stages=stages_for("assess_mlmm_profile"),
        )
    )

    from .skills import list_skills

    skill_names = [s["name"] for s in list_skills()]
    registry.register(
        ToolSpec(
            name="load_skill",
            schema=_schema(
                "load_skill",
                "Load a domain playbook (scientific checklist, caveats, and tool workflow) and unlock that domain's tools when progressive disclosure is active",
                {
                    "name": {
                        "type": "string",
                        "enum": skill_names,
                        "description": "Skill name from the available skills index",
                    }
                },
                ("name",),
            ),
            handler=_h_load_skill,
            scope="general",
            domain="core",
            stages=all_stages,
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

    registry.register(
        ToolSpec(
            name="assess_developability",
            schema=_schema(
                "assess_developability",
                "Evaluate curated expression/membrane liabilities and predicted solubility for a candidate; fails closed and emits an exploration_trigger for soluble homolog search when vetoed/flagged",
                {
                    "accession": _string("Candidate identifier from this task"),
                    "family_id": _string("Optional single InterPro/mechanism family identifier"),
                    "family_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional list of InterPro/mechanism family identifiers"},
                    "predictor_name": _string("Solubility predictor name (default netsolp)"),
                },
                ("accession",),
            ),
            handler=_h_assess_developability,
            scope="general",
            domain="developability",
            stages=stages_for("assess_developability"),
        )
    )

    registry.register(
        ToolSpec(
            name="analyze_pocket_envelope",
            schema=_schema(
                "analyze_pocket_envelope",
                "Enzyme-agnostic asymmetric pocket-envelope and distal-tail exposure analysis; optionally compares target-holo against a native-holo control to separate modeling artifacts, buried steric exclusion, and pocket slack",
                {
                    "accession": _string("Optional candidate identifier for caching results into build_catalytic_profile"),
                    "target_poses": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Authorized target-holo PDB/mmCIF paths"},
                    "native_poses": {"type": "array", "items": {"type": "string"}, "description": "Optional authorized native-holo PDB/mmCIF control paths"},
                    "target_substrate_chains": {"type": "array", "items": {"type": "string"}},
                    "target_substrate_resnames": {"type": "array", "items": {"type": "string"}},
                    "native_substrate_chains": {"type": "array", "items": {"type": "string"}},
                    "native_substrate_resnames": {"type": "array", "items": {"type": "string"}},
                    "reaction_center_xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "clash_cutoff_A": {"type": "number", "minimum": 0.5},
                },
                ("target_poses",),
            ),
            handler=_h_analyze_pocket_envelope,
            scope="general",
            domain="analysis",
            stages=stages_for("analyze_pocket_envelope"),
        )
    )

    atom_ref_schema = {
        "type": "object",
        "properties": {
            "chain": _string("Chain ID (e.g. A, B)"),
            "residue": _string("Residue number/ID as string (e.g. '1', '117')"),
            "atom": _string("PDB atom name (e.g. C1, O1, NZ, PG)"),
        },
        "required": ["chain", "residue", "atom"],
    }
    res_ref_schema = {
        "type": "object",
        "properties": {
            "chain": _string("Chain ID"),
            "residue": _string("Residue number/ID as string"),
        },
        "required": ["chain", "residue"],
    }
    pathway_schema = {
        "type": "object",
        "properties": {
            "name": _string("Descriptive competing-pathway label (e.g. water_attack, futile_atp_hydrolysis, intramolecular_cyclization)"),
            "kind": {"type": "string", "enum": ["solvent", "intramolecular", "intermolecular"]},
            "distance_cutoff_A": {"type": "number", "minimum": 0.5, "description": "Maximum nucleophile-electrophile distance in Angstroms"},
            "ideal_angle_deg": {"type": "number", "description": "Optional ideal approach angle in degrees (e.g. 107 for carbonyl, 180 for inline SN2; omit for distance-only)"},
            "angle_tolerance_deg": {"type": "number", "description": "Optional +/- angle tolerance in degrees"},
            "warning_threshold": {"type": "number", "description": "Occupancy fraction above which to flag risk (default 0.30)"},
            "electrophile": atom_ref_schema,
            "angle_reference": atom_ref_schema,
            "nucleophiles": {"type": "array", "items": atom_ref_schema, "description": "Static pose nucleophile atom refs (mode=poses, kind!=solvent)"},
            "solvent_resnames": {"type": "array", "items": {"type": "string"}, "description": "Optional solvent residue names (mode=poses, kind=solvent)"},
            "solvent_atom_name": _string("Solvent attacking atom name prefix (default 'O')"),
            "electrophile_sel": _string("MDAnalysis 1-atom selection for electrophile (mode=traj)"),
            "angle_reference_sel": _string("Optional MDAnalysis 1-atom selection for angle reference (mode=traj)"),
            "nucleophile_sel": _string("MDAnalysis selection for competing nucleophiles (mode=traj)"),
        },
        "required": ["name", "kind", "distance_cutoff_A"],
    }

    registry.register(
        ToolSpec(
            name="analyze_side_reactions",
            schema=_schema(
                "analyze_side_reactions",
                "Reaction-agnostic competing-pathway geometry analyzer for pose ensembles or MD trajectories (e.g., solvent hydrolysis at any angle/distance, intramolecular cyclization, futile cofactor hydrolysis, or off-target C-H abstraction)",
                {
                    "accession": _string("Optional candidate identifier for caching results into build_catalytic_profile"),
                    "mode": {"type": "string", "enum": ["poses", "traj"]},
                    "pathways": {"type": "array", "items": pathway_schema, "minItems": 1, "description": "Declared competing-attack pathway specs"},
                    "pose_paths": {"type": "array", "items": {"type": "string"}},
                    "top_path": _string("Authorized topology path (mode=traj)"),
                    "traj_paths": {"type": "array", "items": {"type": "string"}},
                    "stride": {"type": "integer", "minimum": 1},
                },
                ("mode", "pathways"),
            ),
            handler=_h_analyze_side_reactions,
            scope="general",
            domain="analysis",
            stages=stages_for("analyze_side_reactions"),
        )
    )

    registry.register(
        ToolSpec(
            name="analyze_preorganization",
            schema=_schema(
                "analyze_preorganization",
                "Enzyme-agnostic preorganization and TS/intermediate stabilization diagnostics: apo-vs-holo catalytic-constellation RMSD, polar/metal stabilization of any declared charge center, and MD-averaged projected electric field (MV/cm) along any reaction axis",
                {
                    "accession": _string("Optional candidate identifier for caching results into build_catalytic_profile"),
                    "apo_path": _string("Authorized apo PDB/mmCIF path"),
                    "holo_path": _string("Authorized holo PDB/mmCIF path"),
                    "catalytic_residues": {"type": "array", "items": res_ref_schema, "description": "Catalytic residues for apo-vs-holo heavy-atom RMSD"},
                    "pose_paths": {"type": "array", "items": {"type": "string"}, "description": "Authorized holo/TI pose paths for charge-center stabilization"},
                    "charge_center": atom_ref_schema,
                    "prmtop_path": _string("Authorized Amber prmtop path for MD electric field"),
                    "traj_paths": {"type": "array", "items": {"type": "string"}},
                    "axis_a_sel": _string("MDAnalysis selection for reaction axis start atom"),
                    "axis_b_sel": _string("MDAnalysis selection for reaction axis end atom"),
                    "environment_sel": _string("MDAnalysis selection for field-producing environment (default protein)"),
                    "expected_favorable_sign": {"type": "integer", "enum": [-1, 1]},
                    "cutoff_A": {"type": "number", "minimum": 2.0},
                    "stride": {"type": "integer", "minimum": 1},
                },
                (),
            ),
            handler=_h_analyze_preorganization,
            scope="general",
            domain="analysis",
            stages=stages_for("analyze_preorganization"),
        )
    )

    registry.register(
        ToolSpec(
            name="build_catalytic_profile",
            schema=_schema(
                "build_catalytic_profile",
                "Assemble a non-collapsed multi-axis CatalyticProfile with explicit failure modes (hard_reject vs soft_evolvable_defect vs modeling_artifact) and residue-level design targets. Automatically reuses cached outputs from analyze_pocket_envelope, analyze_substrate_channels, analyze_side_reactions, analyze_preorganization, assess_mlmm_profile, and assess_developability when omitted.",
                {
                    "accession": _string("Candidate identifier from this task"),
                    "mechanism_gate": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                            "reason": _string("Mechanism compatibility rationale"),
                        },
                    },
                    "pocket_envelope": {"type": "object", "description": "Optional override; auto-filled from cached analyze_pocket_envelope"},
                    "access": {"type": "object", "description": "Optional override; auto-filled from cached analyze_substrate_channels"},
                    "reactive_persistence": {
                        "type": "object",
                        "properties": {
                            "boltz_nac_fraction": {"type": "number"},
                            "md_nac_fraction": {"type": "number"},
                        },
                        "description": "Near-attack geometry fractions from cofold and MD",
                    },
                    "ts_and_preorganization": {"type": "object", "description": "Optional override; auto-filled from cached analyze_preorganization"},
                    "side_reactions": {"type": "object", "description": "Optional override; auto-filled from cached analyze_side_reactions"},
                    "barrier_status": {"type": "object", "description": "Optional override; auto-filled from cached assess_mlmm_profile"},
                    "developability": {"type": "object", "description": "Optional override; auto-filled from cached assess_developability"},
                },
                ("accession",),
            ),
            handler=_h_build_catalytic_profile,
            scope="general",
            domain="core",
            stages=stages_for("build_catalytic_profile"),
        )
    )

    registry.register(
        ToolSpec(
            name="submit_sampling",
            schema=_schema(
                "submit_sampling",
                "Submit a reviewed MD or umbrella-sampling job (openmm_classical or ase_mlip_pilot) once through the configured reservation-aware sampling backend; never runs dynamics in the controller process",
                {
                    "accession": _string("Candidate identifier from this task"),
                    "setup_id": _string("Ready sampling setup ID returned by declare_sampling_setup"),
                    "label": _string("Safe version label for this sampling run"),
                    "engine": {"type": "string", "enum": ["openmm_classical", "ase_mlip_pilot"]},
                    "wall_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                    "inputs": {"type": "object", "description": "Authorized input paths: {prmtop, coordinates, spec_path} for openmm_classical or {config_path} for ase_mlip_pilot"},
                    "settings": {"type": "object", "description": "Optional engine settings (e.g. production_ns, temperature_K, seed, strip_water)"},
                },
                ("accession", "setup_id", "label", "engine", "wall_seconds", "inputs"),
            ),
            handler=_h_submit_sampling,
            scope="general",
            domain="sampling",
            stages=stages_for("submit_sampling"),
        )
    )

    registry.register(
        ToolSpec(
            name="check_sampling",
            schema=_schema(
                "check_sampling",
                "Reconcile an existing sampling job receipt, verify output hashes, and return ready-to-chain analysis_inputs (top_path, traj_paths, geometry_jsonl) on completion",
                {
                    "accession": _string("Candidate identifier from this task"),
                    "job_id": _string("Sampling job ID returned by submit_sampling"),
                },
                ("accession", "job_id"),
            ),
            handler=_h_check_sampling,
            scope="general",
            domain="sampling",
            stages=stages_for("check_sampling"),
            repeatable_reader=True,
        )
    )
