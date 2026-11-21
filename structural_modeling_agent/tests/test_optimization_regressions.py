"""Regression and parity tests for structural_modeling_agent optimizations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from structural_modeling_agent import env_config, io_utils
from structural_modeling_agent.agent import (
    BARRIER_GUIDANCE,
    BARRIER_PROMPT,
    PREPARATION_PROMPT,
    SYSTEM_PROMPT,
    ModelingAgent,
    _merge_model_changes,
    _text_page,
    tool_schemas,
)
from structural_modeling_agent.tool_registry import ToolRegistry, ToolSpec, get_registry
from structural_modeling_agent.tool_schemas import _schema
from structural_modeling_agent.proxies.roles import Atom, Pose
from structural_modeling_agent.proxies.workflow import VDW, contacts, group_sasa
from structural_modeling_agent.tests.test_validation import execution_for, modeling_record, write_pose
from structural_modeling_agent.validation import validate_ensemble


def test_io_utils_digest_and_bounded_workers(tmp_path):
    sample = tmp_path / "payload.bin"
    data = b"enzyme-structural-modeling-check" * 100
    sample.write_bytes(data)
    assert io_utils.file_digest(sample) == hashlib.sha256(data).hexdigest()
    assert io_utils.bounded_workers(4, n_items=2, min_items_for_pool=4) == 1
    assert io_utils.bounded_workers(1, n_items=50, min_items_for_pool=4) == 1
    assert 1 <= io_utils.bounded_workers(None, n_items=50, max_cap=4) <= 4


def test_env_config_defaults():
    assert env_config.NFS_ROOT.name == "rzhu"
    assert "python" in env_config.MD_PYTHON
    assert "python" in env_config.VIZ_PYTHON
    assert "pymol" in env_config.PYMOL_BIN


def test_validate_ensemble_parallel_matches_serial(tmp_path):
    record = modeling_record(tmp_path)
    record["validation_plan"]["poses"]["expected_predictions"] = 8
    record["validation_plan"]["poses"]["minimum_passing_predictions"] = 4
    paths = [
        write_pose(tmp_path / f"pose_{i}.pdb", displacement=0.05 * i)
        for i in range(8)
    ]
    execution = execution_for(record)
    serial = validate_ensemble(paths, execution, max_workers=1)
    parallel = validate_ensemble(paths, execution, max_workers=2)
    assert serial == parallel
    assert serial["status"] == "pass"
    assert serial["n_passing"] == 8


def test_proxy_group_sasa_and_contacts_exact_parity():
    rng = np.random.default_rng(42)
    prot_atoms = [
        Atom("A", str(i + 1), "CA", "ALA", "C", rng.uniform(-5, 5, size=3), True)
        for i in range(120)
    ]
    lig_atoms = [
        Atom("B", "1", f"C{i}", "LIG", "C", rng.uniform(-2, 2, size=3), False)
        for i in range(6)
    ]
    all_atoms = tuple(prot_atoms + lig_atoms)
    pose = Pose(
        ca=np.asarray([a.xyz for a in prot_atoms]),
        protein=np.asarray([a.xyz for a in prot_atoms]),
        donors=np.zeros((0, 3)),
        bases=np.zeros((0, 3)),
        ligands={"B": [(a.name, a.element, a.xyz) for a in lig_atoms]},
        atoms=all_atoms,
    )
    context = {"substrate_chains": ["B"], "protein_chains": ["A"]}

    # Verify contacts against naive O(N*M) reference
    frac, residues = contacts(pose, context, radius=4.0)
    prot_xyz = np.asarray([a.xyz for a in prot_atoms])
    ref_hits, ref_res = [], set()
    for a in lig_atoms:
        mask = np.linalg.norm(prot_xyz - a.xyz, axis=1) < 4.0
        ref_hits.append(bool(mask.any()))
        ref_res.update((p.chain, p.residue) for p, keep in zip(prot_atoms, mask) if keep)
    assert frac == pytest.approx(float(np.mean(ref_hits)))
    assert residues == ref_res

    # Verify group_sasa against naive O(N*M) reference
    group = [{"chain": "B", "residue": "1", "atom": f"C{i}"} for i in range(3)]
    probe, n = 1.4, 96
    fast_sasa = group_sasa(pose, group, context, probe=probe, n=n)

    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (3 - np.sqrt(5)) * i
    unit = np.column_stack((np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z))
    ref_sasa = 0.0
    for a in lig_atoms[:3]:
        radius = VDW[a.element] + probe
        sample = a.xyz + radius * unit
        acc = np.ones(n, dtype=bool)
        for other in all_atoms:
            if other.key == a.key:
                continue
            rr = VDW[other.element] + probe
            if np.linalg.norm(other.xyz - a.xyz) < radius + rr:
                acc &= np.linalg.norm(sample - other.xyz, axis=1) >= rr
        ref_sasa += 4 * np.pi * radius**2 * acc.mean()
    assert fast_sasa == pytest.approx(float(ref_sasa), rel=1e-12, abs=1e-12)


def test_agent_modular_exports_and_unified_tool_registry():
    assert SYSTEM_PROMPT and PREPARATION_PROMPT and BARRIER_PROMPT and BARRIER_GUIDANCE
    assert callable(_text_page) and callable(_merge_model_changes) and ModelingAgent is not None
    reg = get_registry()
    schemas_a = tool_schemas()
    schemas_b = reg.schemas()
    assert schemas_a == schemas_b
    schema_names = {s["function"]["name"] for s in schemas_a}
    assert schema_names == set(reg.names())
    # Newly registered domain capabilities are exposed via the unified registry
    assert {
        "analyze_substrate_contacts",
        "analyze_substrate_channels",
        "assess_mlmm_profile",
        "load_skill",
        "render_channel_visualization",
        "render_contact_figure",
    } <= schema_names
    # Every registered tool has a callable handler and valid scope
    for spec in reg.specs():
        assert callable(spec.handler)
        assert spec.schema["function"]["name"] == spec.name
    # Duplicate registration and mismatched schema names fail closed
    scratch = ToolRegistry()
    spec = ToolSpec(name="demo", schema=_schema("demo", "demo"), handler=lambda a, b: {})
    scratch.register(spec)
    with pytest.raises(ValueError, match="duplicate"):
        scratch.register(spec)
    with pytest.raises(ValueError, match="schema function name"):
        scratch.register(ToolSpec(name="other", schema=_schema("wrong", "d"), handler=lambda a, b: {}))


def test_skills_and_progressive_tool_disclosure(tmp_path):
    from structural_modeling_agent.prompts import SYSTEM_PROMPT, compact_system_prompt
    from structural_modeling_agent.skills import get_skill, list_skills
    from structural_modeling_agent.tool_registry import select_offered_tools

    skills = list_skills()
    names = {s["name"] for s in skills}
    assert {"cofold-setup", "md-preparation", "reaction-barriers", "contacts-and-channels", "visualization"} <= names
    compact = compact_system_prompt()
    assert len(compact) < len(SYSTEM_PROMPT) * 0.25
    assert "contacts-and-channels" in compact

    # Test progressive tool disclosure on a real ModelingAgent
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "target": {"name": "synthetic"},
        "progressive_tool_disclosure": True,
        "candidates": [{"accession": "TEST1", "sequence": "AG", "evidence": {}}],
    }))
    agent = ModelingAgent(task_path=task_path, run_dir=tmp_path / "run")
    initial_names = {t["function"]["name"] for t in select_offered_tools(agent)}
    all_names = {t["function"]["name"] for t in tool_schemas()}
    assert len(initial_names) < len(all_names) * 0.65
    assert "load_skill" in initial_names
    assert "analyze_substrate_channels" not in initial_names
    assert "render_channel_visualization" not in initial_names

    res = agent.dispatch("load_skill", {"name": "visualization"})
    assert "render_channel_visualization" in res["tools_unlocked"]
    assert "visualization" in agent.state["active_skills"]
    after_viz = {t["function"]["name"] for t in select_offered_tools(agent)}
    assert {"render_channel_visualization", "render_contact_figure"} <= after_viz
    assert "analyze_substrate_channels" not in after_viz

    agent.dispatch("load_skill", {"name": "contacts-and-channels"})
    after_analysis = {t["function"]["name"] for t in select_offered_tools(agent)}
    assert {"analyze_substrate_contacts", "analyze_substrate_channels"} <= after_analysis
    assert get_skill("visualization").domains == frozenset({"visualization"})
