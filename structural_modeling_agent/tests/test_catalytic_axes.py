"""Tests for enzyme- and reaction-agnostic multi-axis catalytic evaluators."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from structural_modeling_agent.agent import ModelingAgent
from structural_modeling_agent.analysis.catalytic_profile import build_catalytic_profile
from structural_modeling_agent.analysis.pocket_envelope import analyze_pose_envelope, compare_native_vs_target
from structural_modeling_agent.analysis.preorganization import (
    analyze_charge_center_stabilization,
    analyze_preorganization,
    apo_holo_reorganization,
    compute_projected_electric_field,
)
from structural_modeling_agent.analysis.side_reactions import analyze_side_reactions_poses


def _write_pdb(path: Path, records: list[tuple[str, str, str, str, int, tuple[float, float, float], str]]) -> Path:
    lines = []
    for i, (rec, name, rn, ch, resi, (x, y, z), el) in enumerate(records, 1):
        nm = name if len(name) == 4 else f" {name:<3}"
        lines.append(
            f"{rec:<6}{i:5d} {nm}{rn:>4} {ch}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _cage_protein(radius: float = 7.0, open_plus_x: bool = False) -> list[tuple]:
    """Synthetic spherical protein shell of ALA CA atoms around origin."""
    recs = []
    idx = 1
    for i in range(60):
        z = 1.0 - 2.0 * (i + 0.5) / 60.0
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        phi = math.pi * (3.0 - math.sqrt(5.0)) * i
        x, y = radius * r_xy * math.cos(phi), radius * r_xy * math.sin(phi)
        z_c = radius * z
        if open_plus_x and x > 2.0:
            continue
        recs.append(("ATOM", "CA", "ALA", "A", idx, (x, y, z_c), "C"))
        idx += 1
    return recs


def test_pocket_envelope_distinguishes_exposed_tail_vs_buried_clash(tmp_path):
    # Open-pocket protein (+x open): distal tail pointing into +x is solvent-exposed
    open_prot = _cage_protein(radius=7.0, open_plus_x=True)
    sub_exposed = [
        ("HETATM", "C1", "LIG", "B", 1, (0.0, 0.0, 0.0), "C"),
        ("HETATM", "C2", "LIG", "B", 1, (3.0, 0.0, 0.0), "C"),
        ("HETATM", "C3", "LIG", "B", 1, (9.0, 0.0, 0.0), "C"),
    ]
    p_open = _write_pdb(tmp_path / "open.pdb", open_prot + sub_exposed)
    env_open = analyze_pose_envelope(p_open, reaction_center_xyz=[0.0, 0.0, 0.0])
    assert env_open["tail_orientation"] == "solvent_exposed_tail"
    assert env_open["distal_tail_burial"] < 0.65

    # Closed buried cage: native small substrate fits, larger target clashes with wall at x=7.0
    closed_prot = _cage_protein(radius=7.0, open_plus_x=False)
    native_sub = [
        ("HETATM", "C1", "NAT", "B", 1, (0.0, 0.0, 0.0), "C"),
        ("HETATM", "C2", "NAT", "B", 1, (1.5, 0.0, 0.0), "C"),
    ]
    # Place target distal atom directly overlapping a wall atom
    wall_xyz = closed_prot[10][5]
    target_sub = [
        ("HETATM", "C1", "TGT", "B", 1, (0.0, 0.0, 0.0), "C"),
        ("HETATM", "C2", "TGT", "B", 1, (1.5, 0.0, 0.0), "C"),
        ("HETATM", "C3", "TGT", "B", 1, (3.0, 0.0, 0.0), "C"),
        ("HETATM", "C4", "TGT", "B", 1, (wall_xyz[0] + 0.5, wall_xyz[1], wall_xyz[2]), "C"),
    ]
    p_nat = _write_pdb(tmp_path / "nat.pdb", closed_prot + native_sub)
    p_tgt = _write_pdb(tmp_path / "tgt.pdb", closed_prot + target_sub)
    comp = compare_native_vs_target([p_tgt], [p_nat], reaction_center_xyz=[0.0, 0.0, 0.0])
    assert comp["target_to_native_heavy_atom_ratio"] == 2.0
    assert comp["diagnosis"] == "buried_steric_exclusion"
    assert len(comp["clashing_residues"]) >= 1


def test_side_reactions_general_across_enzyme_classes(tmp_path):
    # One structure encoding three distinct enzyme-class competing geometries:
    # 1. Carbonyl hydrolysis (Burgi-Dunitz ~107 deg, water O at 3.2 A, angle ~107 deg)
    # 2. Inline phosphate hydrolysis (180 deg, water O at 3.4 A along P-O3B axis)
    # 3. Distance-only off-target C-H abstraction (no angle_reference, C at 3.1 A from Fe=O)
    rad107 = math.radians(107.0)
    recs = [
        ("ATOM", "CA", "ALA", "A", 1, (20.0, 20.0, 20.0), "C"),
        # Carbonyl C=O along +x
        ("HETATM", "C1", "INT", "B", 1, (0.0, 0.0, 0.0), "C"),
        ("HETATM", "O1", "INT", "B", 1, (1.2, 0.0, 0.0), "O"),
        ("HETATM", "O", "HOH", "W", 1, (3.2 * math.cos(rad107), 3.2 * math.sin(rad107), 0.0), "O"),
        # Phosphate P-O3B along +z, water inline at -z
        ("HETATM", "PG", "ATP", "C", 1, (10.0, 0.0, 0.0), "P"),
        ("HETATM", "O3B", "ATP", "C", 1, (10.0, 0.0, 1.6), "O"),
        ("HETATM", "O", "HOH", "W", 2, (10.0, 0.0, -3.4), "O"),
        # Radical Fe=O at (0, 10, 0), competing C-H carbon at 3.1 A
        ("HETATM", "O1", "OXO", "D", 1, (0.0, 10.0, 0.0), "O"),
        ("HETATM", "C9", "SUB", "E", 1, (0.0, 13.1, 0.0), "C"),
    ]
    pdb = _write_pdb(tmp_path / "multi_mech.pdb", recs)
    pathways = [
        {
            "name": "acyl_hydrolysis",
            "kind": "solvent",
            "electrophile": {"chain": "B", "residue": "1", "atom": "C1"},
            "angle_reference": {"chain": "B", "residue": "1", "atom": "O1"},
            "distance_cutoff_A": 4.0,
            "ideal_angle_deg": 107.0,
            "angle_tolerance_deg": 20.0,
        },
        {
            "name": "futile_atpase_inline",
            "kind": "solvent",
            "electrophile": {"chain": "C", "residue": "1", "atom": "PG"},
            "angle_reference": {"chain": "C", "residue": "1", "atom": "O3B"},
            "distance_cutoff_A": 4.0,
            "ideal_angle_deg": 180.0,
            "angle_tolerance_deg": 30.0,
        },
        {
            "name": "off_target_ch_abstraction",
            "kind": "intermolecular",
            "electrophile": {"chain": "D", "residue": "1", "atom": "O1"},
            "nucleophiles": [{"chain": "E", "residue": "1", "atom": "C9"}],
            "distance_cutoff_A": 3.5,
        },
    ]
    res = analyze_side_reactions_poses([pdb], pathways)
    by_name = {p["name"]: p for p in res["pathways"]}
    assert by_name["acyl_hydrolysis"]["occupancy_fraction"] == 1.0
    assert by_name["acyl_hydrolysis"]["angle_deg_mean"] == pytest.approx(107.0, abs=0.5)
    assert by_name["futile_atpase_inline"]["occupancy_fraction"] == 1.0
    assert by_name["futile_atpase_inline"]["angle_deg_mean"] == pytest.approx(180.0, abs=0.5)
    assert by_name["off_target_ch_abstraction"]["occupancy_fraction"] == 1.0
    assert by_name["off_target_ch_abstraction"]["angle_deg_mean"] is None


def test_preorganization_rmsd_electric_field_and_charge_stabilization(tmp_path):
    # 1. Apo vs holo reorganization: 4 CA backbone fixed, catalytic sidechain OD1 shifts by 2.0 A
    base = [
        ("ATOM", "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
        ("ATOM", "CA", "ALA", "A", 2, (4.0, 0.0, 0.0), "C"),
        ("ATOM", "CA", "ALA", "A", 3, (0.0, 4.0, 0.0), "C"),
        ("ATOM", "CA", "ASP", "A", 4, (0.0, 0.0, 4.0), "C"),
    ]
    apo = _write_pdb(tmp_path / "apo.pdb", base + [("ATOM", "OD1", "ASP", "A", 4, (1.0, 0.0, 4.0), "O")])
    holo = _write_pdb(tmp_path / "holo.pdb", base + [
        ("ATOM", "OD1", "ASP", "A", 4, (3.0, 0.0, 4.0), "O"),
        ("ATOM", "NZ", "LYS", "A", 5, (0.0, 0.0, 2.8), "N"),
        ("HETATM", "O1", "INT", "B", 1, (0.0, 0.0, 0.0), "O"),
        ("HETATM", "MG", "MG", "C", 1, (0.0, 2.2, 0.0), "MG"),
    ])
    reorg = apo_holo_reorganization(apo, holo, [{"chain": "A", "residue": "4"}])
    assert reorg["global_ca_rmsd_A"] == pytest.approx(0.0, abs=1e-3)
    # 2 matched atoms in res 4 (CA shift=0, OD1 shift=2.0) -> rmsd = sqrt((0 + 4)/2) = sqrt(2)
    assert reorg["catalytic_heavy_rmsd_A"] == pytest.approx(math.sqrt(2.0), abs=0.01)

    # 2. Analytic projected electric field: +1e at (-2, 0, 0), probe at (0, 0, 0), axis along +x
    # E = +1 / 2^2 * 1439.9645 = +359.99 MV/cm
    ef = compute_projected_electric_field(
        coords=np.array([[-2.0, 0.0, 0.0]]),
        charges_e=np.array([1.0]),
        axis_a_xyz=np.array([-1.0, 0.0, 0.0]),
        axis_b_xyz=np.array([1.0, 0.0, 0.0]),
        probe_xyz=np.array([0.0, 0.0, 0.0]),
    )
    assert ef["projected_field_MV_per_cm"] == pytest.approx(1439.9645 / 4.0, abs=0.05)

    # 3. Charge-center stabilization: LYS5 NZ at 2.8 A + MG at 2.2 A
    stab = analyze_charge_center_stabilization([holo], {"chain": "B", "residue": "1", "atom": "O1"})
    assert stab["polar_donor_atoms_mean"] == 1.0
    assert stab["metal_stabilization_freq"] == 1.0


def test_catalytic_profile_and_agent_tool_dispatch(tmp_path):
    profile = build_catalytic_profile(
        "CAND1",
        mechanism_gate={"status": "pass"},
        pocket_envelope={"diagnosis": "pocket_too_loose_slack"},
        reactive_persistence={"boltz_nac_fraction": 0.45, "md_nac_fraction": 0.02},
        developability={"verdict": "ok"},
    )
    assert profile["evolvability_class"] == "evolvable_with_mutations"
    assert "barrier_status" in profile["axes_missing"]
    codes = {fm["code"] for fm in profile["failure_modes"]}
    assert {"pocket_too_loose_drifter", "cofold_poised_md_drifter"} <= codes

    # Verify ModelingAgent can load catalytic-triage skill and dispatch build_catalytic_profile
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "target": {"name": "generic_reaction"},
        "progressive_tool_disclosure": True,
        "candidates": [{"accession": "CAND1", "sequence": "MKKLL", "evidence": {}}],
    }))
    agent = ModelingAgent(task_path=task_path, run_dir=tmp_path / "run")
    skill = agent.dispatch("load_skill", {"name": "catalytic-triage"})
    assert "assess_developability" in skill["tools_unlocked"]
    assert "analyze_side_reactions" in skill["tools_unlocked"]
    built = agent.dispatch("build_catalytic_profile", {
        "accession": "CAND1",
        "mechanism_gate": {"status": "pass"},
        "developability": {"verdict": "ok"},
    })
    assert built["evolvability_class"] == "promising_scaffold"
    assert agent.state["candidates"]["CAND1"]["catalytic_profile"]["evolvability_class"] == "promising_scaffold"


def test_submit_and_check_sampling_workflow_and_evidence_export(tmp_path):
    """End-to-end test of Step 3 (reservation-aware sampling) and Step 4 (evidence + design handoff)."""
    import sys
    from structural_modeling_agent.evidence import modeling_evidence

    run_dir = tmp_path / "run"

    # Mock reservation-aware adapter script that queues on submit and completes on status
    adapter = tmp_path / "mock_sampling_adapter.py"
    adapter.write_text(
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "req = json.loads(sys.stdin.read())\n"
        "plan = json.loads(Path(req['plan_path']).read_text())\n"
        "wd = Path(plan['work_dir'])\n"
        "if req['operation'] == 'submit':\n"
        "    print(json.dumps({'status': 'queued', 'remote_job_id': 'res-42'}))\n"
        "else:\n"
        "    top = wd / 'complex.prmtop'; top.write_text('top')\n"
        "    dcd = wd / 'production.dcd'; dcd.write_text('dcd')\n"
        "    geom = wd / 'geometry.jsonl'; geom.write_text('{}\\n')\n"
        "    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (top, dcd, geom)}\n"
        "    print(json.dumps({'status': 'complete', 'remote_job_id': 'res-42', 'completed_ns': 2.0,\n"
        "                      'topology_path': str(top), 'trajectory_paths': [str(dcd)],\n"
        "                      'geometry_jsonl': str(geom), 'output_sha256': hashes}))\n"
    )

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "target": {"name": "generic_reaction"},
        "candidates": [{"accession": "CAND1", "sequence": "MKKLL", "evidence": {}}],
    }))
    backend_path = tmp_path / "backend.json"
    backend_path.write_text(json.dumps({
        "sampling": {
            "provider": "mock_reserved",
            "supported_engines": ["openmm_classical"],
            "argv": [sys.executable, str(adapter)],
            "resource_policy": {"reservation_managed": True, "max_jobs": 2, "max_wall_seconds": 3600},
        }
    }))
    agent = ModelingAgent(task_path=task_path, run_dir=run_dir, backend_path=backend_path)

    # Create mock prepared input files inside run_dir/artifacts (as produced by preparation)
    prep_dir = run_dir / "artifacts" / "prep_inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)
    prmtop = prep_dir / "sys.prmtop"
    coords = prep_dir / "sys.pdb"
    spec = prep_dir / "rxn.json"
    prmtop.write_text("%FLAG CHARGE\n%FORMAT(5E16.8)\n0.0\n")
    coords.write_text("END\n")
    spec.write_text("{}\n")

    # Seed a ready sampling setup matching declare_sampling_setup's exact schema
    agent.state["candidates"]["CAND1"].setdefault("sampling_setups", {})["setup1"] = {
        "artifact": str(run_dir / "artifacts" / "sampling_setup_setup1.json"),
        "assessment": {"status": "ready_for_preparation"},
        "setup": {"schema_version": "sampling-setup-1.0"},
    }

    # 1. Submit sampling job
    sub = agent.dispatch("submit_sampling", {
        "accession": "CAND1",
        "setup_id": "setup1",
        "label": "md_v1",
        "engine": "openmm_classical",
        "wall_seconds": 600,
        "inputs": {"prmtop": str(prmtop), "coordinates": str(coords), "spec_path": str(spec)},
        "settings": {"production_ns": 2.0},
    })
    assert sub["status"] == "queued"
    job_id = sub["job_id"]

    # 2. Idempotent re-submit with same label does not double-submit
    dup = agent.dispatch("submit_sampling", {
        "accession": "CAND1",
        "setup_id": "setup1",
        "label": "md_v1",
        "engine": "openmm_classical",
        "wall_seconds": 600,
        "inputs": {"prmtop": str(prmtop), "coordinates": str(coords), "spec_path": str(spec)},
    })
    assert dup["status"] == "already_requested"
    assert dup["job_id"] == job_id

    # 3. Check sampling transitions to complete and returns hash-verified analysis_inputs
    chk = agent.dispatch("check_sampling", {"accession": "CAND1", "job_id": job_id})
    assert chk["status"] == "complete"
    assert chk["outputs"]["analysis_inputs"]["top_path"].endswith("complex.prmtop")
    assert len(chk["outputs"]["analysis_inputs"]["traj_paths"]) == 1
    assert "analyze_side_reactions" in chk["outputs"]["recommended_next_tools"]

    # 4. Build catalytic profile with a soft defect and verify Step 4 report + evidence export
    agent.dispatch("build_catalytic_profile", {
        "accession": "CAND1",
        "mechanism_gate": {"status": "pass"},
        "pocket_envelope": {"diagnosis": "distal_tail_buried_clash", "clashing_residues": ["A:PHE120"]},
        "developability": {"verdict": "ok"},
    })
    agent._write_report()
    ev = modeling_evidence(agent.run_dir / "report.json")
    row = ev["candidates"][0]
    assert row["evolvability_class"] == "evolvable_with_mutations"
    assert any("A:PHE120" in (dt.get("residues") or []) for dt in row["design_targets"])
    assert ev["family_feedback"]["n_with_catalytic_profile"] == 1
    assert ev["family_feedback"]["evolvability_counts"]["evolvable_with_mutations"] == 1
    assert "soft_evolvable_defect:distal_tail_clash" in ev["family_feedback"]["failure_mode_counts"]


def test_slim_default_tool_surface_and_auto_caching(tmp_path):
    """Default progressive disclosure hides the 13 deprecated contract/read-back tools and auto-caches analyses."""
    from structural_modeling_agent.controller.tool_registry import get_registry, select_offered_tools

    reg = get_registry()
    assert len(reg.deprecated_names()) == 0
    assert len(reg.names()) == 57
    assert "inspect_validation_contract" not in reg.names()
    assert "inspect_barrier_contract" not in reg.names()

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "target": {"name": "generic_reaction"},
        "candidates": [{"accession": "CAND1", "sequence": "MKKLL", "evidence": {}}],
    }))
    run_dir = tmp_path / "run"
    agent = ModelingAgent(task_path=task_path, run_dir=run_dir)
    offered = {t["function"]["name"] for t in select_offered_tools(agent)}
    # Lean default surface: ~24 tools, zero deprecated contract/read-back tools
    assert len(offered) <= 26
    assert not (offered & set(reg.deprecated_names()))
    assert {"load_skill", "propose_model", "build_catalytic_profile", "finish_task"} <= offered

    # Loading catalytic-triage unlocks analysis + developability without exposing deprecated tools
    agent.dispatch("load_skill", {"name": "catalytic-triage"})
    offered_after = {t["function"]["name"] for t in select_offered_tools(agent)}
    assert {"analyze_pocket_envelope", "analyze_side_reactions", "analyze_preorganization", "assess_developability"} <= offered_after
    assert not (offered_after & set(reg.deprecated_names()))

    # Auto-caching: analyze_pocket_envelope caches onto candidate so build_catalytic_profile picks it up automatically
    pose = run_dir / "artifacts" / "pose.pdb"
    pose.parent.mkdir(parents=True, exist_ok=True)
    _write_pdb(pose, [
        ("ATOM", "CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
        ("ATOM", "CZ", "PHE", "A", 2, (5.5, 0.0, 0.0), "C"),
        ("HETATM", "C1", "LIG", "B", 1, (2.8, 0.0, 0.0), "C"),
        ("HETATM", "C2", "LIG", "B", 1, (4.2, 0.0, 0.0), "C"),
    ])
    agent.dispatch("analyze_pocket_envelope", {"target_poses": [str(pose)]})
    built = agent.dispatch("build_catalytic_profile", {"accession": "CAND1", "mechanism_gate": {"status": "pass"}})
    assert built["axes"]["pocket_envelope"]["status"] == "evaluated"


