"""Execute declared proxy hypotheses and report requirements/coverage per feature.

No activity labels are used here. Coordinate selections, confidence indices and
charge assignments come from a reviewed plan rather than chain-order guesses.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from enzyme_discovery_agent.structural_proxy_planning import load_proxy_catalog, validate_proxy_plan
from .library import (BY_ID, build_ensemble, _near_attack_fraction, _trajectory_angle,
                      _angle_reference)
from .roles import Pose, ReactiveSite, check_resolver, parse_pose


class MissingInput(ValueError):
    pass


def atom_selection(pose: Pose, selectors: list[dict]):
    index = {a.key: a for a in pose.atoms}
    keys = [(str(s["chain"]), str(s["residue"]), str(s["atom"])) for s in selectors]
    if not keys or any(k not in index for k in keys):
        raise MissingInput("declared atom group is empty or missing from this prediction")
    return [index[k] for k in keys]


def substrate_atoms(pose: Pose, context: dict):
    chains = context.get("substrate_chains", [])
    if not chains or any(ch not in pose.ligands for ch in chains):
        raise MissingInput("each target substrate chain must be present explicitly")
    return [a for a in pose.atoms if not a.protein and a.chain in chains]


def protein_atoms(pose: Pose, context: dict):
    chains = context.get("protein_chains")
    atoms = [a for a in pose.atoms if a.protein and (chains is None or a.chain in chains)]
    if not atoms:
        raise MissingInput("no protein atoms in the declared assembly context")
    return atoms


def contacts(pose: Pose, context: dict, radius: float):
    ligand = substrate_atoms(pose, context)
    protein = protein_atoms(pose, context)
    if not ligand:
        raise MissingInput("target substrate has no parsed heavy atoms")
    xyz = np.array([a.xyz for a in protein])
    fraction, residues = [], set()
    for a in ligand:
        mask = np.linalg.norm(xyz - a.xyz, axis=1) < radius
        fraction.append(bool(mask.any()))
        residues.update((p.chain, p.residue) for p, keep in zip(protein, mask) if keep)
    return float(np.mean(fraction)), residues


# Same heavy-atom surface approximation used by the existing analyses. Radii and
# probe settings are part of the reported definition, not inferred chemistry.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
       "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
       "MG": .86, "ZN": 1.39, "NA": 1.02}


def group_sasa(pose: Pose, group: list[dict], context: dict, probe: float, n: int):
    selected = atom_selection(pose, group)
    included = context.get("sasa_occluder_chains")
    atoms = [a for a in pose.atoms if a.residue_name not in ("HOH", "DOD")
             and (included is None or a.chain in included)]
    if any(a.key not in {x.key for x in atoms} for a in selected):
        raise MissingInput("SASA occlusion context must contain the selected group")
    if any(a.element not in VDW for a in atoms):
        raise MissingInput("SASA context contains an element without a declared radius")
    i = np.arange(n) + .5
    z = 1 - 2 * i / n
    phi = np.pi * (3 - np.sqrt(5)) * i
    unit = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    result = 0.
    for a in selected:
        radius = VDW[a.element] + probe
        sample = a.xyz + radius * unit
        accessible = np.ones(n, dtype=bool)
        for other in atoms:
            if other.key == a.key:
                continue
            rr = VDW[other.element] + probe
            if np.linalg.norm(other.xyz-a.xyz) < radius + rr:
                accessible &= np.linalg.norm(sample-other.xyz, axis=1) >= rr
        result += 4 * np.pi * radius**2 * accessible.mean()
    return float(result)


def electrostatics(pose: Pose, site: ReactiveSite, context: dict, cutoff: float):
    target = getattr(site, context.get("potential_target", "electrophile"), None)
    if target is None:
        raise MissingInput("electrostatic target role is unresolved")
    charged = atom_selection(pose, context.get("charges", []))
    if any(not a.protein for a in charged):
        raise MissingInput("this descriptor uses protein charges only")
    if context.get("unlisted_atom_policy") == "error":
        listed = {a.key for a in charged}
        if any(a.key not in listed for a in protein_atoms(pose, context)):
            raise MissingInput("protein partial-charge map is incomplete")
    potential, field = 0., np.zeros(3)
    allowed = context.get("protein_chains")
    for a, spec in zip(charged, context["charges"]):
        if allowed is not None and a.chain not in allowed:
            continue
        delta = target - a.xyz
        distance = np.linalg.norm(delta)
        if distance < .1:
            raise MissingInput("electrostatic target overlaps a charged atom")
        if distance < cutoff:
            charge = float(spec["charge"])
            potential += charge/distance
            field += charge*delta/distance**3
    return float(potential), field


def per_pose(pid: str, pose: Pose, site: ReactiveSite | None, context: dict, params: dict):
    if pid in ("complex_plddt", "complex_iptm", "ligand_iptm"):
        key = "iptm" if pid == "complex_iptm" else pid
        value = pose.confidence.get(key)
        if value is None or not np.isfinite(value) or not 0 <= value <= 1:
            raise MissingInput(f"finite normalized confidence field {key} is absent")
        return float(value)
    if pid in ("protein_substrate_iptm", "substrate_protein_iptm"):
        matrix, mapping = pose.confidence.get("pair_chains_iptm", {}), context.get("confidence_chain_map", {})
        vals = []
        for pair in context.get("confidence_pairs", []):
            a, b = pair["protein_chain"], pair["substrate_chain"]
            if a not in {x.chain for x in pose.atoms if x.protein} or b not in pose.ligands:
                raise MissingInput("confidence pair does not identify a protein and a ligand")
            if a not in mapping or b not in mapping:
                raise MissingInput("explicit confidence chain-index map is missing")
            i, j = str(mapping[a]), str(mapping[b])
            if pid == "substrate_protein_iptm":
                i, j = j, i
            v = matrix.get(i, {}).get(j)
            if v is None or not np.isfinite(v) or not 0 <= v <= 1:
                raise MissingInput("declared confidence matrix entry is absent")
            vals.append(v)
        if not vals:
            raise MissingInput("explicit protein--substrate confidence pairs are required")
        return float(np.mean(vals))
    if pid in ("ligand_contact_fraction", "ligand_contact_residues"):
        fraction, residues = contacts(pose, context, params.get("contact_radius_A", 4.0))
        return fraction if pid == "ligand_contact_fraction" else float(len(residues))
    if pid in ("reactive_group_sasa", "distal_group_sasa"):
        key = "reactive_group_atoms" if pid.startswith("reactive") else "distal_group_atoms"
        return group_sasa(pose, context.get(key, []), context,
                          params.get("probe_radius_A", 1.4), params.get("sphere_points", 96))
    if site is None:
        raise MissingInput("reactive roles unresolved")
    if pid in ("annotated_base_distance", "annotated_donor_contacts"):
        key = "catalytic_base_atoms" if pid.startswith("annotated_base") else "catalytic_donor_atoms"
        group = atom_selection(pose, context.get(key, []))
        if any(not a.protein for a in group):
            raise MissingInput("annotated catalytic groups must belong to the protein")
        target = site.nucleophile if pid.startswith("annotated_base") else site.charge_centre
        if target is None:
            raise MissingInput("charge-accumulating atom is unresolved")
        distances = np.array([np.linalg.norm(a.xyz-target) for a in group])
        return float(distances.min()) if pid.startswith("annotated_base") else float(
            np.sum(distances < params.get("donor_radius_A", 3.5)))
    potential, field = electrostatics(pose, site, context, params.get("charge_cutoff_A", 15.0))
    if pid == "reactive_site_potential":
        return potential
    axis = context.get("field_axis", [])
    if len(axis) != 2:
        raise MissingInput("explicit field-axis roles are required")
    a, b = (getattr(site, role, None) for role in axis)
    if a is None or b is None or np.linalg.norm(b-a) < 1e-6:
        raise MissingInput("field axis is unresolved or degenerate")
    return float(field @ ((b-a)/np.linalg.norm(b-a)))


DEFAULT_PARAMETERS = {
    "near_attack_fraction": {"distance_gate_A": 4., "angle_tolerance_deg": 30.},
    "trajectory_angle_deviation": {"distance_gate_A": 4.},
    "ligand_contact_fraction": {"contact_radius_A": 4.},
    "ligand_contact_residues": {"contact_radius_A": 4.},
    "contact_consistency": {"contact_radius_A": 4.},
    "reactive_group_sasa": {"probe_radius_A": 1.4, "sphere_points": 96},
    "distal_group_sasa": {"probe_radius_A": 1.4, "sphere_points": 96},
    "annotated_donor_contacts": {"donor_radius_A": 3.5},
    "reactive_site_potential": {"charge_cutoff_A": 15.},
    "projected_electric_field": {"charge_cutoff_A": 15.},
    "electric_field_variability": {"charge_cutoff_A": 15.},
}


def evaluate_plan(poses: list[Pose], sites: list[ReactiveSite | None], plan: dict,
                  resolver_report=None) -> dict:
    errors = validate_proxy_plan(plan)
    if errors:
        raise ValueError("Invalid proxy plan: " + "; ".join(errors))
    if not poses or len(poses) != len(sites):
        raise ValueError("provide nonempty parallel pose/reactive-site lists")
    catalog = {d["id"]: d for d in load_proxy_catalog()["descriptors"]}
    ens = build_ensemble(poses, sites)
    context = plan["context"]
    measurements = {}
    for h in plan["hypotheses"]:
        pid, spec = h["proxy_id"], catalog[h["proxy_id"]]
        if pid in BY_ID and h.get("definition_version", BY_ID[pid].version) != BY_ID[pid].version:
            raise ValueError("proxy definition version differs from the current extractor; review/recalibrate before reuse")
        supplied = h.get("parameters", {})
        unknown = set(supplied) - set(DEFAULT_PARAMETERS.get(pid, {}))
        if unknown:
            raise ValueError(f"{pid}: unsupported parameters {sorted(unknown)}")
        params = {**DEFAULT_PARAMETERS.get(pid, {}), **supplied}
        for key, value in params.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{pid}: {key} must be finite and positive")
        if "sphere_points" in params and (not isinstance(params["sphere_points"], int) or params["sphere_points"] < 12):
            raise ValueError("sphere_points must be an integer >= 12")
        item = {"category": spec["category"], "determinant": spec["determinant"],
                "definition_version": BY_ID[pid].version if pid in BY_ID else "1.0",
                "units": spec["units"], "hypothesis": h, "value": None,
                "n_input_poses": len(poses), "n_computed_poses": 0,
                "limitations": spec["limitations"], "parameters": params,
                "evidence_status": spec["evidence_status"], "reasons": []}
        if spec["execution"] != "builtin":
            item.update(status="requires_targeted_extractor", reasons=spec["requirements"])
            measurements[pid] = item
            continue
        if pid in BY_ID:
            if resolver_report is not None and not resolver_report.ok:
                value = None
                item["reasons"].extend(resolver_report.problems)
            elif pid == "near_attack_fraction":
                value = _near_attack_fraction(ens, params.get("distance_gate_A", 4.), params.get("angle_tolerance_deg", 30.))
            elif pid == "trajectory_angle_deviation":
                value = _trajectory_angle(ens, params.get("distance_gate_A", 4.))
            else:
                if params:
                    raise ValueError(f"{pid}: this core definition has no tunable parameters")
                value = BY_ID[pid](ens)
            item["value"] = value
            item["n_computed_poses"] = len(ens.sites) if value is not None else 0
            item["n_distance_gated"] = int(ens.gate(params.get("distance_gate_A", 4.)).sum())
            item["definition"] = BY_ID[pid].doc
            if pid in ("trajectory_angle_deviation", "trajectory_alignment"):
                item["minimum_gated_poses"] = 5
            if value is None:
                item["reasons"].append("missing roles, valid alignment, or sufficient interpretable poses")
        elif pid in ("contact_consistency", "ligand_positional_dispersion"):
            valid = []
            for p in ens.poses if pid == "ligand_positional_dispersion" else poses:
                try:
                    if pid == "contact_consistency":
                        valid.append(contacts(p, context, params.get("contact_radius_A", 4.))[1])
                    else:
                        if not ens.alignment_ok:
                            raise MissingInput("protein alignment unavailable or atom identities differ")
                        atoms = substrate_atoms(p, context)
                        valid.append((tuple(a.key for a in atoms), np.array([a.xyz for a in atoms])))
                except MissingInput as exc:
                    item["reasons"].append(str(exc))
            if len(valid) >= 2:
                if pid == "contact_consistency":
                    # No binding in either pose is not evidence of conserved contacts.
                    pairs = [len(a & b)/len(a | b) if a | b else 0.
                             for a, b in itertools.combinations(valid, 2)]
                    item["value"] = float(np.mean(pairs))
                elif all(v[0] == valid[0][0] for v in valid) and valid[0][0]:
                    xyz = np.array([v[1] for v in valid])
                    item["value"] = float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(0))**2, axis=-1))))
                else:
                    item["reasons"].append("substrate atom correspondence differs across predictions")
                item["n_computed_poses"] = len(valid) if item["value"] is not None else 0
            else:
                item["reasons"].append("at least two comparable predictions required")
        else:
            valid = []
            for p, s in zip(poses, sites):
                try:
                    valid.append(per_pose(pid, p, s, context, params))
                except MissingInput as exc:
                    item["reasons"].append(str(exc))
            if valid and (pid != "electric_field_variability" or len(valid) >= 2):
                item["value"] = float(np.std(valid)) if pid == "electric_field_variability" else float(np.mean(valid))
                item["pose_std"] = float(np.std(valid))
                item["n_computed_poses"] = len(valid)
        item["status"] = ("unavailable" if item["value"] is None else
                          "partial" if item["n_computed_poses"] < len(poses) else "computed")
        item["reasons"] = sorted(set(item["reasons"]))
        measurements[pid] = item
    return {"schema_version": "1.0", "plan": plan, "n_input_poses": len(poses),
            "n_resolved_poses": len(ens.sites), "protein_alignment_available": ens.alignment_ok,
            "resolver_report": asdict(resolver_report) if resolver_report is not None else None,
            "measurements": measurements,
            "note": "Descriptors are hypotheses. This report contains no experimental validation or automatic rank."}


def run_plan(predictions: Path, record: dict, plan: dict, *, use_validated_sites: bool = False) -> dict:
    from structural_modeling_agent.modeling_reasoning.reactive_site import attach_smarts, resolver_from_handoff
    from structural_modeling_agent.modeling_reasoning.reaction import load_atom_selectors
    geometry = record.get("geometry", {})
    if any(h.get("proxy_id") in ("near_attack_fraction", "trajectory_angle_deviation")
           for h in plan.get("hypotheses", [])):
        if "ideal_angle" not in geometry or not np.isfinite(geometry["ideal_angle"]) or not 0 <= geometry["ideal_angle"] <= 180:
            raise ValueError("attack-angle proxies require a declared mechanism-specific ideal angle")
    if plan.get("step_index") is not None and plan["step_index"] != geometry.get("step"):
        raise ValueError("proxy plan and resolved handoff specify different reaction steps")
    if use_validated_sites:
        from structural_modeling_agent.modeling_reasoning.reactive_site import resolver_from_validation
        resolver, role_resolution = resolver_from_validation(record, (plan.get("context") or {}).get("site_id"))
    else:
        record = attach_smarts(record, load_atom_selectors())
        resolver = resolver_from_handoff(record)
        role_resolution = {"basis": "handoff_geometry"}
    paths = sorted(predictions.glob("*_model_*.pdb"))
    poses = [parse_pose(str(p)) for p in paths]
    report = check_resolver(resolver, poses)
    result = evaluate_plan(poses, [resolver(p) for p in poses], plan, report)
    result["role_resolution"] = role_resolution
    result["source_files"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in paths]
    result["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    result["resolved_record_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    result["confidence_sources"] = [{"path": str(p.with_name("confidence_" + p.stem + ".json")),
                                      "sha256": hashlib.sha256(p.with_name("confidence_" + p.stem + ".json").read_bytes()).hexdigest()}
                                     for p in paths if p.with_name("confidence_" + p.stem + ".json").exists()]
    return result
