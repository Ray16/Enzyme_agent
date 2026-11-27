"""Pose-level identity, geometry, and holo-seating analysis for family cofold ensembles."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from ..io_utils import bounded_workers, file_digest, init_single_thread_worker
from ..validation import _angle, _protein_completeness, _read_structure, component_inventories

# Periodic-table catalytic metal elements for generic holo-seating checks
# (works for any metalloenzyme: Mg, Mn, Zn, Fe, Ca, Co, Cu, Ni, Mo, W, etc.).
_METAL_ELEMENTS = frozenset({"MG", "MN", "ZN", "FE", "CA", "CO", "CU", "NI", "MO", "W", "V", "K", "NA"})
_POLAR_COORD_ELEMENTS = frozenset({"O", "N", "S", "P"})


def _holo_seating_metrics(atoms, by_chain, entities):
    """Generic, enzyme-agnostic cofactor and catalytic-metal seating check for one pose."""
    import numpy as np

    protein_xyz = np.array([v["xyz"] for v in atoms.values() if v.get("protein")])
    protein_polar_xyz = np.array([
        v["xyz"] for v in atoms.values()
        if v.get("protein") and str(v.get("element", "")).upper() in _POLAR_COORD_ELEMENTS
    ])

    metal_chains = []
    organic_cofactor_chains = []
    substrate_chains = []
    for key, val in (entities or {}).items():
        if isinstance(val, list):
            for ch in val:
                chain_atoms = by_chain.get(ch, ())
                elems = {str(v.get("element", "")).upper() for _, v in chain_atoms}
                if chain_atoms and elems <= _METAL_ELEMENTS:
                    metal_chains.append(ch)
                elif chain_atoms:
                    organic_cofactor_chains.append(ch)
        elif isinstance(val, str):
            if by_chain.get(val):
                substrate_chains.append(val)

    result = {}
    if organic_cofactor_chains and len(protein_xyz):
        min_contacts = None
        for ch in organic_cofactor_chains:
            c_xyz = np.array([v["xyz"] for _, v in by_chain[ch]])
            dists = np.linalg.norm(protein_xyz[:, None, :] - c_xyz[None, :, :], axis=2)
            n_contacts = int((dists.min(axis=1) <= 4.5).sum())
            min_contacts = n_contacts if min_contacts is None else min(min_contacts, n_contacts)
        result["cofactor_min_protein_contacts"] = min_contacts
        result["cofactor_seated"] = bool(min_contacts is not None and min_contacts >= 4)

    if metal_chains:
        cof_polar_list = [
            v["xyz"] for ch in organic_cofactor_chains for _, v in by_chain.get(ch, ())
            if str(v.get("element", "")).upper() in _POLAR_COORD_ELEMENTS
        ]
        sub_polar_list = [
            v["xyz"] for ch in substrate_chains for _, v in by_chain.get(ch, ())
            if str(v.get("element", "")).upper() in _POLAR_COORD_ELEMENTS
        ]
        all_partner_list = (
            (list(protein_polar_xyz) if len(protein_polar_xyz) else [])
            + cof_polar_list + sub_polar_list
        )
        all_partner_xyz = np.array(all_partner_list) if all_partner_list else np.empty((0, 3))
        cof_polar_xyz = np.array(cof_polar_list) if cof_polar_list else np.empty((0, 3))

        worst_any = 0.0
        worst_cof = 0.0
        for ch in metal_chains:
            m_xyz = np.array([v["xyz"] for _, v in by_chain[ch]])
            if len(all_partner_xyz):
                d_any = float(np.linalg.norm(all_partner_xyz[:, None, :] - m_xyz[None, :, :], axis=2).min())
                worst_any = max(worst_any, d_any)
            else:
                worst_any = float("inf")
            if len(cof_polar_xyz):
                d_cof = float(np.linalg.norm(cof_polar_xyz[:, None, :] - m_xyz[None, :, :], axis=2).min())
                worst_cof = max(worst_cof, d_cof)
        result["metal_nearest_polar_A"] = round(worst_any, 3) if math.isfinite(worst_any) else None
        result["metal_coordinated"] = bool(math.isfinite(worst_any) and worst_any <= 3.2)
        if len(cof_polar_xyz):
            result["metal_to_cofactor_A"] = round(worst_cof, 3)
            result["metal_cofactor_coordinated"] = bool(worst_cof <= 3.5)

    return result


def _summarize_holo_seating(poses):
    """Aggregate per-pose cofactor/metal seating into an ensemble-level diagnostic."""
    n = len(poses)
    if not n:
        return {"status": "not_applicable"}
    cof_vals = [p["cofactor_seated"] for p in poses if "cofactor_seated" in p]
    met_vals = [p["metal_coordinated"] for p in poses if "metal_coordinated" in p]
    mc_vals = [p["metal_cofactor_coordinated"] for p in poses if "metal_cofactor_coordinated" in p]
    if not cof_vals and not met_vals and not mc_vals:
        return {"status": "not_applicable"}
    cof_frac = round(sum(cof_vals) / len(cof_vals), 4) if cof_vals else None
    met_frac = round(sum(met_vals) / len(met_vals), 4) if met_vals else None
    mc_frac = round(sum(mc_vals) / len(mc_vals), 4) if mc_vals else None
    checks = [v for v in (cof_frac, met_frac, mc_frac) if v is not None]
    status = "pass" if all(v >= 0.5 for v in checks) else "warning_seating_artifact"
    summary = {
        "status": status,
        "cofactor_seated_fraction": cof_frac,
        "metal_coordinated_fraction": met_frac,
        "metal_cofactor_coordinated_fraction": mc_frac,
    }
    if status != "pass":
        summary["note"] = (
            "Cofactor and/or catalytic metal failed to seat properly in >50% of poses; "
            "poor target attack geometry may reflect a cofolding seating artifact rather than "
            "true substrate incompatibility (verify with a native-substrate control)."
        )
    return summary


def _analyze_single_prediction(args):
    """Validate identity and compute attack geometry for one prediction file."""
    (prediction, sequence_sha256, expected_protein_copies, entities, ccd_atoms,
     ligand_atom_maps, center_ref, leaving_ref, donor_refs, pocket_protocol) = args
    import numpy as np

    path = Path(prediction["path"])
    if file_digest(path) != prediction["sha256"]:
        raise ValueError("Prediction changed")
    atoms, sequences = _read_structure(path)
    by_chain = defaultdict(list)
    by_chain_atom = defaultdict(list)
    for key, val in atoms.items():
        by_chain[key[0]].append((key, val))
        by_chain_atom[(key[0], key[2])].append(val["xyz"])

    issues = []
    for chain, sequence in sequences.items():
        if hashlib.sha256(sequence.encode()).hexdigest() != sequence_sha256:
            issues.append("protein_sequence:" + chain)
        if any(c["status"] != "pass" for c in _protein_completeness(atoms, chain, {})):
            issues.append("protein_heavy_atoms:" + chain)
    if len(sequences) != expected_protein_copies:
        issues.append("protein_copy_count")
    observed = set(by_chain)
    expected = set(sequences)
    for entity in entities.values():
        expected.update(entity if isinstance(entity, list) else [entity])
    if observed != expected:
        issues.append("entity_inventory")
    for ccd, chains in ((k, v) for k, v in entities.items() if isinstance(v, list)):
        expected_ccd = ccd_atoms.get(ccd)
        for chain in chains:
            actual = {k[2]: v["element"] for k, v in by_chain.get(chain, ())}
            if actual != expected_ccd:
                issues.append("cofactor_identity:" + chain)
    for chain, definition in ligand_atom_maps.items():
        actual = {k[2]: v["element"] for k, v in by_chain.get(chain, ())}
        desired = {v["name"]: v["element"] for v in definition["atoms"].values()}
        if actual != desired:
            issues.append("ligand_atom_inventory:" + chain)

    def coordinate(ref):
        matches = by_chain_atom.get(ref, ())
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous reaction atom " + str(ref))
        return matches[0]

    center = coordinate(center_ref)
    leaving_xyz = coordinate(leaving_ref)
    donors = [(ref, coordinate(ref)) for ref in donor_refs]
    ref, donor = min(donors, key=lambda pair: np.linalg.norm(pair[1] - center))
    distance = float(np.linalg.norm(donor - center))
    angle = _angle(donor, center, leaving_xyz)
    pose = {
        "path": str(path),
        "sha256": prediction["sha256"],
        "distance_A": distance,
        "angle_deg": angle,
        "identity_issues": issues,
        "nucleophile": ref,
        **_holo_seating_metrics(atoms, by_chain, entities),
    }
    if pocket_protocol:
        contacts = {(chain, str(number)) for chain, number in pocket_protocol["contacts"]}
        pocket = np.array([value["xyz"] for key, value in atoms.items() if (key[0], key[1]) in contacts and value["protein"]])
        if not len(pocket):
            raise ValueError("Mapped pocket residues are absent from guided prediction")
        ligand = np.array([value["xyz"] for _, value in by_chain.get(ref[0], ())])
        pose["nucleophile_to_pocket_A"] = float(np.linalg.norm(pocket - donor, axis=1).min())
        pose["ligand_to_pocket_A"] = float(np.linalg.norm(ligand[:, None, :] - pocket[None, :, :], axis=2).min())
        pose["pocket_localized"] = pose["ligand_to_pocket_A"] <= pocket_protocol["max_distance_A"]
    return pose


def analyze_outputs(job, receipt, *, max_workers=None):
    """Measure every pose with the existing geometry metric; retain identity failures."""
    from ..proxies.attack_angle import weighted_attack_angle

    source = json.loads(Path(receipt["prediction_manifest"]).read_text())
    if source["input_sha256"] != job["input_sha256"]:
        raise ValueError("Producer input identity changed")
    predictions = source["predictions"]
    expected_samples = int(job.get("expected_predictions", 50))
    if len(predictions) != expected_samples:
        raise ValueError(f"Require the complete {expected_samples}-pose ensemble")
    geometry = job["modeling_input"]["geometry"]
    entities = job["entities"]
    ccd_atoms = component_inventories(source.get("chemical_component_sources"))

    def references(participant):
        spec = geometry[participant]
        if spec["kind"] == "cofactor":
            return [(entities[spec["ccd"]][0], spec["atom"])]
        chain = entities[spec["role"]]
        mapping = source["ligand_atom_maps"][chain]["atoms"]
        return [(chain, mapping[str(index)]["name"]) for index in spec["reactive_atom_indices"]]

    centers = references("electrophile")
    leaving = references("leaving_group")
    if len(centers) != 1 or len(leaving) != 1:
        raise ValueError("Reaction center is ambiguous")
    donor_refs = references("nucleophile")
    expected_protein_copies = len(job["modeling_input"]["enzyme_state"]["chains"])
    ligand_atom_maps = source.get("ligand_atom_maps", {})
    pocket_protocol = job.get("pocket_protocol")
    tasks = [
        (
            prediction, job["sequence_sha256"], expected_protein_copies, entities,
            ccd_atoms, ligand_atom_maps, centers[0], leaving[0], donor_refs, pocket_protocol,
        )
        for prediction in predictions
    ]
    first_path = Path(predictions[0]["path"]) if predictions else None
    first_size = first_path.stat().st_size if first_path and first_path.is_file() else 0
    if max_workers is None and (len(tasks) < 16 or first_size < 65536):
        workers = 1
    else:
        workers = bounded_workers(max_workers, n_items=len(tasks), min_items_for_pool=8, max_cap=8)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_single_thread_worker) as pool:
            poses = list(pool.map(_analyze_single_prediction, tasks))
    else:
        poses = [_analyze_single_prediction(t) for t in tasks]
    score = weighted_attack_angle([p["distance_A"] for p in poses], [p["angle_deg"] for p in poses])
    holo_seating = _summarize_holo_seating(poses)
    return {
        "poses": poses,
        "geometry": score,
        "holo_seating": holo_seating,
        "identity_pass_poses": sum(not p["identity_issues"] for p in poses),
        **({"pocket_localized_poses": sum(p["pocket_localized"] for p in poses)} if pocket_protocol else {}),
        "step": job["step"],
        "ideal_angle": geometry.get("ideal_angle"),
        "scope": "Raw step-specific structural observations; no enzyme-activity threshold or cross-step activity ranking.",
    }
