"""Family cofold job rendering, prioritization, cross-run reuse indexing, and result publishing."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import string

from enzyme_discovery_agent.state_store import atomic_write_json
from ..msa import valid_csv
from ..scripts.build_step1_inputs import enzyme_chains, modeled_substrates, yaml_for
from .pose_analysis import analyze_outputs
from .protocol import COFOLD_ARM_NAMES, DEFAULT_SAMPLES, DEFAULT_SEED, digest, file_hash


def _prepared_candidates(root):
    result = {}
    for path in (root / "agent_state/production_modeling").glob("*/item.json"):
        item = json.loads(path.read_text())
        task_path = Path(item.get("task", ""))
        if task_path.is_file():
            task = json.loads(task_path.read_text())
            for candidate in task.get("candidates", []):
                result[candidate["accession"]] = candidate
    return result


def prioritize_modeling_jobs(jobs):
    """Order cofold jobs round-robin across families, prioritizing reviewed/annotated leads first."""
    by_family = defaultdict(list)
    seen_member = {}
    for job in jobs:
        fam = job.get("family_id") or "unresolved"
        acc = job.get("accession") or ""
        key = (fam, acc)
        if key not in seen_member:
            seen_member[key] = {
                "is_swissprot": bool(job.get("is_swissprot")),
                "has_reaction": bool(job.get("recorded_activities")),
                "rank": int(job.get("rank") or 999999),
                "accession": acc,
                "jobs": [],
            }
            by_family[fam].append(seen_member[key])
        seen_member[key]["jobs"].append(job)

    for fam in by_family:
        by_family[fam].sort(
            key=lambda m: (
                0 if m["is_swissprot"] else 1,
                0 if m["has_reaction"] else 1,
                m["rank"],
                m["accession"],
            )
        )
        for m in by_family[fam]:
            m["jobs"].sort(key=lambda j: int(j.get("step") or 0))

    family_order = sorted(
        by_family,
        key=lambda f: (
            0 if by_family[f][0]["is_swissprot"] else 1,
            0 if by_family[f][0]["has_reaction"] else 1,
            by_family[f][0]["rank"],
            f,
        ),
    )
    ordered = []
    max_depth = max((len(v) for v in by_family.values()), default=0)
    for slot in range(max_depth):
        for fam in family_order:
            if slot < len(by_family[fam]):
                ordered.extend(by_family[fam][slot]["jobs"])
    return ordered


def render_jobs(run_dir, handoff, directory, candidates=None, *, samples=None, seed=None):
    """Only sequence/identity/MSA expansion is per member; no scientific replanning."""
    root = Path(run_dir)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = candidates if candidates is not None else _prepared_candidates(root)
    jobs = []
    pending = []
    n_samples = int(samples if samples is not None else handoff.get("expected_predictions", DEFAULT_SAMPLES))
    run_seed = int(seed if seed is not None else handoff.get("seed", DEFAULT_SEED))
    for family in handoff["families"]:
        protocol = handoff["protocols"][family["protocol_id"]]
        for member in family["members"]:
            accession = member["accession"]
            candidate = candidates.get(accession)
            try:
                if not candidate:
                    raise ValueError("Exact sequence/MSA preparation pending")
                evidence = candidate["evidence"]
                sequence = candidate["sequence"]
                identity = Path(evidence["identity_source_path"])
                if file_hash(identity) != evidence["identity_source_sha256"]:
                    raise ValueError("Sequence source changed")
                record = json.loads(identity.read_text())
                if record.get("primaryAccession") != accession or record.get("sequence", {}).get("value") != sequence:
                    raise ValueError("Sequence identity mismatch")
                msa = Path(evidence["msa_path"])
                if not valid_csv(msa, sequence):
                    raise ValueError("A validated exact-sequence MSA is required")
                for step, model in protocol["steps"].items():
                    chains = enzyme_chains(sequence, str(msa), model["enzyme_state"]["chains"])
                    text = yaml_for(chains, model["cofactors"], modeled_substrates(model))
                    job_id = digest({"yaml": text, "msa_sha256": file_hash(msa), "samples": n_samples, "profile": "exact", "seed": run_seed})
                    path = directory / (job_id + ".yaml")
                    if not path.exists():
                        path.write_text(text)
                    if path.read_text() != text:
                        raise ValueError("Existing rendered input changed")
                    labels = iter(string.ascii_uppercase)
                    entities = {}
                    for _ in chains:
                        next(labels)
                    for cofactor in model["cofactors"]:
                        for _ in range(cofactor["count"]):
                            entities.setdefault(cofactor["ccd"], []).append(next(labels))
                    for substrate in modeled_substrates(model):
                        entities[substrate["role"]] = next(labels)
                    jobs.append({
                        **member,
                        "id": job_id,
                        "family_id": family["family_id"],
                        "protocol_id": family["protocol_id"],
                        "discovery_run_id": root.name,
                        "discovery_run_dir": str(root.resolve()),
                        "step": int(step),
                        "input_yaml": str(path.resolve()),
                        "input_sha256": file_hash(path),
                        "msa_path": str(msa),
                        "msa_sha256": file_hash(msa),
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                        "expected_predictions": n_samples,
                        "seed": run_seed,
                        "entities": entities,
                        "modeling_input": model,
                    })
            except (KeyError, ValueError, OSError) as exc:
                pending.append({**member, "reason": str(exc)})
    return jobs, pending


def write_input_index(jobs, directory):
    """Expose readable member/stage aliases without changing fingerprinted inputs."""
    from rdkit import Chem

    directory = Path(directory)
    rows = []
    for job in jobs:
        name = job["accession"] if re.fullmatch("[A-Za-z0-9_.-]+", job["accession"]) else digest(job["accession"])[:20]
        folder = directory / "inputs/by_accession" / name
        folder.mkdir(parents=True, exist_ok=True)
        alias = folder / ("step_" + str(job["step"]) + ".yaml")
        temporary = alias.with_name(alias.name + "." + str(os.getpid()) + ".tmp")
        temporary.symlink_to(Path(job["input_yaml"]).resolve())
        os.replace(temporary, alias)
        model = job["modeling_input"]
        rows.append({
            "discovery_run_id": job.get("discovery_run_id", directory.parent.name),
            "enzyme_name": job["enzyme_name"],
            "accession": job["accession"],
            "family_id": job["family_id"],
            "step": job["step"],
            "input_yaml": str(alias),
            "cofactors": "; ".join(f"{v['ccd']} x {v['count']}" for v in model["cofactors"]),
            "ligands": "; ".join(
                f"{v['role']}: {v['modeling_smiles']} (charge {Chem.GetFormalCharge(Chem.MolFromSmiles(v['modeling_smiles']))})"
                for v in model["target_substrates"]
                if v.get("include_in_model")
            ),
        })
    if rows:
        path = directory / "input_index.csv"
        temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)


def build_reusable_index(run):
    """Scan completed sibling runs once and index valid entries by physical_system_key."""
    import yaml
    from campaign_catalog.lookup import physical_system_key

    index = defaultdict(list)
    parent = Path(run).parent
    for arm in COFOLD_ARM_NAMES:
        for path in parent.glob("*/" + arm + "/jobs/*/complete.json"):
            resolved = path.resolve()
            try:
                saved = json.loads(path.read_text())
                old = saved.get("reuse_source_job") or saved["job"]
                old.setdefault("discovery_run_id", Path(saved.get("reuse_source") or path).parents[3].name)
                input_path = Path(old["input_yaml"])
                if not input_path.is_file() or file_hash(input_path) != old["input_sha256"]:
                    continue
                key = physical_system_key(yaml.safe_load(input_path.read_text()))
            except (OSError, ValueError, KeyError):
                continue
            index[key].append((resolved, saved, old))
    return index


def find_reusable(job, run, specification, *, reusable_index=None, analyzer=None):
    """Consult completed family results and the existing registered prediction pools."""
    import yaml
    from campaign_catalog.lookup import lookup_exact_system, physical_system_key

    document = yaml.safe_load(Path(job["input_yaml"]).read_text())
    wanted = physical_system_key(document)
    self_paths = {(Path(run) / arm / "jobs" / job["id"] / "complete.json").resolve() for arm in COFOLD_ARM_NAMES}
    candidates = reusable_index.get(wanted, ()) if reusable_index is not None else build_reusable_index(run).get(wanted, ())
    job_geom_digest = digest(job["modeling_input"]["geometry"])
    analyze_fn = analyzer or analyze_outputs
    for resolved, saved, old in candidates:
        if resolved in self_paths:
            continue
        if digest(old["modeling_input"]["geometry"]) != job_geom_digest:
            return {"status": "exact_coordinates_available_analysis_pending", "source": str(resolved), "saved": saved}
        analysis = analyze_fn(old, saved["receipt"])
        return {"status": "reused", "source": str(resolved), "job": old, "receipt": saved["receipt"], "analysis": analysis}
    for pool in specification.get("prediction_pools", []):
        if not (Path(pool) / "catalog/current.json").is_file():
            continue
        result = lookup_exact_system(pool, document)
        if result["exact_system_matches"]:
            return {
                "status": "exact_coordinates_available_analysis_pending",
                "pool": str(pool),
                "matches": result["exact_system_matches"],
                "physical_system_key": wanted,
            }
    return None


def publish_results(directory, jobs):
    directory = Path(directory)
    rows = []
    for job in jobs:
        folder = directory / "jobs" / job["id"]
        complete = folder / "complete.json"
        pending = folder / "reuse_pending_analysis.json"
        saved = json.loads(complete.read_text()) if complete.exists() else {}
        score = (saved.get("analysis") or {}).get("geometry") or {}
        rows.append({
            "discovery_run_id": job.get("discovery_run_id", directory.parent.name),
            "producer_discovery_run_id": (saved.get("reuse_source_job") or job).get("discovery_run_id", directory.parent.name),
            "enzyme_name": job["enzyme_name"],
            "accession": job["accession"],
            "family_id": job["family_id"],
            "step": job["step"],
            "status": "complete" if saved else "cached_analysis_pending" if pending.exists() else "pending",
            "n_poses": score.get("n_poses"),
            "weighted_attack_angle_deg": score.get("weighted_attack_angle_deg"),
            "harmonic_distance_A": score.get("harmonic_distance_A"),
            "reused": bool(saved.get("reuse_source")),
            "numerical_profile": saved.get("receipt", {}).get("numerical_profile"),
            "input_yaml": job["input_yaml"],
            "result": str(complete) if saved else str(pending) if pending.exists() else "",
        })
    if rows:
        path = directory / "family_results.csv"
        temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
        with temporary.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    from .analysis import refresh

    try:
        refresh(directory)
    except ImportError as exc:
        atomic_write_json(
            directory / "analysis_backend_pending.json",
            {"error": str(exc), "scope": "Plotting unavailable on this frontend; completed coordinates remain available for the central analyzer."},
        )
