"""Optional BioEmu2 preparation and explicitly invoked single-job runner.

Preparation is CPU-only and network-free. Run in a separately provisioned
BioEmu2 environment under the user's resource scheduler; this module does not
acquire reservations, search MSAs, download weights or launch a fleet.
API reference: microsoft/bioemu2 commit 06be369dd555849165a737472d42d936959e46fc.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil

from enzyme_discovery_agent.state_store import atomic_write_json
from .validation import fingerprint, validate_msa, validate_pose_plan, _source_checks

UPSTREAM_REVISION = "06be369dd555849165a737472d42d936959e46fc"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _write_immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("BioEmu2 preparation is immutable; use a new model revision")
    else:
        atomic_write_json(path, value, allow_nan=False)


def prepare(execution_path, checkpoint_path, output_dir, *, seed, mode="equilibrium", baseline_sources=(), num_samples=None):
    if type(seed) is not int or mode not in {"equilibrium", "structure"}:
        raise ValueError("declare an integer seed and equilibrium or structure mode")
    source = Path(execution_path).resolve(strict=True)
    execution = json.loads(source.read_text())
    record, entities = execution["modeling_input"], execution["entity_map"]
    if validate_pose_plan(record, entities)["status"] != "pass":
        raise ValueError("BioEmu2 needs the same complete physical acceptance plan as cofolding")
    declarations = list(record["validation_plan"]["setup"].values()) + [record["validation_plan"]["poses"]]
    policy = record["validation_plan"]["poses"].get("protein_atom_policy") or {}
    declarations += policy.get("allowed_missing", []) + policy.get("modified_residues", [])
    references = []
    for declaration in declarations:
        sources = declaration.get("evidence")
        if any(c["status"] != "pass" for c in _source_checks(sources, "bioemu2.setup")):
            raise ValueError("setup evidence is missing or changed")
        references.extend({"path": str(Path(s["path"]).resolve()), "sha256": s["sha256"]} for s in sources)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    config = checkpoint.with_name("config.yaml")
    if not checkpoint.is_file() or not config.is_file():
        raise ValueError("supply a local checkpoint and its matching config.yaml; no automatic downloads")
    input_yaml = source.with_name("input.yaml")
    if not input_yaml.is_file():
        raise ValueError("the execution must accompany its rendered input.yaml")
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    if num_samples is not None and (type(num_samples) is not int or num_samples < 1):
        raise ValueError("num_samples must be positive")
    system, msas = [], {}
    protein_index = 0
    protein_entities = [e for e in entities if e["kind"] == "protein"]
    if len(execution.get("msa_evidence", [])) != len(protein_entities):
        raise ValueError("MSA evidence must map in rendered protein-chain order")
    for entity in entities:
        chain = entity["chain"]
        if not isinstance(chain, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,4}", chain):
            raise ValueError("unsupported canonical chain ID")
        if entity["kind"] == "protein":
            sequence = entity["sequence"]
            alignment = execution["msa_evidence"][protein_index]
            protein_index += 1
            if alignment.get("query_sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest():
                raise ValueError(f"no fingerprinted chain MSA for {chain}")
            if digest(alignment["path"]) != alignment["sha256"]:
                raise ValueError("MSA changed after rendering")
            validate_msa(alignment["path"], sequence)
            text = Path(alignment["path"]).read_text()
            if text.lstrip().startswith(">"):
                a3m = text
            else:
                rows = list(csv.DictReader(text.splitlines()))
                if not rows or any(not r.get("sequence") for r in rows):
                    raise ValueError("BioEmu2 CSV conversion requires sequence rows")
                a3m = "".join(f">alignment_{i}\n{row['sequence']}\n" for i, row in enumerate(rows))
            destination = output / f"chain_{chain}.a3m"
            if destination.exists() and destination.read_text() != a3m:
                raise ValueError("prepared MSA changed; use a new revision")
            destination.write_text(a3m)
            validate_msa(destination, sequence)
            msas[chain] = str(destination)
            references.append({"path": str(Path(alignment["path"]).resolve()), "sha256": alignment["sha256"]})
            system.append({"entity_type": "protein", "asym_ids": [chain], "sequence": sequence})
        elif entity["kind"] == "cofactor":
            system.append({"entity_type": "ligand", "asym_ids": [chain], "ccd_code": entity["ccd"]})
        elif entity["kind"] == "substrate":
            system.append({"entity_type": "ligand", "asym_ids": [chain], "smiles": entity["modeling_smiles"]})
        else:
            raise ValueError("unsupported rendered entity; composition must not be silently dropped")
    # JSON is a YAML subset accepted by System.from_yaml; no YAML dependency is
    # needed in the controller's preparation environment.
    system_path = output / "system.yaml"
    _write_immutable(system_path, {"entities": system, "bonds": [], "asym_id_to_author_id": {}})
    paths = [source, input_yaml, system_path, checkpoint, config, Path(__file__)] + [Path(p) for p in msas.values()]
    references += [{"path": str(path), "sha256": digest(path)} for path in paths] + list(baseline_sources)
    job = {"schema_version": "bioemu2-job-1.0", "upstream_api_revision": UPSTREAM_REVISION,
           "execution_sha256": fingerprint(execution), "input_sha256": digest(input_yaml),
           "execution_path": str(source), "input_yaml": str(input_yaml),
           "system": str(system_path), "msa_by_chain": msas, "checkpoint_path": str(checkpoint),
           "purpose": "dynamics" if mode == "equilibrium" else "structure_prediction_exploration",
           "mode": mode, "seed": seed, "num_samples": num_samples or record["validation_plan"]["poses"]["expected_predictions"],
           "physicality_mode": "high_quality", "files": references,
           "conditioning": "unrestrained", "scope": "joint modeled complex; biological validity is not established"}
    path = output / "job.json"; _write_immutable(path, job)
    return {"job_path": str(path), "job_sha256": digest(path), "mode": mode, "seed": seed,
            "run_command": ["python", "-m", "structural_modeling_agent.bioemu2_adapter", "run", "--job", str(path), "--output", str(output / "results")]}


def verify_job(job):
    if job.get("schema_version") != "bioemu2-job-1.0":
        raise ValueError("unsupported BioEmu2 job")
    for source in job["files"]:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("BioEmu2 execution dependency changed: " + source["path"])
    required = [job["execution_path"], job["input_yaml"], job["system"], job["checkpoint_path"],
                str(Path(job["checkpoint_path"]).with_name("config.yaml")), *job["msa_by_chain"].values(), str(Path(__file__).resolve())]
    recorded = {str(Path(s["path"]).resolve()) for s in job["files"]}
    if any(str(Path(path).resolve()) not in recorded for path in required):
        raise ValueError("BioEmu2 job references an unfingerprinted input")
    if (job["mode"] not in {"equilibrium", "structure"} or type(job["seed"]) is not int or
            type(job["num_samples"]) is not int or job["num_samples"] < 1 or job["physicality_mode"] != "high_quality" or
            job["conditioning"] != "unrestrained"):
        raise ValueError("invalid or unsupported sampling settings")
    if (fingerprint(json.loads(Path(job["execution_path"]).read_text())) != job["execution_sha256"] or
            digest(job["input_yaml"]) != job["input_sha256"]):
        raise ValueError("BioEmu2 execution/input identity differs")


def run(job_path, output_dir):
    job = json.loads(Path(job_path).read_text()); verify_job(job)
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
        raise RuntimeError("run BioEmu2 only within an externally owned GPU reservation with explicit device visibility")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("sampling output must be empty; reconcile previous execution before retry")
    # Imports and model loading occur only after an explicit run invocation.
    from bioemu2 import System, get_msa, load_model, EmuSession, check_physicality, save_frames
    system = System.from_yaml(Path(job["system"]))
    msa = get_msa(system, source={chain: Path(path) for chain, path in job["msa_by_chain"].items()})
    model = load_model(checkpoint_path=Path(job["checkpoint_path"]))
    sampler = EmuSession(system, model=model, mode=job["mode"], msa=msa)
    samples = sampler.sample(num_samples=job["num_samples"], seed=job["seed"])
    if len(samples) != job["num_samples"]:
        raise ValueError("BioEmu2 returned incomplete sample coverage")
    physicality = check_physicality(samples, mode=job["physicality_mode"], system=system)
    save_frames(samples, output / "raw", physicality=physicality)
    statuses = json.loads((output / "raw" / "physicality.json").read_text())
    if len(statuses) != len(samples) or [row["frame"] for row in statuses] != list(range(len(samples))):
        raise ValueError("physicality report does not match the generated frames")
    predictions = output / "predictions"; predictions.mkdir(parents=True)
    entries = []
    for index, status in enumerate(statuses):
        frame_dir = output / "frames" / str(index)
        save_frames(samples[index:index+1], frame_dir)
        destination = predictions / f"bioemu2_model_{index}.cif"
        shutil.copyfile(frame_dir / "topology.cif", destination)
        entries.append({"path": str(destination), "sha256": digest(destination),
                        "sample_id": f"bioemu2:{job['seed']}:{index}", "engine_physicality": status})
    verify_job(job)
    manifest = {"producer": "bioemu2", "purpose": job["purpose"], "mode": job["mode"], "seed": job["seed"],
                "input_sha256": job["input_sha256"], "execution_sha256": job["execution_sha256"],
                "job_sha256": digest(job_path), "predictions": entries,
                "physicality_report": {"path": str(output / "raw" / "physicality.json"),
                                       "sha256": digest(output / "raw" / "physicality.json")},
                "supporting_files": [{"path": str(path), "sha256": digest(path)} for path in
                                     [Path(job_path).resolve(), output / "raw" / "samples.xtc",
                                      output / "raw" / "topology.cif", output / "raw" / "physicality.json"]],
                "limitations": "All frames, including rejected frames, retained. Engine physicality checks protein geometry only; controller checks the full modeled complex. No Boltz confidence metrics are fabricated."}
    path = output / "prediction_manifest.json"; atomic_write_json(path, manifest, allow_nan=False)
    return {"prediction_manifest": str(path), "predictions": str(predictions)}


def inspect_result(job_path, manifest_path):
    """Inspect a dynamics ensemble without replacing the primary cofold report."""
    import copy
    from .validation import validate_ensemble
    job = json.loads(Path(job_path).read_text()); verify_job(job)
    path = Path(manifest_path).resolve()
    producer = json.loads(path.read_text())
    if (producer.get("producer") != "bioemu2" or producer.get("purpose") != "dynamics" or
            producer.get("job_sha256") != digest(job_path) or producer.get("input_sha256") != job["input_sha256"] or
            producer.get("execution_sha256") != job["execution_sha256"]):
        raise ValueError("dynamics producer is not bound to the prepared job and baseline")
    entries = producer.get("predictions") or []
    ids = [e.get("sample_id") for e in entries]
    frames = [(e.get("engine_physicality") or {}).get("frame") for e in entries]
    if (len(entries) != job["num_samples"] or any(not isinstance(i, str) or not i for i in ids) or
            len(set(ids)) != len(ids) or any(type(i) is not int for i in frames) or sorted(frames) != list(range(len(entries)))):
        raise ValueError("incomplete or duplicate dynamics sample identities/physicality records")
    sources = producer.get("supporting_files") or []
    if not sources:
        raise ValueError("dynamics raw trajectory and physicality evidence are missing")
    physicality_source = producer.get("physicality_report") or {}
    physicality_path = Path(physicality_source.get("path", ""))
    physicality_path = (physicality_path if physicality_path.is_absolute() else path.parent / physicality_path).resolve()
    if (not physicality_path.is_file() or digest(physicality_path) != physicality_source.get("sha256") or
            json.loads(physicality_path.read_text()) != sorted([e["engine_physicality"] for e in entries], key=lambda row: row["frame"])):
        raise ValueError("engine physicality declarations differ from the raw report")
    paths, supplements = [], {}
    for source in entries + sources:
        file = Path(source["path"])
        file = (file if file.is_absolute() else path.parent / file).resolve()
        if digest(file) != source["sha256"]:
            raise ValueError("dynamics supporting file changed")
    for entry in entries:
        file = Path(entry["path"])
        file = (file if file.is_absolute() else path.parent / file).resolve()
        paths.append(file)
        physicality = entry["engine_physicality"]
        status = physicality.get("status")
        supplements[str(file)] = [{"check": "bioemu2.protein_physicality", "status": "pass" if status == "passed" else "fail" if status == "rejected" else "abstain",
                                  "reason": physicality.get("reason") or "engine protein-only physicality check"}]
    if len(set(paths)) != len(paths) or len({digest(p) for p in paths}) != len(paths):
        raise ValueError("copied dynamics frames cannot establish ensemble coverage")
    execution = copy.deepcopy(json.loads(Path(job["execution_path"]).read_text()))
    # These are comparative observations, not the initial cofold acceptance gate.
    execution["modeling_input"]["validation_plan"]["poses"].update(expected_predictions=job["num_samples"], minimum_passing_predictions=1)
    report = validate_ensemble(paths, execution, supplemental_checks=supplements)
    return {"status": "inspected", "purpose": "dynamics", "mode": job["mode"],
            "source_manifest": {"path": str(path), "sha256": digest(path)},
            "cofold_geometry_checks_for_comparison": report,
            "limitations": "All frames retained. Geometry checks describe conformations, not catalytic activity. Equilibrium samples do not establish transition rates; dynamics evidence cannot satisfy the initial cofolding requirement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--execution", required=True); command.add_argument("--checkpoint", required=True)
    command.add_argument("--output", required=True); command.add_argument("--seed", required=True, type=int)
    command.add_argument("--mode", choices=["equilibrium", "structure"], default="equilibrium")
    command = commands.add_parser("run"); command.add_argument("--job", required=True); command.add_argument("--output", required=True)
    args = parser.parse_args()
    result = (prepare(args.execution, args.checkpoint, args.output, seed=args.seed, mode=args.mode)
              if args.operation == "prepare" else run(args.job, args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
