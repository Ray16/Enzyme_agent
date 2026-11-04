"""Run an explicit cofold queue under an externally owned GPU reservation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

from structural_modeling_agent.proxies.attack_angle import pdb_geometry, weighted_attack_angle


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execution_spec(manifest, item):
    """Bind declared execution/scoring inputs and verify their current bytes."""
    from structural_modeling_agent.validation import fingerprint
    for path_key, hash_key in (("input_yaml", "input_sha256"), ("msa_path", "msa_sha256")):
        if digest(Path(item[path_key])) != item[hash_key]:
            raise ValueError(f"current {path_key} differs from its declared fingerprint")
    for source in item.get("msa_evidence", []) + manifest.get("producer_dependencies", []):
        if not source.get("sha256") or digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("declared chain-MSA or producer dependency changed")
    if type(item.get("seed")) is not int or type(item.get("expected_predictions")) is not int or item["expected_predictions"] < 1:
        raise ValueError("declare an integer seed and positive sample count")
    argv = manifest["producer_argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
        raise ValueError("producer_argv must be a nonempty string array")
    producer_files = []
    for argument in argv:
        path = Path(shutil.which(argument) or argument)
        if path.is_file():
            producer_files.append({"path": str(path.resolve()), "sha256": digest(path)})
    from structural_modeling_agent.proxies import attack_angle
    from structural_modeling_agent import validation
    spec = {"schema_version": "queue-execution-2.0", "item": item, "producer_argv": argv,
            "producer_dependencies": manifest.get("producer_dependencies", []),
            "producer_files": producer_files, "ranking": manifest.get("ranking"),
            "scoring_code_sha256": digest(Path(attack_angle.__file__)),
            "structure_validation_code_sha256": digest(Path(validation.__file__)),
            "verification_code_sha256": digest(Path(__file__))}
    return spec, fingerprint(spec)


def verify_completion(manifest, item, saved):
    """A saved status alone never establishes current execution or file identity."""
    from structural_modeling_agent.validation import fingerprint
    spec, identity = execution_spec(manifest, item)
    if saved.get("schema_version") != "queue-completion-2.0" or saved.get("execution_sha256") != identity or saved.get("execution_spec") != spec:
        raise ValueError("cached completion lacks the current execution/scoring identity; explicit revalidation is required")
    checked = verify_outputs(Path(saved["output"]), item)
    for field in ("geometry", "score", "confidence_files"):
        if fingerprint(checked[field]) != fingerprint(saved.get(field)):
            raise ValueError(f"cached {field} differs from recomputed supporting evidence")
    return checked


def verify_outputs(directory, item):
    models = sorted(directory.rglob("*model_*.pdb"))
    expected = item["expected_predictions"]
    if len(models) != expected:
        raise ValueError(f"Expected {expected} model files, found {len(models)}")
    slots = [int(re.search(r"_model_(\d+)\.pdb$", p.name).group(1)) for p in models]
    if sorted(slots) != list(range(expected)):
        raise ValueError("Model slots are duplicated or missing")
    confidences = list(directory.rglob("confidence_*model_*.json"))
    if len(confidences) != expected:
        raise ValueError("Matching confidence outputs are incomplete")
    initial_hashes = {str(path): digest(path) for path in models + confidences}
    if len({initial_hashes[str(path)] for path in models}) != len(models):
        raise ValueError("byte-identical predictions cannot satisfy ensemble coverage")
    matching = {path.with_name("confidence_" + path.stem + ".json") for path in models}
    if matching != set(confidences):
        raise ValueError("confidence files do not match the exact model slots")
    for path in confidences:
        value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite confidence value")))
        if not isinstance(value, dict):
            raise ValueError("confidence sidecar must be a JSON object")
        def finite_tree(node):
            if isinstance(node, float):
                return math.isfinite(node)
            if isinstance(node, dict):
                return all(finite_tree(v) for v in node.values())
            if isinstance(node, list):
                return all(finite_tree(v) for v in node)
            return True
        if not finite_tree(value):
            raise ValueError("nonfinite confidence value")
    from Bio.SeqUtils import seq1
    geometry = []
    for path in models:
        from structural_modeling_agent.validation import _read_structure, _protein_completeness
        atoms, _ = _read_structure(path)
        if any(check["status"] != "pass" for check in _protein_completeness(
                atoms, item["protein_chain"], item.get("protein_atom_policy") or {})):
            raise ValueError("predicted protein heavy-atom completeness is unresolved")
        residues = {}
        atom_chains = {}
        for line in path.read_text().splitlines():
            if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
                continue
            chain = line[21]
            atom_chains.setdefault(chain, []).append(line)
            if chain == item["protein_chain"]:
                residues.setdefault(line[22:27], line[17:20].strip())
        sequence = "".join(seq1(value, custom_map={"MSE": "M"}) for value in residues.values())
        if hashlib.sha256(sequence.encode()).hexdigest() != item["sequence_sha256"]:
            raise ValueError("Predicted protein construct does not match the submitted sequence")
        if set(atom_chains) != set(item["expected_chains"]):
            raise ValueError("Predicted entity inventory differs from the declared model")
        metal = atom_chains[item["metal_chain"]]
        if len(metal) != 1 or metal[0][76:78].strip().upper() != item["metal_element"]:
            raise ValueError("Predicted metal identity/count differs from the declared model")
        d, a = pdb_geometry(path, **item["geometry"])
        geometry.append({"path": str(path), "sha256": digest(path), "distance_A": d, "angle_deg": a})
    if any(digest(Path(path)) != expected_hash for path, expected_hash in initial_hashes.items()):
        raise ValueError("prediction or confidence bytes changed during verification")
    return {"geometry": geometry, "score": weighted_attack_angle(
        [x["distance_A"] for x in geometry], [x["angle_deg"] for x in geometry]),
        "confidence_files": [{"path": str(p), "sha256": digest(p)} for p in confidences]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Run only under a live GPU reservation")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > manifest["cpu_slots_per_worker"]:
        raise RuntimeError("Worker CPU affinity exceeds its allocation")
    if max(os.getloadavg()) + manifest["cpu_slots_per_worker"] > os.cpu_count():
        raise RuntimeError("Host load cannot cover the worker allocation")
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"Missing native-thread cap: {key}")
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
             "worker_id": args.worker_id, "affinity": affinity, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
             "manifest_sha256": digest(args.manifest), "started_at": time.time()}
    write(root / "workers" / (args.worker_id + ".json"), owner)
    failures = 0
    for item in manifest["items"]:
        identity = item["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity):
            raise ValueError("Unsafe queue identity")
        complete = root / "completed" / (identity + ".json")
        if complete.exists():
            try:
                verify_completion(manifest, item, json.loads(complete.read_text()))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                write(root / "failures" / (identity + ".json"), {
                    "item": identity, "status": "unresolved_cached_result", "error": str(exc), "owner": owner})
            continue
        claim = root / "claims" / identity
        claim.parent.mkdir(exist_ok=True)
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write(claim / "owner.json", owner)
        attempt = None
        attempt_path = None
        try:
            if complete.exists():
                continue
            input_path = Path(item["input_yaml"])
            spec, execution_hash = execution_spec(manifest, item)
            attempts = root / "attempts" / identity
            attempts.mkdir(parents=True, exist_ok=True)
            records = list(attempts.glob("attempt_*.json"))
            if any(json.loads(path.read_text()).get("status") in {"running", "unknown"} for path in records):
                raise RuntimeError("prior producer ownership must be reconciled before retry")
            launched = sum(json.loads(path.read_text()).get("status") != "reserved" for path in records)
            if launched >= manifest["maximum_attempts"]:
                continue
            existing_outputs = list((root / "predictions" / identity).glob("attempt_*"))
            used = [int(path.stem.split("_")[-1]) for path in records + existing_outputs
                    if path.stem.split("_")[-1].isdigit()]
            number = max(used, default=0) + 1
            output = root / "predictions" / identity / f"attempt_{number}"
            attempt_path = attempts / f"attempt_{number}.json"
            attempt = {"item": identity, "attempt": number, "owner": owner, "status": "reserved",
                       "output": str(output), "execution_sha256": execution_hash,
                       "preserved_prejournal_outputs": [str(path) for path in existing_outputs if not (attempts / (path.name + ".json")).exists()]}
            write(attempt_path, attempt)
            output.mkdir(parents=True)
            log = root / "logs" / f"{identity}.attempt_{number}.log"
            log.parent.mkdir(exist_ok=True)
            command = manifest["producer_argv"] + ["--input", str(input_path), "--out_dir", str(output),
                "--seed", str(item["seed"]), "--diffusion_samples", str(item["expected_predictions"]),
                "--output_format", "pdb", "--num_workers", "0", "--preprocessing-threads", "1"]
            attempt.update(command=command, status="running", log=str(log), input_sha256=item["input_sha256"])
            write(attempt_path, attempt)
            print(json.dumps({"event": "prediction_start", "item": identity, "log": str(log)}), flush=True)
            with log.open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Predictor exited {result.returncode}; see {log}")
            checked = verify_outputs(output, item)
            # Input and producer identity must also survive the execution.
            if execution_spec(manifest, item)[1] != execution_hash:
                raise ValueError("execution dependencies changed while the producer ran")
            write(complete, {"schema_version": "queue-completion-2.0", "execution_spec": spec,
                  "execution_sha256": execution_hash, "item": identity, "input_sha256": item["input_sha256"],
                  "status": "structural_outputs_complete", "output": str(output), "owner": owner, **checked})
            attempt.update(status="complete", finished_at=time.time())
            write(attempt_path, attempt)
            failures = 0
            print(json.dumps({"event": "prediction_complete", "item": identity, "score": checked["score"]}), flush=True)
        except Exception as exc:
            failures += 1
            if attempt is not None and attempt_path is not None:
                attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
                write(attempt_path, attempt)
            write(root / "failures" / (identity + ".json"), {"item": identity, "error": str(exc), "owner": owner})
            print(json.dumps({"event": "prediction_failed", "item": identity, "error": str(exc)}), flush=True)
            if failures >= manifest["failure_circuit_breaker"]:
                raise
        finally:
            shutil.rmtree(claim)
    print(json.dumps({"event": "worker_exit", "worker_id": args.worker_id}), flush=True)


if __name__ == "__main__":
    main()
