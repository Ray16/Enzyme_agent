"""Inspect saved evidence and retrieve sequence-checked modeling inputs."""

import hashlib
import json
import re
from pathlib import Path

from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.state_store import atomic_write_json


# A turn costs a full context resend, so a page that is too small is not a saving: a
# long source read eighty lines at a time spends far more tokens on repeated context
# than the extra text would ever cost. Pages are sized to cross a typical paper in a
# couple of calls instead of eight.
SOURCE_PAGE_LINES = 320
SOURCE_PAGE_LIMIT = 640
SOURCE_PAGE_CHARS = 40_000


@tool(description="Read a bounded passage from a retrieved text or JSON source under this run's artifacts directory. Search for a literal phrase to locate identity, methods, accession or sequence evidence; inspect neighboring lines before interpreting it. Pages are large: read in a few big passages rather than many small ones, and follow next_start_line to continue.",
      category="reasoning", injected_params=["run_dir"], params={
          "source_path": {"type": "string"},
          "find": {"type": "string", "default": ""},
          "start_line": {"type": "integer", "default": 1},
          "max_lines": {"type": "integer", "default": SOURCE_PAGE_LINES},
          "run_dir": {"type": "string"}})
def inspect_discovery_source(source_path, find="", start_line=1, max_lines=SOURCE_PAGE_LINES, run_dir=None):
    root = Path(run_dir).resolve()
    path = Path(source_path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root / "artifacts") or not path.is_file():
        return {"status": "error", "summary": "Expected a retrieved source under this run's artifacts directory", "data": None}
    if path.suffix.lower() not in {".txt", ".json", ".md", ".xml", ".fasta", ".fa"}:
        return {"status": "error", "summary": "Inspect a text extraction rather than a binary source", "data": None}
    raw = path.read_bytes()
    if len(raw) > 20_000_000:
        return {"status": "error", "summary": "Source exceeds the text inspection size limit", "data": None}
    lines = raw.decode("utf-8", errors="replace").splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if find.casefold() in line.casefold()] if find else []
    start = max(1, int(start_line))
    if find and start == 1 and hits:
        start = max(1, hits[0] - 8)
    end = min(len(lines), start + min(max(1, int(max_lines)), SOURCE_PAGE_LIMIT) - 1)
    passage = "\n".join(f"{i + 1}: {lines[i][:1800]}" for i in range(start - 1, end))[:SOURCE_PAGE_CHARS]
    # Report where reading actually stopped: a page truncated on characters rather than
    # on lines would otherwise look complete and the remainder would never be requested.
    delivered = start - 1 + passage.count("\n") + 1 if passage else start - 1
    return {"status": "success", "summary": f"Retrieved source lines {start}-{min(end, delivered)} of {len(lines)}", "data": {
        "source_path": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "matching_lines": hits[:100], "total_lines": len(lines),
        "next_start_line": min(end, delivered) + 1 if min(end, delivered) < len(lines) else None,
        "passage": passage}}


def _official_record(accession):
    from enzyme_discovery_agent.models.candidate_registry import UNIPROT_ACC_RE
    if UNIPROT_ACC_RE.fullmatch(accession):
        from enzyme_discovery_agent.tools.reasoning import _fetch_uniprot_entry
        entry = _fetch_uniprot_entry(accession)
        if not entry or entry.get("primaryAccession") != accession or entry.get("inactiveReason"):
            raise ValueError("Active authoritative accession could not be verified")
        return entry["sequence"]["value"], "UniProtKB", entry
    from enzyme_discovery_agent.tools.database import fetch_ncbi_protein
    reply = fetch_ncbi_protein(accession)
    entry = reply.get("data") or {}
    if reply.get("status") != "success" or entry.get("accession") != accession:
        raise ValueError("Deposited accession and version could not be verified")
    return entry["sequence"], "NCBI Protein", entry


@tool(description="Retrieve a registered, source-resolved candidate's authoritative sequence and save FASTA plus an identity manifest for downstream modeling. Optionally download an existing AlphaFold model only when its reported sequence exactly matches. This retrieves inputs; it does not submit predictions or validate activity.",
      category="reasoning", parallel_safe=False, injected_params=["run_dir"], params={
          "accession": {"type": "string"},
          "include_existing_model": {"type": "boolean", "default": True},
          "run_dir": {"type": "string"}})
def retrieve_candidate_modeling_inputs(accession, include_existing_model=True, run_dir=None):
    root = Path(run_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", accession):
        return {"status": "error", "summary": "Invalid sequence accession", "data": None}
    entry = load_candidate_registry(root, strict=True)["candidates"].get(accession)
    meta = (entry or {}).get("metadata") or {}
    identity = meta.get("identity_resolution") or {}
    if identity.get("status") != "resolved" or identity.get("resolved_accession") != accession:
        return {"status": "partial", "summary": "Resolve this candidate to a source-supported sequence first", "data": None}
    try:
        sequence, namespace, record = _official_record(accession)
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "error", "summary": str(exc), "data": None}
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence) or digest != identity.get("sequence_sha256"):
        return {"status": "error", "summary": "Retrieved sequence differs from resolved source identity; review before modeling", "data": None}
    destination = root / "artifacts" / "modeling_inputs" / accession
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "authoritative_record.json", record)
    fasta = destination / "sequence.fasta"
    fasta.write_text(f">{accession}\n" + "\n".join(sequence[i:i+80] for i in range(0, len(sequence), 80)) + "\n")
    model = {"status": "not_requested"}
    if include_existing_model and namespace == "UniProtKB":
        from enzyme_discovery_agent.tools._http import _get_session
        session = _get_session()
        try:
            response = session.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=20)
            predictions = response.json() if response.status_code == 200 else []
            matching = [p for p in predictions if p.get("uniprotAccession") == accession and p.get("uniprotSequence") == sequence]
            model = {"status": "unavailable_or_sequence_unverified", "http_status": response.status_code}
            if matching:
                prediction = matching[0]
                url = prediction.get("cifUrl", "")
                if re.fullmatch(r"https://alphafold\.ebi\.ac\.uk/files/[^/]+\.cif", url):
                    coordinates = session.get(url, timeout=30)
                    coordinates.raise_for_status()
                    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
                    from io import StringIO
                    cif = MMCIF2Dict(StringIO(coordinates.text))
                    sequences = [re.sub(r"\s", "", s) for s in cif.get("_entity_poly.pdbx_seq_one_letter_code_can", [])]
                    if sequence not in sequences:
                        raise ValueError("Coordinate sequence does not match the authoritative candidate")
                    path = destination / "existing_alphafold_model.cif"
                    path.write_bytes(coordinates.content)
                    atomic_write_json(destination / "alphafold_metadata.json", prediction)
                    model = {"status": "retrieved_sequence_verified", "path": str(path), "url": url,
                             "sha256": hashlib.sha256(coordinates.content).hexdigest(),
                             "experimental": False, "mean_plddt": prediction.get("globalMetricValue")}
        except Exception as exc:
            model = {"status": "retrieval_failed", "error_type": type(exc).__name__}
    result = {"accession": accession, "namespace": namespace, "sequence_length": len(sequence),
              "sequence_sha256": digest, "fasta_path": str(fasta), "source_identity": identity,
              "existing_model": model, "activity_validated": False, "new_prediction_jobs": 0}
    manifest = destination / "input_manifest.json"
    atomic_write_json(manifest, result)
    update_candidate_registry(root, accession, None, {"sequence": sequence, "sequence_length": len(sequence),
        "modeling_input_files": {"manifest_path": str(manifest), "fasta_path": str(fasta), "existing_model": model}})
    from enzyme_discovery_agent.models.event_log import append_event
    append_event(root, "candidate_modeling_inputs_retrieved", accession=accession, manifest_path=str(manifest),
                 sequence_sha256=digest, existing_model_status=model["status"], new_prediction_jobs=0)
    return {"status": "success", "summary": "Source-resolved sequence saved for modeling; activity remains separately assessed", "data": result}
