#!/usr/bin/env python3
"""Build Boltz cofold YAMLs from discovery-agent handoff records.

The default is fail closed: no candidate is modeled unless its candidate-specific
``modeling_input`` is marked ready and passes local consistency checks. Exploratory
generation requires an explicit override and is recorded in the campaign manifest.
Candidate sequences and MSAs are explicit command-line inputs because they are
structural-modeling artifacts rather than claims contained in the handoff.
"""
import argparse
import hashlib
import json
import os
import stat
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # structural_modeling_agent/
REPO_ROOT = os.path.dirname(ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
OUT = f"{ROOT}/step1"


def read_fasta(path):
    seqs = {}; acc = None; buf = []
    for line in open(path):
        if line.startswith(">"):
            if acc: seqs[acc] = "".join(buf)
            acc = line[1:].strip().split()[0]; buf = []
        else:
            buf.append(line.strip())
    if acc: seqs[acc] = "".join(buf)
    return seqs


def sequence_identity_errors(candidate, sequence, reference, base):
    """Check supplied sequence against a fingerprinted exact-identity record."""
    accession = candidate.get("accession")
    errors = []
    supplied = candidate.get("sequence")
    if supplied and supplied != sequence:
        errors.append("FASTA sequence conflicts with the handoff sequence")
    if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
        return errors + ["a fingerprinted sequence identity record is required"]
    try:
        path = Path(reference["path"])
        path = path if path.is_absolute() else Path(base) / path
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
                raise ValueError("sequence identity source must be a bounded regular file")
            with os.fdopen(os.dup(fd), "rb") as handle:
                content = handle.read(16 * 1024 * 1024 + 1)
            if len(content) > 16 * 1024 * 1024:
                raise ValueError("sequence identity source exceeds the size limit")
        finally:
            os.close(fd)
        if hashlib.sha256(content).hexdigest() != reference["sha256"]:
            raise ValueError("sequence identity source fingerprint mismatch")
        from enzyme_discovery_agent.models.handoff_contract import _unique_object, _reject_constant
        record = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if (record.get("primaryAccession") != accession or record.get("inactiveReason")
                or str(record.get("entryType") or "").casefold() == "inactive"
                or (record.get("sequence") or {}).get("value") != sequence):
            raise ValueError("sequence identity record does not match the accession and FASTA")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(str(exc))
    return errors


def load_handoff_modeling(path):
    """Return the handoff and accession-indexed candidate records."""
    if not path or not os.path.exists(path):
        raise ValueError("--handoff must name an existing schema-2.0 or 3.0 handoff")
    from enzyme_discovery_agent.models.handoff_contract import load_document, validate_handoff
    document = load_document(path)
    if document.get("schema_version") == "3.0":
        validate_handoff(document)
    elif document.get("schema_version") != "2.0":
        raise ValueError(
            f"unsupported handoff schema {document.get('schema_version')!r}; expected '2.0' or '3.0'"
        )
    spec = {}
    for candidate in document.get("candidates", []):
        accession = candidate.get("accession")
        if not accession:
            raise ValueError("a handoff candidate lacks an accession")
        if not isinstance(candidate.get("evidence"), dict):
            raise ValueError(f"candidate {accession} lacks an evidence layer")
        if not isinstance(candidate.get("modeling_input"), dict):
            raise ValueError(f"candidate {accession} lacks a modeling_input layer")
        if not isinstance(candidate.get("selection"), dict):
            raise ValueError(f"candidate {accession} lacks a selection layer")
        spec[accession] = candidate
    return document, spec


def execution_shape_errors(modeling_input):
    """Return malformed-input errors that an exploratory override cannot bypass."""
    errors = []
    substrates = modeling_input.get("target_substrates") or []
    if not (modeling_input.get("enzyme_state") or {}).get("chains"):
        errors.append("enzyme state/chains are missing")
    included = [s for s in substrates if s.get("include_in_model", True)]
    if not included:
        errors.append("no target component is selected for modeling")
    for cofactor in modeling_input.get("cofactors") or []:
        if not cofactor.get("ccd"):
            errors.append("a cofactor lacks a CCD identifier")
        if type(cofactor.get("count")) is not int or cofactor["count"] < 1:
            errors.append("a cofactor lacks a positive integer count")
    for substrate in substrates:
        if substrate.get("include_in_model", True) and not substrate.get("modeling_smiles"):
            errors.append(
                f"component {substrate.get('component_index')} modeling SMILES is missing"
            )
        if not substrate.get("modeling_representation", {}).get("valid", True):
            errors.append(
                f"component {substrate.get('component_index')} modeling SMILES is invalid"
            )
    return list(dict.fromkeys(errors))


def readiness_errors(modeling_input):
    errors = execution_shape_errors(modeling_input)
    if modeling_input.get("status") != "ready":
        errors.append(f"status={modeling_input.get('status', 'missing')}")
    errors.extend(modeling_input.get("unresolved") or [])
    for substrate in modeling_input.get("target_substrates") or []:
        if substrate.get("review_status") == "required":
            errors.append(
                f"component {substrate.get('component_index')} state change is unreviewed"
            )
        stereo = substrate.get("modeling_representation", {}).get(
            "stereochemistry", {},
        ).get("status")
        if stereo == "unspecified":
            errors.append(
                f"component {substrate.get('component_index')} stereochemistry is unspecified"
            )
    return list(dict.fromkeys(errors))


def modeled_substrates(modeling_input):
    return [
        substrate for substrate in modeling_input.get("target_substrates") or []
        if substrate.get("include_in_model", True)
    ]


def load_handoff_modeling_legacy(path):
    """Compatibility shim retained for callers that imported the old helper."""
    _, candidates = load_handoff_modeling(path)
    spec = {}
    for acc, candidate in candidates.items():
        mi = candidate.get("modeling_input") or {}
        if mi:
            spec[acc] = mi
    return spec


def enzyme_chains(sequence, msa, chain_specs):
    """Resolve generic 1-based inclusive sequence ranges from the handoff."""
    chains = []
    for entity in chain_specs:
        source = entity.get("source")
        if source == "full_sequence":
            chains.append({"sequence": sequence, "msa": msa, "source": entity})
        elif source == "sequence_range":
            start, end = entity.get("start"), entity.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("sequence_range requires integer start/end")
            if start < 1 or end < start or end > len(sequence):
                raise ValueError(
                    f"invalid sequence range {start}-{end} for length {len(sequence)}"
                )
            # A full-sequence MSA is not valid for a fragment unless a fragment
            # MSA is supplied explicitly in the reviewed handoff.
            entity_msa = entity.get("msa")
            if not entity_msa:
                raise ValueError(f"sequence range {start}-{end} lacks a fragment MSA")
            chains.append({
                "sequence": sequence[start - 1:end],
                "msa": entity_msa,
                "source": entity,
            })
        else:
            raise ValueError(f"unsupported enzyme-chain source {source!r}")
    return chains


def yaml_for(chains, cofactors, substrates, include_affinity=False):
    """Render enzyme chains, cofactors, and target components in Boltz YAML syntax."""
    def scalar(value):
        # JSON strings are valid YAML scalars and safely preserve paths and SMILES.
        return json.dumps(str(value))

    lines = ["version: 1", "sequences:"]
    ids = iter(string.ascii_uppercase)
    for chain in chains:
        chain_id = next(ids)
        lines += [
            "- protein:",
            f"    id: {chain_id}",
            f"    sequence: {scalar(chain['sequence'])}",
            f"    msa: {scalar(chain['msa'])}",
        ]
    for cof in cofactors:
        ccd = cof["ccd"]
        for _ in range(int(cof.get("count", 1) or 1)):
            lines += [
                "- ligand:",
                f"    id: {next(ids)}",
                f"    ccd: {scalar(ccd)}",
            ]
    affinity_ids = []
    for substrate in substrates:
        substrate_id = next(ids)
        lines += [
            "- ligand:",
            f"    id: {substrate_id}",
            f"    smiles: {scalar(substrate['modeling_smiles'])}",
        ]
        if substrate.get("affinity_target"):
            affinity_ids.append(substrate_id)
    if include_affinity and len(affinity_ids) == 1:
        lines += ["properties:", "- affinity:", f"    binder: {affinity_ids[0]}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True, help="schema-2.0 or validated schema-3.0 structural handoff")
    ap.add_argument(
        "--fasta",
        required=True,
        help="FASTA containing one full precursor or intact sequence per candidate accession",
    )
    ap.add_argument(
        "--msa-index",
        required=True,
        help="tab-separated accession-to-full-sequence-MSA index",
    )
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sequence-identity-index", help="JSON accession-to-fingerprinted identity record index (required for schema 3.0)")
    ap.add_argument("--reviewed-decisions", help="modeling-agent report.json with reviewed candidate decisions for this handoff")
    ap.add_argument("--dry-run", action="store_true", help="report per-enzyme cofactor spec; do not write YAMLs")
    ap.add_argument("--require-validated-setup", action="store_true",
                    help="require evidence-backed scientific setup validation in addition to renderer readiness")
    ap.add_argument(
        "--allow-unreviewed-exploratory",
        action="store_true",
        help="write unresolved inputs for exploration; records all bypassed checks",
    )
    ap.add_argument(
        "--include-affinity",
        action="store_true",
        help="request Boltz-2 affinity output for the single designated target ligand",
    )
    args = ap.parse_args()

    if not os.path.exists(args.fasta):
        ap.error(f"--fasta does not exist: {args.fasta}")
    if not os.path.exists(args.msa_index):
        ap.error(f"--msa-index does not exist: {args.msa_index}")

    seqs = read_fasta(args.fasta)
    msa = {}
    for line in open(args.msa_index):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            msa[parts[0]] = parts[2]
    handoff_document, handoff = load_handoff_modeling(args.handoff)
    v3 = handoff_document.get("schema_version") == "3.0"
    reviewed = {}
    review_root = None
    if args.reviewed_decisions:
        from enzyme_discovery_agent.models.handoff_contract import load_document
        from structural_modeling_agent.validation import fingerprint
        report = load_document(args.reviewed_decisions)
        if (not v3 or not isinstance(report, dict) or report.get("schema_version") != "modeling-agent-1.0"
                or report.get("task_sha256") != fingerprint(handoff_document)
                or not isinstance(report.get("candidates"), list)):
            ap.error("reviewed decisions must be a modeling-agent report bound to this schema-3.0 handoff")
        review_root = Path(args.reviewed_decisions).resolve().parent
        for row in report["candidates"]:
            if not isinstance(row, dict) or not row.get("accession") or row["accession"] in reviewed:
                ap.error("reviewed decisions require unique candidate identifiers")
            reviewed[row["accession"]] = row
    identity_index = {}
    if args.sequence_identity_index:
        from enzyme_discovery_agent.models.handoff_contract import load_document
        identity_index = load_document(args.sequence_identity_index)
        if not isinstance(identity_index, dict):
            ap.error("sequence identity index must be a JSON object")
    print(f"loaded modeling specs for {len(handoff)} accessions from {args.handoff}")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    n, miss, blocked = 0, [], {}
    executions = []
    for acc, candidate in handoff.items():
        seq = seqs.get(acc)
        if not seq:
            blocked[acc] = ["candidate sequence is missing from FASTA"]
            continue
        if v3:
            base = Path(args.sequence_identity_index).resolve().parent if args.sequence_identity_index else Path.cwd()
            identity_errors = sequence_identity_errors(candidate, seq, identity_index.get(acc), base)
            if identity_errors:
                blocked[acc] = identity_errors
                continue
        mp = msa.get(acc)
        if not mp or not os.path.exists(mp):
            miss.append(acc)
            blocked[acc] = ["full-sequence MSA is missing from the MSA index or filesystem"]
            continue
        mi = candidate.get("modeling_input") or {}
        if args.reviewed_decisions:
            decision = reviewed.get(acc) or {}
            mi = decision.get("modeling_input")
            if not isinstance(mi, dict):
                blocked[acc] = ["reviewed candidate decision is missing"]
                continue
            declarations = ((mi.get("validation_plan") or {}).get("setup") or {}).values()
            paths = [source.get("path") for decl in declarations if isinstance(decl, dict)
                     for source in (decl.get("evidence") or []) if isinstance(source, dict) and source.get("path")]
            approved = [(review_root / name).resolve() for name in ("sources", "artifacts")]
            if any(not any(Path(path).resolve().is_relative_to(folder) for folder in approved) for path in paths):
                blocked[acc] = ["reviewed source path is outside the modeling run's sources or artifacts"]
                continue
        from structural_modeling_agent.validation import validate_setup, entity_map_for, validate_msa, file_digest
        scientific_validation = validate_setup(mi, seq)
        errors = readiness_errors(mi)
        if (v3 or args.require_validated_setup) and scientific_validation["status"] != "pass":
            errors.append("scientific setup validation is " + scientific_validation["status"])
        shape_errors = execution_shape_errors(mi)
        if shape_errors:
            blocked[acc] = shape_errors
            continue
        if errors and (v3 or not args.allow_unreviewed_exploratory):
            blocked[acc] = errors
            continue
        substrates = modeled_substrates(mi)
        if any(not item.get("modeling_smiles") for item in substrates):
            blocked[acc] = list(dict.fromkeys(errors + ["a modeling substrate is missing"]))
            continue
        cofactors = mi.get("cofactors") or []
        try:
            chains = enzyme_chains(
                seq,
                os.path.abspath(mp),
                (mi.get("enzyme_state") or {}).get("chains") or [],
            )
            missing_chain_msas = [
                chain["msa"] for chain in chains if not os.path.exists(chain["msa"])
            ]
            if missing_chain_msas:
                raise ValueError(
                    "enzyme-chain MSA does not exist: "
                    + ", ".join(missing_chain_msas)
                )
            msa_evidence = [validate_msa(chain["msa"], chain["sequence"]) for chain in chains]
        except ValueError as exc:
            blocked[acc] = list(dict.fromkeys(errors + [str(exc)]))
            continue
        cofactor_summary = ", ".join(
            f"{item.get('ccd')}x{item.get('count')}" for item in cofactors
        ) or "(none)"
        substrate_summary = ", ".join(item["modeling_smiles"] for item in substrates)
        if v3:
            identity_errors = sequence_identity_errors(candidate, seq, identity_index.get(acc), base)
            if identity_errors:
                blocked[acc] = identity_errors
                continue
        if args.dry_run:
            print(
                f"  {acc:12} status={mi.get('status')} cofactors={cofactor_summary} "
                f"substrates={substrate_summary}"
            )
        else:
            output_path = f"{args.out}/step1_{acc}.yaml"
            if any(file_digest(entry["path"]) != entry["sha256"] for entry in msa_evidence):
                blocked[acc] = ["MSA changed during input preparation"]
                continue
            with open(output_path, "w") as handle:
                handle.write(yaml_for(
                    chains, cofactors, substrates, include_affinity=args.include_affinity,
                ))
        executions.append({
            "accession": acc,
            "modeling_status": mi.get("status"),
            "exploratory_override": bool(errors),
            "bypassed_checks": errors,
            "cofactors": cofactors,
            "enzyme_state": mi.get("enzyme_state"),
            "target_substrates": substrates,
            "geometry": mi.get("geometry"),
            "modeling_input": mi,
            "entity_map": entity_map_for(chains, cofactors, substrates),
            "setup_validation": scientific_validation,
            "msa_evidence": msa_evidence,
            "input_yaml": os.path.abspath(output_path) if not args.dry_run else None,
            "input_sha256": file_digest(output_path) if not args.dry_run else None,
            **({"sequence_identity_source": {
                "path": str((Path(identity_index[acc]["path"]) if Path(identity_index[acc]["path"]).is_absolute()
                             else base / identity_index[acc]["path"]).resolve()),
                "sha256": identity_index[acc]["sha256"],
                "source_attribution": "asserted by caller"}} if v3 else {}),
        })
        n += 1
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {n} step-1 YAMLs to {args.out}")
    print(f"  blocked by handoff validation: {len(blocked)}")
    for acc, errors in sorted(blocked.items()):
        print(f"  BLOCKED {acc}: {'; '.join(errors)}")
    if miss:
        print(f"  missing MSA (skipped): {miss}")
    if not args.dry_run:
        manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_handoff": os.path.abspath(args.handoff),
            "source_handoff_schema": handoff_document.get("schema_version"),
            "exploratory_override_enabled": args.allow_unreviewed_exploratory,
            "affinity_enabled": args.include_affinity,
            "scientific_validation_required": v3 or args.require_validated_setup,
            "reviewed_decisions": ({"path": os.path.abspath(args.reviewed_decisions),
                                      "sha256": hashlib.sha256(Path(args.reviewed_decisions).read_bytes()).hexdigest()}
                                     if args.reviewed_decisions else None),
            "executions": executions,
            "blocked": blocked,
            "missing_msa": miss,
        }
        from enzyme_discovery_agent.state_store import atomic_write_json
        atomic_write_json(os.path.join(args.out, "campaign_manifest.json"), manifest)


if __name__ == "__main__":
    main()
