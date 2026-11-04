"""Index supplied discovery and prediction records; never seed discovery runs."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import sys


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value if isinstance(value, bytes) else encoded(value) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class Sources:
    def __init__(self, project):
        self.project, self.files, self.content = Path(project), {}, {}

    def path(self, name):
        p = Path(name)
        return (p if p.is_absolute() else self.project / p).resolve()

    def read(self, name):
        p = self.path(name)
        if str(p) not in self.content:
            self.content[str(p)] = p.read_bytes()
        self.ref(p)
        return self.content[str(p)]

    def ref(self, name, pointer=None):
        p = self.path(name); key = str(p)
        if key not in self.files:
            stat = p.stat()
            checksum = hashlib.sha256()
            if key in self.content:
                checksum.update(self.content[key])
            else:
                with p.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(chunk)
            self.files[key] = {"path": key, "sha256": checksum.hexdigest(), "bytes": stat.st_size}
        return {**self.files[key], **({"pointer": pointer} if pointer is not None else {})}

    def json(self, name):
        return json.loads(self.read(name))

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode())))


def state_file(root, name):
    modern = root / "agent_state" / name
    return modern if modern.exists() else root / name


def identity(accession, run_id, key):
    accession = str(accession or "").strip()
    if re.fullmatch(r"literature:[a-f0-9]{20}", accession):
        return accession
    if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}", accession):
        return "uniprot:" + accession
    if re.fullmatch(r"(?:[A-Z]{1,4}_?)[0-9]{5,12}(?:\.[0-9]+)?", accession):
        return "sequence_identifier:" + accession
    return "unresolved:" + digest([run_id, key])[:24]


def iso_time(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return 0


def import_discovery(source, files, tables, proteins, memberships):
    root = files.path(source["path"])
    manifest = files.json(root / "manifest.json")
    run_id = source.get("run_id") or manifest["run_id"]
    if run_id != manifest["run_id"]:
        raise ValueError("Discovery run identifier must match its source manifest")
    registry_path = state_file(root, "candidate_registry.json")
    registry = files.json(registry_path).get("candidates", {})
    rank_path = files.path(source["ranking"])
    ranks = {r["accession"]: r for r in files.rows(rank_path)}
    selected = set(ranks) if source.get("select_ranked_only") else None
    count = 0
    for key, entry in registry.items():
        accession = str(entry.get("accession") or key)
        if selected is not None and accession not in selected: continue
        pid = identity(accession, run_id, key)
        meta = entry.get("metadata") or {}
        data = {**(meta.get("evaluation_data") or {}), **meta}
        family = data.get("interpro_family_id") or data.get("screening_family_id")
        protein = proteins.setdefault(pid, {"protein_id": pid, "accession": accession,
                    "aliases": [], "sequence_versions": [], "discovery_run_ids": [], "names": [],
                    "family_ids": [], "gene_names": [], "organisms": [], "literature_names": [], "database_names": []})
        for field, values in {"aliases": [key], "names": [data.get("protein_name"), data.get("literature_name"), entry.get("enzyme_name")],
                              "literature_names": [data.get("literature_name")],
                              "database_names": [] if data.get("is_literature_lead") else [data.get("protein_name")],
                              "family_ids": [family], "gene_names": data.get("gene_names") or [],
                              "organisms": [data.get("organism")], "discovery_run_ids": [run_id]}.items():
            protein[field] = sorted(set(protein[field]) | {str(v) for v in values if v})
        sequence = data.get("sequence")
        if isinstance(sequence, str) and sequence:
            version = {"sha256": hashlib.sha256(sequence.encode()).hexdigest(), "length": len(sequence), "source_run_id": run_id}
            if version not in protein["sequence_versions"]: protein["sequence_versions"].append(version)
        latest = (entry.get("statuses") or [{}])[-1]
        rank = ranks.get(accession, {})
        try: number = int(rank.get("rank") or 0) or None
        except (TypeError, ValueError): number = None
        aid = "assessment:" + digest([source["target_id"], run_id, key])[:24]
        assessment = {"assessment_id": aid, "revision_sha256": digest(entry), "protein_id": pid,
                      "target_id": source["target_id"], "source_run_id": run_id, "source_label": source["label"],
                      "source_record_key": key, "status": latest.get("status", "unknown"),
                      "registration_run_id": data.get("registration_run_id"),
                      "record_type": "literature_lead" if data.get("is_literature_lead") else "sequence_candidate",
                      "observed_at": entry.get("last_updated") or latest.get("timestamp"),
                      "reported_rank": number, "family_id": family,
                      "native_reactions": data.get("catalytic_activities") or [],
                      "expression_data": data.get("expression_data") or {},
                      "rejection_reason": data.get("rejection_reason"),
                      "critical_warnings": data.get("critical_warnings") or [],
                      "mechanism_source": (data.get("mechanism_evidence") or {}).get("paper") or {},
                      "registry_source": files.ref(registry_path, "/candidates/" + key.replace("~", "~0").replace("/", "~1")),
                      "rank_source": files.ref(rank_path), "raw_record": entry}
        comparisons = (data.get("substrate_similarity") or {}).get("property_comparisons") or []
        flags = []
        if any(c.get("functional_group_similarity") == 0 and c.get("functional_groups_target") for c in comparisons):
            flags.append("zero_functional_group_overlap_in_scored_pair")
        if any("*" in str(c.get("native_smiles") or "") for c in comparisons):
            flags.append("generic_native_structure_used_in_property_comparison")
        assessment["quality_flags"] = flags
        tables["assessments"].append(assessment)
        member = {"membership_id": "member:" + digest([run_id, pid, key])[:24], "run_id": run_id,
                  "protein_id": pid, "role": "discovered", "assessment_id": aid,
                  "baseline_selected": bool(source.get("baseline"))}
        tables["memberships"].append(member); memberships[run_id].append(member); count += 1
    if source.get("expected_candidates") is not None and count != source["expected_candidates"]:
        raise ValueError(f"Selected candidate count mismatch for {run_id}: {count}")
    tables["runs"].append({"run_id": run_id, "kind": "discovery", "label": source["label"],
                          "source_path": str(root), "status": manifest.get("status"), "target_id": source["target_id"],
                          "timing_as_recorded": manifest.get("timing"), "selected_candidates": count,
                          "manifest_source": files.ref(root / "manifest.json"),
                          "source_alias": source.get("source_alias"), "baseline": bool(source.get("baseline"))})
    leads_path = root / "artifacts/fulltext/papers_enzymes.json"
    if leads_path.exists():
        seen_leads = {}
        for paper_index, paper in enumerate(files.json(leads_path)):
            for bucket in ("enzymes", "homologs"):
                for lead_index, lead in enumerate(paper.get(bucket) or []):
                    content = {"source_run_id": run_id, "paper_doi": paper.get("doi"), "paper_title": paper.get("title"),
                               "name": lead.get("name"), "organism": lead.get("organism"),
                               "associated_identifiers": lead.get("accessions") or [], "relationship": lead.get("relationship"),
                               "lookup_failed": lead.get("lookup_failed"),
                               "identity_scope": "Saved name/identifier association; sequence identity and target activity require review"}
                    key = digest(content)
                    pointer = files.ref(leads_path, f"/{paper_index}/{bucket}/{lead_index}")
                    if key in seen_leads:
                        seen_leads[key]["sources"].append(pointer)
                    else:
                        row = {"lead_id": "lead:" + key[:24], **content, "sources": [pointer]}
                        seen_leads[key] = row; tables["literature_leads"].append(row)
    events_path = root / "logs/events.jsonl"
    if not events_path.exists(): events_path = root / "events.jsonl"
    if events_path.exists():
        events = []
        for line in files.read(events_path).decode().splitlines():
            try: events.append(json.loads(line))
            except ValueError: continue
        starts = [i for i, event in enumerate(events) if event.get("type") in {"run_start", "run_resume"}]
        for position, start in enumerate(starts):
            window = events[start: starts[position + 1] if position + 1 < len(starts) else len(events)]
            terminal = next((i for i, event in enumerate(window) if event.get("type") == "run_budget_checkpoint"), None)
            if terminal is not None: window = window[:terminal + 1]
            first, last = window[0], window[-1]
            calls = [e for e in window if e.get("type") == "llm_call"]
            stops = [e for e in window if e.get("type") == "run_budget_checkpoint"]
            tables["sessions"].append({"session_id": run_id + ":" + str(first.get("seq", start)), "run_id": run_id,
                    "started_at": first.get("ts"), "last_recorded_event_at": last.get("ts"),
                    "start_iteration": first.get("from_iteration", 0),
                    "last_iteration": next((e.get("iteration") for e in reversed(window) if isinstance(e.get("iteration"), int)), None),
                    "provider_reported_tokens": sum(e.get("total_tokens", 0) for e in calls),
                    "models_used": sorted({str(e.get("model")) for e in calls}),
                    "stop_reason": stops[-1].get("reason") if stops else "No terminal resource checkpoint recorded in this event window",
                    "source": files.ref(events_path, f"event seq {first.get('seq')} through {last.get('seq')}"),
                    "timing_scope": "recorded event window; not inferred continuous runtime"})
    registration_index = root / "agent_state/literature_registration_runs.json"
    if registration_index.exists():
        for row in files.json(registration_index).get("runs", []):
            tables["runs"].append({**row, "target_id": source["target_id"], "source": files.ref(row["manifest_path"])})
    return run_id


def import_modeling(source, files, tables, proteins):
    run_id = source["run_id"]
    selected = files.rows(source["selection"])
    modeled = files.rows(source["modeled_manifest"])
    excluded = files.rows(source["excluded_manifest"])
    selected_ids, modeled_ids, excluded_ids = (set(r["accession"] for r in rows) for rows in (selected, modeled, excluded))
    if modeled_ids & excluded_ids or modeled_ids | excluded_ids != selected_ids:
        raise ValueError("Modeling baseline is not a disjoint, complete selected/modeled/excluded partition")
    for key, actual in [("selected", len(selected_ids)), ("modeled", len(modeled_ids)), ("excluded", len(excluded_ids))]:
        if key in source.get("expected", {}) and actual != source["expected"][key]:
            raise ValueError(f"Modeling {key} count mismatch: {actual}")
    entries = {r["accession"]: r for r in files.json(source["candidate_index"])["candidates"]}
    origins = {str(files.path(r["active_path"])): r for r in files.json(source["provenance_index"]).get("files", [])} if source.get("provenance_index") else {}
    for row in excluded:
        pid = "uniprot:" + row["accession"]
        tables["memberships"].append({"membership_id": "member:" + digest([run_id, pid])[:24],
                    "run_id": run_id, "protein_id": pid, "role": "excluded_from_modeling",
                    "reason": row.get("exclusion_reason"), "source": files.ref(source["excluded_manifest"])})
    for row in modeled:
        accession = row["accession"]; pid = "uniprot:" + accession
        if pid not in proteins: raise ValueError(f"Modeling candidate lacks a discovery record: {accession}")
        entry = entries[accession]; folder = files.path(entry["prediction_dir"])
        coordinates = sorted(folder.glob("*.pdb")) + sorted(folder.glob("*.cif"))
        confidence = sorted(folder.glob("confidence*.json"))
        expected = int(row.get("n_models") or 0)
        input_file = files.path(source["inputs_dir"]) / ("step1_" + accession + ".yaml")
        input_ref = files.ref(input_file) if input_file.exists() else None
        setup_id = "setup:" + digest([pid, input_ref, {k: row.get(k) for k in ("phosphate", "mg", "mechanism_basis")}])[:24]
        tables["modeling_setups"].append({"setup_id": setup_id, "protein_id": pid, "target_id": source["target_id"],
                    "collection_run_id": run_id, "input_source": input_ref, "declared_legacy_recipe": row,
                    "prediction_input_binding": "not_verified_from_legacy_manifest",
                    "scientific_setup_validation": "not established by output inventory"})
        producer_label = row.get("source", "unknown")
        producer = source["producer_runs"][producer_label]
        execution_id = "execution:" + digest([producer["run_id"], pid, str(folder)])[:24]
        file_refs = [files.ref(p) for p in coordinates + confidence]
        for ref in file_refs:
            if ref["path"] in origins:
                origin = origins[ref["path"]]
                ref.update(historical_source=str(files.path(origin["historical_source"])),
                           declared_copy_sha256=origin.get("sha256"),
                           copied_artifact_integrity=ref["sha256"] == origin.get("sha256"))
                if not ref["copied_artifact_integrity"]:
                    tables["issues"].append({"kind": "legacy_copy_checksum_changed", "path": ref["path"]})
        present = len(coordinates) == expected and len(confidence) == expected
        tables["modeling_executions"].append({"execution_id": execution_id, "protein_id": pid,
                    "setup_id": setup_id, "collection_run_id": run_id, "producer_run_id": producer["run_id"],
                    "producer_label_as_recorded": producer_label, "producer_identifier_scope": producer["scope"],
                    "prediction_dir": str(folder), "expected_poses": expected, "coordinate_files": len(coordinates),
                    "confidence_files": len(confidence), "output_presence": "complete" if present else "incomplete",
                    "reuse_policy": "Review existing setup and predictions before any new submission",
                    "source": files.ref(source["modeled_manifest"]), "artifacts": file_refs})
        tables["memberships"].append({"membership_id": "member:" + digest([run_id, pid])[:24],
                    "run_id": run_id, "protein_id": pid, "role": "modeled", "execution_id": execution_id,
                    "source_discovery_run_id": source["discovery_run_id"]})
        if not present:
            tables["issues"].append({"kind": "prediction_count_mismatch", "protein_id": pid, "execution_id": execution_id})
    tables["runs"].append({"run_id": run_id, "kind": "modeling_collection", "label": source["label"],
                    "source_path": str(files.path(source["path"])), "target_id": source["target_id"],
                    "discovery_run_id": source["discovery_run_id"], "selected_candidates": len(selected_ids),
                    "modeled_candidates": len(modeled_ids), "excluded_candidates": len(excluded_ids),
                    "producer_counts": dict(Counter(r.get("source") for r in modeled)),
                    "source": files.ref(source["modeled_manifest"])})
    for producer in source["producer_runs"].values():
        if producer["run_id"] != run_id:
            tables["runs"].append({"run_id": producer["run_id"], "kind": "legacy_model_producer",
                        "label": producer["label"], "identifier_scope": producer["scope"],
                        "source": files.ref(source["modeled_manifest"]), "target_id": source["target_id"]})
    import_validation_bundle(source, files, tables, proteins, run_id)


def import_modeling_agent(source,files,tables,proteins):
    """Index new agent runs in place; preserve their input binding and scientific scope."""
    from structural_modeling_agent.evidence import modeling_evidence
    report_path=files.path(source['report']);report=files.json(report_path)
    evidence=modeling_evidence(report_path)
    run_id=source.get('run_id') or (report.get('result_layout') or {}).get('run_id')
    if not run_id:raise ValueError('A modeling-agent source requires a run identity')
    by_accession={row['accession']:row for row in evidence['candidates']}
    completed=0;reviewed=0
    for candidate in report['candidates']:
        accession=candidate['accession']
        identities=[pid for pid,protein in proteins.items() if protein['accession']==accession
                    and source['discovery_run_id'] in protein.get('discovery_run_ids',[])]
        if len(identities)!=1:raise ValueError('Modeling identity requires one source-scoped discovery record: '+accession)
        pid=identities[0]
        existing=candidate.get('existing_prediction_review')
        if existing:
            execution_id=existing['source_execution_id']
            if not any(row['execution_id']==execution_id for row in tables['modeling_executions']):
                raise ValueError('Index the original producer before its prediction review')
            tables['memberships'].append({'membership_id':'member:'+digest([run_id,pid,'review'])[:24],
                'run_id':run_id,'protein_id':pid,'role':'existing_prediction_review','execution_id':execution_id,
                'source_discovery_run_id':source['discovery_run_id']})
            tables['validations'].append({'validation_id':'reuse-review:'+digest([run_id,pid])[:24],
                'protein_id':pid,'collection_run_id':run_id,'kind':'existing_prediction_review',
                'review':existing,'source':files.ref(report_path)})
            reviewed+=1
            continue
        poses=(candidate.get('pose_validation') or {}).get('poses') or []
        if not poses:continue
        input_file=files.path(candidate['input_yaml']);input_ref=files.ref(input_file)
        setup_id='setup:'+digest([pid,input_ref,candidate.get('modeling_input')])[:24]
        execution_id='execution:'+digest([run_id,pid,candidate.get('execution_sha256')])[:24]
        row=by_accession[accession];coordinates=[files.path(p['path']) for p in poses]
        folders={p.parent for p in coordinates}
        if len(folders)!=1:raise ValueError('One indexed execution must have one prediction directory')
        folder=next(iter(folders));confidence=sorted(folder.glob('confidence*.json'))
        refs=[files.ref(path) for path in coordinates+confidence]
        provenance=(candidate.get('pose_validation') or {}).get('prediction_provenance') or {}
        if provenance.get('path'):refs.append(files.ref(provenance['path']))
        for ref in provenance.get('supporting_files') or []:refs.append(files.ref(ref['path']))
        expected=(((candidate.get('modeling_input') or {}).get('validation_plan') or {}).get('poses') or {}).get('expected_predictions')
        present=row.get('execution_status')=='complete' and len(coordinates)==expected and len(confidence)==expected
        completed+=present
        tables['modeling_setups'].append({'setup_id':setup_id,'protein_id':pid,'target_id':source['target_id'],
            'collection_run_id':run_id,'input_source':input_ref,'declared_modeling_input':candidate.get('modeling_input'),
            'prediction_input_binding':'verified_from_current_producer_manifest' if present else 'unverified_or_incomplete',
            'scientific_setup_validation':(candidate.get('setup_validation') or {}).get('status'),
            'scope':'Source/consistency checks under the declared model; conditional assumptions are retained, not verified as native facts.'})
        tables['modeling_executions'].append({'execution_id':execution_id,'protein_id':pid,'setup_id':setup_id,
            'collection_run_id':run_id,'producer_run_id':run_id,'producer_label_as_recorded':'modeling_agent',
            'producer_identifier_scope':'agent run with fingerprinted producer binding','prediction_dir':str(folder),
            'expected_poses':expected,'coordinate_files':len(coordinates),'confidence_files':len(confidence),
            'output_presence':'complete' if present else 'incomplete','reuse_policy':'Review existing setup and predictions before any new submission',
            'source':files.ref(report_path),'artifacts':refs,'geometry_metrics':row.get('geometry_metrics'),
            'pose_qc_status':row.get('pose_qc_status'),'scientific_status':row.get('scientific_status'),
            'decision_origin':report.get('decision_origin')})
        tables['memberships'].append({'membership_id':'member:'+digest([run_id,pid])[:24],'run_id':run_id,
            'protein_id':pid,'role':'modeled','execution_id':execution_id,'source_discovery_run_id':source['discovery_run_id']})
    tables['runs'].append({'run_id':run_id,'kind':'modeling_agent','label':source.get('label',run_id),
        'source_path':str(report_path.parent),'target_id':source['target_id'],'discovery_run_id':source['discovery_run_id'],
        'modeled_candidates':completed,'reviewed_existing_candidates':reviewed,'status':report['status'],'source':files.ref(report_path)})
    import_validation_bundle(source, files, tables, proteins, run_id)


def import_validation_bundle(source, files, tables, proteins, run_id):
    if source.get("validation_bundle"):
        path = source["validation_bundle"]; bundle = files.json(path)
        for entry in bundle.get("candidates", []):
            identities = [pid for pid, protein in proteins.items() if protein['accession'] == entry['accession']
                          and source['discovery_run_id'] in protein.get('discovery_run_ids', [])]
            if len(identities) != 1:
                raise ValueError('Validation identity requires one source-scoped discovery record: ' + entry['accession'])
            tables["validations"].append({"validation_id": "validation:" + digest(entry)[:24],
                        "protein_id": identities[0], "collection_run_id": run_id,
                        "kind": "legacy_proxy_application", "bundle_id": entry.get("bundle_id"),
                        "transfer_status": entry.get("transfer_status"), "calibration_status": entry.get("calibration_status"),
                        "score": entry.get("score"), "reasons": entry.get("reasons"), "source": files.ref(path),
                        "limitation": "Proxy application is not proof of valid setup or target activity"})


def audit_events(path):
    previous, events = None, []
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line); checksum = event.pop("event_sha256")
            if event.get("previous_sha256") != previous or digest(event) != checksum:
                raise ValueError("Catalog audit chain failed verification")
            event["event_sha256"] = checksum; events.append(event); previous = checksum
    return events


def append_audit(path, event):
    events = audit_events(path)
    event.update(timestamp=datetime.now(timezone.utc).isoformat(), sequence=len(events) + 1,
                 previous_sha256=events[-1]["event_sha256"] if events else None)
    event["event_sha256"] = digest(event)
    with path.open("ab") as stream:
        stream.write(encoded(event) + b"\n"); stream.flush(); os.fsync(stream.fileno())


def build(campaign_dir):
    root = Path(campaign_dir).resolve(); config = json.loads((root / "sources.json").read_text())
    catalog = root / "catalog"; catalog.mkdir(exist_ok=True)
    with (catalog / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        audit_events(catalog / "events.jsonl")
        files = Sources(config["project_root"])
        tables = {k: [] for k in ("proteins", "assessments", "memberships", "runs", "sessions",
                                  "modeling_setups", "modeling_executions", "validations", "host_reviews", "policies", "feedback", "literature_leads", "issues")}
        proteins, memberships = {}, defaultdict(list)
        contracts = {}
        for source in config["discovery_sources"]:
            context = files.json(state_file(files.path(source["path"]), "target_context.json"))
            chemistry = context.get("target_reasoning") or {}
            contract = {"substrates": sorted(str(context.get("substrate_smiles") or "").split(".")),
                        **{key: chemistry.get(key) for key in ("reaction_direction", "bond_types", "substrate_topology")}}
            if source["target_id"] in contracts and contracts[source["target_id"]] != contract:
                raise ValueError("Sources assigned to one target have different chemistry contracts")
            contracts[source["target_id"]] = contract
            import_discovery(source, files, tables, proteins, memberships)
        for source in config.get("modeling_sources", []):
            import_modeling(source, files, tables, proteins)
        for source in config.get('structural_agent_sources',[]):
            import_modeling_agent(source,files,tables,proteins)
        mechanism_sources = defaultdict(set)
        for assessment in tables["assessments"]:
            paper = assessment["mechanism_source"]
            key = paper.get("doi") or paper.get("pmid")
            if key and assessment.get("family_id"):
                mechanism_sources[key].add(assessment["family_id"])
        for assessment in tables["assessments"]:
            paper = assessment["mechanism_source"]; key = paper.get("doi") or paper.get("pmid")
            if key and len(mechanism_sources[key]) > 1:
                assessment["quality_flags"].append("shared_mechanism_source_across_families_verify_candidate_linkage")
        tables["proteins"] = sorted(proteins.values(), key=lambda r: r["protein_id"])
        files.ref(root / "sources.json")
        if config.get("host_review_file") and files.path(config["host_review_file"]).exists():
            for review in files.json(config["host_review_file"])["reviews"]:
                tables["host_reviews"].append({**review, "source": files.ref(config["host_review_file"]),
                                              "scope": "New source review; original run assessments remain unchanged"})
        if config.get("prioritization_policy"):
            policy = files.json(config["prioritization_policy"])
            if not policy.get("version") or not policy.get("change_reason"):
                raise ValueError("Priority policy must have a version and a recorded reason")
            policy_hash = digest(policy)
            archived = root / "policies" / (policy["version"] + "-" + policy_hash[:16] + ".json")
            if archived.exists() and json.loads(archived.read_text()) != policy:
                raise ValueError("Archived policy revision has changed")
            if not archived.exists(): write(archived, policy)
            tables["policies"].append({"policy_sha256": policy_hash, "policy": policy,
                                      "source": files.ref(config["prioritization_policy"])})
        if (catalog / "feedback_events.jsonl").exists():
            tables["feedback"] = audit_events(catalog / "feedback_events.jsonl")
            files.ref(catalog / "feedback_events.jsonl")
        write(root / "campaign.json", {"schema_version": 1, "campaign_id": config["campaign_id"],
                                      "target_contracts": contracts,
                                      "purpose": "Explicit pool and modeling reuse index; not an implicit discovery input"})
        tables["files"] = sorted(files.files.values(), key=lambda r: r["path"])
        payloads = {name + ".jsonl": b"".join(encoded(row) + b"\n" for row in rows) for name, rows in tables.items()}
        producer_code = {}
        for source_path in sorted(Path(__file__).parent.glob("*.py")):
            raw = source_path.read_bytes()
            producer_code[source_path.name] = hashlib.sha256(raw).hexdigest()
            payloads["implementation/" + source_path.name] = raw
        payloads["implementation/runtime.json"] = encoded({"python": sys.version, "executable": sys.executable})
        signature = digest({"config": digest(config), "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}})
        snapshot_id = signature[:24]; destination = catalog / "snapshots" / snapshot_id
        fresh = not destination.exists()
        if fresh:
            destination.parent.mkdir(exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=destination.parent))
            try:
                manifest = {"snapshot_id": snapshot_id, "content_sha256": signature,
                            "generated_at": datetime.now(timezone.utc).isoformat(), "config_sha256": digest(config),
                            "producer_code_sha256": producer_code,
                            "counts": {name: len(rows) for name, rows in tables.items()},
                            "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}}
                for name, raw in payloads.items(): write(temporary / name, raw)
                write(temporary / "manifest.json", manifest)
                os.rename(temporary, destination)
            finally:
                if temporary.exists(): shutil.rmtree(temporary)
        else:
            manifest = json.loads((destination / "manifest.json").read_text())
            for name, checksum in manifest["files"].items():
                if hashlib.sha256((destination / name).read_bytes()).hexdigest() != checksum:
                    raise ValueError("Existing immutable snapshot has changed")
        from .views import publish_views
        publish_views(root, tables, manifest)
        for run in tables["runs"]:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run["run_id"])
            kind = "discovery" if run["kind"] in {"discovery", "literature_registration_followup"} else "modeling"
            write(root / "runs" / kind / safe / "source.json", {**run, "catalog_snapshot_id": snapshot_id,
                         "note": "Index record only; original artifacts remain at the source path"})
        append_audit(catalog / "events.jsonl", {"kind": "snapshot_created" if fresh else "sources_checked_no_change",
                     "snapshot_id": snapshot_id, "config_sha256": digest(config), "counts": manifest["counts"],
                     "priority_policies": [{"version": p["policy"]["version"], "sha256": p["policy_sha256"]} for p in tables["policies"]]})
        write(catalog / "current.json", {"snapshot_id": snapshot_id, "manifest": str(destination / "manifest.json")})
        return {"snapshot_id": snapshot_id, "new_snapshot": fresh, "counts": manifest["counts"],
                "view": str(root / "views/index.html"), "issues": tables["issues"]}


def verify(campaign_dir):
    root = Path(campaign_dir).resolve(); current = json.loads((root / "catalog/current.json").read_text())
    manifest_path = Path(current["manifest"]); manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Catalog snapshot checksum mismatch: " + name)
    events = audit_events(root / "catalog/events.jsonl")
    checksum = digest({"config": manifest["config_sha256"], "files": manifest["files"]})
    if checksum != manifest["content_sha256"] or checksum[:24] != current["snapshot_id"]:
        raise ValueError("Snapshot identity does not match its content manifest")
    if not any(e.get("snapshot_id") == current["snapshot_id"] for e in events):
        raise ValueError("Published snapshot is missing its audit event")
    return {"snapshot_id": current["snapshot_id"], "snapshot_verified": True, "audit_events_verified": len(events)}


def record_feedback(campaign_dir, feedback_file):
    root = Path(campaign_dir).resolve(); data = json.loads(Path(feedback_file).read_text())
    required = {"protein_id", "source_run_id", "kind", "result", "evidence_files"}
    if not required <= set(data) or not data["evidence_files"]:
        raise ValueError("Feedback requires a candidate, source run, observation kind/result and evidence files")
    allowed = {"expression_observation", "setup_validation", "native_control_validation", "pose_validation", "experimental_activity", "priority_review"}
    if data["kind"] not in allowed: raise ValueError("Unknown feedback kind")
    current = json.loads((root / "catalog/current.json").read_text()); folder = Path(current["manifest"]).parent
    proteins = {json.loads(line)["protein_id"] for line in (folder / "proteins.jsonl").read_text().splitlines()}
    runs = {json.loads(line)["run_id"] for line in (folder / "runs.jsonl").read_text().splitlines()}
    if data["protein_id"] not in proteins or data["source_run_id"] not in runs:
        raise ValueError("Feedback candidate and run must already be registered")
    memberships = [json.loads(line) for line in (folder / "memberships.jsonl").read_text().splitlines()]
    executions = [json.loads(line) for line in (folder / "modeling_executions.jsonl").read_text().splitlines()]
    matched = any(m["protein_id"] == data["protein_id"] and m["run_id"] == data["source_run_id"] for m in memberships)
    matched = matched or any(e["protein_id"] == data["protein_id"] and data["source_run_id"] in {
        e["producer_run_id"], e["collection_run_id"]} for e in executions)
    if not matched: raise ValueError("Feedback candidate does not belong to its stated source run")
    config = json.loads((root / "sources.json").read_text()); files = Sources(config["project_root"])
    event = {**data, "evidence_files": [files.ref(path) for path in data["evidence_files"]],
             "input_record": files.ref(Path(feedback_file).resolve()),
             "scope": "Evidence observation, not an automatic activity claim or policy rewrite"}
    event["observation_id"] = "feedback:" + digest(event)[:24]
    with (root / "catalog/.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = root / "catalog/feedback_events.jsonl"
        existing = audit_events(path)
        if not any(e.get("observation_id") == event["observation_id"] for e in existing):
            append_audit(path, event)
    return {"observation_id": event["observation_id"], "registered": True}
