"""Register saved paper-derived leads in an inactive run, without model calls."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.models.literature_leads import register_literature_lead
from enzyme_discovery_agent.run_layout import run_file
from enzyme_discovery_agent.state_store import atomic_write_json


def backfill(run_dir, output_dir, paper_ids):
    root, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.exists(): raise ValueError("Follow-up output directory already exists; preserve it and choose a new run identifier")
    source_manifest = json.loads((root / "manifest.json").read_text())
    if source_manifest.get("status") == "running": raise ValueError("Checkpoint the active discovery run before registration")
    with (root / "agent_state/discovery_orchestrator.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        papers_file = root / "artifacts/fulltext/papers_enzymes.json"
        papers = json.loads(papers_file.read_text())
        metadata = {p.get("doi") or p.get("key"): p for p in json.loads((root / "artifacts/fulltext/manifest.json").read_text())}
        selected = [p for p in papers if p.get("doi") in set(paper_ids)]
        if set(paper_ids) - {p.get("doi") for p in selected}: raise ValueError("Requested paper is not present in saved literature records")
        for paper in selected:
            if not (root / "artifacts/fulltext" / metadata[paper["doi"]]["text_file"]).is_file():
                raise ValueError("Saved paper text is unavailable")
        before = load_candidate_registry(root, strict=True).get("candidates", {})
        output.mkdir(parents=True)
        registry_file = run_file(root, "candidate_registry.json")
        shutil.copy2(registry_file, output / "before_candidate_registry.json")
        record = {"run_id": output.name, "kind": "literature_registration_followup", "status": "running",
                  "parent_discovery_run_id": source_manifest["run_id"], "source_path": str(root),
                  "source_papers": paper_ids, "started_at": datetime.now(timezone.utc).isoformat(),
                  "before_registry_sha256": hashlib.sha256(registry_file.read_bytes()).hexdigest(),
                  "papers_record_sha256": hashlib.sha256(papers_file.read_bytes()).hexdigest(),
                  "model_calls": 0, "new_compute_jobs": 0,
                  "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), Path(__file__).parent / "models/literature_leads.py")}}
        atomic_write_json(output / "manifest.json", record)
        ids = set()
        try:
            for paper in selected:
                meta = metadata[paper["doi"]]
                text_path = root / "artifacts/fulltext" / meta["text_file"]
                for lead in paper.get("enzymes") or []:
                    if lead.get("accessions"):
                        for accession in lead["accessions"]:
                            existing = load_candidate_registry(root, strict=True).get("candidates", {}).get(accession)
                            if existing and lead.get("name") and existing.get("metadata", {}).get("literature_name") != lead["name"]:
                                update_candidate_registry(root, accession, None, {"literature_name": lead["name"]})
                        continue
                    lead_id = register_literature_lead(root, {**lead, "relationship": lead.get("relationship") or "named"},
                                                        meta, text_path, registration_run_id=output.name)
                    if lead_id: ids.add(lead_id)
            record.update(status="completed", registered_lead_ids=sorted(ids),
                          newly_registered_lead_ids=sorted(ids - set(before)),
                          after_registry_sha256=hashlib.sha256(registry_file.read_bytes()).hexdigest())
            append_event(root, "literature_lead_registration_followup", registration_run_id=output.name,
                         source_discovery_run_id=source_manifest["run_id"], registered_lead_ids=sorted(ids),
                         manifest_path=str(output / "manifest.json"))
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(output / "manifest.json", record)
        index = root / "agent_state/literature_registration_runs.json"
        registrations = json.loads(index.read_text()) if index.exists() else {"runs": []}
        registrations["runs"].append({"run_id": output.name, "kind": record["kind"], "status": record["status"],
                 "source_path": str(output), "parent_discovery_run_id": source_manifest["run_id"],
                 "label": "Saved literature lead registration", "manifest_path": str(output / "manifest.json")})
        atomic_write_json(index, registrations)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper", action="append", required=True, help="Saved source paper DOI")
    args = parser.parse_args()
    print(json.dumps(backfill(args.run_dir, args.output_dir, args.paper), indent=2))
