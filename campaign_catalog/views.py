"""Human-readable projections; rankings remain attached to their source runs."""

from collections import Counter, defaultdict
import csv
import html
import io
import json
from .catalog import write, iso_time, digest


def publish_views(root, tables, manifest):
    assessments = defaultdict(list)
    executions = defaultdict(list)
    baseline, excluded = set(), set()
    for row in tables["assessments"]: assessments[row["protein_id"]].append(row)
    for row in tables["modeling_executions"]: executions[row["protein_id"]].append(row)
    for row in tables["memberships"]:
        if row.get("baseline_selected"): baseline.add(row["protein_id"])
        if row.get("role") == "excluded_from_modeling": excluded.add(row["protein_id"])
    reviews = {r["protein_id"]: r for r in tables.get("host_reviews", [])}
    setup_by_id={row['setup_id']:row for row in tables['modeling_setups']}
    pool, coverage, queue = [], [], []
    for protein in tables["proteins"]:
        pid = protein["protein_id"]; history = sorted(assessments[pid], key=lambda a: iso_time(a.get("observed_at")))
        latest = history[-1]
        expression = reviews.get(pid, {}).get("assessment") or latest.get("expression_data") or {}
        level = (expression.get("preferred_host_assessment") or {}).get("level", "unknown")
        row = {"protein_id": pid, "accession": protein["accession"], "name": " | ".join(protein.get("literature_names") or protein["names"]),
               "database_names": " | ".join(protein.get("database_names") or []), "record_type": latest.get("record_type", "sequence_candidate"),
               "registration_run_id": latest.get("registration_run_id") or "",
               "genes": " | ".join(protein["gene_names"]), "families": " | ".join(protein["family_ids"]),
               "latest_status": latest["status"], "latest_source_run": latest["source_run_id"],
               "latest_run_rank": latest.get("reported_rank") or "", "host_evidence": level,
               "host_evidence_scope": "new source review" if pid in reviews else "saved run assessment",
               "discovery_runs": " | ".join(protein["discovery_run_ids"]),
               "previously_modeled": bool(executions[pid]), "baseline_panel_member": pid in baseline,
               "assessment_count": len(history)}
        row["assessment_quality"] = " | ".join(latest.get("quality_flags") or [])
        priority=((latest.get('raw_record') or {}).get('metadata') or {}).get('priority_assessment') or {}
        row['work_priority']=priority.get('decision','')
        pool.append(row)
        covered = {"accession": protein["accession"], "name": row["name"],
                   "baseline_selected": pid in baseline, "baseline_excluded": pid in excluded,
                   "indexed_executions": len(executions[pid]),
                   "coordinate_files": sum(e["coordinate_files"] for e in executions[pid]),
                   "producer_runs": " | ".join(sorted({e["producer_run_id"] for e in executions[pid]})),
                   "collection_runs": " | ".join(sorted({e["collection_run_id"] for e in executions[pid]})),
                   "reuse_status": "review_existing_setup_first" if executions[pid] else "no_indexed_predictions",
                   "setup_validation": ' | '.join(sorted({str(setup_by_id[e['setup_id']].get('scientific_setup_validation','unknown')) for e in executions[pid]})) if executions[pid] else 'not_prepared'}
        coverage.append(covered)
        hypothesis=(priority.get('decision') in {'prioritize','investigate'} and priority.get('strengths')
                    and priority.get('evidence_refs') and latest.get('record_type')!='literature_lead'
                    and latest.get('status')!='rejected')
        if (row["latest_run_rank"] or hypothesis) and not executions[pid] and pid not in excluded:
            queue.append({**row, "queue_status": "review_scoring_and_evidence" if latest.get("quality_flags") else "review_candidate_and_setup", "submission_authorized": False})
    parent = {row["protein_id"]: row["protein_id"] for row in pool}
    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]; pid = parent[pid]
        return pid
    reaction_groups = {}
    for row in pool:
        latest = max(assessments[row["protein_id"]], key=lambda a: iso_time(a.get("observed_at")))
        for reaction in latest.get("native_reactions") or []:
            key = (row["families"], " ".join(reaction.lower().split()))
            if key in reaction_groups: parent[find(row["protein_id"])] = find(reaction_groups[key])
            else: reaction_groups[key] = row["protein_id"]
    annotation_groups = defaultdict(list)
    for row in pool: annotation_groups[find(row["protein_id"])].append(row)
    preferences = {"reported_active": 4, "structural_construct_expressed": 3, "reported_expression": 2,
                   "unknown": 1, "expression_problem_reported": 0}
    representatives = []
    for members in annotation_groups.values():
        group_id = "annotation_group:" + digest(sorted(r["protein_id"] for r in members))[:16]
        eligible = [r for r in members if r["latest_status"] in {"evaluated", "ranked"}]
        eligible = eligible or [r for r in members if r["latest_status"] != "rejected"]
        representative = max(eligible or members, key=lambda r: (preferences.get(r["host_evidence"], 1),
                               bool(r["previously_modeled"]), -(r["latest_run_rank"] or 1000000)))
        chosen = representative["accession"] if eligible else ""
        for row in members:
            row.update(annotation_group=group_id, expression_preferred_representative=chosen)
        representatives.append({"annotation_group": group_id, "family": representative["families"],
                "representative": chosen, "host_evidence": representative["host_evidence"],
                "members": " | ".join(r["accession"] for r in members),
                "selection_basis": "documented preferred-host evidence, then existing modeling coverage, then source rank",
                "grouping_scope": "shared recorded native reaction and family; not a phylogenetic orthology claim"})
    modeled_families = {f for p in tables["proteins"] if executions[p["protein_id"]] for f in p["family_ids"]}
    by_pid = {row["protein_id"]: row for row in pool}
    for row in queue:
        row.update(annotation_group=by_pid[row["protein_id"]]["annotation_group"],
                   expression_preferred_representative=by_pid[row["protein_id"]]["expression_preferred_representative"],
                   family_already_modeled=bool(set(row["families"].split(" | ")) & modeled_families))
        if row["family_already_modeled"]:
            row["queue_status"] = "require_advantage_over_existing_family_models"
    policy = tables.get("policies", [{}])[0].get("policy") if tables.get("policies") else None
    if policy:
        from .prioritization import review_priority
        for row in queue: row.update(review_priority(row, policy))
    pool.sort(key=lambda row: (row["families"], row["name"], row["accession"]))
    queue.sort(key=lambda row: (0 if row["latest_source_run"] not in {
        r["run_id"] for r in tables["runs"] if r.get("baseline")} else 1,
        row["latest_source_run"], row["latest_run_rank"] or float('inf'), row['accession']))
    def save_csv(name, rows):
        stream = io.StringIO(); columns = list(rows[0]) if rows else ["accession"]
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
        write(root / "views" / name, stream.getvalue().encode())
    save_csv("candidate_pool.csv", pool)
    save_csv("modeling_coverage.csv", coverage)
    save_csv("new_modeling_review.csv", queue)
    run_rows = [{"run_id": r["run_id"], "kind": r["kind"], "label": r["label"],
                 "status": r.get("status", "historical"), "source_path": r.get("source_path", "")}
                for r in tables["runs"]]
    save_csv("run_history.csv", run_rows)
    family_summary = []
    groups = defaultdict(list)
    for row in pool: groups[row["families"] or "unresolved"].append(row)
    for family, members in sorted(groups.items()):
        family_summary.append({"family": family, "protein_records": len(members),
                               "baseline_members": sum(r["baseline_panel_member"] for r in members),
                               "previously_modeled": sum(r["previously_modeled"] for r in members),
                               "ranked_records": sum(bool(r["latest_run_rank"]) for r in members),
                               "documented_host_records": sum(r["host_evidence"] not in {"unknown", "expression_problem_reported"} for r in members)})
    save_csv("family_coverage.csv", family_summary)
    save_csv("expression_representatives.csv", representatives)
    lead_rows = [{"name": r["name"], "organism": r["organism"], "paper_doi": r["paper_doi"],
                  "associated_identifiers": " | ".join(r["associated_identifiers"]), "relationship": r["relationship"],
                  "source_run_id": r["source_run_id"], "identity_scope": r["identity_scope"]}
                 for r in tables.get("literature_leads", [])]
    save_csv("literature_leads.csv", lead_rows)
    feedback_rows = [{"protein_id": r["protein_id"], "kind": r["kind"], "source_run_id": r["source_run_id"],
                      "result": json.dumps(r["result"], ensure_ascii=False),
                      "evidence_files": " | ".join(e["path"] for e in r["evidence_files"])}
                     for r in tables.get("feedback", [])]
    save_csv("feedback.csv", feedback_rows)
    identity_queue = []
    for row in pool:
        if row["record_type"] == "literature_lead":
            latest = max(assessments[row["protein_id"]], key=lambda a: iso_time(a.get("observed_at")))
            lead = (latest.get("raw_record", {}).get("metadata", {}).get("literature_lead") or {})
            identity_queue.append({"name": row["name"], "lead_id": row["accession"],
                "source_paper": (lead.get("paper") or {}).get("doi"),
                "source_discovery_run_id": row["latest_source_run"], "registration_run_id": row["registration_run_id"],
                "identity_status": "pending", "next_action": "Resolve the exact candidate sequence from the source paper and deposited identifiers"})
    save_csv("identity_resolution_queue.csv", identity_queue)
    def table(rows, columns):
        headings = "".join(f"<th>{html.escape(c.replace('_', ' '))}</th>" for c in columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>" for row in rows)
        return f"<div class='scroll'><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>"
    modeled_count = sum(bool(executions[p["protein_id"]]) for p in tables["proteins"])
    summary = {"unique_candidate_records": len(pool), "selected_reference_panel": len(baseline),
               "paper_scoped_candidate_leads": sum(row["record_type"] == "literature_lead" for row in pool),
               "reference_panel_modeled": sum(bool(executions[p]) for p in baseline),
               "reference_panel_excluded": len(excluded), "new_modeling_review_records": len(queue),
               "scope": "Accession-level records; family coverage and expression evidence are separate from pool size"}
    write(root / "views/summary.json", summary)
    body = ("<!doctype html><html><head><meta charset='utf-8'><title>Candidate pool and run provenance</title>"
            "<style>body{font:15px system-ui;margin:30px;color:#243447}table{border-collapse:collapse;width:100%}"
            "th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}th{background:#eef2f6}"
            ".scroll{overflow:auto;max-height:650px}input{padding:9px;width:50%}details{margin:25px 0}summary{cursor:pointer}"
            ".note{color:#58677b}</style></head><body><h1>Candidate pool and run provenance</h1>"
            f"<p>{len(pool)} candidate records · {len(baseline)} reference candidates · {modeled_count} with indexed predictions · {len(queue)} modeling-review entries</p>"
            f"<p class='note'>Snapshot {manifest['snapshot_id']} · built {manifest['generated_at']}. This is a source index, "
            "not a new discovery or modeling run. Rankings and decisions remain specific to their source run. "
            "Existing outputs require setup review before reuse; output presence alone does not validate a prediction.</p>"
            "<p><a href='candidate_pool.csv'>Candidate table</a> · <a href='modeling_coverage.csv'>Modeling coverage</a> · "
            "<a href='new_modeling_review.csv'>Modeling review queue</a> · <a href='family_coverage.csv'>Family coverage</a> · "
            "<a href='expression_representatives.csv'>Expression-based representatives</a> · "
            "<a href='literature_leads.csv'>Retrieved literature leads</a> · <a href='feedback.csv'>Source-linked feedback</a> · "
            "<a href='identity_resolution_queue.csv'>Registered candidates awaiting identity</a> · "
            "<a href='run_history.csv'>Run history</a> · <a href='../catalog/current.json'>Snapshot pointer</a> · "
            "<a href='../catalog/events.jsonl'>Append-only audit log</a></p>"
            + (f"<p>Modeling review policy: {html.escape(policy['version'])}. Every review row records its policy version. "
               "Policy changes require a recorded reason and produce a new immutable snapshot. No queue row authorizes compute.</p>" if policy else "") +
            "<h2>Source runs</h2>" + table(run_rows, ["run_id", "kind", "label", "status", "source_path"]) +
            "<h2>Candidate pool</h2><p><input id='filter' placeholder='Filter by accession, gene, family, status or source run'></p>"
            + ("<h3>Registered literature candidates awaiting identity</h3>" + table(identity_queue, ["name", "lead_id", "source_paper", "registration_run_id", "identity_status"]) if identity_queue else "") +
            "<p class='note'>Multiple species variants within a family do not count as independent new catalytic directions. "
            "Preferred-host evidence distinguishes reported activity, deposited structural constructs, expression reports, "
            "problems and unknowns. Origin alone is not expression evidence.</p><div id='pool'>" +
            table(pool, ["accession", "name", "families", "latest_status", "latest_run_rank", "host_evidence", "expression_preferred_representative", "assessment_quality", "previously_modeled", "latest_source_run", "discovery_runs"]) + "</div>"
            "<details><summary>Modeling coverage and reuse</summary>" + table(coverage, list(coverage[0]) if coverage else ["accession"]) + "</details>"
            "<details><summary>Additional candidates requiring modeling review</summary>" + table(queue, ["accession", "name", "families", "host_evidence", "latest_source_run", "queue_status"]) + "</details>"
            "<details><summary>Family coverage and redundancy</summary>" + table(family_summary, list(family_summary[0]) if family_summary else ["family"]) + "</details>"
            "<details><summary>Retrieved names requiring identity and relevance review</summary>" +
            table(lead_rows, ["name", "organism", "associated_identifiers", "relationship", "paper_doi", "source_run_id"]) + "</details>"
            "<details><summary>Source-linked feedback observations</summary>" + table(feedback_rows, ["protein_id", "kind", "result", "source_run_id", "evidence_files"]) + "</details>"
            "<details><summary>Refresh and verification</summary><p>Sources are explicitly registered in <code>../sources.json</code>. "
            "This catalog is a published snapshot; source runs are indexed in place. Refresh after adding or completing a run.</p>"
            f"<p><code>python -m campaign_catalog {html.escape(str(root))}</code></p>"
            f"<p><code>python -m campaign_catalog {html.escape(str(root))} --verify</code></p>"
            "<p>New source-linked observations can be recorded with <code>--feedback PATH_TO_FEEDBACK_JSON</code>. "
            "Every snapshot preserves its tables, processor source code, input hashes and policy revision. "
            "Future criteria changes require a versioned policy and recorded reason.</p></details>"
            "<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();"
            "document.querySelectorAll('#pool tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script>"
            "</body></html>")
    write(root / "views/index.html", body.encode())
