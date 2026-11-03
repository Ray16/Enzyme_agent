"""Resolve paper-scoped leads without treating a shared symbol as identity."""

import hashlib
import json
import re
import time
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
from enzyme_discovery_agent.models.event_log import append_event
from enzyme_discovery_agent.tool_registry import tool


def _norm(value):
    return " ".join(str(value or "").casefold().split())


def _contains(text, value):
    return bool(value) and bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I))


def _source(path, source_id):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "source_id": source_id, "text": data.decode("utf-8")}


def source_associations(name, sources):
    """Extract only explicit local name/taxon or name/deposit associations."""
    from enzyme_discovery_agent.models.literature_leads import sequence_name_leads
    associations = []
    pattern = re.compile(r"(?<!\w)" + re.escape(name) +
                         r"\s+from\s+([A-Z][a-z]+\s+[a-z][a-z-]+)(?![a-z])")
    cluster = re.compile(r'\b(?:enzymes|proteins|genes)\s+([A-Za-z0-9_,/() -]{1,160}?)\s*,?\s*encoded\s+in\s+the\s+'
        r'([A-Z][a-z]+\s+[a-z][a-z-]+)[^.!?\n]{0,100}?(?:gene|biosynthetic)\s+cluster')
    for source in sources:
        text = source["text"]
        for match in pattern.finditer(text):
            associations.append({"organism": match.group(1), "quote": match.group(0),
                                 **{k: source[k] for k in ("path", "sha256", "source_id")}})
        for match in cluster.finditer(text):
            if _contains(match.group(1),name):
                associations.append({'organism':match.group(2),'quote':match.group(0),
                    **{k:source[k] for k in ('path','sha256','source_id')}})
        for lead in sequence_name_leads(text):
            if _norm(lead["name"]) == _norm(name):
                associations.append({"identifier": lead["genbank"], "quote": lead["source_quote"],
                                     **{k: source[k] for k in ("path", "sha256", "source_id")}})
    return associations


def _matching_organism(entry, organism):
    taxon = entry.get("organism") or {}
    names = [taxon.get("scientificName", ""), *(taxon.get("synonyms") or [])]
    expected = _norm(organism)
    return any(_norm(name) == expected or _norm(name).startswith(expected + " (")
               or _norm(name).startswith(expected + " strain ") for name in names)


def select_sequence(entries, name, organism, *, identifier=None):
    """Return a unique, active, complete sequence identity or an abstention."""
    from enzyme_discovery_agent.tools.fulltext import _entry_gene_names
    candidates = []
    for entry in entries:
        accession = entry.get("primaryAccession")
        sequence = (entry.get("sequence") or {}).get("value", "")
        if not accession or entry.get("inactiveReason") or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence):
            continue
        if organism and not _matching_organism(entry, organism):
            continue
        if identifier:
            crossrefs = entry.get("uniProtKBCrossReferences") or []
            identifiers = {accession}
            for ref in crossrefs:
                if ref.get("database") == "EMBL":
                    identifiers.add(ref.get("id"))
                    identifiers.update(p.get("value") for p in ref.get("properties", [])
                                       if p.get("key") == "ProteinId")
            if identifier not in identifiers:
                continue
        elif not any(_norm(gene) == _norm(name) for gene in _entry_gene_names(entry)):
            continue
        pd = entry.get("proteinDescription") or {}
        if pd.get("flag") == "Fragment":
            continue
        candidates.append(entry)
    sequences = {entry["sequence"]["value"] for entry in candidates}
    if len(sequences) != 1:
        return None, {"status": "ambiguous" if sequences else "pending",
                      "matching_accessions": sorted(e["primaryAccession"] for e in candidates),
                      "distinct_sequences": len(sequences)}
    candidates.sort(key=lambda e: ("Swiss-Prot" not in e.get("entryType", ""), e["primaryAccession"]))
    return candidates[0], {"status": "resolved", "matching_accessions": sorted(e["primaryAccession"] for e in candidates),
                           "distinct_sequences": 1}


@tool(description="Resolve a registered paper-scoped lead to a sequence. Inspect explicit source associations, retain ambiguous matches, and link a resolved accession back to its original lead. Extra sources must be retrieved evidence belonging to this run, never evaluation data.",
      category="reasoning", parallel_safe=False, injected_params=["run_dir"], params={
          "lead_id": {"type": "string", "description": "Registered literature lead identifier."},
          "source_paths": {"type": "array", "items": {"type": "string"}, "default": [],
                           "description": "Optional retrieved text snapshots under this run's artifacts directory."},
          "run_dir": {"type": "string"}})
def resolve_literature_lead(lead_id, source_paths=None, run_dir=None):
    """Use evidence-scoped identities; service errors and ambiguity remain pending."""
    from enzyme_discovery_agent.tools.fulltext import _uniprot_search, _resolve_embl_accession, LookupUnavailable
    from enzyme_discovery_agent.tools.reasoning import _fetch_uniprot_entry
    from enzyme_discovery_agent.state_store import atomic_write_json
    root = Path(run_dir).resolve()
    from enzyme_discovery_agent.models.lead_followup import reconcile_directions
    reconcile_directions(root,accessions=[lead_id])
    registry = load_candidate_registry(root, strict=True)["candidates"]
    lead = registry.get(lead_id)
    if not lead or not (lead.get("metadata") or {}).get("is_literature_lead"):
        return {"status": "error", "summary": "Unknown paper-scoped lead", "data": None}
    meta = lead["metadata"]
    if meta.get('target_relevance_status')=='excluded_opposite_direction':
        return {'status':'partial','summary':'Source-supported opposite reaction direction; identity lookup skipped.', 'data':None,'count':0}
    origin = meta.get("literature_lead")
    if not isinstance(origin, dict) or not origin.get("text_path") or not origin.get("text_sha256"):
        return {"status": "partial", "summary": "This alias lacks its original paper-scoped identity source; recover the source before resolving it", "data": None}
    sources = []
    original = _source(origin["text_path"], (origin.get("paper") or {}).get("doi") or lead_id)
    if original["sha256"] != origin["text_sha256"]:
        return {"status": "error", "summary": "Original source changed; re-extract its lead before resolution", "data": None}
    sources.append(original)
    for path in source_paths or []:
        path = Path(path).resolve()
        if not path.is_relative_to(root / "artifacts"):
            return {"status": "error", "summary": "Additional identity evidence must be retrieved into this run's artifacts directory", "data": None}
        sources.append(_source(path, str(path.relative_to(root))))
    name = meta["literature_name"]
    associations = source_associations(name, sources)
    organisms = {_norm(a["organism"]): a["organism"] for a in associations if a.get("organism")}
    if meta.get("organism"):
        organisms.setdefault(_norm(meta["organism"]), meta["organism"])
    identifiers = {a["identifier"] for a in associations if a.get("identifier")}
    if origin.get("deposited_identifier"):
        identifiers.add(origin["deposited_identifier"])
    basis = {"lead_id": lead_id, "name": name, "associations": associations,
             "sources": [{k: s[k] for k in ("path", "sha256", "source_id")} for s in sources]}
    if len(organisms) > 1 or len(identifiers) > 1:
        result = {**basis, "status": "conflict", "reason": "Conflicting source identity associations"}
    elif not organisms and not identifiers:
        result = {**basis, "status": "pending", "reason": "Retrieve a source linking this name to a taxon or deposited identifier"}
    else:
        organism = next(iter(organisms.values()), "")
        identifier = next(iter(identifiers), None)
        try:
            if identifier:
                accessions = _resolve_embl_accession(identifier)
                entries = [_fetch_uniprot_entry(a) for a in accessions]
                if any(e is None for e in entries):
                    raise LookupUnavailable("An authoritative sequence record could not be retrieved")
                if not entries:
                    from enzyme_discovery_agent.tools.database import fetch_ncbi_protein
                    reply = fetch_ncbi_protein(identifier)
                    record = reply.get("data") or {}
                    if reply.get("status") != "success":
                        raise LookupUnavailable(reply.get("summary") or "Deposited sequence unavailable")
                    if record.get("accession") != identifier:
                        raise LookupUnavailable("Deposited identifier or version changed; inspect the current record")
                    entries = [{"primaryAccession": identifier, "sequence": {"value": record.get("sequence", "")},
                                "entryType": "NCBI Protein", "organism": {"scientificName": record.get("organism", "")},
                                "proteinDescription": {"submissionNames": [{"fullName": {"value": record.get("title", "")}}]}}]
            else:
                query = f'gene_exact:"{name.replace(chr(34), " ")}" AND organism_name:"{organism.replace(chr(34), " ")}"'
                entries = _uniprot_search(query, size=101, fields="accession,gene_names,organism_name,sequence,protein_name,reviewed,xref_embl")
                if len(entries) >= 101:
                    raise LookupUnavailable("Identity query exceeds the bounded result window; narrow its source-derived scope")
            entry, selection = select_sequence(entries, name, organism, identifier=identifier)
            result = {**basis, **selection, "organism": organism, "identifier": identifier}
            if entry:
                accession = entry["primaryAccession"]
                sequence = entry["sequence"]["value"]
                identity = {**result, "resolved_accession": accession,
                            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                            "database_record_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
                            "activity_validated": False, "construct_identity": "source_association; exact assayed construct not established"}
                snapshot = root / "artifacts" / "identity_resolution" / (identity["database_record_sha256"] + ".json")
                atomic_write_json(snapshot, entry)
                identity["database_record_path"] = str(snapshot)
                previous = registry.get(accession, {})
                links = set((previous.get("metadata") or {}).get("resolved_literature_leads") or []) | {lead_id}
                update_candidate_registry(root, accession, None if previous else "discovered", {
                    "literature_name": name, "sequence": sequence, "sequence_length": len(sequence),
                    "organism": entry.get("organism", {}).get("scientificName", organism),
                    "identity_resolution": identity, "resolved_literature_leads": sorted(links),
                    "source_tool": "resolve_literature_lead"})
                update_candidate_registry(root, lead_id, None, {"identity_resolution_status": "resolved",
                    "resolved_accession": accession, "identity_resolution": identity,
                    "claim_validation": {"promotion_eligible": False, "blocking_claims": ["activity_evidence"],
                                         "reason": "Resolved lead alias; evaluate its linked accession, not the literature identifier"}})
                append_event(root, "literature_lead_resolved", lead_id=lead_id, accession=accession,
                             sequence_sha256=identity["sequence_sha256"], activity_validated=False)
                return {"status": "success", "summary": "Literature lead linked to a unique sequence; activity remains unvalidated", "data": identity, "count": 1}
        except (LookupUnavailable, OSError, ValueError, TypeError) as exc:
            result = {**basis, "status": "unavailable", "reason": str(exc)}
    update_candidate_registry(root, lead_id, None, {"identity_resolution": result})
    append_event(root, "literature_lead_resolution_pending", lead_id=lead_id, resolution_status=result["status"])
    return {"status": "partial", "summary": result.get("reason") or "Sequence identity is ambiguous", "data": result, "count": 0}


@tool(description='Resolve a bounded batch of current paper-scoped identities with explicit retrieved name/taxon or deposited-ID associations. Interleave primary-title matches and other ready leads; do not infer activity from identity.',
      category='reasoning',parallel_safe=False,injected_params=['run_dir'],params={
          'run_dir':{'type':'string'},'limit':{'type':'integer','minimum':1,'maximum':8,'default':4}})
def resolve_actionable_literature_leads(run_dir,limit=4):
    from enzyme_discovery_agent.models.research_decisions import decision_context
    if type(limit) is not int or not 1<=limit<=8:return {'status':'error','summary':'Identity batch limit must be 1..8.'}
    root=Path(run_dir);registry=load_candidate_registry(root,strict=True)['candidates']
    packet=decision_context(root);results=[]
    for lead in packet['recent_unassessed_leads']:
        meta=(registry.get(lead['accession']) or {}).get('metadata') or {}
        if not meta.get('is_literature_lead'):continue
        paths=list(dict.fromkeys(str(root/paper['source_path']) for paper in lead.get('primary_title_matches') or []))
        origin=meta.get('literature_lead') or {};all_paths=([origin['text_path']] if origin.get('text_path') else [])+paths
        sources=[_source(path,str(path)) for path in all_paths if Path(path).is_file()]
        associations=source_associations(meta.get('literature_name',''),sources)
        if not associations and not meta.get('organism') and not origin.get('deposited_identifier'):continue
        basis=hashlib.sha256(json.dumps({'name':meta.get('literature_name'),
            'sources':[(source['path'],source['sha256']) for source in sources]},sort_keys=True).encode()).hexdigest()
        if meta.get('identity_batch_attempt_basis')==basis and time.time()<meta.get('identity_batch_retry_after',0):continue
        result=resolve_literature_lead(lead['accession'],source_paths=paths,run_dir=root)
        update_candidate_registry(root,lead['accession'],None,{'identity_batch_attempt_basis':basis,
            'identity_batch_retry_after':time.time()+3600})
        results.append({'lead_id':lead['accession'],'name':lead['name'],'result':result})
        if len(results)>=limit:break
    return {'status':'success','summary':'Bounded source-grounded identity work; unresolved or ambiguous records stay pending.',
            'data':results,'count':sum(row['result'].get('count',0) for row in results)}
