"""Recover explicit sequence/name associations as unvalidated discovery leads."""

import re
import hashlib
import json
from pathlib import Path

# An author-coined name such as "ShABS" exists only in its paper: UniProt returns nothing
# for it under any query, so the deposited accession the paper cites is the ONLY route from
# that name to a sequence. Recognise the usual ways papers write one, while still demanding
# an explicit database cue next to the name so this stays an explicit local association
# rather than a scrape of every accession-shaped token in the text.
_DEPOSIT_CUE = r"(?:GenBank|GenBank\s+accession|NCBI(?:\s+Reference\s+Sequence)?|RefSeq|UniProt(?:KB)?|EMBL|DDBJ|accession)"
_ACCESSION = (r"(?:[A-Z]{2,4}\d{5,9}(?:\.\d+)?"                      # GenBank / EMBL / DDBJ
              r"|(?:WP|NP|YP|XP|AP|EF|QQ)_\d{6,9}(?:\.\d+)?"          # RefSeq
              r"|[OPQ][0-9][A-Z0-9]{3}[0-9]"                          # UniProt
              r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})")
_ASSOCIATION = re.compile(
    r"\b([A-Z][A-Za-z0-9]{1,14})\s*(?:\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\d+|\^[\d{}]+|\{\d+\})?\s*"
    r"[(\[]?\s*" + _DEPOSIT_CUE + r"\s*(?:no\.?|number|ID)?\s*:?\s*(?:\d{1,6}\s+)?"
    + r"(" + _ACCESSION + r")\s*[)\]]?", re.I)


def sequence_name_leads(text):
    leads = []
    for match in _ASSOCIATION.finditer(text):
        context = text[max(0, match.start() - 220):min(len(text), match.end() + 220)]
        if not re.search(r"encod|synthesiz|express|purif|clon|characteriz", context, re.I):
            continue
        leads.append({"name": match.group(1), "genbank": match.group(2), "organism": "",
                      "source_quote": match.group(0), "relationship": "sequence_identifier_lead"})
    return leads


# A class label names a group, not an individual enzyme, so no identifier lookup can
# ever resolve it. Registering one costs a pool slot and a share of every later
# resolution pass while never being able to yield a sequence. Detection stays purely
# grammatical -- plural heads, citation phrases, collection words -- so it carries no
# assumption about which groups a given target makes relevant.
_CLASS_CUES = re.compile(r"(\bet\s+al\b|\blibrar(?:y|ies)\b|\bfamil(?:y|ies)\b|\bsuperfamily\b|"
                         r"\bclass\b|\bgroup\b|\(general\)|\bsuch\s+as\b|\bvarious\b|\bother\b|"
                         # A domain is a part of an enzyme, and "used by"/"from" introduce a
                         # provenance phrase rather than an enzyme's own name.
                         r"\bdomains?\b|\benzymes\b|\bproteins\b|\bused\s+by\b)", re.I)
_PLURAL_HEAD = re.compile(r"\b\w{3,}(?:ases|ases\b)\b", re.I)
_BARE_GENERIC = re.compile(r"^\s*(?:putative\s+|probable\s+)?\w+(?:ase|domain|enzyme|protein)\s*$", re.I)


def enzyme_class_label(name):
    """True when the name denotes a category of enzymes rather than one enzyme."""
    text = str(name or "").strip()
    if not text:
        return False
    if _CLASS_CUES.search(text) or _PLURAL_HEAD.search(text):
        return True
    # A single generic head with no distinguishing qualifier ("Lipase", "Acyltransferase")
    # is a class; the same head with a qualifier ("Coronafacate ligase") is an enzyme.
    return bool(_BARE_GENERIC.match(text))


def register_literature_lead(run_dir, record, paper, text_path, *, registration_run_id=None):
    """Register a paper-scoped candidate before a sequence identity is known."""
    from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, update_candidate_registry
    from enzyme_discovery_agent.models.event_log import append_event
    from enzyme_discovery_agent.run_layout import run_file
    root = Path(run_dir); name = str(record.get("name") or "").strip()
    if not name: return None
    if enzyme_class_label(name) and not str(record.get("genbank") or "").strip():
        # Keep the observation in the mined text; it just does not become a candidate
        # that resolution passes will retry forever. A deposited identifier overrides
        # this, because then the lookup is determinate regardless of how it is named.
        return None
    path = Path(text_path).resolve(); contents = path.read_bytes()
    organism = str(record.get("organism") or "").strip()
    if organism.casefold() in {"not specified", "unknown", "not stated", "unspecified"}: organism = ""
    identifier = str(record.get("genbank") or "").strip()
    source_id = paper.get("doi") or paper.get("pmid") or paper.get("pmcid") or hashlib.sha256(contents).hexdigest()
    key = json.dumps([str(source_id).casefold(), name.casefold(), organism.casefold(), identifier], sort_keys=True)
    lead_id = "literature:" + hashlib.sha256(key.encode()).hexdigest()[:20]
    candidates = load_candidate_registry(root, strict=True).get("candidates", {})
    existing = candidates.get(lead_id)
    old = (existing or {}).get("metadata") or {}
    relationship = record.get("relationship") or "named"
    source = {"paper": {k: paper.get(k) for k in ("doi", "pmid", "pmcid", "title")},
              "text_path": str(path), "text_sha256": hashlib.sha256(contents).hexdigest(),
              "name_anchor_found": bool(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", contents.decode(errors="replace"), re.I)),
              "deposited_identifier": identifier or None,
              "relationships": sorted(set((old.get("literature_lead") or {}).get("relationships") or []) | {relationship})}
    try: origin = json.loads((root / "manifest.json").read_text())["run_id"]
    except (OSError, ValueError, KeyError): origin = root.name
    metadata = {"is_literature_lead": True, "literature_name": name, "protein_name": old.get("protein_name") or name,
                "source_function_evidence":{k:record[k] for k in ('mechanism','reaction_direction','native_substrates','evidence_quote') if k in record},
                "organism": organism, "identity_resolution_status": "pending", "literature_lead": source,
                "discovered_in_run_id": origin, "registration_run_id": registration_run_id or origin,
                "source_tool": "paper_named_literature_lead",
                "claim_validation": {"promotion_eligible": False, "blocking_claims": ["sequence_identity"],
                                     "reason": "Paper-scoped enzyme lead; accession and sequence require resolution"}}
    if old.get("literature_lead") == source and old.get('source_function_evidence')==metadata['source_function_evidence']: return lead_id
    if old.get("identity_resolution_status") == "resolved" and old.get('literature_lead')==source:
        metadata.update(identity_resolution_status='resolved',resolved_accession=old.get('resolved_accession'))
    elif old.get("identity_resolution_status") == "resolved":
        metadata.update(identity_resolution_status="pending", resolved_accession=None,
                        identity_resolution={"status": "stale", "reason": "Source identity evidence changed"})
    update_candidate_registry(root, lead_id, None if existing else "evidence_pending", metadata)
    from .lead_followup import reconcile_directions
    reconcile_directions(root,accessions=[lead_id])
    append_event(root, "literature_lead_registered", lead_id=lead_id, literature_name=name,
                 source_paper=source["paper"], source_discovery_run_id=origin,
                 registration_run_id=registration_run_id or origin, identity_status="pending")
    return lead_id
