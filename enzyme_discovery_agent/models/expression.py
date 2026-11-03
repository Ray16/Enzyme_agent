"""Separate preferred-host observations from native origin and sequence risks."""

import hashlib
import json
import re

_HOST = re.compile(r"\b(?:Escherichia\s+coli|E\.\s*coli)\b", re.I)
_EXPRESSION = re.compile(r"(?:express(?:ed|ion)|produc(?:ed|tion)|purifi(?:ed|cation)).{0,100}(?:Escherichia\s+coli|E\.\s*coli)", re.I)
_PROBLEM = re.compile(r"insoluble|inclusion bod|failed to express|not expressed|inactive", re.I)


def expression_preference(data):
    expression = data.get("expression_data") or {}
    assessment = expression.get("preferred_host_assessment") or {}
    return {"reported_active": 4, "structural_construct_expressed": 3,
            "reported_expression": 2, "expression_problem_reported": 0}.get(assessment.get("level"), 1)


def assess_expression(accession, entry, pdb_ids=(), fetch_json=None):
    """Use explicit expression annotations; absence stays unknown.

    Structural host records apply to the deposited construct, and do not prove
    catalytic activity, soluble yield, or suitability of another construct.
    """
    lineage = entry.get("organism", {}).get("lineage", [])
    lineage = [row.get("scientificName", "") if isinstance(row, dict) else row for row in lineage]
    native = entry.get("organism", {}).get("scientificName", "")
    observations, risks, failures = [], [], []
    for comment in entry.get("comments", []):
        for passage in comment.get("texts", []):
            text = str(passage.get("value") or "")
            if not _EXPRESSION.search(text):
                continue
            level = "expression_problem_reported" if _PROBLEM.search(text) else (
                "reported_active" if re.search(r"catalyz|catalys|enzymatic activity", text, re.I) else "reported_expression")
            observations.append({"level": level, "host": "Escherichia coli", "source_type": "UniProt expression statement",
                                 "source": f"https://rest.uniprot.org/uniprotkb/{accession}.json",
                                 "comment_type": comment.get("commentType"), "quote": text,
                                 "source_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
                                 "references": passage.get("evidences") or []})
    for feature in entry.get("features", []):
        kind = feature.get("type")
        if kind in {"Transmembrane", "Signal", "Glycosylation", "Disulfide bond"}:
            risks.append({"annotation": kind, "description": feature.get("description"),
                          "interpretation": "Review construct, localization and host requirements; annotation alone does not establish expression failure."})
    for pdb in list(pdb_ids)[:2] if fetch_json else []:
        try:
            index = fetch_json(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}")
            entities = index.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
            for entity_id in entities:
                url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity_id}"
                entity = fetch_json(url)
                ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
                accessions = set(ids.get("uniprot_ids") or []) | {
                    item.get("database_accession") for item in ids.get("reference_sequence_identifiers", [])
                    if item.get("database_name") == "UniProt"}
                if accession not in accessions:
                    continue
                for host in entity.get("rcsb_entity_host_organism") or []:
                    name = host.get("ncbi_scientific_name") or host.get("scientific_name") or ""
                    if _HOST.search(name):
                        observations.append({"level": "structural_construct_expressed", "host": name,
                                             "source_type": "candidate-matched PDB expression host", "source": url,
                                             "pdb_id": pdb, "polymer_entity_id": str(entity_id),
                                             "quote": json.dumps(host, sort_keys=True),
                                             "source_sha256": hashlib.sha256(json.dumps(entity, sort_keys=True).encode()).hexdigest(),
                                             "construct_scope": "deposited experimental construct; native full-length soluble expression and activity remain separate questions"})
        except Exception as exc:
            failures.append({"pdb_id": pdb, "error_type": type(exc).__name__})
    preference = {"reported_active": 4, "structural_construct_expressed": 3,
                  "reported_expression": 2, "expression_problem_reported": 0}
    best = max(observations, key=lambda row: preference[row["level"]]) if observations else None
    level = best["level"] if best else "unknown"
    documented = level in {"reported_active", "reported_expression", "structural_construct_expressed"}
    return {"ecoli_expression": documented, "bacterial_expression": documented,
            "is_eukaryote": "Eukaryota" in lineage, "native_organism": native,
            "native_host_match": bool(_HOST.search(native)),
            "expression_evidence": [f"{row['source_type']}: {row['source']}" for row in observations],
            "expression_host": best["host"] if documented else None,
            "preferred_host_assessment": {"host": "Escherichia coli", "level": level,
                                          "evidence": observations, "sequence_annotation_risks": risks,
                                          "lookup_failures": failures,
                                          "limitation": "Evidence concerns reported constructs and native-function assays, not activity on a new target reaction."}}
