"""Source-grounded reaction evidence for proteins missing database annotations."""

import hashlib
from pathlib import Path
import re


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def build_literature_evidence(accession: str, record: dict, paper: dict,
                               text_path: str | Path, *, identity_basis: str) -> dict:
    """Retain the exact source and quotation behind a resolved enzyme record.

    The caller supplies an accession resolved from the deposited identifier or
    enzyme name and organism. Cited analogues do not establish native reactions.
    """
    path = Path(text_path).resolve()
    contents = path.read_bytes()
    return {
        "accession": accession,
        "enzyme_name": record.get("name", ""),
        "deposited_accession": record.get("genbank", ""),
        "organism": record.get("organism", ""),
        "identity_basis": identity_basis,
        "paper": {k: paper.get(k) for k in ("doi", "pmid", "pmcid", "title")},
        "text_path": str(path),
        "text_sha256": hashlib.sha256(contents).hexdigest(),
        "evidence_quote": record.get("evidence_quote", ""),
        "mechanism": record.get("reaction") or {},
        "native_substrates": record.get("native_substrates") or [],
        "characterization": record.get("characterization") or [],
        "evidence_basis": "source_grounded_literature_extraction",
    }


def verify_literature_evidence(evidence: dict | None, accession: str, entry: dict | None = None) -> bool:
    """Verify source integrity, candidate linkage and an actual reaction quote."""
    if not isinstance(evidence, dict) or evidence.get("accession") != accession:
        return False
    if evidence.get("identity_basis") not in {"resolved_name_and_organism", "deposited_accession"}:
        return False
    paper = evidence.get("paper") or {}
    if not any(paper.get(k) for k in ("doi", "pmid", "pmcid")):
        return False
    try:
        contents = Path(evidence["text_path"]).read_bytes()
    except (OSError, KeyError, TypeError):
        return False
    if hashlib.sha256(contents).hexdigest() != evidence.get("text_sha256"):
        return False
    quote = _normalized(str(evidence.get("evidence_quote") or ""))
    if len(quote) < 20 or quote not in _normalized(contents.decode(errors="replace")):
        return False
    anchors = [evidence.get("enzyme_name"), evidence.get("deposited_accession"), accession]
    if not any(a and re.search(r"(?<!\w)" + re.escape(_normalized(str(a))) + r"(?!\w)", quote)
               for a in anchors):
        return False
    # A source quote and a coincidentally matching accession do not establish
    # that this sequence is the enzyme characterized in the source.
    if not isinstance(entry, dict) or entry.get("primaryAccession") != accession:
        return False
    source = _normalized(contents.decode(errors="replace"))
    name = _normalized(str(evidence.get("enzyme_name") or ""))
    if not name:
        return False
    if evidence["identity_basis"] == "resolved_name_and_organism":
        organism = _normalized(str(evidence.get("organism") or ""))
        record_organism = _normalized(str((entry.get("organism") or {}).get("scientificName") or ""))
        if not organism or not (record_organism == organism or record_organism.startswith(organism + " (")):
            return False
        if not re.search(r"(?<!\w)" + re.escape(name) + r"\s+from\s+" + re.escape(organism) + r"(?!\w)", source):
            return False
        names = set()
        for gene in entry.get("genes") or []:
            names.add(_normalized((gene.get("geneName") or {}).get("value", "")))
            names.update(_normalized(s.get("value", "")) for s in gene.get("synonyms") or [])
        description = entry.get("proteinDescription") or {}
        for item in [description.get("recommendedName") or {}, *(description.get("alternativeNames") or []),
                     *(description.get("submissionNames") or [])]:
            names.add(_normalized((item.get("fullName") or {}).get("value", "")))
        if name not in names:
            return False
    else:
        identifier = _normalized(str(evidence.get("deposited_accession") or ""))
        if not identifier:
            return False
        identifiers = {_normalized(accession)}
        for ref in entry.get("uniProtKBCrossReferences") or []:
            if ref.get("database") == "EMBL":
                # An EMBL nucleotide/cluster accession can contain many
                # proteins. Only a protein-level identifier establishes this
                # entry's sequence, not the shared nucleotide record.
                identifiers.update(_normalized(p.get("value") or "") for p in ref.get("properties") or []
                                   if p.get("key") == "ProteinId")
        if identifier not in identifiers:
            return False
        if not re.search(r"(?<!\w)" + re.escape(name) + r"[^.!?]{0,120}(?<!\w)" +
                         re.escape(identifier) + r"(?!\w)", source):
            return False
    return isinstance(evidence.get("mechanism"), dict) and bool(evidence["mechanism"])
