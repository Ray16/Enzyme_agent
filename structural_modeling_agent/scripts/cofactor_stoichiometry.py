#!/usr/bin/env python3
"""Data-driven catalytic-metal stoichiometry for cofold setup.

The step-1 cofold used a blanket 2 Mg for every enzyme. That is wrong for a large
class of ATP-dependent enzymes that use a SINGLE catalytic Mg2+ (the adenylate-forming
ANL enzymes — e.g. SfaB) and right for others that use two (e.g. glutamine synthetase).
Rather than hardcode any mechanism->metal table (which for the discovery agent would also
be answer leakage), we READ the count from UniProt annotation:

  1. COFACTOR note "Binds N Mg(2+) ions per subunit"     -> explicit N   (most reliable)
  2. distinct metal-ion BINDING labels                    -> ion count
  3. a divalent-metal COFACTOR with no count/binding      -> assume 1
  4. candidate has NO metal annotation (e.g. TrEMBL)      -> modal count of its FAMILY's
                                                             reviewed members (data-driven)
  5. nothing anywhere                                      -> generic default (1)

Everything is per-enzyme annotation; no reaction/mechanism/family name is baked in.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import Counter

DIVALENT = ("Mg", "Mn", "Zn", "Co", "Ni", "Fe", "Ca")
_UNIPROT = "https://rest.uniprot.org/uniprotkb"


def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def _first_divalent(names: list[str]) -> str | None:
    for n in names:
        for m in DIVALENT:
            if n and n.startswith(m):
                return "MG" if m == "Mg" else m.upper()
    return None


def metal_count_from_uniprot_json(data: dict) -> tuple[int | None, str, str | None]:
    """(count, basis, metal_ccd) from one UniProt entry, or (None, basis, None)."""
    cof_names: list[str] = []
    note_text = ""
    for c in data.get("comments", []):
        if c.get("commentType") != "COFACTOR":
            continue
        for co in c.get("cofactors", []):
            if co.get("name"):
                cof_names.append(co["name"])
        for t in c.get("note", {}).get("texts", []):
            note_text += " " + t.get("value", "")

    metal = _first_divalent(cof_names)

    # (1) explicit "Binds N <metal>" in the cofactor note
    m = re.search(r"binds?\s+(\d+)\s+([A-Z][a-z]?)\(?", note_text, re.I)
    if m and metal:
        return int(m.group(1)), "uniprot_cofactor_note", metal

    # (2) distinct metal-ion binding labels (numeric labels distinguish ions;
    #     a bare ChEBI-id label means residues of a single unlabelled ion)
    labels_by_metal: dict[str, set] = {}
    for f in data.get("features", []):
        if f.get("type") != "Binding site":
            continue
        lig = f.get("ligand", {})
        name = lig.get("name", "")
        ccd = _first_divalent([name])
        if not ccd:
            continue
        lab = lig.get("label") or lig.get("id") or ""
        labels_by_metal.setdefault(ccd, set()).add(lab)
    if labels_by_metal:
        ccd = metal or next(iter(labels_by_metal))
        labs = labels_by_metal.get(ccd, set())
        numeric = {l for l in labs if str(l).isdigit()}
        count = len(numeric) if numeric else 1
        return count, "uniprot_binding_labels", ccd

    # (3) a divalent metal is named as cofactor but no count/binding detail
    if metal:
        return 1, "uniprot_cofactor_single", metal

    return None, "no_metal_annotation", None


def interpro_ids_from_json(data: dict) -> list[str]:
    """Candidate's own InterPro accessions (used to find annotated reviewed relatives)."""
    return [x.get("id") for x in data.get("uniProtKBCrossReferences", [])
            if x.get("database") == "InterPro" and x.get("id")]


def _reviewed_family_members(family_query: str, limit: int = 25) -> list[str]:
    """Accessions of reviewed entries matching a family/EC query."""
    params = urllib.parse.urlencode({
        "query": f"({family_query}) AND reviewed:true",
        "fields": "accession", "format": "json", "size": str(limit),
    })
    data = _fetch_json(f"{_UNIPROT}/search?{params}")
    if not data:
        return []
    return [r.get("primaryAccession", "") for r in data.get("results", []) if r.get("primaryAccession")]


def derive_metal_cofactor(
    accession: str,
    family_query: str | None = None,
    default_count: int = 1,
    fetch=_fetch_json,
) -> dict:
    """Return {ccd, count, role, basis, metal} for the catalytic metal of `accession`.

    Falls back to the modal count of the reviewed members named by `family_query`
    (e.g. 'xref:interpro-IPR050237' or 'ec:6.3.5.-') when the candidate itself is
    unannotated, then to `default_count`.
    """
    data = fetch(f"{_UNIPROT}/{accession}.json") if accession else None
    if data:
        count, basis, metal = metal_count_from_uniprot_json(data)
        if count:
            return {"ccd": metal or "MG", "count": count, "role": "catalytic_metal",
                    "basis": basis, "metal": metal or "MG"}

    # candidate itself is unannotated (e.g. TrEMBL): poll its reviewed relatives. Use an
    # explicit family_query if given, else the candidate's OWN InterPro memberships.
    queries: list[str] = [family_query] if family_query else []
    if not queries and data:
        queries = [f"xref:interpro-{i}" for i in interpro_ids_from_json(data)]
    for q in queries:
        counts: Counter = Counter()
        metals: Counter = Counter()
        for acc in _reviewed_family_members(q):
            if acc == accession:
                continue
            md = fetch(f"{_UNIPROT}/{acc}.json")
            if not md:
                continue
            c, _, m = metal_count_from_uniprot_json(md)
            if c:
                counts[c] += 1
                metals[m or "MG"] += 1
        if counts:
            count = counts.most_common(1)[0][0]
            metal = metals.most_common(1)[0][0]
            return {"ccd": metal, "count": count, "role": "catalytic_metal",
                    "basis": f"family_reviewed_modal({q},n={sum(counts.values())})", "metal": metal}

    return {"ccd": "MG", "count": default_count, "role": "catalytic_metal",
            "basis": "generic_default", "metal": "MG"}


if __name__ == "__main__":
    import sys
    acc = sys.argv[1] if len(sys.argv) > 1 else "A0A2H4T912"
    fam = sys.argv[2] if len(sys.argv) > 2 else "xref:interpro-IPR050237"
    print(acc, "->", derive_metal_cofactor(acc, family_query=fam))
