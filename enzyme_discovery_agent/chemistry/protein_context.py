"""Extract active-site architecture, tractability, and promiscuity cues from UniProt records.

Operates strictly on already-fetched UniProt JSON and local evaluation metadata
(zero extra network or LLM calls). Reaction-agnostic: encodes no benchmark
substrates, answer families, or enzyme names.
"""

from __future__ import annotations

import re

_MEMBRANE_CUES = re.compile(r"\b(?:membrane|transmembrane|integral\s+component\s+of\s+membrane)\b", re.I)
_HETEROMER_CUES = re.compile(r"\b(?:heterodimer|heterotetramer|heterohexamer|heterooctamer|heteromultimer|heterooligomer|complex\s+with|interacts\s+with\s+\w+\s+subunit)\b", re.I)
_HOMOMER_CUES = re.compile(r"\b(?:homodimer|homotrimer|homotetramer|homohexamer|homooctamer|homodecamer|homododecamer|homooligomer|homomultimer)\b", re.I)
_MONOMER_CUES = re.compile(r"\bmonomer\b", re.I)

_BROAD_SPECIFICITY_CUES = re.compile(
    r"\b(?:broad\s+substrate\s+specificit\w*|promiscu\w*|wide\s+range\s+of|"
    r"variety\s+of\s+substrates?|multiple\s+substrates?|various\s+\w+\s+substrates?|"
    r"relaxed\s+specificit\w*|accepts?\s+several|can\s+also\s+use)\b",
    re.I,
)
_SECONDARY_METAB_CUES = re.compile(
    r"\b(?:secondary\s+metabol\w*|antibiotic\s+biosynthesis|natural\s+product|"
    r"xenobiotic|detoxification|biosynthetic\s+gene\s+cluster|specialized\s+metabol\w*)\b",
    re.I,
)
_PRIMARY_HOUSEKEEPING_CUES = re.compile(
    r"\b(?:aminoacyl-tRNA|tRNA\s+synthetase|ribosom\w*|translation\s+elongation|"
    r"housekeeping|essential\s+primary)\b",
    re.I,
)


def _feature_position(feature: dict) -> str | None:
    loc = feature.get("location") or {}
    start = (loc.get("start") or {}).get("value")
    end = (loc.get("end") or {}).get("value")
    if start is not None and end is not None:
        return str(start) if start == end else f"{start}-{end}"
    if start is not None:
        return str(start)
    return None


def extract_architecture_and_tractability(
    entry: dict, structural_data: dict | None = None
) -> dict:
    """Summarize catalytic residues, ligand binding sites, membrane flags, and subunit state."""
    if not isinstance(entry, dict):
        return {}

    catalytic_residues: list[dict] = []
    binding_sites: list[dict] = []
    transmem_count = 0

    for feat in entry.get("features") or []:
        if not isinstance(feat, dict):
            continue
        ftype = feat.get("type") or ""
        pos = _feature_position(feat)
        desc = str(feat.get("description") or "").strip()
        if ftype in {"Active site", "ACT_SITE"}:
            catalytic_residues.append({k: v for k, v in (("position", pos), ("description", desc)) if v})
        elif ftype in {"Binding site", "BINDING"}:
            ligand = ((feat.get("ligand") or {}).get("name") or desc or "").strip()
            binding_sites.append({k: v for k, v in (("position", pos), ("ligand", ligand)) if v})
        elif ftype in {"Transmembrane", "Intramembrane", "TRANSMEM", "INTRAMEM"}:
            transmem_count += 1

    subunit_texts: list[str] = []
    subcellular_membrane = False
    for comment in entry.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        ctype = comment.get("commentType") or ""
        if ctype == "SUBUNIT":
            for t in comment.get("texts") or []:
                val = str(t.get("value") or "").strip()
                if val:
                    subunit_texts.append(val)
        elif ctype == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations") or []:
                val = str((loc.get("location") or {}).get("value") or "")
                if _MEMBRANE_CUES.search(val):
                    subcellular_membrane = True

    subunit_joined = " ".join(subunit_texts)
    if _HETEROMER_CUES.search(subunit_joined):
        subunit_class = "heterooligomer"
    elif _HOMOMER_CUES.search(subunit_joined):
        subunit_class = "homooligomer"
    elif _MONOMER_CUES.search(subunit_joined):
        subunit_class = "monomer"
    else:
        subunit_class = "unknown"

    pdb_ids = (structural_data or {}).get("pdb_ids") or []
    result = {
        "subunit_class": subunit_class,
        "is_membrane_protein": bool(transmem_count > 0 or subcellular_membrane),
        "transmembrane_helices": transmem_count,
        "has_experimental_structure": bool(pdb_ids),
        "pdb_count": len(pdb_ids),
    }
    if subunit_joined:
        result["subunit_note"] = subunit_joined[:220]
    if catalytic_residues:
        result["catalytic_residues"] = catalytic_residues[:8]
    if binding_sites:
        # Collapse by ligand name to stay compact
        grouped: dict[str, list[str]] = {}
        for bs in binding_sites:
            lig = bs.get("ligand") or "unspecified"
            if bs.get("position"):
                grouped.setdefault(lig, []).append(bs["position"])
            else:
                grouped.setdefault(lig, [])
        result["binding_sites"] = [
            {"ligand": lig, "positions": positions[:8]}
            for lig, positions in list(grouped.items())[:6]
        ]
    return result


def extract_promiscuity_indicators(
    entry: dict,
    reaction_groups: list[dict] | None = None,
    literature: dict | None = None,
) -> dict:
    """Summarize observable promiscuity/generalist vs. specialist cues from retrieved metadata."""
    if not isinstance(entry, dict):
        return {}

    distinct_rxns = len({
        str(g.get("reaction") or "").strip().casefold()
        for g in (reaction_groups or [])
        if str(g.get("reaction") or "").strip()
    })
    if distinct_rxns == 0:
        distinct_rxns = sum(
            1 for c in (entry.get("comments") or [])
            if isinstance(c, dict) and c.get("commentType") == "CATALYTIC ACTIVITY"
        )

    function_texts = []
    pathway_texts = []
    for c in entry.get("comments") or []:
        if not isinstance(c, dict):
            continue
        ctype = c.get("commentType") or ""
        if ctype == "FUNCTION":
            for t in c.get("texts") or []:
                if t.get("value"):
                    function_texts.append(str(t["value"]))
        elif ctype == "PATHWAY":
            for t in c.get("texts") or []:
                if t.get("value"):
                    pathway_texts.append(str(t["value"]))

    keywords = [
        str(k.get("name") or "") for k in (entry.get("keywords") or []) if isinstance(k, dict)
    ]
    combined = " ".join(function_texts + pathway_texts + keywords)

    broad_cues = sorted({m.lower() for m in _BROAD_SPECIFICITY_CUES.findall(" ".join(function_texts))})
    if _PRIMARY_HOUSEKEEPING_CUES.search(combined):
        metabolic_context = "primary_or_housekeeping"
    elif _SECONDARY_METAB_CUES.search(combined):
        metabolic_context = "secondary_or_specialized"
    else:
        metabolic_context = "unspecified"

    eng_papers = int((literature or {}).get("engineering_papers") or 0)
    result = {
        "distinct_annotated_reactions": distinct_rxns,
        "metabolic_context": metabolic_context,
        "engineering_precedent_papers": eng_papers,
    }
    if broad_cues:
        result["broad_specificity_cues"] = broad_cues[:6]
    return result
