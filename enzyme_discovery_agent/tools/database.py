"""Database search tools: UniProt, KEGG, InterPro.

Provides the agent with access to protein/enzyme databases for finding
candidate enzymes by EC number, keyword, gene name, or domain family.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from enzyme_discovery_agent import config
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")


# ---------------------------------------------------------------------------
# UniProt — shared parsing
# ---------------------------------------------------------------------------


def _parse_uniprot_json(data: dict) -> dict:
    """Parse common fields from a UniProt JSON entry.

    Returns dict with: protein_name, ec_numbers, gene_names, organism,
    catalytic_activities, function.
    """
    pd = data.get("proteinDescription", {})
    rec = pd.get("recommendedName", {})
    protein_name = rec.get("fullName", {}).get("value", "") if rec else ""
    if not protein_name:
        for sub in pd.get("submissionNames", []):
            pn = sub.get("fullName", {}).get("value", "")
            if pn:
                protein_name = pn
                break

    ec_numbers: list[str] = []
    # From recommended name
    if rec:
        for ec in rec.get("ecNumbers", []):
            val = ec.get("value", "")
            if val:
                ec_numbers.append(val)
    # From submission names
    for sub in pd.get("submissionNames", []):
        for ec in sub.get("ecNumbers", []):
            val = ec.get("value", "")
            if val and val not in ec_numbers:
                ec_numbers.append(val)
    # From alternative names
    for alt in pd.get("alternativeNames", []):
        for ec in alt.get("ecNumbers", []):
            val = ec.get("value", "")
            if val and val not in ec_numbers:
                ec_numbers.append(val)

    gene_names: list[str] = []
    for gene in data.get("genes", []):
        gn = gene.get("geneName", {}).get("value", "")
        if gn:
            gene_names.append(gn)
        for syn in gene.get("synonyms", []):
            sv = syn.get("value", "")
            if sv and sv not in gene_names:
                gene_names.append(sv)

    organism = data.get("organism", {}).get("scientificName", "")

    catalytic_activities: list[str] = []
    for comment in data.get("comments", []):
        if comment.get("commentType") == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            name = reaction.get("name", "")
            if name:
                catalytic_activities.append(name)

    function_text = ""
    for comment in data.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            for t in comment.get("texts", []):
                function_text = t.get("value", "")
                break

    return {
        "protein_name": protein_name,
        "ec_numbers": ec_numbers,
        "gene_names": gene_names,
        "organism": organism,
        "catalytic_activities": catalytic_activities,
        "function": function_text,
    }


# ---------------------------------------------------------------------------
# UniProt — search & fetch
# ---------------------------------------------------------------------------

def _relax_uniprot_query(query: str) -> list[str]:
    """Progressively simpler variants of a free-text UniProt query.

    UniProt ANDs free-text terms, so an over-specified query zeros out when one
    term misses.  Qualifiers (organism, keyword tags) usually trail the core
    identity, so drop terms from the END one at a time, down to the first two.
    Returns the full query first, then the relaxations, de-duplicated.
    """
    terms = query.split()
    variants = [query]
    for k in range(len(terms) - 1, 1, -1):
        variants.append(" ".join(terms[:k]))
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


@tool(
    description=(
        "Search UniProt for enzymes matching a query. Supports UniProt query syntax "
        "(e.g., 'ec:3.5.1.* AND amidase'). Use gene_name for precise gene-field search. "
        "Use function_filter to disambiguate short or ambiguous names."
    ),
    category="database",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": (
                "Optional UniProt query string (UniProt search syntax). Ignored when "
                "gene_name is provided. Syntax, e.g. 'ec:<number>', 'gene:<name>', "
                "'organism_id:<taxid>', 'reviewed:true', combined with AND/OR."
            ),
            "default": "",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 50).",
            "default": 50,
        },
        "gene_name": {
            "type": "string",
            "description": (
                "If provided, searches using the UniProt gene: field "
                "(e.g., gene_name='example_gene'). Much more precise than keyword search."
            ),
            "default": "",
        },
        "function_filter": {
            "type": "string",
            "description": (
                "If provided, AND-ed with the query to disambiguate. "
                "E.g., function_filter='ATP-dependent ligase' when a gene name is ambiguous."
            ),
            "default": "",
        },
    },
)
def search_uniprot(
    query: str = "",
    max_results: int = 50,
    gene_name: str = "",
    function_filter: str = "",
) -> dict:
    """Search UniProt for enzymes matching a query."""
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    if gene_name:
        query = f"gene:{gene_name}"

    if not query:
        return {
            "status": "error",
            "summary": "No query or gene_name provided",
            "data": [],
            "suggestions": ["Provide a query string or gene_name."],
        }

    # UniProt free-text search implicitly ANDs the terms, so an over-specified
    # query (gene name + full description + organism + a keyword tag) returns 0 the
    # moment ONE term does not match, even when the core 2-3 terms have plenty of
    # hits.  So on an empty result, RELAX: retry progressively simpler variants
    # (drop trailing qualifier terms) until hits appear, mirroring the InterPro
    # simplifier.  A structured gene: query is left as-is (precise by construction).
    fields = ("accession,protein_name,organism_name,sequence,ec,gene_names,"
              "cc_function,cc_catalytic_activity")
    variants = [query] if gene_name else _relax_uniprot_query(query)

    def _fetch(q: str):
        full = f"({q}) AND ({function_filter})" if function_filter else q
        params = {"query": full, "format": "json",
                  "size": str(min(max_results, config.UNIPROT_MAX_RESULTS_CAP)),
                  "fields": fields}
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        return json.loads(request_with_retry(req, timeout=30).decode())

    data = {}
    used_query = query
    relaxed = False
    try:
        for i, variant in enumerate(variants):
            data = _fetch(variant)
            if data.get("results"):
                used_query = variant
                relaxed = i > 0
                break
    except Exception as e:
        return {
            "status": "error",
            "summary": f"UniProt search failed: {e}",
            "data": [],
            "suggestions": ["Try a simpler query or different search terms."],
        }
    if relaxed:
        logger.info("UniProt search relaxed '%s' -> '%s' (%d hits)",
                    query, used_query, len(data.get("results", [])))

    entries = []
    for entry in data.get("results", []):
        # Protein name
        protein_name = ""
        pd = entry.get("proteinDescription", {})
        rec = pd.get("recommendedName")
        if rec:
            protein_name = rec.get("fullName", {}).get("value", "")
        if not protein_name:
            for sub in pd.get("submissionNames", []):
                pn = sub.get("fullName", {}).get("value", "")
                if pn:
                    protein_name = pn
                    break

        # EC numbers
        ec_numbers = []
        if rec:
            for ec_entry in rec.get("ecNumbers", []):
                ec_numbers.append(ec_entry.get("value", ""))
        for sub in pd.get("submissionNames", []):
            for ec_entry in sub.get("ecNumbers", []):
                val = ec_entry.get("value", "")
                if val and val not in ec_numbers:
                    ec_numbers.append(val)

        # Gene names
        entry_gene_names = []
        for gene_entry in entry.get("genes", []):
            if "geneName" in gene_entry:
                entry_gene_names.append(gene_entry["geneName"].get("value", ""))
            for syn in gene_entry.get("synonyms", []):
                entry_gene_names.append(syn.get("value", ""))

        seq = entry.get("sequence", {}).get("value", "")
        organism = entry.get("organism", {}).get("scientificName", "")

        catalytic_activities = []
        function_text = ""
        for comment in entry.get("comments", []):
            ct = comment.get("commentType", "")
            if ct == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                name = reaction.get("name", "")
                if name:
                    catalytic_activities.append(name)
            elif ct == "FUNCTION":
                for t in comment.get("texts", []):
                    val = t.get("value", "")
                    if val:
                        function_text = val
                        break

        entries.append({
            "accession": entry.get("primaryAccession", ""),
            "name": protein_name,
            "organism": organism,
            "ec_number": "; ".join(ec_numbers),
            "gene_names": entry_gene_names,
            "seq_length": len(seq),
            "sequence": seq,
            "function": function_text[:200],
            "catalytic_activities": catalytic_activities[:3],
        })

    _relaxed_note = (
        f" (relaxed from '{query[:50]}' — the full query had no hits)"
        if relaxed else ""
    )
    return {
        "status": "success" if entries else "partial",
        "summary": (f"Found {len(entries)} UniProt entries for query: "
                    f"{used_query[:60]}{_relaxed_note}"),
        "data": entries,
        "count": len(entries),
        "suggestions": (
            ["Try searching by gene_name for more precise results.",
             "Try broader keywords or partial EC numbers."]
            if not entries else []
        ),
    }


@tool(
    description=(
        "Fetch full UniProt annotations for a protein by accession ID. Returns "
        "protein name, gene names, organism, EC numbers, catalytic activities, "
        "function description, cross-references (PDB, GenBank/EMBL, InterPro, "
        "KEGG, Pfam), sequence length, and the sequence itself. Use this to "
        "verify a candidate's identity, check cross-database links, and get "
        "detailed annotations beyond what search_uniprot returns."
    ),
    category="database",
    parallel_safe=True,
    params={
        "accession": {
            "type": "string",
            "description": "UniProt accession ID (e.g., 'Q9X4W9').",
        },
    },
)
def fetch_uniprot_sequence(accession: str) -> dict:
    """Fetch full UniProt annotations for a protein by accession."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        data = json.loads(request_with_retry(req, timeout=15).decode())
    except Exception as e:
        return {
            "status": "error",
            "summary": f"Failed to fetch {accession}: {e}",
            "data": {"accession": accession},
            "suggestions": ["Check that the accession ID is correct."],
        }

    # Shared field parsing
    parsed = _parse_uniprot_json(data)
    protein_name = parsed["protein_name"]
    ec_numbers = parsed["ec_numbers"]
    gene_names = parsed["gene_names"]
    organism = parsed["organism"]
    catalytic_activities = parsed["catalytic_activities"]
    function_text = parsed["function"]

    # Alternative names (unique to full fetch)
    pd = data.get("proteinDescription", {})
    alternative_names = []
    for alt in pd.get("alternativeNames", []):
        an = alt.get("fullName", {}).get("value", "")
        if an:
            alternative_names.append(an)

    # Taxonomy (unique to full fetch)
    lineage = data.get("organism", {}).get("lineage", [])
    taxonomy = [t if isinstance(t, str) else t.get("scientificName", "") for t in lineage]

    # Entry type
    entry_type = data.get("entryType", "")
    is_swissprot = "Swiss-Prot" in entry_type

    # Sequence
    seq = data.get("sequence", {}).get("value", "")
    seq_length = data.get("sequence", {}).get("length", len(seq))

    # Cross-references (PDB, GenBank/EMBL, InterPro, KEGG, Pfam)
    xref_pdb = []
    xref_embl = []
    xref_interpro = []
    xref_kegg = []
    xref_pfam = []
    for xref in data.get("uniProtKBCrossReferences", []):
        db = xref.get("database", "")
        xid = xref.get("id", "")
        if db == "PDB":
            # Extract resolution and method from properties
            props = {p.get("key", ""): p.get("value", "") for p in xref.get("properties", [])}
            xref_pdb.append({
                "id": xid,
                "method": props.get("Method", ""),
                "resolution": props.get("Resolution", ""),
                "chains": props.get("Chains", ""),
            })
        elif db == "EMBL":
            props = {p.get("key", ""): p.get("value", "") for p in xref.get("properties", [])}
            xref_embl.append({
                "nucleotide_id": xid,
                "protein_id": props.get("ProteinId", ""),
            })
        elif db == "InterPro":
            props = {p.get("key", ""): p.get("value", "") for p in xref.get("properties", [])}
            xref_interpro.append({
                "id": xid,
                "name": props.get("EntryName", ""),
            })
        elif db == "KEGG":
            xref_kegg.append(xid)
        elif db == "Pfam":
            props = {p.get("key", ""): p.get("value", "") for p in xref.get("properties", [])}
            xref_pfam.append({
                "id": xid,
                "name": props.get("EntryName", ""),
            })

    result_data = {
        "accession": accession,
        "protein_name": protein_name,
        "alternative_names": alternative_names[:5],
        "gene_names": gene_names,
        "organism": organism,
        "taxonomy": taxonomy[-3:],  # last 3 levels (most specific)
        "entry_type": entry_type,
        "is_swissprot": is_swissprot,
        "ec_numbers": ec_numbers,
        "catalytic_activities": catalytic_activities[:5],
        "function": function_text[:500],
        "sequence_length": seq_length,
        "sequence": seq,
        "cross_references": {
            "pdb": xref_pdb[:10],
            "genbank_embl": xref_embl[:10],
            "interpro": xref_interpro,
            "kegg": xref_kegg[:5],
            "pfam": xref_pfam,
        },
    }

    parts = [f"{accession} ({protein_name or 'unnamed'})"]
    parts.append(f"{seq_length} aa")
    if organism:
        parts.append(organism.split("(")[0].strip())
    if ec_numbers:
        parts.append(f"EC {', '.join(ec_numbers)}")
    if is_swissprot:
        parts.append("Swiss-Prot")
    if xref_pdb:
        parts.append(f"{len(xref_pdb)} PDB")

    return {
        "status": "success",
        "summary": f"Fetched {'; '.join(parts)}",
        "data": result_data,
    }


# ---------------------------------------------------------------------------
# KEGG
# ---------------------------------------------------------------------------

def _parse_kegg_enzyme_entry(ec_number: str, text: str) -> dict:
    """Parse a single KEGG enzyme flat-file response into a dict."""
    entry = {"ec": ec_number, "names": [], "genes": [], "reactions": []}
    current_field = None
    for line in text.split("\n"):
        if line.startswith("NAME"):
            current_field = "names"
            entry["names"].append(line[12:].strip().rstrip(";"))
        elif line.startswith("REACTION"):
            current_field = "reactions"
            entry["reactions"].append(line[12:].strip())
        elif line.startswith("GENES"):
            current_field = "genes"
            entry["genes"].append(line[12:].strip())
        elif line.startswith(" ") and current_field:
            entry[current_field].append(line.strip().rstrip(";"))
        elif line.startswith("///"):
            break
        else:
            current_field = None
    return entry


@tool(
    description=(
        "Search KEGG for enzyme entries by EC number. Supports both complete EC "
        "numbers (e.g., '6.3.1.2') and partial prefixes (e.g., '6.3.1'). "
        "Returns enzyme names, genes, and reactions."
    ),
    category="database",
    parallel_safe=True,
    params={
        "ec_number": {
            "type": "string",
            "description": "EC number (complete like '6.3.1.2' or partial like '6.3.1').",
        },
    },
)
def search_kegg_enzyme(ec_number: str) -> dict:
    """Search KEGG for enzyme entries by EC number."""
    ec_number = ec_number.strip()

    if not re.match(r'^\d+(\.\d+){0,3}$', ec_number):
        return {
            "status": "error",
            "summary": f"Invalid EC number format: '{ec_number}'",
            "data": [],
            "suggestions": [
                "EC numbers should be in the format X.X.X.X (e.g., '6.3.1.2') "
                "or a partial prefix (e.g., '6.3.1' or '6.3')."
            ],
        }

    parts = ec_number.split(".")
    is_partial = len(parts) < 4

    if is_partial:
        # Use KEGG list endpoint for partial EC
        list_url = f"https://rest.kegg.jp/list/ec:{ec_number}"
        list_req = urllib.request.Request(list_url)
        try:
            list_text = request_with_retry(list_req, timeout=15).decode()
        except Exception:
            # Fallback to find endpoint
            find_url = f"https://rest.kegg.jp/find/enzyme/{ec_number}"
            find_req = urllib.request.Request(find_url)
            try:
                list_text = request_with_retry(find_req, timeout=15).decode()
            except Exception as e2:
                return {
                    "status": "error",
                    "summary": f"KEGG search failed for {ec_number}: {e2}",
                    "data": [],
                    "suggestions": ["Try a complete EC number or different prefix."],
                }

        prefix = ec_number + "."
        complete_ecs = []
        for line in list_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            first_col = line.split("\t")[0]
            full_ec = first_col.replace("ec:", "").strip()
            if full_ec and full_ec.startswith(prefix):
                complete_ecs.append(full_ec)

        if not complete_ecs:
            return {
                "status": "partial",
                "summary": f"No KEGG enzymes found matching EC {ec_number}",
                "data": [],
                "suggestions": ["Try a broader EC prefix or keyword search in UniProt."],
            }

        # Fetch details for up to 25 complete ECs in parallel
        ecs_to_fetch = complete_ecs[:config.KEGG_EC_FETCH_LIMIT]
        if len(ecs_to_fetch) == 1:
            return search_kegg_enzyme(ecs_to_fetch[0])

        all_entries = []
        with ThreadPoolExecutor(max_workers=config.KEGG_THREAD_POOL_WORKERS) as pool:
            futures = [pool.submit(_fetch_kegg_single, ec) for ec in ecs_to_fetch]
            for fut in futures:
                result = fut.result()
                if result:
                    all_entries.append(result)

        return {
            "status": "success" if all_entries else "partial",
            "summary": f"Found {len(all_entries)} KEGG entries for EC {ec_number}",
            "data": all_entries,
            "count": len(all_entries),
        }

    # Complete EC number — fetch directly
    entry = _fetch_kegg_single(ec_number)
    if entry:
        return {
            "status": "success",
            "summary": f"Found KEGG entry for EC {ec_number}: {', '.join(entry.get('names', [])[:2])}",
            "data": [entry],
            "count": 1,
        }
    return {
        "status": "partial",
        "summary": f"No KEGG entry found for EC {ec_number}",
        "data": [],
        "suggestions": ["Try a broader EC prefix or search UniProt by keyword."],
    }


def _fetch_kegg_single(ec_number: str) -> dict | None:
    """Fetch a single KEGG enzyme entry."""
    url = f"https://rest.kegg.jp/get/ec:{ec_number}"
    req = urllib.request.Request(url)
    try:
        text = request_with_retry(req, timeout=15).decode()
    except Exception as e:
        logger.warning("KEGG fetch failed for ec:%s: %s", ec_number, e)
        return None
    entry = _parse_kegg_enzyme_entry(ec_number, text)
    return entry if entry["names"] else None


# ---------------------------------------------------------------------------
# InterPro
# ---------------------------------------------------------------------------

def _interpro_fetch(urls_to_try: list[str], context: str = "InterPro") -> dict | None:
    """Try multiple InterPro URLs, return first valid JSON response."""
    for url in urls_to_try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            raw = request_with_retry(req, timeout=30)
            stripped = raw.decode().strip()
            if not stripped:
                # 204 No Content — try next URL
                continue
            if stripped[0] not in ('{', '['):
                continue
            return json.loads(stripped)
        except Exception:
            continue
    return None


def _simplify_interpro_query(query: str) -> list[str]:
    """Generate progressively simpler query variants for InterPro.

    InterPro's search returns 204 for multi-word hyphenated queries.
    Strategy: try full query, then individual significant words.
    """
    variants = [query]
    # Split into individual words, keep only significant ones (>3 chars)
    words = [w for w in query.split() if len(w) > 3
             and w.lower() not in {"with", "from", "that", "this", "have", "been"}]
    # Try each significant word individually
    for word in words[:3]:
        if word != query:
            variants.append(word)
    return variants


@tool(
    description=(
        "Search InterPro for protein domain/family entries. Returns domain accessions, "
        "names, types, and member database cross-references."
    ),
    category="database",
    parallel_safe=True,
    params={
        "query": {
            "type": "string",
            "description": "Search terms (an enzyme family / fold name, EC, or keyword).",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 20).",
            "default": 20,
        },
    },
)
def search_interpro(query: str, max_results: int = 20) -> dict:
    """Search InterPro for protein domain/family entries."""
    # InterPro returns 204 for complex multi-word queries.
    # Try progressively simpler variants.
    query_variants = _simplify_interpro_query(query)

    data = None
    tried_query = query
    for variant in query_variants:
        params_base = {
            "search": variant,
            "page_size": str(min(max_results, config.INTERPRO_PAGE_SIZE_CAP)),
        }
        url = (
            f"https://www.ebi.ac.uk/interpro/api/entry/interpro?"
            f"{urllib.parse.urlencode(params_base)}"
        )
        data = _interpro_fetch([url], context="InterPro search")
        if data is not None and data.get("results"):
            tried_query = variant
            break

    if data is None:
        return {
            "status": "error",
            "summary": "InterPro search failed after retries",
            "data": [],
            "suggestions": ["Try simpler search terms."],
        }

    results = []
    for entry in data.get("results", []):
        metadata = entry.get("metadata", {})
        results.append({
            "accession": metadata.get("accession", ""),
            "name": metadata.get("name", ""),
            "type": metadata.get("type", ""),
            "description": (metadata.get("description", [""])[0]
                            if metadata.get("description") else ""),
            "source_database": metadata.get("source_database", ""),
        })

    return {
        "status": "success" if results else "partial",
        "summary": f"Found {len(results)} InterPro entries for '{tried_query}'",
        "data": results,
        "count": len(results),
    }


@tool(
    description=(
        "Get all InterPro domain annotations for a UniProt protein. "
        "Returns domain accessions, names, types, and coordinate ranges."
    ),
    category="database",
    parallel_safe=True,
    params={
        "uniprot_accession": {
            "type": "string",
            "description": "UniProt accession ID (e.g., 'Q9X4W9').",
        },
    },
)
def get_protein_domains(uniprot_accession: str) -> dict:
    """Get all InterPro domain annotations for a given UniProt protein."""
    urls = [
        f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/"
        f"{uniprot_accession}?page_size=50",
        f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/"
        f"{uniprot_accession}",
    ]

    data = _interpro_fetch(urls, context=f"InterPro domains ({uniprot_accession})")
    if data is None:
        return {
            "status": "error",
            "summary": f"InterPro domain lookup failed for {uniprot_accession}",
            "data": [],
            "suggestions": ["Check that the UniProt accession is correct."],
        }

    results = []
    for entry in data.get("results", []):
        metadata = entry.get("metadata", {})
        coordinates = []
        for protein_match in entry.get("proteins", []):
            for location in protein_match.get("entry_protein_locations", []):
                for fragment in location.get("fragments", []):
                    coordinates.append({
                        "start": fragment.get("start"),
                        "end": fragment.get("end"),
                    })
        results.append({
            "accession": metadata.get("accession", ""),
            "name": metadata.get("name", ""),
            "type": metadata.get("type", ""),
            "coordinates": coordinates,
        })

    return {
        "status": "success" if results else "partial",
        "summary": f"Found {len(results)} domains for {uniprot_accession}",
        "data": results,
        "count": len(results),
    }


@tool(
    description=(
        "Look up UniProt entries by GenBank/EMBL accession. Use this when a "
        "paper references a specific GenBank protein accession (e.g., "
        "ATY72527.1) to find the exact UniProt entry, avoiding gene-name "
        "ambiguity. This is the most precise way to identify a protein when "
        "a sequence database accession is available."
    ),
    category="database",
    parallel_safe=True,
    params={
        "genbank_accession": {
            "type": "string",
            "description": (
                "GenBank or EMBL accession ID (e.g., 'ATY72527' or 'ATY72527.1'). "
                "Version suffix (.1) is stripped automatically."
            ),
        },
    },
)
def lookup_genbank_to_uniprot(
    genbank_accession: str,
) -> dict:
    """Look up UniProt entries by GenBank/EMBL accession."""
    acc = genbank_accession.split(".")[0].strip()
    if not acc:
        return {
            "status": "error",
            "summary": "No accession provided",
            "data": [],
            "suggestions": ["Provide a GenBank/EMBL accession ID."],
        }

    query = f"xref:embl-{acc}"
    params = {
        "query": query,
        "format": "json",
        "size": "10",
        "fields": "accession,protein_name,organism_name,ec,gene_names,cc_function,cc_catalytic_activity",
    }
    url = f"https://rest.uniprot.org/uniprotkb/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        raw = request_with_retry(req, timeout=20)
        data = json.loads(raw.decode())
    except Exception as e:
        return {
            "status": "error",
            "summary": f"UniProt xref lookup failed for {acc}: {e}",
            "data": [],
            "suggestions": ["Check that the GenBank accession is correct."],
        }

    entries = []
    for entry in data.get("results", []):
        parsed = _parse_uniprot_json(entry)
        entries.append({
            "accession": entry.get("primaryAccession", ""),
            "name": parsed["protein_name"],
            "organism": parsed["organism"],
            "ec_number": "; ".join(parsed["ec_numbers"]),
            "gene_names": parsed["gene_names"],
            "function": parsed["function"][:200],
            "catalytic_activities": parsed["catalytic_activities"][:3],
            "genbank_source": genbank_accession,
        })

    return {
        "status": "success" if entries else "partial",
        "summary": (
            f"Found {len(entries)} UniProt entries for GenBank {genbank_accession}"
            + (f" — {entries[0]['accession']} ({entries[0]['name'][:50]})" if entries else "")
        ),
        "data": entries,
        "count": len(entries),
        "suggestions": (
            [f"No UniProt entry found for GenBank {genbank_accession}. "
             "The protein may not be in UniProt yet — try NCBI Protein directly."]
            if not entries else []
        ),
    }
