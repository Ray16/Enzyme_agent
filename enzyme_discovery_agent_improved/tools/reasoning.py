"""Reasoning tools: candidate screening and multi-source evaluation.

Contains only the two @tool-decorated functions (screen_candidates,
evaluate_candidate) and their private helpers. All data models, static
data, and chemistry logic live in their own modules.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from enzyme_discovery_agent_improved import config
from enzyme_discovery_agent_improved.chemistry_logic import (
    check_bond_type_mismatch,
    classify_reaction_chemistry,
    compare_substrate_properties,
    derive_activation_mechanism,
    derive_modeling_cofactors,
    extract_target_bond_type,
    get_chebi_name,
    resolve_chebi_smiles,
)
from enzyme_discovery_agent_improved.models.candidate_registry import (
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent_improved.models.exploration_log import append_exploration_log
from enzyme_discovery_agent_improved.models.target_profile import OPPOSITE_DIRECTIONS
from enzyme_discovery_agent_improved.substrate_fit import assign_one_to_one
from enzyme_discovery_agent_improved.static.cofactors import COFACTOR_CHEBI_IDS
from enzyme_discovery_agent_improved.tool_registry import tool
from enzyme_discovery_agent_improved.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _merge_activation_evidence(existing: dict | None, activation: dict | None) -> dict | None:
    """Attach a deterministic activation record under mechanism_evidence["activation"].

    Preserves any other keys already present (e.g. literature-extracted
    "facts"/"paper") so the two provenance sources coexist. Returns ``existing``
    unchanged when there is no activation record to add.
    """
    if not activation:
        return existing
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged["activation"] = activation
    return merged

def _fetch_uniprot_entry_light(accession: str) -> dict | None:
    """Fetch minimal UniProt data for screening (single API call)."""
    from enzyme_discovery_agent_improved.tools.database import _parse_uniprot_json

    url = (
        f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        f"?fields=accession,protein_name,ec,organism_name,gene_names,cc_catalytic_activity,xref_interpro,length"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        data = json.loads(request_with_retry(req, timeout=15).decode())
    except Exception as e:
        logger.warning("Screen fetch failed for %s: %s", accession, e)
        return None

    parsed = _parse_uniprot_json(data)
    from enzyme_discovery_agent_improved.models.claim_validation import validate_uniprot_claims
    identity_validation = validate_uniprot_claims(
        data, expected_accession=accession, require_native_reaction=False,
    )
    # InterPro fold ids let the screening step group representatives by fold
    # family for relevance budgeting without a second lookup.
    interpro_ids = [
        x.get("id", "") for x in data.get("uniProtKBCrossReferences", [])
        if x.get("database") == "InterPro" and x.get("id")
    ]
    return {
        "accession": accession,
        "protein_name": parsed["protein_name"],
        "ec_numbers": parsed["ec_numbers"],
        "catalytic_activities": parsed["catalytic_activities"][:3],
        "function": parsed["function"][:300],
        "organism": parsed["organism"],
        "gene_names": parsed["gene_names"],
        "interpro_ids": interpro_ids,
        "seq_length": data.get("sequence", {}).get("length", 0),
        "claim_validation": identity_validation,
    }


def _target_direction_for_run(run_dir: Path) -> str | None:
    """Read the declared target direction (delegates to the single target-profile
    reader so the file/parse logic lives in one place)."""
    from enzyme_discovery_agent_improved.tools.family_reactions import _target_profile_for_run
    return _target_profile_for_run(run_dir)[0]


def _target_claims_for_run(run_dir: Path) -> tuple[str | None, set[str], str | None]:
    """Return the normalized target claims used for authoritative validation."""
    from enzyme_discovery_agent_improved.tools.family_reactions import (
        _target_profile_for_run,
        _target_topology_for_run,
    )
    direction, bonds = _target_profile_for_run(run_dir)
    return direction, bonds, _target_topology_for_run(run_dir)


def _fetch_uniprot_entry(accession: str) -> dict | None:
    """Fetch full UniProt JSON entry."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        return json.loads(request_with_retry(req, timeout=20).decode())
    except Exception as e:
        logger.warning("Failed to fetch UniProt entry for %s: %s", accession, e)
        return None


def _check_pdb_availability(accession: str, entry: dict | None = None) -> dict:
    """Check PDB and AlphaFold availability for a UniProt accession."""
    pdb_ids = []
    has_alphafold = False

    source = entry
    if source is None:
        source = _fetch_uniprot_entry(accession) or {}

    for xref in source.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "PDB":
            pdb_ids.append(xref.get("id", ""))

    af_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    req = urllib.request.Request(af_url, headers={"Accept": "application/json"})
    try:
        request_with_retry(req, timeout=10)
        has_alphafold = True
    except Exception:
        pass

    return {
        "pdb_ids": pdb_ids,
        "has_alphafold": has_alphafold,
        "has_structure": bool(pdb_ids) or has_alphafold,
    }


def _check_bacterial_expression(accession: str, entry: dict, pdb_ids: list[str]) -> dict:
    """Check for evidence of expression in bacteria, prioritising E. coli."""
    result = {
        "ecoli_expression": False,
        "bacterial_expression": False,
        "is_eukaryote": False,
        "expression_evidence": [],
        "expression_host": None,
    }

    # 1. Native organism is E. coli or another bacterium
    organism = entry.get("organism", {}).get("scientificName", "")
    # UniProt lineage is a list of strings (taxon names)
    lineage = entry.get("organism", {}).get("lineage", [])
    if isinstance(lineage, list) and lineage and isinstance(lineage[0], dict):
        lineage = [t.get("scientificName", "") for t in lineage]
    is_bacterial = "Bacteria" in lineage
    is_eukaryote = "Eukaryota" in lineage

    if "Escherichia coli" in organism:
        result["ecoli_expression"] = True
        result["bacterial_expression"] = True
        result["expression_evidence"].append("Native E. coli enzyme")
        result["expression_host"] = "Escherichia coli (native)"
        return result
    result["is_eukaryote"] = is_eukaryote
    if is_bacterial:
        result["bacterial_expression"] = True
        result["expression_evidence"].append(f"Native bacterium: {organism}")

    # 2. UniProt comments mentioning recombinant expression
    for comment in entry.get("comments", []):
        text = json.dumps(comment)
        ctype = comment.get("commentType", "")
        if ctype in ("BIOTECHNOLOGY", "MISCELLANEOUS", "DISRUPTION PHENOTYPE"):
            if "Escherichia coli" in text or "E. coli" in text:
                result["ecoli_expression"] = True
                result["bacterial_expression"] = True
                result["expression_evidence"].append(f"UniProt {ctype} mentions E. coli expression")
                result["expression_host"] = "Escherichia coli (UniProt annotation)"

    # 3. Check PDB expression systems via RCSB API (check first 2 structures max)
    for pdb_id in pdb_ids[:2]:
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            data = json.loads(request_with_retry(req, timeout=10).decode())
            data_str = json.dumps(data)
            if "Escherichia coli" in data_str:
                result["ecoli_expression"] = True
                result["bacterial_expression"] = True
                result["expression_evidence"].append(
                    f"PDB {pdb_id} expressed in E. coli"
                )
                if not result["expression_host"]:
                    result["expression_host"] = f"Escherichia coli (PDB {pdb_id})"
                break
            elif any(b in data_str for b in [
                "Bacillus", "Lactococcus", "Streptomyces", "Corynebacterium"
            ]):
                result["bacterial_expression"] = True
                result["expression_evidence"].append(
                    f"PDB {pdb_id} expressed in bacteria"
                )
        except Exception as e:
            logger.debug("RCSB expression check failed for %s: %s", pdb_id, e)

    return result


def _extract_mutagenesis(entry: dict) -> dict:
    """Extract mutagenesis/variant data from a UniProt entry."""
    mutations = []
    pmids: set[str] = set()

    for feat in entry.get("features", []):
        if feat.get("type") != "Mutagenesis":
            continue

        loc = feat.get("location", {})
        position = loc.get("start", {}).get("value", "")
        alt_seq = feat.get("alternativeSequence", {})
        original = alt_seq.get("originalSequence", "") if isinstance(alt_seq, dict) else ""
        alternatives = alt_seq.get("alternativeSequences", []) if isinstance(alt_seq, dict) else []
        description = feat.get("description", "")

        for ev in feat.get("evidences", []):
            if ev.get("source") == "PubMed":
                pmids.add(ev["id"])

        if original and alternatives:
            mutation_str = f"{original}{position}{alternatives[0]}"
        elif position:
            mutation_str = f"pos{position}"
        else:
            mutation_str = "unknown"

        mutations.append({
            "mutation": mutation_str,
            "effect": description[:150],
        })

    return {
        "n_mutations": len(mutations),
        "n_references": len(pmids),
        "pmids": sorted(pmids),
        "mutations": mutations[:10],
    }


def _count_pubmed_for_term(
    search_term: str,
    organism: str = "",
) -> tuple[int, int]:
    """Count PubMed characterization and engineering papers for a term."""
    from enzyme_discovery_agent_improved.tools.literature import _pubmed_esearch_count

    org_filter = ""
    if len(search_term) <= 4 and organism:
        genus = organism.split()[0] if organism else ""
        if genus and len(genus) > 3:
            org_filter = f' AND "{genus}"[Title/Abstract]'

    char_query = (
        f'("{search_term}"[Title/Abstract]){org_filter} AND '
        f'(characterization OR "substrate specificity" OR "crystal structure" '
        f'OR "kinetic" OR "catalytic mechanism")'
    )
    char_count = _pubmed_esearch_count(char_query)

    eng_query = (
        f'("{search_term}"[Title/Abstract]){org_filter} AND '
        f'(engineering OR "directed evolution" OR "rational design" '
        f'OR "site-directed mutagenesis" OR "protein engineering")'
    )
    eng_count = _pubmed_esearch_count(eng_query)

    return char_count, eng_count


def _count_pubmed_characterization(
    protein_name: str,
    accession: str,
    gene_names: list[str] | None = None,
    alternative_names: list[str] | None = None,
    organism: str = "",
) -> dict:
    """Count PubMed papers on characterization/engineering for an enzyme."""
    specific_terms = []
    generic_terms = []

    if gene_names:
        for gn in gene_names:
            if gn and gn not in specific_terms:
                specific_terms.append(gn)
    if accession:
        specific_terms.append(accession)
    if alternative_names:
        for an in alternative_names:
            if an and an not in specific_terms and an not in generic_terms:
                if len(an.split()) <= 2:
                    specific_terms.append(an)
                else:
                    generic_terms.append(an)
    if protein_name:
        if len(protein_name.split()) <= 2:
            specific_terms.append(protein_name)
        else:
            generic_terms.append(protein_name)

    search_terms = specific_terms + generic_terms
    if not search_terms:
        search_terms = [accession]

    best_char = 0
    best_eng = 0
    best_term = search_terms[0]
    used_generic = False

    ew = config.ENGINEERING_PAPER_WEIGHT
    for term in specific_terms[:3]:
        char_count, eng_count = _count_pubmed_for_term(term, organism=organism)
        total = char_count + eng_count * ew
        if total > best_char + best_eng * ew:
            best_char = char_count
            best_eng = eng_count
            best_term = term

    if best_char + best_eng == 0:
        for term in generic_terms[:2]:
            char_count, eng_count = _count_pubmed_for_term(term, organism=organism)
            total = char_count + eng_count * ew
            if total > best_char + best_eng * ew:
                best_char = char_count
                best_eng = eng_count
                best_term = term
                used_generic = True

    return {
        "characterization_papers": best_char,
        "engineering_papers": best_eng,
        "search_term_used": best_term,
        "used_generic_term": used_generic,
    }


# --------------------------------------------------------------------------
# Tool: screen_candidates
# --------------------------------------------------------------------------

@tool(
    description=(
        "Quick screen of a UniProt accession — returns EC number, protein name, "
        "catalytic activities, and function in a single API call. Use this to "
        "pre-filter candidates BEFORE calling the expensive evaluate_candidate. "
        "Accepts a list of up to 10 accessions for batch screening."
    ),
    category="reasoning",
    parallel_safe=True,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "accessions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of UniProt accession IDs to screen (max 10).",
        },
    },
)
def screen_candidates(run_dir: Path | str, accessions: list[str]) -> dict:
    """Lightweight batch screening — single UniProt fetch per accession."""
    run_dir = Path(run_dir)
    target_direction = _target_direction_for_run(run_dir)
    results = []
    for acc in accessions[:config.SCREENING_BATCH_SIZE]:
        entry = _fetch_uniprot_entry_light(acc)
        if entry:
            identity_validation = entry.get("claim_validation") or {}
            if identity_validation and identity_validation.get("promotion_eligible") is not True:
                reason = (
                    "Rejected at screening: authoritative record does not support "
                    "the asserted active accession identity (blocking claims: "
                    + ", ".join(identity_validation.get("blocking_claims") or ["unknown"])
                    + ")."
                )
                entry["screening_decision"] = "rejected"
                entry["screening_reason"] = reason
                results.append(entry)
                update_candidate_registry(run_dir, acc, "rejected", metadata={
                    "claim_validation": identity_validation,
                    "rejection_reason": reason,
                })
                continue
            chemistry = classify_reaction_chemistry(
                entry.get("ec_numbers", []), entry.get("catalytic_activities", [])
            )
            native_direction = chemistry.get("reaction_direction")
            if (native_direction, target_direction) in OPPOSITE_DIRECTIONS:
                reason = (
                    f"Rejected at screening: native reaction direction is {native_direction}, "
                    f"but the declared target requires {target_direction}."
                )
                entry["screening_decision"] = "rejected"
                entry["screening_reason"] = reason
                results.append(entry)
                append_exploration_log(
                    run_dir, entry_type="REJECT", title=f"{acc} direction mismatch",
                    details={"Reason": reason},
                )
                update_candidate_registry(run_dir, acc, "rejected", metadata={
                    "protein_name": entry.get("protein_name", ""),
                    "organism": entry.get("organism", ""),
                    "ec_numbers": entry.get("ec_numbers", []),
                    "catalytic_activities": entry.get("catalytic_activities", []),
                    "gene_names": entry.get("gene_names", []),
                    "rejection_reason": reason,
                })
                continue
            results.append(entry)
            append_exploration_log(
                run_dir,
                entry_type="SCREEN",
                title=f"{acc} ({entry.get('protein_name', 'unnamed')}, {entry.get('organism', '')})",
                details={
                    "EC": ", ".join(entry.get("ec_numbers", [])) or "none",
                    "Function": entry.get("function", "")[:200] or "N/A",
                },
            )
            update_candidate_registry(run_dir, acc, "screened", metadata={
                "protein_name": entry.get("protein_name", ""),
                "organism": entry.get("organism", ""),
                "ec_numbers": entry.get("ec_numbers", []),
                "catalytic_activities": entry.get("catalytic_activities", []),
                "gene_names": entry.get("gene_names", []),
                "function": entry.get("function", ""),
                "seq_length": entry.get("seq_length", 0),
                # InterPro fold ids fetched free with the light screen call; kept
                # so the registry can group by fold before evaluation resolves the
                # single catalytic family.
                "interpro_ids": entry.get("interpro_ids", []),
                "claim_validation": identity_validation,
            })
        else:
            results.append({
                "accession": acc,
                "error": f"Could not fetch {acc}",
            })

    # Candidate evidence must be inspectable while discovery is still in
    # progress, not only after evaluation/finalization.  Keep one durable
    # screen record per accession alongside the agent-level tool artifact.
    candidate_dir = run_dir / "artifacts" / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    registry = load_candidate_registry(run_dir).get("candidates", {})
    for result in results:
        accession = result.get("accession", "")
        if not accession:
            continue
        (candidate_dir / f"screen_{accession}.json").write_text(json.dumps({
            "tool_name": "screen_candidates",
            "accession": accession,
            "target_direction": target_direction,
            "result": result,
            "registry_entry": registry.get(accession, {}),
        }, indent=2, default=str) + "\n")

    return {
        "status": "success",
        "summary": f"Screened {len(results)} candidates",
        "data": results,
        "count": len(results),
    }


# --------------------------------------------------------------------------
# Substrate fallback lookup (Rhea → KEGG)
# --------------------------------------------------------------------------

def _fallback_substrate_lookup(
    rhea_ids: list[str],
) -> list[dict]:
    """Resolve native substrate SMILES from the enzyme's OWN Rhea reaction IDs.

    Rhea IDs come from the enzyme's UniProt catalytic-activity annotation, so the
    participants are THIS enzyme's actual substrates.  We deliberately do NOT
    fall back to EC->reaction (KEGG/BRENDA): an EC number is a reaction CLASS, so
    its archetype substrate is often not this enzyme's substrate — a plausible-
    but-wrong value would silently corrupt the substrate-accommodation score.
    The reported-but-reliable fallback is the characterization paper, handled by
    the caller.
    """
    results: list[dict] = []

    # Rhea: fetch compound participants from Rhea reaction IDs
    for rhea_id in rhea_ids[:3]:
        try:
            # Use undirected Rhea ID (always even; Rhea IDs come in sets of 4)
            rid = rhea_id.replace("RHEA:", "").strip()
            url = f"https://www.rhea-db.org/rhea/{rid}.json"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            data = json.loads(request_with_retry(req, timeout=15).decode())
            participants = data.get("reactionParticipants", [])
            seen = set()
            for p in participants:
                compound = p.get("compound", {})
                chebi_id = compound.get("chebiId", "")
                name = compound.get("name", "")
                if not chebi_id or chebi_id in seen:
                    continue
                seen.add(chebi_id)
                full_id = f"CHEBI:{chebi_id}" if not chebi_id.startswith("CHEBI:") else chebi_id
                if full_id in COFACTOR_CHEBI_IDS:
                    continue
                smi = resolve_chebi_smiles(full_id)
                if smi:
                    results.append({"chebi_id": full_id, "name": name, "smiles": smi,
                                    "source": "rhea_fallback"})
        except Exception as e:
            logger.debug("Rhea fallback failed for %s: %s", rhea_id, e)

    return results


_COFACTOR_NAMES = frozenset({
    "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph", "coa", "fad", "fadh2",
    "fmn", "gtp", "gdp", "ump", "ctp", "h2o", "water", "h+", "proton",
    "diphosphate", "pyrophosphate", "phosphate", "co2", "oxygen", "o2",
    "ppi", "pi", "h2o2", "hydrogen peroxide", "ammonia", "nh3",
})


def _is_cofactor_name(name: str) -> bool:
    return name.strip().lower() in _COFACTOR_NAMES


def _chebi_smiles_by_name(name: str) -> str | None:
    """Search ChEBI by compound name and return SMILES if found."""
    if _is_cofactor_name(name):
        return None
    try:
        encoded = urllib.parse.quote(name)
        ols_url = (f"https://www.ebi.ac.uk/ols4/api/search"
                   f"?q={encoded}&ontology=chebi&rows=3&exact=false")
        req = urllib.request.Request(ols_url, headers={"Accept": "application/json"})
        data = json.loads(request_with_retry(req, timeout=10).decode())
        docs = data.get("response", {}).get("docs", [])
        if docs:
            obo_id = docs[0].get("obo_id", "")
            if obo_id.startswith("CHEBI:"):
                return resolve_chebi_smiles(obo_id)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Tool: evaluate_candidate
# --------------------------------------------------------------------------

# Generic non-enzyme domain classes.  This list is deliberately ANSWER-AGNOSTIC:
# it names regulatory / transport / DNA-binding / structural roles and NEVER any
# enzyme family, so it cannot bias selection toward the target's mechanism (which
# would be leakage — cf. the leakage firewall).  "Enzyme unless positively a
# non-enzyme" is also conservative: it can only ever drop a proven non-enzyme,
# never a relevant enzyme.  Matched against InterPro/Pfam DOMAIN names only, not
# the protein name (which carries pathway context like "... biosynthesis
# transcriptional regulator" that would misclassify a regulator).
_NON_ENZYME_DOMAIN_TOKENS = (
    "regulat", "transcription", "dna-binding", "dna binding", "helix-turn-helix",
    " hth", "winged helix", "homeodomain", "receptor", "transporter",
    "membrane fusion", "multidrug", "efflux", "channel", "porin",
    "two-component", "sigma factor", "repressor", "fusion protein",
)


def _catalytic_family(domain_architecture, ec_numbers, catalytic_activities):
    """Resolve the enzyme FAMILY from the catalytic domain, and flag non-enzymes.

    Returns ``(is_enzyme, family_name, family_id)``.  The family is taken from a
    catalytic InterPro/Pfam domain, never the first domain — which for a
    multidomain enzyme is often a fold or accessory motif (so asnB / murC land
    under the wrong entry), and for co-cluster contaminants (Fur-type regulators,
    membrane-fusion transporters) is the only domain, so those resolve to no
    enzyme family and are filtered out rather than polluting the family view.

    An entry is treated as an enzyme unless it is POSITIVELY a non-enzyme: a
    curated EC number or catalytic activity makes it an enzyme outright;
    otherwise it is an enzyme as long as it has any family/domain that is not a
    known non-enzyme class.  No positive "enzyme vocabulary" is used, so the
    classification stays answer-agnostic.
    """
    doms = [d for d in (domain_architecture or []) if isinstance(d, dict)]

    def _non_enzyme_dom(dm):
        name = (dm.get("name") or "").lower()
        return any(tok in name for tok in _NON_ENZYME_DOMAIN_TOKENS)

    enzyme_doms = [
        d for d in doms
        if not _non_enzyme_dom(d) and d.get("type") in ("family", "domain")
    ]
    is_enzyme = bool(ec_numbers or catalytic_activities or enzyme_doms)
    # Prefer a family-type enzyme domain (most specific), then any enzyme domain.
    fam = next((d for d in enzyme_doms if d.get("type") == "family"), None) \
        or (enzyme_doms[0] if enzyme_doms else None)
    if fam is None:
        # EC/activity-positive but no non-regulatory domain: group by any
        # family-type domain if one exists, else the first domain.
        fam = next((d for d in doms if d.get("type") == "family"), None) \
            or (doms[0] if doms else None)
    if fam is None:
        return is_enzyme, "", ""
    return is_enzyme, fam.get("name", ""), fam.get("accession", "")


@tool(
    description=(
        "Evaluate an enzyme candidate by gathering data from UniProt, InterPro, "
        "PubMed, PDB/AlphaFold, and ChEBI. Returns structured data about annotation "
        "quality, catalytic activities, domain architecture, literature support, "
        "structural availability, and substrate similarity. The LLM interprets this "
        "data and decides the ranking — no composite score is computed."
    ),
    category="reasoning",
    parallel_safe=True,
    injected_params=["run_dir", "llm_client"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "accession": {
            "type": "string",
            "description": "UniProt accession ID to evaluate.",
        },
        "target_ec_class": {
            "type": "string",
            "description": "Target EC class for relevance (e.g., '6.3.2').",
            "default": "",
        },
        "target_reaction_type": {
            "type": "string",
            "description": "Description of the target reaction type.",
            "default": "",
        },
        "target_substrate_description": {
            "type": "string",
            "description": "Description of target substrates.",
            "default": "",
        },
        "target_substrate_smiles": {
            "type": "string",
            "description": "Target substrate SMILES for fingerprint comparison. Use dot-separated for multiple substrates.",
            "default": "",
        },
    },
)
def evaluate_candidate(
    run_dir: Path | str,
    accession: str,
    target_ec_class: str = "",
    target_reaction_type: str = "",
    target_substrate_description: str = "",
    target_substrate_smiles: str = "",
    llm_client=None,
) -> dict:
    """Evaluate an enzyme candidate using multiple data sources."""
    run_dir = Path(run_dir)

    # A direction mismatch is a hard screening decision.  Without this guard,
    # a later direct tool call could silently revive a rejected candidate and
    # allow it to contaminate the final ranking.
    from enzyme_discovery_agent_improved.models.candidate_registry import (
        current_candidate_status,
        load_candidate_registry,
    )
    existing = load_candidate_registry(run_dir).get("candidates", {}).get(accession, {})
    if current_candidate_status(existing) == "rejected":
        reason = existing.get("metadata", {}).get("rejection_reason", "screening rejection")
        return {
            "status": "error",
            "summary": f"Evaluation skipped for {accession}: {reason}",
            "data": None,
            "suggestions": ["Resolve new, contradictory primary evidence before registering a new candidate."],
        }

    entry = _fetch_uniprot_entry(accession)
    if not entry:
        return {
            "status": "error",
            "summary": f"Could not fetch UniProt entry for {accession}",
            "data": None,
            "suggestions": ["Check that the accession ID is correct."],
        }

    from enzyme_discovery_agent_improved.models.claim_validation import validate_uniprot_claims
    identity_validation = validate_uniprot_claims(
        entry, expected_accession=accession, require_native_reaction=False,
    )
    if identity_validation.get("promotion_eligible") is not True:
        reason = ", ".join(identity_validation.get("blocking_claims") or ["unknown"])
        update_candidate_registry(run_dir, accession, "rejected", metadata={
            "claim_validation": identity_validation,
            "rejection_reason": f"authoritative identity validation failed: {reason}",
        })
        return {
            "status": "error",
            "summary": f"Evaluation skipped for {accession}: authoritative identity validation failed ({reason})",
            "data": None,
            "suggestions": ["Resolve the candidate to an active authoritative accession."],
        }

    # Extract basic info
    pd = entry.get("proteinDescription", {})
    rec = pd.get("recommendedName", {})
    protein_name = rec.get("fullName", {}).get("value", "") if rec else ""
    if not protein_name:
        for sub in pd.get("submissionNames", []):
            pn = sub.get("fullName", {}).get("value", "")
            if pn:
                protein_name = pn
                break

    organism = entry.get("organism", {}).get("scientificName", "")
    seq_length = entry.get("sequence", {}).get("length", 0)
    entry_type = entry.get("entryType", "")
    is_swissprot = "Swiss-Prot" in entry_type

    gene_names = []
    for gene in entry.get("genes", []):
        gn = gene.get("geneName", {}).get("value", "")
        if gn:
            gene_names.append(gn)
        for syn in gene.get("synonyms", []):
            sv = syn.get("value", "")
            if sv and sv not in gene_names:
                gene_names.append(sv)

    alternative_names = []
    for alt in pd.get("alternativeNames", []):
        an = alt.get("fullName", {}).get("value", "")
        if an:
            alternative_names.append(an)

    ec_numbers = []
    for ec in rec.get("ecNumbers", []):
        val = ec.get("value", "")
        if val:
            ec_numbers.append(val)
    for alt in pd.get("alternativeNames", []):
        for ec in alt.get("ecNumbers", []):
            val = ec.get("value", "")
            if val and val not in ec_numbers:
                ec_numbers.append(val)
    for sub in pd.get("submissionNames", []):
        for ec in sub.get("ecNumbers", []):
            val = ec.get("value", "")
            if val and val not in ec_numbers:
                ec_numbers.append(val)

    catalytic_activities = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            name = reaction.get("name", "")
            if name:
                catalytic_activities.append(name)

    function_comments = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            for t in comment.get("texts", []):
                function_comments.append(t.get("value", ""))

    required_cofactors = []
    cofactor_notes = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "COFACTOR":
            for cof in comment.get("cofactors", []):
                name = cof.get("name", "")
                if name:
                    required_cofactors.append(name)
            for text in (comment.get("note", {}) or {}).get("texts", []):
                value = text.get("value", "")
                if value:
                    cofactor_notes.append(value)

    # Concurrent data gathering
    from enzyme_discovery_agent_improved.tools.database import get_protein_domains

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_domains = pool.submit(get_protein_domains, uniprot_accession=accession)
        fut_literature = pool.submit(
            _count_pubmed_characterization,
            protein_name, accession,
            gene_names=gene_names,
            alternative_names=alternative_names,
            organism=organism,
        )
        fut_structural = pool.submit(_check_pdb_availability, accession, entry)

        domain_result = fut_domains.result()
        literature = fut_literature.result()
        structural_data = fut_structural.result()

        pdb_ids = structural_data.get("pdb_ids", [])
        fut_expression = pool.submit(_check_bacterial_expression, accession, entry, pdb_ids)
        expression_data = fut_expression.result()

    domain_architecture = domain_result.get("data", []) if isinstance(domain_result, dict) else []

    # Native substrate info
    native_substrates_info: list[dict] = []
    rhea_ids: list[str] = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "CATALYTIC ACTIVITY":
            for xref in (comment.get("reaction", {})).get("reactionCrossReferences", []):
                if xref.get("database") == "Rhea" and xref.get("id"):
                    if xref["id"] not in rhea_ids:
                        rhea_ids.append(xref["id"])
    # Native SUBSTRATES from catalytic activity — REACTANT side only.  A product's
    # reactive groups would spuriously match a target substrate and mask that the
    # enzyme's true partner for it is cofactor-scale; ranking's substrate signal
    # (_accommodation_score) must judge what the enzyme CONSUMES.  Shared with the
    # family-survey resolution so promotion and ranking see the same substrates.
    from enzyme_discovery_agent_improved.tools.family_reactions import _reactant_side_chebi_ids

    def _name_of(cid):
        resolve_chebi_smiles(cid)  # populates the name cache
        return get_chebi_name(cid)

    for xid in _reactant_side_chebi_ids(entry, _name_of):
        native_substrates_info.append({
            "chebi_id": xid,
            "name": get_chebi_name(xid),
            "smiles": resolve_chebi_smiles(xid),
        })

    # Substrate-source cascade — measured or reported ONLY, never inferred.
    # 1) UniProt catalyticActivity -> ChEBI -> SMILES (above; most reliable).
    # 2) The enzyme's own Rhea reaction IDs -> ChEBI -> SMILES (still measured).
    literature_name = ""
    if not native_substrates_info:
        native_substrates_info = _fallback_substrate_lookup(rhea_ids)

    # 3) Characterization paper (reported): for sparse entries (often TrEMBL)
    # with no structured reaction, read THIS enzyme's paper and take the native
    # substrates it reports, plus the name the paper gives the enzyme.  No
    # EC-archetype or homology guessing feeds the substrate axis.
    if not native_substrates_info and llm_client is not None:
        try:
            from enzyme_discovery_agent_improved.tools.literature_extraction import (
                extract_substrates_and_mechanism,
            )
            source_tool = existing.get("metadata", {}).get("source_tool", "")
            lit = extract_substrates_and_mechanism(
                run_dir, accession, source_tool, protein_name, llm_client,
                query=target_reaction_type, organism=organism,
            )
            if lit and lit.get("paper_relevant"):
                for ns in lit.get("native_substrates") or []:
                    if ns.get("smiles"):
                        native_substrates_info.append({
                            "chebi_id": "", "name": ns.get("name", ""),
                            "smiles": ns["smiles"], "source": "literature",
                        })
                literature_name = (lit.get("enzyme_name_in_paper") or "").strip()
        except Exception as e:  # never let the fallback block evaluation
            logger.debug("literature substrate fallback failed for %s: %s",
                         accession, e)

    # Substrate similarity
    substrate_similarity = {}
    if target_substrate_smiles:
        from enzyme_discovery_agent_improved.tools.chemistry import compute_fingerprint_similarity
        from enzyme_discovery_agent_improved.tools.substrate import compute_substrate_properties

        native_smiles_list = [
            ns["smiles"] for ns in native_substrates_info if ns.get("smiles")
        ]
        target_molecules = [s.strip() for s in target_substrate_smiles.split(".")
                            if s.strip()]

        if native_smiles_list and target_molecules:
            # ONE-TO-ONE correspondence (shared with the promotion path via
            # assign_one_to_one): each native substrate serves AT MOST ONE target
            # substrate, so a two-partner target cannot flatter itself by matching
            # both partners onto its single best native.  Fingerprint similarity is
            # the per-pair score; it already carries the role signal (an amine target
            # is most similar to an amine native), so the assignment pairs amine↔amine
            # and carboxyl↔carboxyl and forces the target's amine partner onto the
            # family's amine partner even when that is a tiny FG-less donor (ammonia),
            # which then scores as the poor fit it is.  Answer-agnostic throughout.
            _fp_cache: dict = {}

            def _fp(t_smi, n_smi):
                key = (t_smi, n_smi)
                if key not in _fp_cache:
                    _fp_cache[key] = compute_fingerprint_similarity(
                        t_smi, n_smi).get("data", {}).get("similarity", 0.0)
                return _fp_cache[key]

            per_target_best = []
            match_details = []
            property_comparisons = []
            for t_smi, n_smi, sim in assign_one_to_one(
                    target_molecules, native_smiles_list, _fp):
                per_target_best.append(sim)
                match_details.append(
                    f"{t_smi[:40]}->{(n_smi or 'none')[:40]}: {sim:.3f}")
                if n_smi:
                    prop = compare_substrate_properties(
                        t_smi, n_smi, compute_substrate_properties)
                    if prop:
                        property_comparisons.append(prop)

            # The enzyme must accommodate EVERY target substrate, so the substrate
            # score is the WORST assigned pair (the bottleneck), not a geometric
            # mean.  A mean lets a candidate that fits ONE target substrate well but
            # the other poorly (e.g. 0.29 vs 0.09) score deceptively high (0.16) and
            # sneak above a balanced both-substrate fit.  The worst substrate is the
            # honest limit on whether the pocket can host the whole reaction.
            base_score = min(per_target_best) if per_target_best else 0.0

            substrate_similarity = {
                "tanimoto_score": round(base_score, 3),
                "n_native_smiles": len(native_smiles_list),
                "n_target_molecules": len(target_molecules),
                "per_target_best": [round(s, 3) for s in per_target_best],
                "match_details": match_details,
                "property_comparisons": property_comparisons,
            }
            # Role-aware accommodation over the REACTANT-side substrates: which
            # target substrate (if any) has no genuine native partner.  Ranking
            # uses this to DEMOTE a candidate that cannot host every target
            # substrate, rather than scoring an unresolvable partner as neutral.
            from enzyme_discovery_agent_improved.substrate_accommodation import accommodation_facts
            substrate_similarity["accommodation"] = accommodation_facts(
                target_molecules, native_smiles_list)
            if len(property_comparisons) < len(target_molecules):
                substrate_similarity["property_note"] = (
                    "Property comparison unavailable for some substrates "
                    "-- interpret Tanimoto with caution"
                )
        else:
            substrate_similarity = {
                "tanimoto_score": None,
                "n_native_smiles": len(native_smiles_list) if target_substrate_smiles else 0,
                "note": "No native substrate SMILES resolved from catalytic activities" if not native_smiles_list else "No target SMILES provided",
            }

    # EC match analysis
    ec_match = {}
    if target_ec_class and ec_numbers:
        target_parts = target_ec_class.rstrip(".").split(".")
        best_match = 0
        best_ec = ""
        for ec in ec_numbers:
            ec_parts = ec.split(".")
            match_depth = 0
            for i, tp in enumerate(target_parts):
                if i < len(ec_parts) and ec_parts[i] == tp:
                    match_depth += 1
                else:
                    break
            if match_depth > best_match:
                best_match = match_depth
                best_ec = ec
        ec_match = {
            "best_ec": best_ec,
            "match_depth": best_match,
            "target_ec": target_ec_class,
            "total_levels": len(target_parts),
        }

    mutagenesis_data = _extract_mutagenesis(entry)

    # InterPro family, resolved from the CATALYTIC domain (not the first domain),
    # with a non-enzyme flag so regulators/transporters that co-occur in the same
    # gene clusters are filtered rather than grouped as spurious families.
    is_enzyme, interpro_family, interpro_family_id = _catalytic_family(
        domain_architecture, ec_numbers, catalytic_activities
    )

    # Identity validity is necessary but says nothing about function.  Build a
    # second, target-specific bundle from the authoritative catalytic-activity
    # fields and require every applicable functional claim to be supported.
    target_direction, target_bonds, target_topology = _target_claims_for_run(run_dir)
    function_validation = validate_uniprot_claims(
        entry,
        expected_accession=accession,
        expected_fold_id=interpro_family_id or "",
        target_direction=target_direction,
        target_bond_types=target_bonds,
        target_topology=target_topology,
        require_native_reaction=True,
    )

    # Deterministic activation-chemistry annotation from the candidate's own
    # native reaction products. Nested under mechanism_evidence["activation"] so
    # it coexists with any literature-extracted mechanism facts (which live under
    # "facts"/"paper") without either clobbering the other.
    existing_mech = existing.get("metadata", {}).get("mechanism_evidence")
    activation = derive_activation_mechanism(catalytic_activities)
    mechanism_evidence = _merge_activation_evidence(existing_mech, activation)

    # Deterministic co-fold input spec: which nucleotide + metal(s) to include,
    # derived from the candidate's own reaction + cofactor annotation, plus the
    # activation phosphorus for scoring. None for a non-NTP reaction.
    cofactor_spec = derive_modeling_cofactors(
        catalytic_activities, required_cofactors, cofactor_notes)
    modeling_input = None
    if cofactor_spec:
        modeling_input = dict(cofactor_spec)
        if activation:
            modeling_input["reactive_phosphate"] = activation["reactive_phosphate"]
            modeling_input["activation_mode"] = activation["activation_mode"]
        # Whether the adipate near-attack frame even applies to this enzyme's
        # native reaction (a carboxylic-acid + amine ligation) or not (formate/
        # bicarbonate/carbamoyl donors) -- so downstream cofolding/scoring can skip
        # out-of-frame enzymes instead of measuring a meaningless attack angle.
        modeling_input["ligation_frame"] = classify_ligation_frame(catalytic_activities)

    # Build evaluation result
    eval_data = {
        "accession": accession,
        "is_enzyme": is_enzyme,
        "protein_name": protein_name,
        "gene_names": gene_names,
        "alternative_names": alternative_names,
        "organism": organism,
        "sequence_length": seq_length,
        "is_swissprot": is_swissprot,
        "ec_numbers": ec_numbers,
        "ec_match": ec_match,
        "catalytic_activities": catalytic_activities,
        "function_comments": function_comments[:3],
        "required_cofactors": required_cofactors,
        "domain_architecture": domain_architecture,
        "interpro_family": interpro_family,
        "interpro_family_id": interpro_family_id,
        "literature": literature,
        "structural_data": structural_data,
        "expression_data": expression_data,
        "substrate_similarity": substrate_similarity,
        "mutagenesis": mutagenesis_data,
        "native_substrates": [
            ns for ns in native_substrates_info if ns.get("smiles")
        ],
        "rhea_ids": rhea_ids,
        "identity_validation": identity_validation,
        "claim_validation": function_validation,
        "mechanism_evidence": mechanism_evidence,
        "modeling_input": modeling_input,
    }

    # Strengths and weaknesses
    strengths = []
    weaknesses = []

    if is_swissprot:
        strengths.append("Swiss-Prot reviewed entry")
    if ec_match.get("match_depth", 0) >= config.EC_MATCH_DEPTH_STRONG:
        strengths.append(f"Strong EC match: {ec_match.get('best_ec')}")
    elif ec_match.get("match_depth", 0) >= config.EC_MATCH_DEPTH_PARTIAL:
        strengths.append(f"Partial EC match: {ec_match.get('best_ec')}")
    if literature["characterization_papers"] >= config.WELL_CHARACTERIZED_THRESHOLD:
        strengths.append(f"Well-characterized ({literature['characterization_papers']} papers)")
    if literature["engineering_papers"] >= config.ENGINEERING_PRECEDENT_THRESHOLD:
        strengths.append(f"Engineering precedent ({literature['engineering_papers']} papers)")
    if expression_data.get("ecoli_expression"):
        host = expression_data.get("expression_host", "E. coli")
        strengths.append(f"E. coli expression demonstrated ({host})")
    elif expression_data.get("bacterial_expression"):
        evid = expression_data.get("expression_evidence", ["bacteria"])[0]
        strengths.append(f"Bacterial expression demonstrated ({evid})")
    elif not expression_data.get("bacterial_expression"):
        lineage = entry.get("organism", {}).get("lineage", [])
        if isinstance(lineage, list) and lineage and isinstance(lineage[0], dict):
            lineage = [t.get("scientificName", "") for t in lineage]
        if "Eukaryota" in lineage:
            weaknesses.append(
                "Eukaryotic origin — codon optimisation and soluble "
                "expression in E. coli not guaranteed"
            )
    if mutagenesis_data["n_mutations"] > 0:
        strengths.append(f"Mutagenesis data ({mutagenesis_data['n_mutations']} variants, {mutagenesis_data['n_references']} refs)")
    if structural_data.get("pdb_ids"):
        strengths.append(f"Crystal structure ({len(structural_data['pdb_ids'])} PDB)")
    elif structural_data.get("has_alphafold"):
        strengths.append("AlphaFold structure available")

    native_polymer_substrate = False
    _polymer_re = re.compile(r"\(n\)|\(n[+\-]\d\)|poly[-(]", re.IGNORECASE)
    for cat in catalytic_activities:
        if _polymer_re.search(cat):
            native_polymer_substrate = True
            break
    # A polymeric native substrate is a limitation for a *formation* target
    # (it may require a primer), but it is positive scope evidence for a
    # hydrolysis target.  Do not apply a synthesis-specific warning before
    # inspecting the target direction.
    target_direction = extract_target_bond_type(target_reaction_type).get("direction")
    requires_polymer = native_polymer_substrate and target_direction == "formation"
    if requires_polymer:
        weaknesses.append(
            "POLYMER SUBSTRATE: native reaction extends a growing polymer chain -- "
            "requires primer; cannot ligate two free small molecules"
        )
    elif native_polymer_substrate and target_direction == "cleavage":
        strengths.append("Native oligomer/polymer substrate scope")
    eval_data["requires_polymer_substrate"] = requires_polymer

    # Bond-type and reaction chemistry validation
    critical_warnings = []
    enzyme_chem = {}

    if target_reaction_type:
        enzyme_chem = classify_reaction_chemistry(ec_numbers, catalytic_activities)
        target_chem = extract_target_bond_type(target_reaction_type)
        bond_warnings = check_bond_type_mismatch(enzyme_chem, target_chem)
        for w in bond_warnings:
            if "BOND TYPE MISMATCH" in w or "REACTION DIRECTION MISMATCH" in w:
                critical_warnings.append(w)
            else:
                weaknesses.append(w)
        eval_data["reaction_chemistry"] = enzyme_chem

    # Substrate size mismatch detection
    if substrate_similarity and substrate_similarity.get("property_comparisons"):
        worst_mw_sim = min(
            (p.get("mw_similarity", 1.0)
             for p in substrate_similarity["property_comparisons"]),
            default=1.0,
        )
        if worst_mw_sim < config.MW_SEVERE_MISMATCH_THRESHOLD:
            p = substrate_similarity["property_comparisons"][0]
            critical_warnings.append(
                f"SEVERE SIZE MISMATCH: native substrate MW "
                f"({p.get('mw_native', 0):.0f} Da) vs target "
                f"({p.get('mw_target', 0):.0f} Da) -- "
                f"{worst_mw_sim:.0%} similarity"
            )
        elif worst_mw_sim < config.MW_MODERATE_MISMATCH_THRESHOLD:
            p = substrate_similarity["property_comparisons"][0]
            weaknesses.append(
                f"SUBSTRATE SIZE MISMATCH: native substrate MW "
                f"({p.get('mw_native', 0):.0f} Da) vs target "
                f"({p.get('mw_target', 0):.0f} Da)"
            )

    # Per-substrate mismatch warning
    per_target = substrate_similarity.get("per_target_best", []) if substrate_similarity else []
    if len(per_target) > 1:
        bottleneck = min(per_target)
        best = max(per_target)
        if bottleneck > 0 and best / bottleneck > 3.0:
            idx = per_target.index(bottleneck)
            weaknesses.append(
                f"ONE-SIDED SUBSTRATE MISMATCH: substrate {idx + 1} of "
                f"{len(per_target)} has Tanimoto {bottleneck:.2f} while "
                f"another has {best:.2f} -- aggregate score masks a "
                f"poor match on one substrate"
            )

    # Substrate class detection
    substrate_class = []
    for cat in catalytic_activities:
        if _polymer_re.search(cat):
            substrate_class.append("polymer")
        if re.search(
            r"\[acp\]|\[arcp\]|holo-\[|carrier\s+protein|pantetheine",
            cat, re.IGNORECASE,
        ):
            substrate_class.append("carrier_protein_tethered")
    eval_data["substrate_class"] = sorted(set(substrate_class)) or ["unclassified"]

    # Per-substrate match summary
    match_summary = []
    if substrate_similarity and substrate_similarity.get("property_comparisons"):
        for i, p in enumerate(substrate_similarity["property_comparisons"]):
            tani = per_target[i] if i < len(per_target) else None
            fg_t = p.get("functional_group_counts_target", {})
            fg_n = p.get("functional_group_counts_native", {})
            entry_item = {
                "substrate_index": i + 1,
                "tanimoto": round(tani, 3) if tani is not None else None,
                "mw_target": p.get("mw_target"),
                "mw_native": p.get("mw_native"),
                "mw_similarity": p.get("mw_similarity"),
                "fg_similarity": p.get("functional_group_similarity"),
                "fg_counts_target": fg_t,
                "fg_counts_native": fg_n,
            }
            match_summary.append(entry_item)
    eval_data["per_substrate_match"] = match_summary or None

    eval_data["critical_warnings"] = critical_warnings

    if not is_swissprot:
        weaknesses.append("TrEMBL (unreviewed) entry")
    if target_ec_class and not ec_match:
        weaknesses.append(f"No EC match to target {target_ec_class}")
    if literature["characterization_papers"] == 0:
        weaknesses.append("No characterization papers found")
    if not structural_data.get("has_structure"):
        weaknesses.append("No structural data available")

    eval_data["strengths"] = strengths
    eval_data["weaknesses"] = weaknesses

    append_exploration_log(
        run_dir,
        entry_type="EVALUATION",
        title=f"{accession} ({protein_name or 'unnamed'}, {organism})",
        details={
            "EC": ", ".join(ec_numbers) or "none",
            "Strengths": "; ".join(strengths) or "none",
            "Weaknesses": "; ".join(weaknesses) or "none",
            "Literature": f"{literature['characterization_papers']} char, {literature['engineering_papers']} eng papers",
            "Structure": "yes" if structural_data.get("has_structure") else "no",
        },
    )
    # A non-enzyme (no EC, no catalytic activity, no catalytic domain) is a
    # co-cluster contaminant, not a candidate: mark it rejected so it is excluded
    # from the family view and ranking rather than registered as a family.
    function_supported = function_validation.get("promotion_eligible") is True
    if not function_supported:
        blocked = ", ".join(function_validation.get("blocking_claims") or ["unknown"])
        weaknesses.append(f"Authoritative function claims not supported: {blocked}")
        eval_data["weaknesses"] = weaknesses
    eval_status = "evaluated" if is_enzyme and function_supported else "rejected"
    update_candidate_registry(run_dir, accession, eval_status, metadata={
        "protein_name": protein_name,
        "gene_names": gene_names,
        "literature_name": literature_name,
        "organism": organism,
        "sequence_length": seq_length,
        "ec_numbers": ec_numbers,
        "catalytic_activities": catalytic_activities[:3],
        "is_enzyme": is_enzyme,
        "interpro_family": interpro_family,
        "interpro_family_id": interpro_family_id,
        "literature": literature,
        "structural_data": structural_data,
        "substrate_similarity": substrate_similarity,
        "reaction_chemistry": enzyme_chem,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "critical_warnings": critical_warnings,
        "is_swissprot": is_swissprot,
        "identity_validation": identity_validation,
        "claim_validation": function_validation,
        "mechanism_evidence": mechanism_evidence,
        "modeling_input": modeling_input,
        "rejection_reason": (
            None if eval_status == "evaluated" else
            "authoritative function validation failed: "
            + ", ".join(function_validation.get("blocking_claims") or ["not an enzyme"])
        ),
    })

    crit_summary = f", {len(critical_warnings)} CRITICAL" if critical_warnings else ""
    return {
        "status": "success",
        "summary": (
            f"Evaluated {accession} ({protein_name or 'unnamed'}): "
            f"{len(strengths)} strengths, {len(weaknesses)} weaknesses"
            f"{crit_summary}"
        ),
        "data": eval_data,
    }
