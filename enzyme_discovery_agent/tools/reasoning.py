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

from enzyme_discovery_agent import config
from enzyme_discovery_agent.chemistry_logic import (
    check_bond_type_mismatch,
    classify_reaction_chemistry,
    compare_substrate_properties,
    derive_activation_mechanism,
    derive_modeling_cofactors,
    extract_target_bond_type,
    get_chebi_name,
    resolve_chebi_smiles,
)
from enzyme_discovery_agent.models.modeling_handoff import build_modeling_input
from enzyme_discovery_agent.models.candidate_registry import (
    load_candidate_registry,
    update_candidate_registry,
)
from enzyme_discovery_agent.models.exploration_log import append_exploration_log
from enzyme_discovery_agent.models.target_profile import OPPOSITE_DIRECTIONS
from enzyme_discovery_agent.substrate_fit import assign_one_to_one
from enzyme_discovery_agent.static.cofactors import COFACTOR_CHEBI_IDS
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry
from enzyme_discovery_agent.run_layout import run_file

logger = logging.getLogger("enzyme_agent")


@tool(description="Read a candidate's saved identity, native-function evidence, expression assessment and structural observations without new network calls.",
      category="reasoning", parallel_safe=True, injected_params=["run_dir"],
      params={"accession": {"type": "string", "description": "Candidate accession to inspect."}})
def get_candidate_evidence(accession: str, run_dir: str = "") -> dict:
    entry = load_candidate_registry(Path(run_dir), strict=True).get("candidates", {}).get(accession)
    if not entry:
        return {"status": "empty", "data": None, "summary": "Candidate is not registered in this run."}
    from enzyme_discovery_agent.models.expansion import seed_evidence
    return {"status": "success", "summary": f"Saved candidate evidence for {accession}; no new evaluation performed.",
            "data": seed_evidence(accession, entry)}


@tool(description="Inspect this campaign's complete saved exploration frontier, modeling observations and unresolved questions. Supports bounded pagination beyond the examples in working context. This read-only view does not rank candidates, establish global saturation, approve an interpretation or authorize compute.",
      category="reasoning", parallel_safe=True, injected_params=["run_dir"],
      params={"section": {"type": "string", "enum": ["overview", "families", "hypotheses", "methods", "modeling", "questions", "searches"],
                          "default": "questions", "description": "Saved state to inspect."},
              "offset": {"type": "integer", "default": 0, "description": "Nonnegative pagination offset."},
              "limit": {"type": "integer", "default": 10, "description": "Records per page, between 1 and 20."},
              "query": {"type": "string", "default": "", "description": "Optional text filter over the requested saved-state section."}})
def get_exploration_state(section: str = "questions", offset: int = 0, limit: int = 10,
                          query: str = "", run_dir: str = "") -> dict:
    from enzyme_discovery_agent.models.exploration_state import exploration_page
    try:
        data = exploration_page(Path(run_dir), section, offset=offset, limit=limit, query=query)
    except ValueError as exc:
        return {"status": "error", "summary": str(exc), "data": None}
    return {"status": "success" if data["records"] else "empty", "data": data,
            "summary": f"Saved {section} diagnostics: {len(data['records'])} of {data['total']} matching records."}


@tool(description="Recall this campaign's saved searches, unresolved literature identities, family decisions or notebook passages without network calls. Results are paginated; notes and decisions are not primary evidence.",
      category="reasoning", parallel_safe=True, injected_params=["run_dir"],
      params={"topic": {"type": "string", "enum": ["searches", "leads", "notes", "decisions", "reasoning"],
                        "description": "Research memory to inspect."},
              "query": {"type": "string", "default": "", "description": "Optional text filter over saved records."},
              "offset": {"type": "integer", "default": 0, "description": "Pagination offset from a previous response."},
              "limit": {"type": "integer", "default": 10, "description": "Maximum records, between 1 and 20."}})
def get_research_memory(topic: str, query: str = "", offset: int = 0, limit: int = 10, run_dir: str = "") -> dict:
    from enzyme_discovery_agent.models.research_memory import recall_memory
    try:
        data = recall_memory(Path(run_dir), topic, query, offset, limit)
    except ValueError as exc:
        return {"status": "error", "summary": str(exc), "data": None}
    return {"status": "success" if data["records"] else "empty", "data": data,
            "summary": f"Recalled {len(data['records'])} saved {topic} records; {data['total']} matching records."}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _fetch_uniprot_entry_light(accession: str) -> dict | None:
    """Fetch minimal UniProt data for screening (single API call)."""
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

    return _screening_annotation(data, accession)


def _screening_annotation(data: dict, accession: str) -> dict:
    """Project a retrieved UniProt record for screening without another request."""
    from enzyme_discovery_agent.tools.database import _parse_uniprot_json

    parsed = _parse_uniprot_json(data)
    from enzyme_discovery_agent.models.claim_validation import validate_uniprot_claims
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
    from enzyme_discovery_agent.tools.family_reactions import _target_profile_for_run
    return _target_profile_for_run(run_dir)[0]


def _target_claims_for_run(run_dir: Path) -> tuple[str | None, set[str], str | None]:
    """Return the normalized target claims used for authoritative validation."""
    from enzyme_discovery_agent.tools.family_reactions import (
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


def _evaluate_metagenomic_candidate(
    run_dir: Path,
    accession: str,
    metadata: dict,
) -> dict:
    """Evaluate an unannotated Atlas hit without inventing UniProt evidence."""
    sequence = re.sub(r"[^A-Za-z]", "", str(metadata.get("sequence") or "")).upper()
    family_id = str(
        metadata.get("screening_family_id")
        or (metadata.get("interpro_ids") or [""])[0]
        or ""
    )
    homology = metadata.get("structural_homology") or {}
    problems = []
    if not sequence:
        problems.append("sequence_missing")
    if not family_id:
        problems.append("seed_family_unresolved")
    if homology.get("probability") is None:
        problems.append("structural_match_probability_missing")
    eligible = not problems
    validation = {
        "schema_version": "metagenomic-structural-candidate-v1",
        "subject": accession,
        "source": "ESM Metagenomic Atlas + Foldseek",
        "promotion_eligible": eligible,
        "blocking_claims": problems,
        "evidence_scope": "candidate_identity_and_structural_family_only",
    }
    warnings = [
        "UNANNOTATED METAGENOMIC LEAD: structural family similarity does not establish "
        "catalytic activity, reaction direction, or substrate scope."
    ]
    data = {
        "accession": accession,
        "is_enzyme": None,
        "is_metagenomic": True,
        "protein_name": metadata.get(
            "protein_name", f"Metagenomic structural homolog {accession}"
        ),
        "gene_names": [],
        "organism": "",
        "sequence_length": len(sequence),
        "sequence": sequence,
        "is_swissprot": False,
        "ec_numbers": [],
        "catalytic_activities": [],
        "function_comments": [],
        "required_cofactors": [],
        "cofactor_notes": [],
        "domain_architecture": [],
        "interpro_family": "",
        "interpro_family_id": family_id,
        "literature": {"characterization_papers": 0, "engineering_papers": 0},
        "structural_data": {
            "has_structure": True,
            "source": homology.get("source_database", "ESMAtlas"),
            "structural_homology": homology,
        },
        "substrate_similarity": {
            "tanimoto_score": None,
            "note": "No native-substrate annotation exists for this metagenomic sequence.",
        },
        "reaction_chemistry": {},
        "strengths": [
            "Sequence recovered from the metagenomic structure atlas",
            "Foldseek structural-family connection to an evaluated seed",
        ],
        "weaknesses": warnings,
        "critical_warnings": warnings,
        "claim_validation": validation,
        "evaluation_scope": "exploratory_structural_lead",
        "structural_homology": homology,
    }
    update_candidate_registry(
        run_dir,
        accession,
        "evaluated" if eligible else "rejected",
        metadata=data,
    )
    return {
        "status": "success" if eligible else "error",
        "summary": (
            f"Evaluated {accession} as an unannotated metagenomic structural lead"
            if eligible
            else f"Could not validate metagenomic lead {accession}: {', '.join(problems)}"
        ),
        "data": data if eligible else None,
    }


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
    """Assess documented preferred-host expression, preserving uncertainty."""
    from enzyme_discovery_agent.models.expression import assess_expression

    def fetch_json(url):
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        return json.loads(request_with_retry(request, timeout=10).decode())

    return assess_expression(accession, entry, pdb_ids, fetch_json)


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
    from enzyme_discovery_agent.tools.literature import _pubmed_esearch_count

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
    registry_candidates = load_candidate_registry(run_dir).get("candidates", {})
    results = []
    for acc in accessions[:config.SCREENING_BATCH_SIZE]:
        candidate_entry = registry_candidates.get(acc, {})
        metadata = candidate_entry.get("metadata", {})
        if metadata.get("is_literature_lead"):
            results.append({"accession": acc, "screening_decision": "evidence_pending",
                            "screening_reason": "Paper-scoped identity requires independent resolution."})
            continue
        from enzyme_discovery_agent.models.candidate_registry import defer_non_uniprot_deposit
        if defer_non_uniprot_deposit(run_dir, acc, candidate_entry):
            results.append({"accession": acc, "screening_decision": "evidence_pending",
                            "screening_reason": "Deposited identity needs independent resolution."})
            continue
        if metadata.get("is_metagenomic"):
            sequence = str(metadata.get("sequence") or "")
            entry = {
                "accession": acc,
                "protein_name": metadata.get("protein_name", ""),
                "organism": "",
                "ec_numbers": [],
                "catalytic_activities": [],
                "gene_names": [],
                "function": "Unannotated metagenomic structural homolog",
                "seq_length": len(sequence),
                "interpro_ids": metadata.get("interpro_ids", []),
                "is_metagenomic": True,
            }
        else:
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
                "is_metagenomic": bool(entry.get("is_metagenomic")),
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
        import hashlib
        safe = accession if re.fullmatch(r"[A-Za-z0-9_.-]+", accession) else "identifier_" + hashlib.sha256(accession.encode()).hexdigest()
        (candidate_dir / f"screen_{safe}.json").write_text(json.dumps({
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
    from enzyme_discovery_agent.models.candidate_registry import (
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
    existing_metadata = existing.get("metadata", {})
    from enzyme_discovery_agent.models.candidate_registry import defer_non_uniprot_deposit
    if defer_non_uniprot_deposit(run_dir, accession, existing):
        return {"status": "partial", "summary": "Deposited identifier requires identity resolution before evaluation.",
                "data": {"accession": accession, "identity_status": "pending"},
                "suggestions": ["Inspect the source-scoped identity evidence before selecting an accession."]}
    if existing_metadata.get("is_literature_lead"):
        resolved = existing_metadata.get("resolved_accession")
        return {"status": "partial", "summary": "Literature identifiers are evidence aliases, not database accessions.",
                "data": {"resolved_accession": resolved}, "suggestions": [
                    f"Evaluate linked accession {resolved}." if resolved else
                    "Resolve the paper's deposited identifier or candidate-specific sequence evidence before functional evaluation."]}
    if existing_metadata.get("is_metagenomic"):
        return _evaluate_metagenomic_candidate(run_dir, accession, existing_metadata)

    entry = _fetch_uniprot_entry(accession)
    if not entry:
        return {
            "status": "error",
            "summary": f"Could not fetch UniProt entry for {accession}",
            "data": None,
            "suggestions": ["Check that the accession ID is correct."],
        }

    from enzyme_discovery_agent.models.claim_validation import validate_uniprot_claims
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
            for text in (comment.get("note") or {}).get("texts", []):
                value = text.get("value", "")
                if value:
                    cofactor_notes.append(value)

    # Concurrent data gathering
    from enzyme_discovery_agent.tools.database import get_protein_domains

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
    from enzyme_discovery_agent.tools.family_reactions import _reactant_side_chebi_ids

    def _name_of(cid):
        resolve_chebi_smiles(cid)  # populates the name cache
        from enzyme_discovery_agent.chemistry_logic import get_chebi_names
        return get_chebi_names(cid)

    from enzyme_discovery_agent.tools.native_reactions import reaction_groups, select_reaction_group, substrate_occurrences
    groups=reaction_groups(entry,lambda one:_reactant_side_chebi_ids(one,_name_of))
    for group in groups:
        group['native_substrates']=[{'chebi_id':cid,'name':get_chebi_name(cid),
            'smiles':resolve_chebi_smiles(cid)} for cid in group['chebi_ids']]
        group['native_substrates']=substrate_occurrences(group['reaction'],group['native_substrates'])
    # Use the same campaign contract for reaction selection and final validation.
    from enzyme_discovery_agent.models.target_reasoning import load_target_reasoning
    contract = load_target_reasoning(run_dir)
    if contract is not None:
        target_chemistry = {'direction': contract.get('reaction_direction'),
                            'bond_types': contract.get('bond_types') or []}
    else:
        target_chemistry = extract_target_bond_type(target_reaction_type)
    target_parts=[part.strip() for part in target_substrate_smiles.split('.') if part.strip()]
    def _group_fit(group):
        from enzyme_discovery_agent.tools.family_reactions import _substrate_facts
        from enzyme_discovery_agent.substrate_accommodation import accommodation_facts
        from enzyme_discovery_agent.substrate_fit import accommodation_score
        native=[item['smiles'] for item in group['native_substrates'] if item.get('smiles')]
        if not native or not target_parts:
            return None
        try:
            facts=_substrate_facts(native,target_parts) or {}
            from enzyme_discovery_agent.tools.native_reactions import annotate_indirect_donor_roles
            roles=annotate_indirect_donor_roles(entry,group['native_substrates'],accommodation_facts(target_parts,native))
            return accommodation_score({'property_comparisons':facts.get('property_comparisons'),
                                        'accommodation':roles})
        except (ValueError,TypeError,RuntimeError):
            return None
    selected_reaction=select_reaction_group(groups,target_chemistry.get('direction'),
                                           target_chemistry.get('bond_types') or [],_group_fit)
    if selected_reaction:
        native_substrates_info=selected_reaction['native_substrates']
        # The fallback must use this same reaction's identifiers as well.
        rhea_ids=selected_reaction['rhea_ids']

    # Substrate-source cascade — measured or reported ONLY, never inferred.
    # 1) UniProt catalyticActivity -> ChEBI -> SMILES (above; most reliable).
    # 2) The enzyme's own Rhea reaction IDs -> ChEBI -> SMILES (still measured).
    literature_name = ""
    literature_evidence = existing_metadata.get("literature_evidence")
    extracted_literature_substrates = []
    if not native_substrates_info:
        native_substrates_info = _fallback_substrate_lookup(rhea_ids)

    # 3) Characterization paper (reported): for sparse entries (often TrEMBL)
    # with no structured reaction, read THIS enzyme's paper and take the native
    # substrates it reports, plus the name the paper gives the enzyme.  No
    # EC-archetype or homology guessing feeds the substrate axis.
    if not native_substrates_info and llm_client is not None:
        try:
            from enzyme_discovery_agent.tools.literature_extraction import (
                extract_substrates_and_mechanism,
            )
            source_tool = existing.get("metadata", {}).get("source_tool", "")
            lit = extract_substrates_and_mechanism(
                run_dir, accession, source_tool, protein_name, llm_client,
                query=target_reaction_type, organism=organism,
            )
            if lit and lit.get("paper_relevant"):
                literature_evidence = lit.get("literature_evidence") or literature_evidence
                extracted_literature_substrates = lit.get("native_substrates") or []
                literature_name = (lit.get("enzyme_name_in_paper") or "").strip()
        except Exception as e:  # never let the fallback block evaluation
            logger.debug("literature substrate fallback failed for %s: %s",
                         accession, e)

    from enzyme_discovery_agent.models.literature_evidence import verify_literature_evidence
    literature_verified = verify_literature_evidence(literature_evidence, accession, entry)
    if literature_verified and not native_substrates_info:
        for ns in extracted_literature_substrates:
            if isinstance(ns, dict) and ns.get("smiles"):
                native_substrates_info.append({"chebi_id": "", "name": ns.get("name", ""),
                                               "smiles": ns["smiles"], "source": "literature"})
    if literature_verified and not native_substrates_info:
        from enzyme_discovery_agent.chemistry_logic import resolve_substrate_names_to_smiles
        for native in literature_evidence.get("native_substrates") or []:
            name = native.get("name", "") if isinstance(native, dict) else str(native)
            smiles = native.get("smiles") if isinstance(native, dict) else None
            smiles = smiles or resolve_substrate_names_to_smiles(name)
            if smiles:
                native_substrates_info.append({"name": name, "smiles": smiles,
                                              "source": "literature"})

    # Substrate similarity
    substrate_similarity = {}
    if target_substrate_smiles:
        from enzyme_discovery_agent.tools.chemistry import compute_fingerprint_similarity
        from enzyme_discovery_agent.tools.substrate import compute_substrate_properties

        native_smiles_list = [
            ns["smiles"] for ns in native_substrates_info if ns.get("smiles")
        ]
        target_molecules = [s.strip() for s in target_substrate_smiles.split(".")
                            if s.strip()]

        if native_smiles_list and target_molecules:
            # Match distinct source-supported occurrences, including repeated
            # copies of one species. Fingerprints supply structural context;
            # reacting roles require an explicit step interpretation.
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
            _property_cache={}
            def _pair(t,n):
                if (t,n) not in _property_cache:
                    _property_cache[t,n]=compare_substrate_properties(t,n,compute_substrate_properties)
                return _property_cache[t,n]
            def _pair_fit(t,n):
                from enzyme_discovery_agent.chemistry_logic import whole_package_pair_fit
                value=whole_package_pair_fit(_pair(t,n))
                return value if value is not None else -1.0
            for component,(t_smi,n_smi,fit) in enumerate(assign_one_to_one(
                    target_molecules,native_smiles_list,_pair_fit),1):
                sim=_fp(t_smi,n_smi) if n_smi else 0.0
                per_target_best.append(sim)
                match_details.append(
                    f"{t_smi[:40]}->{(n_smi or 'none')[:40]}: {sim:.3f}")
                if n_smi:
                    prop = _pair(t_smi,n_smi)
                    if prop:
                        property_comparisons.append({**prop,'target_component_index':component})

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
                "target_smiles": target_molecules,
                "assignment_basis":"one-to-one maximin over the shared complete property-pair fit; fingerprints describe those same selected pairs",
                "per_target_best": [round(s, 3) for s in per_target_best],
                "match_details": match_details,
                "property_comparisons": property_comparisons,
            }
            # Role-aware accommodation over the REACTANT-side substrates: which
            # target substrate (if any) has no genuine native partner.  Ranking
            # uses this to DEMOTE a candidate that cannot host every target
            # substrate, rather than scoring an unresolvable partner as neutral.
            from enzyme_discovery_agent.substrate_accommodation import accommodation_facts
            substrate_similarity["accommodation"] = accommodation_facts(
                target_molecules, native_smiles_list)
            from enzyme_discovery_agent.tools.native_reactions import annotate_indirect_donor_roles
            substrate_similarity['accommodation']=annotate_indirect_donor_roles(
                entry,native_substrates_info,substrate_similarity['accommodation'])
            from enzyme_discovery_agent.models.reaction_roles import apply_roles
            role_basis={'catalytic_activities':catalytic_activities,'function_comments':function_comments[:3],
                        'native_substrates':[ns for ns in native_substrates_info if ns.get('smiles')],
                        'native_substrate_reaction_scope':selected_reaction,'required_cofactors':required_cofactors,
                        'reaction_step':existing_metadata.get('reaction_step')}
            substrate_similarity['accommodation']=apply_roles(role_basis,existing_metadata.get('reaction_roles'),
                substrate_similarity['accommodation'])
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
    if literature_verified and not catalytic_activities:
        is_enzyme = True

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
        literature_evidence=literature_evidence,
    )

    existing_mechanism = existing.get("metadata", {}).get("mechanism_evidence")
    from enzyme_discovery_agent.tools.native_reactions import scoped_activation
    activation_context=scoped_activation(selected_reaction,catalytic_activities)
    activation = activation_context['activation']
    mechanism_evidence = dict(existing_mechanism or {})
    if (selected_reaction and mechanism_evidence.get('activation') and
            mechanism_evidence.get('modeled_reaction_scope')!=activation_context['reaction_scope']):
        mechanism_evidence['previous_unscoped_activation']=mechanism_evidence.pop('activation')
    if activation:
        mechanism_evidence["activation"] = activation
    if selected_reaction:
        mechanism_evidence['modeled_reaction_scope']=activation_context['reaction_scope']
    if activation_context.get('all_activity_observations'):
        mechanism_evidence['all_activity_activation_observations']=activation_context['all_activity_observations']

    modeling_candidate = {
        "catalytic_activities": activation_context['activities'],
        "required_cofactors": required_cofactors,
        "cofactor_notes": cofactor_notes,
        "mechanism_evidence": mechanism_evidence or None,
        "modeling_specification": (
            existing.get("metadata", {}).get("modeling_specification")
        ),
    }
    try:
        target_context = json.loads((run_file(run_dir, 'target_context.json')).read_text())
    except (OSError, json.JSONDecodeError):
        target_context = {
            "substrate_smiles": target_substrate_smiles,
            "target_reasoning": {
                "bond_types": sorted(target_bonds),
                "reaction_direction": target_direction,
            },
        }
    modeling_input = build_modeling_input(modeling_candidate, target_context)

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
        "cofactor_notes": cofactor_notes,
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
        "native_substrate_reaction_scope": selected_reaction,
        "native_reaction_groups": groups,
        "rhea_ids": rhea_ids,
        "identity_validation": identity_validation,
        "claim_validation": function_validation,
        "literature_evidence": literature_evidence,
        "mechanism_evidence": mechanism_evidence or None,
        "modeling_input": modeling_input,
    }

    # Strengths and weaknesses
    strengths = []
    weaknesses = []
    if (substrate_similarity.get('accommodation') or {}).get('indirect_donor_role_mappings'):
        weaknesses.append('INDIRECT SUBSTRATE ROLE: the fitted native partner is annotated as a hydrolyzed nitrogen source. '
            'Whole-molecule resemblance does not establish direct acceptance at the ligation site; target role transfer requires evidence.')

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
        level = (expression_data.get("preferred_host_assessment") or {}).get("level", "reported_expression")
        strengths.append(f"Preferred-host evidence: {level} ({host}); construct and activity scope retained")
    elif expression_data.get("bacterial_expression"):
        evid = expression_data.get("expression_evidence", ["bacteria"])[0]
        strengths.append(f"Bacterial expression demonstrated ({evid})")
    else:
        weaknesses.append("Preferred-host expression is unresolved; native organism alone does not establish recombinant expression")
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
    target_direction = target_chemistry.get("direction")
    requires_polymer = native_polymer_substrate and target_direction == "formation"
    if requires_polymer:
        weaknesses.append(
            "POLYMER SUBSTRATE: native annotation describes chain extension on a primer. "
            "Initiation from two free inputs is unestablished; primer-compatible "
            "elongation needs a separate substrate-compatibility assessment."
        )
    elif native_polymer_substrate and target_direction == "cleavage":
        strengths.append("Native oligomer/polymer substrate scope")
    eval_data["requires_polymer_substrate"] = requires_polymer

    # Bond-type and reaction chemistry validation
    critical_warnings = []
    enzyme_chem = {}

    if target_reaction_type or contract is not None:
        enzyme_chem = classify_reaction_chemistry(ec_numbers, catalytic_activities)
        target_chem = target_chemistry
        bond_warnings = check_bond_type_mismatch(enzyme_chem, target_chem)
        for w in bond_warnings:
            if "BOND TYPE MISMATCH" in w or "REACTION DIRECTION MISMATCH" in w:
                critical_warnings.append(w)
            else:
                weaknesses.append(w)
        eval_data["reaction_chemistry"] = enzyme_chem
    if literature_verified and not catalytic_activities:
        mechanism = literature_evidence.get("mechanism") or {}
        enzyme_chem = {
            "bond_types": mechanism.get("bond_types") or [mechanism.get("bond_formed")],
            "reaction_direction": mechanism.get("direction"),
            "source": "primary_literature",
        }
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
    contradictions = [c for c in function_validation.get("claims", [])
                      if c.get("verdict") == "contradicted"
                      and c.get("claim_type") in function_validation.get("blocking_claims", [])]
    eval_status = ("evaluated" if is_enzyme and function_supported else
                   "rejected" if contradictions else "evidence_pending")
    update_candidate_registry(run_dir, accession, eval_status, metadata={
        "protein_name": protein_name,
        "gene_names": gene_names,
        "literature_name": literature_name,
        "organism": organism,
        "sequence_length": seq_length,
        "ec_numbers": ec_numbers,
        "catalytic_activities": catalytic_activities[:3],
        "required_cofactors": required_cofactors,
        "cofactor_notes": cofactor_notes,
        "is_enzyme": is_enzyme,
        "interpro_family": interpro_family,
        "interpro_family_id": interpro_family_id,
        "literature": literature,
        "structural_data": structural_data,
        "substrate_similarity": substrate_similarity,
        "expression_data": expression_data,
        "reaction_chemistry": enzyme_chem,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "critical_warnings": critical_warnings,
        "is_swissprot": is_swissprot,
        "identity_validation": identity_validation,
        "claim_validation": function_validation,
        "literature_evidence": literature_evidence,
        "mechanism_evidence": mechanism_evidence or None,
        "modeling_input": modeling_input,
        "evaluation_data": dict(eval_data),
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
