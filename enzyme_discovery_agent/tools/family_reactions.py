"""Family native-reaction survey.

Once the agent has a candidate enzyme, this tool looks up that candidate's FOLD
family, enumerates the fold's reviewed members TOGETHER WITH THEIR NATIVE
REACTIONS, and compares each native reaction against the target reaction the run
is pursuing.  The agent can then SEE what a family actually catalyses and steer
expansion toward the family whose members' native reactions match the target
(same direction + shared bond type), instead of guessing a family name.

Answer-agnostic by construction: the fold comes from a DISCOVERED candidate and
the target comes from the run's own context, so nothing here encodes an answer.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from enzyme_discovery_agent.chemistry_logic import (
    classify_reaction_chemistry,
    compare_substrate_properties,
    extract_target_bond_type,
    resolve_chebi_smiles,
)
from enzyme_discovery_agent.static.cofactors import COFACTOR_CHEBI_IDS
from enzyme_discovery_agent.tool_registry import tool
from enzyme_discovery_agent.tools._http import request_with_retry

logger = logging.getLogger("enzyme_agent")

# UniProt caps a search page at 500 rows, so the cheap "list" probe saturates
# there: a fold returning a full page has AT LEAST this many reviewed members and
# is treated as a whole superfamily (not the most-specific fold we want), while a
# short page is the fold's exact reviewed-member count.
_LIST_PAGE_MAX = 500
_MEMBER_CAP = 120

# Directions that are genuine opposites (a cleavage fold cannot be steered into a
# formation target and vice versa).  This is a grounded chemical fact, not a tuned
# knob — the ONE hard call the survey makes; everything else is a reported number
# the agent reasons over.
_OPPOSITE_DIRECTIONS = {("formation", "cleavage"), ("cleavage", "formation")}


# --------------------------------------------------------------------------
# Pure classification (offline-testable — no network)
# --------------------------------------------------------------------------

def classify_member_reaction(
    ec_numbers: list[str],
    native_reaction: str,
    target_direction: str | None,
    target_bond_types: set[str],
    substrate_fit: float | None = None,
) -> dict:
    """Score one family member's native reaction against the target reaction.

    WARM  -> same direction as the target AND (when substrate structures were
             resolved) native substrates that RESEMBLE the target substrates.
             This member does the target chemistry ON target-like substrates.
    tepid -> right direction but substrates DON'T resemble the target (e.g. a
             fold that forms the target bond on a large cofactor / nucleotide /
             polymer substrate) — real chemistry, wrong substrate.
    COLD  -> opposite direction (formation vs cleavage); a dead end.
    neutral -> direction unknown or unrelated.

    ``substrate_fit`` is a 0..1 native-vs-target substrate similarity, or None
    when substrate structures were not resolved (then tagging falls back to
    direction only, the historical behaviour).  Pure and side-effect-free.
    """
    chem = classify_reaction_chemistry(
        ec_numbers, [native_reaction] if native_reaction else []
    )
    ndir = chem.get("reaction_direction")
    nbonds = set(chem.get("bond_types") or [])

    same_direction = bool(ndir and target_direction and ndir == target_direction)
    opposite_direction = (ndir, target_direction) in _OPPOSITE_DIRECTIONS
    shared_bond = bool(nbonds & target_bond_types)

    # Ordering score: same direction dominates, a shared bond type refines within
    # it, substrate resemblance refines further, an opposite direction sinks it.
    score = 0
    if same_direction:
        score += 2
    if opposite_direction:
        score -= 2
    if shared_bond:
        score += 1
    if substrate_fit is not None:
        score += round(2.0 * substrate_fit, 2)

    if opposite_direction:
        tag = "cold"
    elif substrate_fit is None:
        # Direction-only fallback (no substrate structures resolved).
        tag = "warm" if same_direction else "neutral"
    elif same_direction and substrate_fit >= _WARM_FIT_THRESHOLD:
        tag = "warm"
    elif same_direction:
        tag = "tepid"                       # right direction, substrate mismatch
    elif substrate_fit >= _WARM_FIT_THRESHOLD:
        tag = "tepid"                       # substrate matches but direction unclear
    else:
        tag = "neutral"

    return {
        "direction": ndir,
        "bond_types": sorted(nbonds),
        "shared_bond": shared_bond,
        "substrate_fit": substrate_fit,
        "score": score,
        "tag": tag,
    }


def summarise_family_profile(members: list[dict]) -> dict:
    """Condense scored members into the family's distinct native-reaction profile."""
    directions: dict[str, int] = {}
    distinct_reactions: list[str] = []
    counts = {"warm": 0, "tepid": 0, "cold": 0, "neutral": 0}
    for m in members:
        counts[m["tag"]] = counts.get(m["tag"], 0) + 1
        d = m.get("direction")
        if d:
            directions[d] = directions.get(d, 0) + 1
        rxn = (m.get("native_reaction") or "").strip()
        if rxn and rxn not in distinct_reactions and len(distinct_reactions) < 12:
            distinct_reactions.append(rxn)
    return {
        "directions": directions,
        "distinct_reactions": distinct_reactions,
        "warm_members": counts["warm"],
        "tepid_members": counts["tepid"],
        "cold_members": counts["cold"],
        "neutral_members": counts["neutral"],
    }


# --------------------------------------------------------------------------
# Substrate resemblance (network: ChEBI -> SMILES -> property comparison)
# --------------------------------------------------------------------------

def _member_substrate_chebis(entry: dict) -> list[str]:
    """Non-cofactor ChEBI participant ids from a member's catalytic activity.

    The enumeration response carries each reaction's ChEBI participants, so the
    member's actual substrates are available without a second fetch.  Cofactors
    (ATP/AMP/PPi/H2O/...) are dropped so only the transformed substrates remain.
    """
    ids: list[str] = []
    for c in entry.get("comments", []):
        if c.get("commentType") != "CATALYTIC ACTIVITY":
            continue
        for x in c.get("reaction", {}).get("reactionCrossReferences", []):
            if x.get("database") != "ChEBI":
                continue
            cid = x.get("id", "")
            if cid and cid not in COFACTOR_CHEBI_IDS and cid not in ids:
                ids.append(cid)
    return ids


def _substrate_fit(member_smiles: list[str], target_smiles: list[str]) -> float | None:
    """Coarse native-vs-target substrate resemblance in [0,1], or None.

    For EACH target substrate, take the best functional-group / chain-length /
    size match among the member's substrates, then average across the target
    substrates (the enzyme must handle every target substrate).  Aromatic/ring
    mismatches are SOFT down-weights, never hard caps — this steers where to dig,
    it does not disqualify.
    """
    if not member_smiles or not target_smiles:
        return None
    from enzyme_discovery_agent.tools.substrate import compute_substrate_properties

    per_target = []
    for t in target_smiles:
        best = 0.0
        for n in member_smiles:
            cmp = compare_substrate_properties(t, n, compute_substrate_properties)
            if not cmp:
                continue
            dims = [cmp.get(k) for k in (
                "functional_group_similarity", "chain_length_similarity",
                "mw_similarity",
            )]
            dims = [d for d in dims if isinstance(d, (int, float))]
            if not dims:
                continue
            fit = sum(dims) / len(dims)
            if cmp.get("aromaticity_match") is False:
                fit *= 0.7          # soft down-weight, NOT a hard cap
            if cmp.get("cyclicity_match") is False:
                fit *= 0.85
            best = max(best, fit)
        per_target.append(best)
    return round(sum(per_target) / len(per_target), 3) if per_target else None


# --------------------------------------------------------------------------
# Helpers (network)
# --------------------------------------------------------------------------

def _target_profile_for_run(run_dir: Path) -> tuple[str | None, set[str]]:
    """Read the run's declared target direction + bond types from its context.

    Mirrors tools/reasoning.py::_target_direction_for_run so the target is taken
    from the run, never passed in as free text.
    """
    try:
        context = json.loads((run_dir / "target_context.json").read_text())
        reaction = (context.get("target_profile") or {}).get("reaction") or {}
        target_text = " ".join(filter(None, [
            str(reaction.get("class", "")),
            str(reaction.get("direction", "")),
            str(context.get("query", "")),
        ]))
        info = extract_target_bond_type(target_text)
        return info.get("direction"), set(info.get("bond_types") or [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return None, set()


def _fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    return json.loads(request_with_retry(req, timeout=timeout).decode())


def _seed_fold_ids(seed_accession: str) -> list[str]:
    """Return the fold-family cross-reference ids annotated on a candidate."""
    url = (
        f"https://rest.uniprot.org/uniprotkb/{seed_accession}.json"
        f"?fields=xref_interpro"
    )
    data = _fetch_json(url, timeout=20)
    ids = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "InterPro":
            xid = xref.get("id", "")
            if xid and xid not in ids:
                ids.append(xid)
    return ids


def _reviewed_member_count(fold_id: str) -> int:
    """Reviewed-member count of a fold via the cheap list endpoint.

    Returns the exact count for a fold below one page, or ``_LIST_PAGE_MAX`` as a
    saturation sentinel (>= a full page) for a superfamily-sized fold.
    """
    q = urllib.parse.quote(f"(xref:interpro-{fold_id}) AND (reviewed:true)")
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?query={q}"
        f"&format=list&size={_LIST_PAGE_MAX}"
    )
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    body = request_with_retry(req, timeout=30).decode()
    return len([line for line in body.splitlines() if line.strip()])


def _reviewed_members(fold_id: str, size: int) -> list[dict]:
    """Enumerate reviewed members of a fold with their native reactions in one query."""
    q = urllib.parse.quote(f"(xref:interpro-{fold_id}) AND (reviewed:true)")
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?query={q}"
        f"&fields=accession,protein_name,ec,cc_catalytic_activity"
        f"&format=json&size={size}"
    )
    return _fetch_json(url, timeout=45).get("results", [])


def _member_name(entry: dict) -> str:
    rec = entry.get("proteinDescription", {}).get("recommendedName", {})
    name = (rec.get("fullName") or {}).get("value", "")
    if name:
        return name
    for sub in entry.get("proteinDescription", {}).get("submissionNames", []):
        val = (sub.get("fullName") or {}).get("value", "")
        if val:
            return val
    return ""


def _member_ec(entry: dict) -> list[str]:
    rec = entry.get("proteinDescription", {}).get("recommendedName", {})
    return [e.get("value", "") for e in rec.get("ecNumbers", []) if e.get("value")]


def _member_native_reaction(entry: dict) -> str:
    for c in entry.get("comments", []):
        if c.get("commentType") == "CATALYTIC ACTIVITY":
            return c.get("reaction", {}).get("name", "")
    return ""


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------

@tool(
    description=(
        "Survey a FOLD FAMILY by its native reactions. Give it EITHER a candidate "
        "accession you have already found (its most specific fold family is "
        "surveyed) OR an InterPro fold id directly (e.g. one returned by "
        "search_interpro, so you can go target reaction -> derived family -> "
        "search_interpro -> survey without needing a candidate in hand first). It "
        "enumerates that fold's reviewed members WITH THEIR NATIVE REACTIONS from "
        "UniProt and compares each native reaction to the target reaction this run "
        "is pursuing (direction + bond type). Use it to SEE what a family actually "
        "catalyses and decide whether to concentrate expansion on it (its members' "
        "native reactions match the target) or move on (they are opposite/"
        "unrelated), turning family navigation into a data-grounded decision "
        "instead of a guess."
    ),
    category="reasoning",
    parallel_safe=True,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "seed_accession": {
            "type": "string",
            "description": (
                "UniProt accession of a candidate you have already found; its fold "
                "family is the one surveyed. Provide this OR interpro_id."
            ),
            "default": "",
        },
        "interpro_id": {
            "type": "string",
            "description": (
                "InterPro fold id (e.g. IPR012345) to survey directly, such as one "
                "returned by search_interpro. Provide this OR seed_accession; if "
                "both are given, interpro_id takes precedence."
            ),
            "default": "",
        },
    },
)
def survey_family_reactions(run_dir: Path | str, seed_accession: str = "",
                            interpro_id: str = "") -> dict:
    """Enumerate a fold family and rank members by native-reaction fit.

    Entry point is EITHER a candidate accession (its most specific fold is
    resolved) OR an InterPro fold id supplied directly.
    """
    run_dir = Path(run_dir)
    seed_accession = (seed_accession or "").strip()
    interpro_id = (interpro_id or "").strip().upper()
    if not seed_accession and not interpro_id:
        return {
            "status": "error",
            "summary": "Provide either a seed_accession or an interpro_id.",
            "data": None,
            "suggestions": [
                "Pass a found candidate's accession, or an InterPro fold id "
                "(e.g. one from search_interpro)."
            ],
        }

    target_direction, target_bonds = _target_profile_for_run(run_dir)

    # Determine the fold to survey.  Direct InterPro id honours the caller's
    # choice; a seed accession resolves the candidate's most specific fold.
    if interpro_id:
        source = f"InterPro {interpro_id}"
        fold_id = interpro_id
        try:
            fold_size = _reviewed_member_count(fold_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Member count failed for %s: %s", fold_id, e)
            fold_size = 0
        if fold_size <= 1:
            return {
                "status": "partial",
                "summary": f"InterPro {fold_id} has no surveyable reviewed members.",
                "data": {"seed_accession": seed_accession or None, "fold_id": fold_id,
                         "fold_size": fold_size, "members": []},
            }
    else:
        source = f"{seed_accession} fold"
        try:
            fold_ids = _seed_fold_ids(seed_accession)
        except Exception as e:  # noqa: BLE001 - a failed lookup should not crash the run
            logger.warning("Fold lookup failed for %s: %s", seed_accession, e)
            return {
                "status": "error",
                "summary": f"Could not resolve fold ids for {seed_accession}: {e}",
                "data": None,
            }

        if not fold_ids:
            return {
                "status": "partial",
                "summary": f"{seed_accession} has no fold-family annotation to survey.",
                "data": {"seed_accession": seed_accession, "fold_id": None, "members": []},
            }

        # Pick the MOST SPECIFIC fold: the fewest reviewed members, but more than
        # one (a singleton fold tells us nothing) and not an entire superfamily.
        counts: dict[str, int] = {}
        for fid in fold_ids:
            try:
                counts[fid] = _reviewed_member_count(fid)
            except Exception as e:  # noqa: BLE001
                logger.debug("Member count failed for %s: %s", fid, e)
        fold_id = min(
            (f for f, c in counts.items() if 1 < c < _LIST_PAGE_MAX),
            key=lambda f: counts[f],
            default=None,
        )
        if not fold_id:
            return {
                "status": "partial",
                "summary": (
                    f"No fold of a surveyable size found for {seed_accession} "
                    f"(folds seen: {counts or 'none'})."
                ),
                "data": {"seed_accession": seed_accession, "fold_id": None, "members": []},
            }
        fold_size = counts[fold_id]
        source = f"{seed_accession} fold {fold_id}"
    try:
        raw_members = _reviewed_members(fold_id, size=_MEMBER_CAP)
    except Exception as e:  # noqa: BLE001
        logger.warning("Member enumeration failed for %s: %s", fold_id, e)
        return {
            "status": "error",
            "summary": f"Could not enumerate members of {fold_id}: {e}",
            "data": {"seed_accession": seed_accession, "fold_id": fold_id, "members": []},
        }

    # Target substrate structures — enable the substrate-resemblance dimension so
    # a member that makes the target bond on the WRONG substrate is only tepid.
    try:
        ctx = json.loads((run_dir / "target_context.json").read_text())
        target_smiles = [s.strip() for s in (ctx.get("substrate_smiles") or "").split(".")
                         if s.strip()]
    except (OSError, json.JSONDecodeError):
        target_smiles = []

    members: list[dict] = []
    chebi_cache: dict[str, str | None] = {}
    n_substrate_scored = 0
    for entry in raw_members:
        acc = entry.get("primaryAccession", "")
        if not acc:
            continue
        ec = _member_ec(entry)
        native_reaction = _member_native_reaction(entry)

        # Structurally score substrate resemblance for non-opposite members only,
        # up to a cap (ChEBI lookups cost network) — direction alone tags the rest.
        substrate_fit = None
        if target_smiles and n_substrate_scored < _SUBSTRATE_SCORE_CAP:
            direction_only = classify_member_reaction(
                ec, native_reaction, target_direction, target_bonds
            )
            if direction_only["tag"] != "cold":
                smiles = []
                for cid in _member_substrate_chebis(entry):
                    if cid not in chebi_cache:
                        try:
                            chebi_cache[cid] = resolve_chebi_smiles(cid)
                        except Exception:  # noqa: BLE001
                            chebi_cache[cid] = None
                    if chebi_cache[cid]:
                        smiles.append(chebi_cache[cid])
                substrate_fit = _substrate_fit(smiles, target_smiles)
                if substrate_fit is not None:
                    n_substrate_scored += 1

        scored = classify_member_reaction(
            ec, native_reaction, target_direction, target_bonds, substrate_fit
        )
        members.append({
            "accession": acc,
            "name": _member_name(entry),
            "ec": ec,
            "native_reaction": native_reaction,
            "direction": scored["direction"],
            "bond_types": scored["bond_types"],
            "substrate_fit": scored["substrate_fit"],
            "tag": scored["tag"],
            "score": scored["score"],
        })

    members.sort(key=lambda m: (-m["score"], m["accession"]))
    profile = summarise_family_profile(members)

    warm = profile["warm_members"]
    tepid = profile["tepid_members"]
    cold = profile["cold_members"]
    logger.info(
        "Family survey: %s (fold %s, %d reviewed members); %d warm / %d tepid / %d cold vs target",
        source, fold_id, fold_size, warm, tepid, cold,
    )

    return {
        "status": "success",
        "summary": (
            f"{source} ({fold_size} reviewed members): {warm} WARM (target "
            f"direction + target-like substrates), {tepid} tepid (right direction, "
            f"different substrate), {cold} opposite-direction."
        ),
        "data": {
            "seed_accession": seed_accession or None,
            "fold_id": fold_id,
            "fold_size": fold_size,
            "target_direction": target_direction,
            "target_bond_types": sorted(target_bonds),
            "family_reaction_profile": profile,
            "members": members,
        },
    }
