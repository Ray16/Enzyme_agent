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
) -> dict:
    """Report GROUNDED facts about a member's native reaction vs the target.

    Emits facts, not a verdict — no tag, no threshold, no tuned score.  The one
    non-empirical classification is ``direction_vs_target`` (formation vs
    cleavage is a real chemical fact, not a knob).  Substrate resemblance is
    attached separately as reported numbers; the AGENT weighs them and decides
    which family to expand.  Pure and side-effect-free.
    """
    chem = classify_reaction_chemistry(
        ec_numbers, [native_reaction] if native_reaction else []
    )
    ndir = chem.get("reaction_direction")
    nbonds = set(chem.get("bond_types") or [])

    if not ndir or not target_direction:
        direction_vs_target = "unknown"
    elif ndir == target_direction:
        direction_vs_target = "same"
    elif (ndir, target_direction) in _OPPOSITE_DIRECTIONS:
        direction_vs_target = "opposite"
    else:
        direction_vs_target = "unrelated"

    return {
        "direction": ndir,
        "direction_vs_target": direction_vs_target,
        "bond_types": sorted(nbonds),
        "shared_bond_type": bool(nbonds & target_bond_types),
    }


def _distribution(values: list[float]) -> dict | None:
    """Min/median/max of a list of numbers (or None if empty) — a grounded
    summary the agent reads, with no threshold applied."""
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"n": n, "min": round(vals[0], 3), "median": round(med, 3), "max": round(vals[-1], 3)}


def summarise_family_profile(members: list[dict]) -> dict:
    """Condense members into the family's GROUNDED native-reaction picture.

    Reports the direction breakdown and the distribution of substrate-resemblance
    numbers across members — no warm/tepid tally, no cutoff.  The agent compares
    these family-level distributions to decide which fold's members most resemble
    the target and expand there.
    """
    direction_counts: dict[str, int] = {}
    distinct_reactions: list[str] = []
    fg_sims: list[float] = []
    size_ratios: list[float] = []
    chain_sims: list[float] = []
    n_aromatic_mismatch = 0
    n_ring_mismatch = 0
    for m in members:
        dv = m.get("direction_vs_target") or "unknown"
        direction_counts[dv] = direction_counts.get(dv, 0) + 1
        sub = m.get("substrate") or {}
        if sub.get("functional_group_similarity") is not None:
            fg_sims.append(sub["functional_group_similarity"])
        if sub.get("size_ratio") is not None:
            size_ratios.append(sub["size_ratio"])
        if sub.get("chain_length_similarity") is not None:
            chain_sims.append(sub["chain_length_similarity"])
        if sub.get("native_aromatic_target_lacks"):
            n_aromatic_mismatch += 1
        if sub.get("native_ring_target_lacks"):
            n_ring_mismatch += 1
        rxn = (m.get("native_reaction") or "").strip()
        if rxn and rxn not in distinct_reactions and len(distinct_reactions) < 12:
            distinct_reactions.append(rxn)
    return {
        "n_members": len(members),
        "direction_vs_target_counts": direction_counts,
        "functional_group_similarity": _distribution(fg_sims),
        "size_ratio": _distribution(size_ratios),
        # Main-chain-length match distribution and how many members carry
        # aromatic/ring character the target lacks — the axes that discriminate
        # once functional-group overlap saturates.  Facts, not a verdict.
        "chain_length_similarity": _distribution(chain_sims),
        "members_native_aromatic_target_lacks": n_aromatic_mismatch,
        "members_native_ring_target_lacks": n_ring_mismatch,
        "distinct_reactions": distinct_reactions,
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


def _substrate_facts(member_smiles: list[str], target_smiles: list[str]) -> dict | None:
    """Report GROUNDED substrate-resemblance facts (no blended score, no weights).

    For EACH target substrate, pick the member substrate with the best functional-
    group overlap and keep its raw comparison numbers.  Report the WORST-matched
    target as the headline (the enzyme must accommodate every target substrate — a
    'handle all' modelling choice, not a tuned weight), plus the raw functional
    groups and size ratio.  Returns None when nothing resolves.  The agent weighs
    these; nothing here converts them into a verdict.
    """
    if not member_smiles or not target_smiles:
        return None
    from enzyme_discovery_agent.tools.substrate import compute_substrate_properties

    per_target = []
    for t in target_smiles:
        best = None
        for n in member_smiles:
            cmp = compare_substrate_properties(t, n, compute_substrate_properties)
            if not cmp:
                continue
            fg = cmp.get("functional_group_similarity")
            if fg is None:
                continue
            cl = cmp.get("chain_length_similarity")
            # Pick a member's own closest substrate by functional-group overlap,
            # breaking ties by MAIN-CHAIN closeness — the FG axis saturates (a
            # carboxyl/amine is near-universal here), so without the chain tiebreak
            # this would pick an arbitrarily-sized substrate among FG ties.
            key = (fg, cl if cl is not None else -1.0)
            if best is None or key > best["_key"]:
                best = {
                    "_key": key,
                    "fg": fg,
                    "size_ratio": cmp.get("mw_similarity"),
                    "chain_length_similarity": cl,
                    "chain_length_target": cmp.get("chain_length_target"),
                    "chain_length_native": cmp.get("chain_length_native"),
                    "aromaticity_match": cmp.get("aromaticity_match"),
                    "cyclicity_match": cmp.get("cyclicity_match"),
                    "fg_target": cmp.get("functional_groups_target") or [],
                    "fg_native": cmp.get("functional_groups_native") or [],
                }
        if best is not None:
            per_target.append(best)
    if not per_target:
        return None

    def _worst(field: str):
        vals = [p[field] for p in per_target if p.get(field) is not None]
        return round(min(vals), 3) if vals else None

    # Report size/chain numbers for the WORST-fitting target substrate — the one
    # that governs whether the pocket can host every target (facts, not a verdict).
    worst_cl = min(
        per_target,
        key=lambda p: (p["chain_length_similarity"]
                       if p.get("chain_length_similarity") is not None else 1.0),
    )
    return {
        "functional_group_similarity": round(min(p["fg"] for p in per_target), 3),
        "size_ratio": _worst("size_ratio"),
        "chain_length_similarity": _worst("chain_length_similarity"),
        "chain_length_target": worst_cl.get("chain_length_target"),
        "chain_length_native": worst_cl.get("chain_length_native"),
        "functional_groups_target": sorted({g for p in per_target for g in p["fg_target"]}),
        "functional_groups_native": sorted({g for p in per_target for g in p["fg_native"]}),
        "native_aromatic_target_lacks": any(p.get("aromaticity_match") is False for p in per_target),
        "native_ring_target_lacks": any(p.get("cyclicity_match") is False for p in per_target),
    }


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
    for entry in raw_members:
        acc = entry.get("primaryAccession", "")
        if not acc:
            continue
        ec = _member_ec(entry)
        native_reaction = _member_native_reaction(entry)
        facts = classify_member_reaction(ec, native_reaction, target_direction, target_bonds)

        # Substrate-resemblance FACTS for every non-opposite member (opposite
        # direction is already a grounded dead end, so skip its ChEBI lookups).
        substrate = None
        if target_smiles and facts["direction_vs_target"] != "opposite":
            smiles = []
            for cid in _member_substrate_chebis(entry):
                if cid not in chebi_cache:
                    try:
                        chebi_cache[cid] = resolve_chebi_smiles(cid)
                    except Exception:  # noqa: BLE001
                        chebi_cache[cid] = None
                if chebi_cache[cid]:
                    smiles.append(chebi_cache[cid])
            substrate = _substrate_facts(smiles, target_smiles)

        members.append({
            "accession": acc,
            "name": _member_name(entry),
            "ec": ec,
            "native_reaction": native_reaction,
            "direction": facts["direction"],
            "direction_vs_target": facts["direction_vs_target"],
            "bond_types": facts["bond_types"],
            "shared_bond_type": facts["shared_bond_type"],
            "substrate": substrate,
        })

    # Presentation order over GROUNDED axes (not a verdict): same-direction first,
    # then higher functional-group overlap, then closer size.  No cutoff — the
    # agent reads the numbers and decides where to expand.
    _dir_rank = {"same": 0, "unknown": 1, "unrelated": 2, "opposite": 3}

    def _order_key(m: dict):
        sub = m.get("substrate") or {}
        fg = sub.get("functional_group_similarity")
        sz = sub.get("size_ratio")
        return (
            _dir_rank.get(m.get("direction_vs_target"), 1),
            -(fg if fg is not None else -1.0),
            -(sz if sz is not None else -1.0),
            m["accession"],
        )

    members.sort(key=_order_key)
    profile = summarise_family_profile(members)

    fg = profile.get("functional_group_similarity")
    dv = profile.get("direction_vs_target_counts", {})
    fg_txt = (f"functional-group similarity max {fg['max']}, median {fg['median']}"
              if fg else "substrate similarity unavailable")
    logger.info(
        "Family survey: %s (fold %s, %d members); direction %s; %s",
        source, fold_id, fold_size, dv, fg_txt,
    )

    return {
        "status": "success",
        "summary": (
            f"{source} ({fold_size} reviewed members). Direction vs target: {dv}. "
            f"Substrate {fg_txt}. These are grounded numbers — compare this family's "
            f"distribution against the others you survey and reason about which "
            f"fold's members most resemble the target before expanding."
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


@tool(
    description=(
        "Act on the family landscape: decide which fold families to EXPAND "
        "(keep pursuing / evaluate in depth) and which to STOP (set aside, not "
        "worth evaluation budget). Call this AFTER reasoning over the surveyed "
        "family landscape. Pass InterPro fold ids exactly as shown in the "
        "landscape (e.g. IPR011810). STOP a fold when its reviewed members' "
        "native reactions do not resemble the target — e.g. the native substrate "
        "is a peptide, amino acid, nucleotide, sugar, or cofactor while the target "
        "is a small aliphatic substrate, or the reaction runs the opposite "
        "direction. A stopped fold is then waived from the evaluation floor, so "
        "the budget concentrates on the folds you judged relevant instead of the "
        "framework forcing coverage of an off-target family. Decisions are "
        "revisitable: call again to resume a fold if later evidence warrants. "
        "ALWAYS give a one-line reason per fold so the decision is auditable."
    ),
    category="reasoning",
    parallel_safe=False,
    injected_params=["run_dir"],
    params={
        "run_dir": {"type": "string", "description": "Agent run directory (injected)."},
        "stop": {
            "type": "array", "items": {"type": "string"}, "default": [],
            "description": (
                "InterPro fold ids to STOP expanding/evaluating (off-target chemistry)."
            ),
        },
        "expand": {
            "type": "array", "items": {"type": "string"}, "default": [],
            "description": (
                "InterPro fold ids to keep EXPANDING (native reactions resemble the target)."
            ),
        },
        "reasons": {
            "type": "object", "default": {},
            "description": (
                "Map of fold id -> one-line reason for the decision (why stop / why expand)."
            ),
        },
    },
)
def set_family_focus(run_dir: Path | str, stop: list | None = None,
                     expand: list | None = None, reasons: dict | None = None) -> dict:
    """Record the agent's expand/stop decisions over surveyed fold families.

    Persists to ``family_decisions.json`` (revisitable) and emits ``family_decision``
    events; the pre-evaluation budget then waives the breadth floor for stopped
    folds.  This is the action channel that lets the agent's reasoning over the
    family landscape actually steer where evaluation budget is spent.
    """
    from enzyme_discovery_agent.models.family_decisions import record_family_decisions

    stop = stop or []
    expand = expand or []
    if not stop and not expand:
        return {
            "status": "partial",
            "summary": "No fold ids given; pass stop=[...] and/or expand=[...] from the landscape.",
            "data": None,
        }
    decisions = record_family_decisions(
        run_dir, expand=expand, stop=stop, reasons=reasons or {})
    stopped_now = sorted(k for k, v in decisions.items() if v.get("decision") == "stop")
    return {
        "status": "success",
        "summary": (
            f"Family focus recorded: {len(expand)} expand, {len(stop)} stop. "
            f"{len(stopped_now)} fold(s) now set aside from the evaluation budget."
        ),
        "data": {"stopped": stopped_now,
                 "expanded": sorted(k for k, v in decisions.items()
                                    if v.get("decision") == "expand")},
    }
