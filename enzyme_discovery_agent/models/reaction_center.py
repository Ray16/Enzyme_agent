"""Deterministic reactant-vs-product reaction-center analysis.

Compares resolved non-cofactor reactant and product occurrences of a single
recorded native reaction using RDKit functional-group counts and ring counts.
Identifies which functional groups were consumed, which were formed, which
remained unchanged spectators, and the net molecularity (intermolecular
condensation, cleavage, intramolecular cyclization, or group exchange).

This module is strictly reaction-agnostic: it encodes no benchmark substrates,
target families, or enzyme names, and never predicts target activity.
"""

from __future__ import annotations

from collections import Counter
import logging

logger = logging.getLogger("enzyme_agent")

_GROUP_FIELDS = (
    ("carboxyl", "num_carboxyl"),
    ("primary_amine", "num_primary_amine"),
    ("secondary_amine", "num_secondary_amine"),
    ("tertiary_amine", "num_tertiary_amine"),
    ("amide", "num_amide"),
    ("hydroxyl", "num_hydroxyl"),
    ("thiol", "num_thiol"),
    ("ketone", "num_ketone"),
    ("aldehyde", "num_aldehyde"),
    ("ester", "num_ester"),
    ("ether", "num_ether"),
    ("epoxide", "num_epoxide"),
    ("phosphate", "num_phosphate"),
    ("sulfonate", "num_sulfonate"),
    ("nitro", "num_nitro"),
    ("nitrile", "num_nitrile"),
    ("imine", "num_imine"),
    ("guanidinium", "num_guanidinium"),
    ("aromatic_ring", "num_aromatic_ring"),
)


def _ring_count(smiles: str) -> int:
    try:
        from rdkit import Chem
    except Exception:  # noqa: BLE001
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return int(mol.GetRingInfo().NumRings())


def _side_group_inventory(occurrences: list[dict]) -> tuple[Counter, int, list[dict], int]:
    """Return (total_groups, total_rings, per_occurrence_groups, unresolved_count)."""
    from enzyme_discovery_agent.tools.substrate import compute_substrate_properties

    totals: Counter = Counter()
    total_rings = 0
    per_occ: list[dict] = []
    unresolved = 0

    for item in occurrences or []:
        smiles = (item.get("smiles") or "").strip() if isinstance(item, dict) else ""
        if not smiles:
            unresolved += 1
            continue
        try:
            props = compute_substrate_properties(smiles)
        except Exception:  # noqa: BLE001 - one unparseable SMILES must not fail the diff
            unresolved += 1
            continue
        counts = {}
        for label, attr in _GROUP_FIELDS:
            val = int(getattr(props, attr, 0) or 0)
            if val > 0:
                counts[label] = val
                totals[label] += val
        rings = _ring_count(smiles)
        total_rings += rings
        per_occ.append({
            "occurrence_id": item.get("occurrence_id"),
            "name": item.get("name"),
            "smiles": smiles,
            "groups": counts,
            "rings": rings,
        })
    return totals, total_rings, per_occ, unresolved


def _classify_molecularity(
    n_reactants: int, n_products: int, ring_delta: int
) -> str:
    if n_reactants > n_products:
        return "intermolecular_condensation"
    if n_reactants < n_products:
        return "cleavage"
    if n_reactants == 1 and n_products == 1:
        return "intramolecular_cyclization" if ring_delta > 0 else "intramolecular_transformation"
    return "group_transfer_or_exchange"


def analyze_reaction_center(
    reactants: list[dict],
    products: list[dict],
    reaction_text: str = "",
) -> dict | None:
    """Compute a deterministic functional-group and molecularity diff for one native reaction.

    Args:
        reactants: Resolved non-cofactor reactant occurrence dicts (with ``smiles``).
        products: Resolved non-cofactor product occurrence dicts (with ``smiles``).
        reaction_text: Optional recorded reaction string for provenance.

    Returns:
        Structured dict with ``consumed_groups``, ``formed_groups``,
        ``unchanged_spectator_groups``, ``molecularity``, ``ring_delta``, and
        ``per_reactant_reactivity`` — or ``None`` if RDKit or both sides are
        unavailable.
    """
    if not reactants or not products:
        return None
    try:
        from enzyme_discovery_agent.tools.substrate import _HAS_RDKIT
        if not _HAS_RDKIT:
            return None
    except Exception:  # noqa: BLE001
        return None

    r_totals, r_rings, r_items, r_unresolved = _side_group_inventory(reactants)
    p_totals, p_rings, p_items, p_unresolved = _side_group_inventory(products)
    if not r_items or not p_items:
        return None

    all_keys = sorted(set(r_totals) | set(p_totals))
    consumed = {}
    formed = {}
    spectators = {}
    for key in all_keys:
        diff = p_totals[key] - r_totals[key]
        if diff < 0:
            consumed[key] = -diff
        elif diff > 0:
            formed[key] = diff
        elif r_totals[key] > 0:
            spectators[key] = r_totals[key]

    ring_delta = p_rings - r_rings
    molecularity = _classify_molecularity(len(r_items), len(p_items), ring_delta)

    per_reactant = []
    for item in r_items:
        groups = item["groups"]
        reacting = [g for g in groups if g in consumed]
        spectator_only = [g for g in groups if g in spectators and g not in consumed]
        entry = {
            "occurrence_id": item["occurrence_id"],
            "name": item["name"],
            "candidate_reacting_groups": reacting,
            "spectator_only_groups": spectator_only,
        }
        per_reactant.append({k: v for k, v in entry.items() if v not in (None, "", [])})

    status = "complete" if (r_unresolved == 0 and p_unresolved == 0) else "partial"
    result = {
        "status": status,
        "molecularity": molecularity,
        "non_cofactor_reactant_count": len(r_items),
        "non_cofactor_product_count": len(p_items),
        "ring_delta": ring_delta,
        "consumed_groups": consumed,
        "formed_groups": formed,
        "unchanged_spectator_groups": spectators,
        "per_reactant_reactivity": per_reactant,
        "scope": (
            "Deterministic reactant-vs-product functional-group diff across resolved "
            "non-cofactor species; identifies net group changes and spectators, not "
            "enzyme mechanism or target activity."
        ),
    }
    if r_unresolved or p_unresolved:
        result["unresolved_participants"] = {
            "reactants": r_unresolved,
            "products": p_unresolved,
        }
    return result
