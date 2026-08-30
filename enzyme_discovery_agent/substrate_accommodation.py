"""Faithful, role-aware substrate-accommodation facts for AGENT reasoning.

The pipeline must decide whether an enzyme can accommodate the target's
substrates.  A single blended scalar (an equal-weight mean over ~15
physicochemical axes) is the wrong instrument: it lets high size/shape
similarity drown the ONE axis that actually gates the reaction — whether a
native counterpart shares the target substrate's reactive functional groups.
That is how an enzyme whose partner for one target substrate is only a
single-atom molecule can score as high as one whose partner is a genuine
multi-atom analogue of that target substrate.

This module does NOT decide relevance and applies NO weights or thresholds to
rank.  It EXTRACTS the grounded, role-aware facts — per target substrate, its
best distinct native counterpart, whether they share functional groups, and how
small that counterpart is — and hands them to the agent to reason over together
with the target chemistry contract.  Nothing weak is averaged away: a target
substrate whose only native counterpart is functional-group-mismatched or
cofactor-scale is surfaced as an explicit fact, not hidden.

Deterministic and offline-testable.  ``native_smiles`` should be the enzyme's
native SUBSTRATES (from UniProt/Rhea/ChEBI catalytic activity), the same source
the rest of the pipeline uses.
"""

from __future__ import annotations

import logging

from enzyme_discovery_agent.substrate_fit import assign_one_to_one

logger = logging.getLogger("enzyme_agent")


def _heavy_atoms(smiles: str) -> int | None:
    try:
        from rdkit import Chem
    except Exception:  # noqa: BLE001 - RDKit unavailable
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else None


def accommodation_facts(
    target_smiles: list[str], native_smiles: list[str],
) -> dict | None:
    """Role-aware accommodation facts for one enzyme vs the target substrates.

    For EACH target substrate, assign its best DISTINCT native counterpart
    (one-to-one maximin by functional-group overlap — the shared assignment the
    survey and evaluation paths use) and report the raw facts, KEEPING every
    target substrate.  A target whose counterpart shares no functional groups, is
    absent, or is cofactor-scale is listed in ``unaccommodated_targets`` so the
    "the enzyme must host every target substrate" constraint is visible, not
    blended away.  Returns None only when nothing resolves.
    """
    targets = [s for s in (target_smiles or []) if s and s.strip()]
    natives = [s for s in (native_smiles or []) if s and s.strip()]
    if not targets:
        return None

    from enzyme_discovery_agent.chemistry_logic import compare_substrate_properties
    from enzyme_discovery_agent.tools.substrate import compute_substrate_properties

    _cmp_cache: dict = {}

    def _cmp(t: str, n: str) -> dict:
        key = (t, n)
        if key not in _cmp_cache:
            try:
                _cmp_cache[key] = compare_substrate_properties(
                    t, n, compute_substrate_properties) or {}
            except Exception:  # noqa: BLE001 - one bad pair must not sink the member
                _cmp_cache[key] = {}
        return _cmp_cache[key]

    def _fg(t: str, n: str) -> float:
        val = _cmp(t, n).get("functional_group_similarity")
        # -1 sentinel so an unresolved pair never wins the maximin assignment.
        return val if isinstance(val, (int, float)) else -1.0

    per_target: list[dict] = []
    unaccommodated: list[str] = []
    for t, n, _score in assign_one_to_one(targets, natives, _fg):
        if n is None:
            per_target.append({
                "target_smiles": t,
                "native_smiles": None,
                "functional_group_similarity": None,
                "shares_functional_groups": False,
                "note": "no distinct native counterpart for this target substrate",
            })
            unaccommodated.append(t)
            continue
        cmp = _cmp(t, n)
        fg = cmp.get("functional_group_similarity")
        fg = fg if isinstance(fg, (int, float)) else None
        heavy = _heavy_atoms(n)
        # A one- or two-heavy-atom native (e.g. water or a single-atom ion)
        # cannot be a structural analogue of a multi-atom target substrate — a
        # structural FACT, not a tuned relevance threshold.
        minimal = heavy is not None and heavy <= 2
        shares = bool(fg and fg > 0)
        entry = {
            "target_smiles": t,
            "target_functional_groups": cmp.get("functional_groups_target"),
            "native_smiles": n,
            "native_functional_groups": cmp.get("functional_groups_native"),
            "functional_group_similarity": (round(fg, 3) if fg is not None else None),
            "shares_functional_groups": shares,
            "native_heavy_atoms": heavy,
            "native_is_cofactor_scale": minimal,
            "mw_target": cmp.get("mw_target"),
            "mw_native": cmp.get("mw_native"),
            "chain_length_target": cmp.get("chain_length_target"),
            "chain_length_native": cmp.get("chain_length_native"),
        }
        per_target.append(entry)
        # A target substrate is UNACCOMMODATED when its only counterpart shares
        # none of its reactive groups OR is cofactor-scale — the enzyme lacks a
        # genuine partner for it, however well other axes (size/shape) match.
        if not shares or minimal:
            unaccommodated.append(t)

    matched_fg = [p["functional_group_similarity"] for p in per_target
                  if isinstance(p.get("functional_group_similarity"), (int, float))]
    return {
        "per_target": per_target,
        "n_target_substrates": len(targets),
        "unaccommodated_targets": unaccommodated,
        "all_target_substrates_accommodated": not unaccommodated,
        "weakest_functional_group_similarity": (
            round(min(matched_fg), 3) if matched_fg else None),
    }
