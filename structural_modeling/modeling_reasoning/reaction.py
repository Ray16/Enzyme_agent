#!/usr/bin/env python3
"""Reaction-coordinate knowledge: which step is being modeled, and what it implies.

This is the step-generality core. A mechanism is an ordered set of steps; each step
declares the species present at that step (its Michaelis state), the species it
consumes/produces (mass balance), and the nucleophile/electrophile ROLES of its
near-attack. Everything is expressed as roles, never enzyme- or target-specific
atoms, so the resolver can map a declared step onto any candidate.

The seed coordinates in ``knowledge/reaction_coordinates.json`` are intended to be
populated by the reasoning module's exact-grounded ``reaction-step`` extraction, so
this knowledge self-evolves with the literature exactly like the cofolding rules.
"""
from __future__ import annotations

import json
import os
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COORDINATES = os.path.join(HERE, "knowledge", "reaction_coordinates.json")
DEFAULT_SELECTORS = os.path.join(HERE, "knowledge", "atom_selectors.json")

# Metal is catalytic scaffolding present across steps; it never triggers a
# species-set mismatch on its own.
UBIQUITOUS_SPECIES = {"catalytic_metal"}


class ReactionCoordinates:
    def __init__(self, document: dict[str, Any], *, source: str | None = None):
        self.source = source
        self.schema_version = document.get("schema_version")
        self.mechanisms: dict[str, Any] = document.get("mechanisms") or {}

    @classmethod
    def load(cls, path: str = DEFAULT_COORDINATES) -> "ReactionCoordinates":
        with open(path) as handle:
            return cls(json.load(handle), source=os.path.abspath(path))

    def select(
        self,
        *,
        family_ids: tuple[str, ...] | list[str] = (),
        activation_mode: str | None = None,
        override: str | None = None,
    ) -> tuple[str | None, dict | None, str | None]:
        """Select a coordinate for a candidate; return (key, mechanism, basis).

        FAMILY FIRST: mechanism is a family-level property, so a candidate whose
        InterPro family is listed by a coordinate inherits it -- this is how the
        mechanism transfers to unannotated homologs of a known family. Falls back to
        the activation mode, then abstains. ``basis`` records which rule matched.
        """
        if override:
            return override, self.mechanisms.get(override), "override"
        family_ids = set(family_ids or ())
        for key, mech in self.mechanisms.items():
            fams = set(((mech.get("keys") or {}).get("families")) or [])
            if family_ids & fams:
                return key, mech, "family"
        for key, mech in self.mechanisms.items():
            modes = ((mech.get("keys") or {}).get("activation_mode")) or []
            if activation_mode in modes:
                return key, mech, "activation_mode"
        return None, None, None

    def select_mechanism(self, activation_mode: str | None, override: str | None = None) -> tuple[str | None, dict | None]:
        """Backward-compatible activation-mode selection."""
        key, mech, _ = self.select(activation_mode=activation_mode, override=override)
        return key, mech

    @staticmethod
    def step(mechanism: dict, index: int) -> dict | None:
        for step in mechanism.get("steps") or []:
            if step.get("index") == index:
                return step
        return None

    @staticmethod
    def n_steps(mechanism: dict) -> int:
        return len(mechanism.get("steps") or [])


def load_atom_selectors(path: str = DEFAULT_SELECTORS) -> dict:
    with open(path) as handle:
        return json.load(handle)


def cofactor_species_role(ccd: str | None, role: str | None, nucleotide_ccd: str | None) -> str | None:
    """Map a modeled cofactor to a reaction-coordinate species role."""
    if role == "catalytic_metal":
        return "catalytic_metal"
    if ccd and nucleotide_ccd and ccd == nucleotide_ccd:
        return "NTP"
    if ccd in {"ATP", "GTP", "CTP", "UTP", "ITP"}:
        return "NTP"
    return None


def modeled_species_roles(mi: dict, nucleotide_ccd: str | None) -> set[str]:
    """The set of non-protein species roles actually in the modeled system."""
    roles: set[str] = set()
    for cof in mi.get("cofactors") or []:
        role = cofactor_species_role(cof.get("ccd"), cof.get("role"), nucleotide_ccd)
        if role:
            roles.add(role)
    for sub in mi.get("target_substrates") or []:
        if sub.get("include_in_model", True) and sub.get("role"):
            roles.add(sub["role"])
    return roles
