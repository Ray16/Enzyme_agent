"""Check a reviewed protonation assignment against prepared coordinates.

These checks establish bookkeeping and local geometry, not pKa accuracy or a
unique catalytic microstate. Uncertain microstates need separate prepared inputs.
"""
from __future__ import annotations

import math

import numpy as np

HISTIDINE_STATES = {"HID": (1, 0), "HIE": (0, 1), "HIP": (1, 1)}


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing protonation review: {label}")


def validate_protonation_review(atoms, report):
    """Fail closed on unreviewed key residues or inconsistent histidine H atoms."""
    conditions = report.get("assay_conditions", {})
    ph = conditions.get("pH")
    if isinstance(ph, bool) or not isinstance(ph, (int, float)) or not math.isfinite(ph) or not 0 <= ph <= 14:
        raise ValueError("A documented assay pH is required for protonation review")
    text(conditions.get("pH_source"), "assay pH source")
    review = report.get("key_residue_review", {})
    text(review.get("coverage_evidence"), "coverage of catalytic and nearby titratable residues")
    sites = review.get("histidines")
    if not isinstance(sites, list):
        raise ValueError("Explicit key-histidine review is required")
    if not sites:
        text(review.get("no_key_histidines_reason"), "reason no key histidines were identified")
    identifiers, used_nitrogens = set(), set()
    hydrogen_indices = np.flatnonzero(atoms.numbers == 1)
    names = atoms.arrays.get("atomnames", atoms.arrays.get("atomtypes"))
    for site in sites:
        text(site.get("residue_id"), "unambiguous histidine chain/residue identifier")
        if site["residue_id"] in identifiers:
            raise ValueError("Duplicate histidine review identifier")
        identifiers.add(site["residue_id"])
        state = site.get("state")
        if state not in HISTIDINE_STATES:
            raise ValueError("Histidine must explicitly select HID, HIE or HIP")
        considered = site.get("considered_states", [])
        if not considered or state not in considered or not set(considered) <= HISTIDINE_STATES.keys():
            raise ValueError("Record the histidine states considered and the selected state")
        text(site.get("role"), "histidine catalytic/binding/environment role")
        text(site.get("rationale"), "histidine state rationale")
        if not isinstance(site.get("evidence"), list) or not site["evidence"] or not all(isinstance(x, str) and x.strip() for x in site["evidence"]):
            raise ValueError("Histidine state needs traceable evidence")
        indices = [site.get("ND1"), site.get("NE2")]
        if len(set(indices)) != 2 or any(type(i) is not int or not 0 <= i < len(atoms)
                                         or atoms[i].symbol != "N" for i in indices):
            raise ValueError("Histidine ND1/NE2 indices must identify distinct nitrogen atoms")
        if used_nitrogens & set(indices):
            raise ValueError("Histidine nitrogen assigned to more than one residue")
        used_nitrogens.update(indices)
        if names is None or [str(names[i]).strip() for i in indices] != ["ND1", "NE2"]:
            raise ValueError("Prepared coordinates must retain ND1/NE2 atom names for the declared histidine")
        counts = []
        for index in indices:
            distances = atoms.get_distances(index, hydrogen_indices, mic=bool(atoms.pbc.any()))
            if (distances < 0.7).any():
                raise ValueError("Unphysical histidine N-H contact")
            counts.append(int((distances <= 1.25).sum()))
        if tuple(counts) != HISTIDINE_STATES[state]:
            raise ValueError(f"{site['residue_id']}: prepared hydrogens disagree with {state}: {counts}")
        bindings = site.get("metal_bindings", [])
        for binding in bindings:
            nitrogen = binding.get("nitrogen")
            if nitrogen not in ("ND1", "NE2"):
                raise ValueError("Histidine metal-binding nitrogen must be ND1 or NE2")
            nitrogen_position = ("ND1", "NE2").index(nitrogen)
            if counts[nitrogen_position]:
                raise ValueError("A declared metal-donor histidine nitrogen carries a proton")
            metal = binding.get("metal_index")
            if type(metal) is not int or not 0 <= metal < len(atoms) or atoms[metal].symbol not in {
                    "Mg", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}:
                raise ValueError("Invalid declared histidine metal partner")
            cutoff = binding.get("max_distance_A")
            if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("Specify a reviewed histidine-metal distance limit")
            if atoms.get_distance(indices[nitrogen_position], metal, mic=bool(atoms.pbc.any())) > cutoff:
                raise ValueError("Declared histidine metal coordination is absent from prepared geometry")
    return {"pH": ph, "histidines_checked": len(sites),
            "scope": "review completeness and starting H geometry; not pKa validation"}
