"""Versioned JSON Schema for `extract_reaction_mechanism` output — the single source of
truth for the class-agnostic mechanism-facts contract.

Why a schema module: the schema is the interface between the extractor (facts) and the main
agent (which reasons over those facts to choose a model). Pinning it here lets us (a) validate
the tool's output at runtime so drift/typos surface immediately, and (b) assert in the
tool-eval harness that the RIGHT fields populate for the RIGHT enzyme class. Keep this in sync
with `_MECH_SYS` in ``mechanism_extraction.py``.
"""

from __future__ import annotations

MECHANISM_FACTS_SCHEMA_VERSION = "1.0"

_MECHANISM_CLASSES = [
    "SN2", "SN1", "SNAr", "addition_elimination", "acyl_transfer", "hydrolysis",
    "redox_hydride", "redox_electron", "radical", "pericyclic", "isomerization",
    "elimination", "carboxylation", "aldol", "metal_catalyzed", "acid_base", "other", None,
]
_PARTICIPANT_ROLES = [
    "nucleophile", "electrophile", "leaving_group", "hydride_donor", "hydride_acceptor",
    "proton_donor", "proton_acceptor", "electron_donor", "electron_acceptor",
    "radical_initiator", "metal_center", "electrophile_activator", "oxyanion_stabilizer",
    "stabilizer", "base", "acid", "other",
]
_ON = ["substrate", "cofactor", "enzyme", "water", "metal", None]

# A permissive JSON Schema: it constrains TYPES and ENUMS (so a bad mechanism_class or a
# non-list `participants` is caught) but does not force every field to be present, since the
# extractor legitimately nulls the substitution fields for non-substitution enzymes.
MECHANISM_FACTS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": f"enzyme mechanism facts v{MECHANISM_FACTS_SCHEMA_VERSION}",
    "type": "object",
    "required": ["mechanism_class", "confidence"],
    "additionalProperties": True,
    "properties": {
        "reaction": {"type": ["string", "null"]},
        "mechanism_class": {"enum": _MECHANISM_CLASSES},
        "chemical_step": {"type": ["string", "null"]},
        "bonds_broken": {"type": ["array", "null"], "items": {"type": "object"}},
        "bonds_formed": {"type": ["array", "null"], "items": {"type": "object"}},
        "participants": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "role": {"enum": _PARTICIPANT_ROLES},
                    "identity": {"type": ["string", "null"]},
                    "atom": {"type": ["string", "null"]},
                    "on": {"enum": _ON},
                },
            },
        },
        "reactive_pair": {
            "type": ["object", "null"],
            "properties": {
                "atom_a": {"type": ["string", "null"]},
                "atom_b": {"type": ["string", "null"]},
                "note": {"type": ["string", "null"]},
            },
        },
        "substrate": {
            "type": ["object", "null"],
            "properties": {
                "identity": {"type": ["string", "null"]},
                "reactive_atom": {"type": ["string", "null"]},
                "smiles": {"type": ["string", "null"]},
            },
        },
        "cofactors": {"type": ["array", "null"], "items": {"type": "object"}},
        "metals": {"type": ["array", "null"], "items": {"type": "object"}},
        "catalytic_residues": {"type": ["array", "null"], "items": {"type": "object"}},
        "stereochemistry": {"enum": ["inversion", "retention", "racemization", "none", None]},
        "covalent_intermediate": {
            "type": ["object", "null"],
            "properties": {
                "forms": {"type": ["boolean", "null"]},
                "residue": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
            },
        },
        "enzyme_state": {
            "type": ["object", "null"],
            "properties": {
                "state": {"enum": ["intact", "processed", "unknown"]},
                "cleavage_sites": {"type": ["array", "null"], "items": {"type": "object"}},
                "mature_chains": {"type": ["array", "null"], "items": {"type": "object"}},
                "evidence": {"type": ["string", "null"]},
            },
        },
        "michaelis_structure": {"type": ["string", "null"]},
        "reference_structures": {"type": ["array", "null"], "items": {"type": "string"}},
        "substitution_center": {"enum": ["carbon", "phosphorus", "sulfur", "other", None]},
        "is_clean_inline_substitution": {"type": ["boolean", "null"]},
        "nucleophile": {"type": ["object", "null"]},
        "electrophile": {"type": ["object", "null"]},
        "leaving_group": {"type": ["object", "null"]},
        "evidence": {"type": ["string", "null"]},
        "confidence": {"enum": ["high", "medium", "low"]},
    },
}


def validate_mechanism_facts(facts: dict) -> list[str]:
    """Return a list of human-readable schema violations (empty list == valid).

    Uses ``jsonschema`` when available; falls back to a minimal hand check so the validator
    never hard-depends on the package."""
    if not isinstance(facts, dict):
        return ["output is not a JSON object"]
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(MECHANISM_FACTS_SCHEMA)
        out = []
        for err in sorted(validator.iter_errors(facts), key=lambda e: list(e.path)):
            loc = "/".join(str(p) for p in err.path) or "(root)"
            out.append(f"{loc}: {err.message}")
        return out
    except Exception:  # noqa: BLE001 - jsonschema missing or internal error: minimal fallback
        problems = []
        for key in ("mechanism_class", "confidence"):
            if key not in facts:
                problems.append(f"(root): missing required field '{key}'")
        if facts.get("mechanism_class") not in _MECHANISM_CLASSES:
            problems.append(f"mechanism_class: {facts.get('mechanism_class')!r} not an allowed value")
        if facts.get("confidence") not in ("high", "medium", "low"):
            problems.append(f"confidence: {facts.get('confidence')!r} not in high|medium|low")
        for arr in ("participants", "bonds_broken", "bonds_formed", "cofactors", "metals",
                    "catalytic_residues", "reference_structures"):
            v = facts.get(arr)
            if v is not None and not isinstance(v, list):
                problems.append(f"{arr}: expected an array, got {type(v).__name__}")
        return problems
