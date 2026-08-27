"""Golden cases for the per-tool eval harness.

Each mechanism case PINS an open-access paper (verified fetchable) so the case tests the
EXTRACTION in isolation, not the run-to-run variance of paper discovery. Expectations are
deliberately lenient where the literature itself is ambiguous (e.g. L-DEX's first SN2 step is
variously called "SN2" or "acyl_transfer") and strict where the chemistry is unambiguous
(e.g. a serine hydrolase is NOT a clean in-line substitution; a dehydrogenase has no
nucleophile but does have a hydride donor/acceptor pair).
"""

from __future__ import annotations

# --- mechanism cases (extract_reaction_mechanism) ---
MECHANISM_CASES = [
    {
        "kind": "mechanism",
        "name": "L-DEX (SN2 substitution)",
        "enzyme": "L-2-haloacid dehalogenase (L-DEX)",
        "paper": "PMC6100074",  # Dehalogenases review (OA, fetchable)
        "expect": {
            "mechanism_class_in": ["SN2", "SN1", "acyl_transfer", "hydrolysis",
                                   "addition_elimination"],
            "is_clean_inline_substitution": True,
            "participant_roles_any": ["nucleophile", "leaving_group"],
            "nucleophile_source_in": ["protein_residue"],
            "reactive_pair_present": True,
            "min_confidence": "low",
            # modeling-field correctness:
            "stereochemistry_in": ["inversion"],          # SN2 inverts config
            "covalent_intermediate_forms": True,          # aspartyl-ester intermediate
            "substrate_reactive_atom_regex": r"C2|alpha|α",
        },
    },
    {
        "kind": "mechanism",
        "name": "PETase (serine hydrolase)",
        "enzyme": "PETase poly(ethylene terephthalate) hydrolase Ideonella sakaiensis",
        "paper": "PMC5785972",  # Joo 2018 IsPETase structure paper — names the Ser-His-Asp triad
        "expect": {
            "mechanism_class_in": ["hydrolysis", "addition_elimination", "acyl_transfer"],
            "is_clean_inline_substitution": False,
            "participant_roles_any": ["nucleophile", "base"],
            "catalytic_residues_regex": r"Ser|His|Asp",
            "reactive_pair_present": True,
            "min_confidence": "medium",
            # modeling-field correctness:
            "covalent_intermediate_forms": True,          # acyl-enzyme intermediate
            "covalent_residue_regex": r"Ser",
            "substrate_reactive_atom_regex": r"carbonyl|ester",
            # Joo 2018 reports apo / wild-type IsPETase structures (5XJH, 5XG0), NOT a
            # substrate-bound Michaelis complex — so the tool correctly files those under
            # reference_structures and leaves michaelis_structure null. Check what the paper
            # actually provides (a reference PDB), not a Michaelis complex it does not contain.
            "reference_structures_present": True,         # e.g. 5XJH / 5XG0
        },
    },
    {
        "kind": "mechanism",
        "name": "Alcohol dehydrogenase (redox / hydride)",
        "enzyme": "alcohol dehydrogenase (NAD-dependent, zinc)",
        "paper": "PMC12858457",  # ADH-catalyzed reduction paper (OA, fetchable)
        "expect": {
            "mechanism_class_in": ["redox_hydride", "redox_electron", "other"],
            "participant_roles_any": ["hydride_donor", "hydride_acceptor"],
            "nucleophile_null": True,          # a dehydrogenase has NO nucleophile
            "metals_regex": r"[Zz]n",
            # modeling-field correctness:
            "covalent_intermediate_forms": False,         # no covalent intermediate in ADH
            "substrate_reactive_atom_regex": r"carbonyl|C=?O",
            "cofactor_ccd_includes": ["NAD"],             # Boltz CCD code
            "metal_ccd_includes": ["ZN"],
        },
    },
]

# --- measurement cases (extract_experimental_measurements) ---
MEASUREMENT_CASES = [
    {
        "kind": "measurement",
        "name": "PET-hydrolase kinetics (PMC9086285)",
        "identifier": "PMC9086285",
        "context_hint": "PET hydrolase / cutinase kinetic parameters (kcat, Km)",
        "expect": {
            "min_count": 20,
            "rows_include": [
                {"enzyme_re": "MtCut", "substrate_re": r"pNP-?C2", "metric": "kcat",
                 "value": 4.47, "tol": 0.6},
                {"enzyme_re": "MtCut", "substrate_re": r"pNP-?C2", "metric": "Km",
                 "value": 0.55, "tol": 0.15},
            ],
        },
    },
]

CASES = MECHANISM_CASES + MEASUREMENT_CASES
