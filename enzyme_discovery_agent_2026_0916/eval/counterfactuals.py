"""Counterfactual paired target sets for constraint-sensitivity testing.

The question these pairs answer is behavioral, not linguistic: when the target
chemistry changes while answer recall becomes unhelpful, do the pipeline's
DECISIONS change appropriately?  Each pair changes exactly ONE chemical
dimension; the test harness asserts that the shared chemistry contract and the
deterministic guards react to that one change.

The pairs are deliberately generic (an ester/amide toy chemistry on unnamed
small molecules) so they exercise constraint sensitivity WITHOUT leaking any
benchmark substrate, product, or enzyme — testing reasoning, not memorization.

A future extension (plan step 7) runs the full LLM agent on these pairs to score
the Variant A/B/C behavioral metrics end-to-end; that runner is out of scope for
the deterministic harness here and is intentionally not invoked in tests.
"""

from __future__ import annotations

# Generic toy substrates (no benchmark chemistry).
_ALCOHOL = "CCO"
_ACID = "CC(=O)O"
_AMINE = "CCN"

# Each pair: one changed dimension, plus the contract constraint that must flip.
COUNTERFACTUAL_PAIRS: list[dict] = [
    {
        "id": "direction_formation_to_cleavage",
        "dimension": "reaction_direction",
        "base": {
            "query": "enzymes that catalyze the formation of an ester bond",
            "substrate_smiles": f"{_ALCOHOL}.{_ACID}",
        },
        "variant": {
            "query": "enzymes that catalyze the hydrolysis of an ester bond",
            "substrate_smiles": None,
        },
        "expect": {"field": "reaction_direction",
                   "base": "formation", "variant": "cleavage"},
        "required_behavior": "Do not retain formation-only strategies for a cleavage target.",
    },
    {
        "id": "topology_free_to_carrier",
        "dimension": "substrate_topology",
        "base": {
            "query": "enzymes that catalyze the formation of an amide bond",
            "substrate_smiles": f"{_AMINE}.{_ACID}",
            "profile": {"substrate": {"topology": "free_small_molecule"}},
        },
        "variant": {
            "query": "enzymes that catalyze the formation of an amide bond",
            "substrate_smiles": None,
            "profile": {"substrate": {"topology": "carrier_protein_tethered"}},
        },
        "expect": {"field": "substrate_topology",
                   "base": "free_small_molecule", "variant": "carrier_protein_tethered"},
        "required_behavior": (
            "Do not reject carrier-dependent strategies merely because they are "
            "carrier-dependent when the target itself is carrier-bound."),
    },
    {
        "id": "topology_free_to_polymer",
        "dimension": "substrate_topology",
        "base": {
            "query": "enzymes that catalyze the formation of an amide bond",
            "substrate_smiles": f"{_AMINE}.{_ACID}",
            "profile": {"substrate": {"topology": "free_small_molecule"}},
        },
        "variant": {
            "query": "enzymes that catalyze the formation of an amide bond",
            "substrate_smiles": None,
            "profile": {"substrate": {"topology": "polymer"}},
        },
        "expect": {"field": "substrate_topology",
                   "base": "free_small_molecule", "variant": "polymer"},
        "required_behavior": "Do not promote strictly monomer-only evidence as equivalent for a polymeric target.",
    },
    {
        "id": "bond_ester_to_amide",
        "dimension": "bond_types",
        "base": {
            "query": "enzymes that catalyze the formation of an ester bond",
            "substrate_smiles": f"{_ALCOHOL}.{_ACID}",
        },
        "variant": {
            "query": "enzymes that catalyze the formation of an amide bond",
            "substrate_smiles": f"{_AMINE}.{_ACID}",
        },
        "expect": {"field": "bond_types", "base": "ester", "variant": "amide"},
        "required_behavior": "Search/rank on the changed activation chemistry, not scaffold similarity.",
    },
]

# A synthetic-analog case is asserted directly in the harness rather than as a
# contract-flip pair: the required behavior is that a member with the correct
# native mechanism (direction + bond) is RETAINED even though its native
# substrate differs from a synthetic target — mechanism possibility is kept
# without claiming native-substrate evidence.
SYNTHETIC_ANALOG_CASE = {
    "id": "synthetic_analog_retains_mechanism",
    "target_direction": "formation",
    "target_bond_types": ["amide"],
    "member_native_reaction": "carboxylic acid + amine + ATP = amide + AMP + diphosphate",
    "member_ec": "6.3.1.1",
    "required_behavior": (
        "Retain mechanism-level possibility for a synthetic-analog target without "
        "asserting native-substrate evidence."),
}
