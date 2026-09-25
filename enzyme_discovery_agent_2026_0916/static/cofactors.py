"""Common cofactors/byproducts to exclude from substrate comparison."""

COFACTOR_CHEBI_IDS = frozenset({
    "CHEBI:15377",  # water
    "CHEBI:15378",  # H+
    "CHEBI:15379",  # dioxygen
    "CHEBI:30616",  # ATP
    "CHEBI:456215", # ADP
    "CHEBI:456216", # AMP (4-)
    "CHEBI:16027",  # AMP
    "CHEBI:33019",  # diphosphate (PPi)
    "CHEBI:43474",  # phosphate (Pi)
    "CHEBI:57287",  # CoA
    "CHEBI:57540",  # NAD+
    "CHEBI:57945",  # NADH
    "CHEBI:58349",  # NADP+
    "CHEBI:57783",  # NADPH
    "CHEBI:15422",  # ATP (neutral)
    "CHEBI:16761",  # ADP (neutral)
    "CHEBI:16708",  # AMP (neutral)
    "CHEBI:29888",  # Pi (3-)
})
