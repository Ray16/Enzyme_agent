"""Comprehensive IUBMB EC -> bond-type mapping covering all 7 EC classes.

Keys are (ec_class, ec_subclass) tuples; values are dicts with:
  bond_type: primary bond type acted on
  direction: "formation", "cleavage", "transfer", "redox", "isomerization", "translocation"
  context:   additional sub-subclass context
"""

EC_BOND_TYPE_MAP: dict[tuple[int, int], dict] = {
    # EC 1 — Oxidoreductases
    (1, 1): {"bond_type": "C-OH", "direction": "redox", "context": "alcohol oxidation"},
    (1, 2): {"bond_type": "C=O", "direction": "redox", "context": "aldehyde/ketone oxidation"},
    (1, 3): {"bond_type": "C-C", "direction": "redox", "context": "C-C desaturation"},
    (1, 4): {"bond_type": "C-N", "direction": "redox", "context": "amine oxidation"},
    (1, 5): {"bond_type": "C-H", "direction": "redox", "context": "CH-NH oxidation"},
    (1, 6): {"bond_type": "NADH", "direction": "redox", "context": "NADH/NADPH oxidation"},
    (1, 7): {"bond_type": "other", "direction": "redox", "context": "other nitrogenous donors"},
    (1, 8): {"bond_type": "sulfur", "direction": "redox", "context": "sulfur group oxidation"},
    (1, 9): {"bond_type": "heme", "direction": "redox", "context": "heme/Cu electron transfer"},
    (1, 10): {"bond_type": "diphenol", "direction": "redox", "context": "diphenol oxidation"},
    (1, 11): {"bond_type": "peroxide", "direction": "redox", "context": "peroxidase"},
    (1, 12): {"bond_type": "hydrogen", "direction": "redox", "context": "hydrogen donor"},
    (1, 13): {"bond_type": "C-H", "direction": "redox", "context": "single donor + O2"},
    (1, 14): {"bond_type": "C-H", "direction": "redox", "context": "paired donors + O2 (oxygenase)"},
    (1, 15): {"bond_type": "superoxide", "direction": "redox", "context": "superoxide as acceptor"},
    (1, 16): {"bond_type": "iron-sulfur", "direction": "redox", "context": "Fe-S protein oxidation"},
    (1, 17): {"bond_type": "C-H", "direction": "redox", "context": "CH2 group oxidation"},
    (1, 18): {"bond_type": "metal", "direction": "redox", "context": "metal ion as donor"},
    (1, 19): {"bond_type": "reduced_flavo", "direction": "redox", "context": "reduced flavodoxin"},
    (1, 20): {"bond_type": "phosphorus", "direction": "redox", "context": "phosphorus/arsenic donor"},
    (1, 21): {"bond_type": "X-H", "direction": "redox", "context": "X-H + Y-H = X-Y"},
    (1, 23): {"bond_type": "ferredoxin", "direction": "redox", "context": "ferredoxin reductase"},
    (1, 97): {"bond_type": "other", "direction": "redox", "context": "other oxidoreductases"},
    # EC 2 — Transferases
    (2, 1): {"bond_type": "C1_group", "direction": "transfer", "context": "methyl/formyl/carboxyl transfer"},
    (2, 2): {"bond_type": "aldehyde_ketone", "direction": "transfer", "context": "aldehyde/ketone transfer"},
    (2, 3): {"bond_type": "acyl", "direction": "transfer", "context": "acyl group transfer"},
    (2, 4): {"bond_type": "glycosidic", "direction": "transfer", "context": "glycosyl transfer"},
    (2, 5): {"bond_type": "alkyl_aryl", "direction": "transfer", "context": "alkyl/aryl transfer (prenyl, etc.)"},
    (2, 6): {"bond_type": "amino", "direction": "transfer", "context": "amino group transfer"},
    (2, 7): {"bond_type": "phosphoester", "direction": "transfer", "context": "phosphoryl transfer (kinase)"},
    (2, 8): {"bond_type": "sulfur", "direction": "transfer", "context": "sulfur group transfer"},
    (2, 9): {"bond_type": "selenium", "direction": "transfer", "context": "selenium group transfer"},
    (2, 10): {"bond_type": "molybdenum", "direction": "transfer", "context": "Mo/W transfer"},
    # EC 3 — Hydrolases
    (3, 1): {"bond_type": "ester", "direction": "cleavage", "context": "ester bond hydrolysis"},
    # EC 3.1 spans two different electrophiles. These sub-subclasses hydrolyse
    # a bond to PHOSPHORUS, not to a carbonyl carbon, so they must not satisfy a
    # target whose electrophile is an acyl carbon.
    (3, 1, 3): {"bond_type": "phosphoester", "direction": "cleavage",
                "context": "phosphoric monoester hydrolysis"},
    (3, 1, 4): {"bond_type": "phosphoester", "direction": "cleavage",
                "context": "phosphoric diester hydrolysis"},
    (3, 1, 8): {"bond_type": "phosphoester", "direction": "cleavage",
                "context": "phosphoric triester hydrolysis"},
    (3, 2): {"bond_type": "glycosidic", "direction": "cleavage", "context": "glycosidic bond hydrolysis"},
    (3, 3): {"bond_type": "ether", "direction": "cleavage", "context": "ether bond hydrolysis"},
    (3, 4): {"bond_type": "peptide", "direction": "cleavage", "context": "peptide bond hydrolysis (protease)"},
    (3, 5): {"bond_type": "amide", "direction": "cleavage", "context": "non-peptide C-N bond hydrolysis"},
    (3, 6): {"bond_type": "acid_anhydride", "direction": "cleavage", "context": "acid anhydride hydrolysis"},
    (3, 7): {"bond_type": "C-C", "direction": "cleavage", "context": "C-C bond hydrolysis"},
    (3, 8): {"bond_type": "C-halide", "direction": "cleavage", "context": "C-halide bond hydrolysis"},
    (3, 9): {"bond_type": "P-N", "direction": "cleavage", "context": "P-N bond hydrolysis"},
    (3, 10): {"bond_type": "S-N", "direction": "cleavage", "context": "S-N bond hydrolysis"},
    (3, 11): {"bond_type": "C-P", "direction": "cleavage", "context": "C-P bond hydrolysis"},
    (3, 12): {"bond_type": "S-S", "direction": "cleavage", "context": "S-S bond hydrolysis"},
    (3, 13): {"bond_type": "C-S", "direction": "cleavage", "context": "C-S bond hydrolysis"},
    # EC 4 — Lyases
    (4, 1): {"bond_type": "C-C", "direction": "cleavage", "context": "C-C lyase"},
    (4, 2): {"bond_type": "C-O", "direction": "cleavage", "context": "C-O lyase (dehydratase)"},
    (4, 3): {"bond_type": "C-N", "direction": "cleavage", "context": "C-N lyase (ammonia-lyase)"},
    (4, 4): {"bond_type": "C-S", "direction": "cleavage", "context": "C-S lyase"},
    (4, 5): {"bond_type": "C-halide", "direction": "cleavage", "context": "C-halide lyase"},
    (4, 6): {"bond_type": "P-O", "direction": "cleavage", "context": "P-O lyase (cyclase)"},
    (4, 99): {"bond_type": "other", "direction": "cleavage", "context": "other lyases"},
    # EC 5 — Isomerases
    (5, 1): {"bond_type": "stereocenter", "direction": "isomerization", "context": "racemase/epimerase"},
    (5, 2): {"bond_type": "cis_trans", "direction": "isomerization", "context": "cis-trans isomerase"},
    (5, 3): {"bond_type": "intramolecular_redox", "direction": "isomerization", "context": "intramolecular oxidoreductase"},
    (5, 4): {"bond_type": "intramolecular_transfer", "direction": "isomerization", "context": "intramolecular transferase (mutase)"},
    (5, 5): {"bond_type": "intramolecular_lyase", "direction": "isomerization", "context": "intramolecular lyase"},
    (5, 6): {"bond_type": "topoisomerase", "direction": "isomerization", "context": "topoisomerase"},
    (5, 99): {"bond_type": "other", "direction": "isomerization", "context": "other isomerases"},
    # EC 6 — Ligases
    (6, 1): {"bond_type": "ester", "direction": "formation", "context": "C-O ligase (aminoacyl-tRNA synthetase)"},
    (6, 2): {"bond_type": "thioester", "direction": "formation", "context": "C-S ligase (acid-thiol ligase)"},
    (6, 3): {"bond_type": "amide", "direction": "formation", "context": "C-N ligase (amide bond formation)"},
    (6, 4): {"bond_type": "C-C", "direction": "formation", "context": "C-C ligase (carboxylase)"},
    (6, 5): {"bond_type": "C-halide", "direction": "formation", "context": "C-halide ligase"},
    (6, 6): {"bond_type": "metal-N", "direction": "formation", "context": "metal-nitrogen ligase"},
    # EC 7 — Translocases
    (7, 1): {"bond_type": "translocation", "direction": "translocation", "context": "hydrolysis-coupled transport"},
    (7, 2): {"bond_type": "translocation", "direction": "translocation", "context": "decarboxylation-coupled transport"},
    (7, 3): {"bond_type": "translocation", "direction": "translocation", "context": "reaction-coupled transport"},
    (7, 4): {"bond_type": "translocation", "direction": "translocation", "context": "electrochemical-coupled transport"},
    (7, 5): {"bond_type": "translocation", "direction": "translocation", "context": "redox-coupled transport"},
    (7, 6): {"bond_type": "translocation", "direction": "translocation", "context": "light-coupled transport"},
}
