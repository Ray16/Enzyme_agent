"""Ground-truth benchmark campaigns for literature-exploration evaluation.

DESIGN: goals change, so recall is **data-driven**. Each benchmark declares any
number of named ``recall_targets``; the scorer (metrics.py) computes each one
generically, so redefining or adding a goal is a benchmark edit — never a code
change. A target has:
    scope  - "papers"    : match against retrieved paper titles+abstracts
             "artifacts" : match against ALL run artifacts (papers + resolved
                           UniProt/candidate records, so accessions/enzyme
                           functions count)
    values - lowercase substrings; a target "hits" if ANY value is found, and
             its recall = fraction of values found (useful when you want ALL of
             a specific set, e.g. both answer accessions).
``primary_targets`` names which targets define ``found_answer`` (strict ground
truth); other targets are reported as diagnostics (e.g. mechanism-class recall).

Answer identities are used ONLY to score after the fact — never fed to the
agent — preserving the leakage firewall.
"""

BENCHMARKS = {
    # Amide-synthetase ligation. Strict answers = the dapdiamide ATP-grasp
    # ligases (recall failure under generic queries). Mechanism-class target
    # credits finding the broader ATP-dependent amide-synthetase space (e.g. the
    # NIS/siderophore synthetases the agent reaches via mechanistic reasoning).
    "amide_synthetase": {
        "query": ("find amide synthetases that ligates adipic acid and "
                  "hexamethylenediamine for nylon-6,6 synthesis"),
        "substrate_name": "adipic acid and hexamethylenediamine (nylon-6,6 monomers)",
        "recall_targets": {
            "specific_accession": {"scope": "artifacts",
                                   "values": ["e2ja32", "a0a2h4t912"]},
            "specific_member": {"scope": "papers",
                                "values": ["dapdiamide", "ddag", "sfab"]},
            "mechanism_class": {"scope": "artifacts",
                                "values": ["amide ligase", "amide bond synthetase",
                                           "atp-grasp", "adenylation",
                                           "nonribosomal", "siderophore synthetase",
                                           "nis synthetase"]},
        },
        "primary_targets": ["specific_accession", "specific_member"],
        # Coverage: the distinct amide-BOND-FORMING enzyme families the agent
        # should span. A run is "narrow" if it covers only one or two. NIS
        # synthetases couple dicarboxylate + diamine — closest to nylon chemistry.
        "family_coverage": {
            "atp_grasp_ligase": ["atp-grasp", "d-ala-d-ala", "mur ligase",
                                 "glutathione synthetase", "dapdiamide"],
            "nis_synthetase": ["nis synthetase", "nrps-independent siderophore",
                               "iuca", "iucc", "asba", "asbb", "desd",
                               "aerobactin", "petrobactin", "rhizobactin"],
            "adenylation_nrps": ["adenylation domain", "nonribosomal peptide",
                                 "amide bond synthetase", "mcba", "sfab"],
            "amidotransferase": ["asparagine synthetase", "glutamine synthetase",
                                 "amidotransferase"],
            "acyl_transfer": ["n-acyltransferase", "acyl-coa", "gnat acetyltransferase"],
            "amidase_reverse": ["amidase", "acylase", "aminoacylase", "peptidase"],
        },
    },
    # Nylon hydrolysis: recall-OK / precision-imperfect (NylC vs NylB/NylA).
    "nylon_hydrolase": {
        "query": "enzymes that hydrolyze nylon-6 oligomers",
        "substrate_name": "nylon-6 oligomer (6-aminohexanoate linear oligomer)",
        "recall_targets": {
            "specific_member": {"scope": "papers",
                                "values": ["nylc", "nylb", "nyla",
                                           "6-aminohexanoate", "nylon oligomer hydrolase"]},
            "mechanism_class": {"scope": "artifacts",
                                "values": ["amidase", "hydrolase", "beta-lactamase fold",
                                           "n-terminal nucleophile"]},
        },
        "primary_targets": ["specific_member"],
    },
    # Carbamate (urethane) hydrolysis -- a third chemistry, independent of the
    # amide-formation and amide-hydrolysis benchmarks above.  The characterised
    # answer set is small and recent (most since 2023), which makes it a clean
    # blind-recall test: the agent is given only the substrate and the bond.
    #
    # PROVENANCE OF THE ANSWER KEY (verified 2026-10-11, each resolved against
    # the live database before being listed here):
    #   WBR49956.1 / WBR49957.1 / WBR49958.1  GenBank, the three metagenomic
    #       urethanases from one PU-contaminated soil library (partial CDS)
    #   A0ABH7FG12  UniProt TrEMBL, 100% identical to WBR49957.1
    #   A0A7H0LPJ3  UniProt TrEMBL, Sphingomonas alpina amidase (SaAmd)
    #   A0A679EIJ6  UniProt TrEMBL, Candida parapsilosis urethanase (PDB 8WV2)
    #   9FVF / 9IZT / 8WV2  PDB
    # Deliberately ABSENT because no accession could be verified: AbPURase,
    # pQR3139/3141/3144, CCPUR1/2, TflABH, MthABH, GatA250, NfPolyA -- their
    # sequences are published only in supplementary tables or on request.  They
    # are therefore matched by NAME in the papers scope, not by accession, and
    # accession recall is scored only over the six that genuinely exist.
    # CAUTION: the PDB->UniProt cross-links for 9FVF and 9FZW point at SaAmd,
    # not at the UMG sequences; use the GenBank accessions for those.
    "aromatic_carbamate_hydrolase": {
        "query": ("find enzymes that hydrolyse the carbamate (urethane) linkages in "
                  "this aromatic biscarbamate, releasing the free aromatic diamine, "
                  "the oligo-ether alcohol and CO2"),
        "substrate_name": ("bis[2-(2-hydroxyethoxy)ethyl] "
                           "N,N'-[methanediylbis(4,1-phenylene)]dicarbamate"),
        "recall_targets": {
            "specific_accession": {"scope": "artifacts",
                                   "values": ["wbr49956", "wbr49957", "wbr49958",
                                              "a0abh7fg12", "a0a7h0lpj3", "a0a679eij6"]},
            "specific_member": {"scope": "papers",
                                "values": ["umg-sp", "abpurase", "ccpur", "pqr3139",
                                           "pqr3141", "pqr3144", "aes72", "saamd",
                                           "tflabh", "mthabh", "gata250"]},
            "structure_anchor": {"scope": "artifacts",
                                 "values": ["9fvf", "9izt", "8wv2", "8s7z", "8wdw"]},
            "mechanism_class": {"scope": "artifacts",
                                "values": ["urethanase", "carbamate hydrolase",
                                           "amidase", "aryl acylamidase",
                                           "alpha/beta-hydrolase", "carbamate"]},
        },
        "primary_targets": ["specific_accession", "specific_member"],
        # The distinct folds that carry demonstrated carbamate-cleaving activity.
        # A run that reaches only one of these is narrow even if it scores well:
        # the two families are unrelated and were found by different routes.
        "family_coverage": {
            "amidase_signature": ["amidase signature", "ser-cisser-lys", "ggssgs",
                                  "umg-sp", "ccpur", "saamd", "malonamidase",
                                  "fatty acid amide hydrolase", "peptide amidase"],
            "alpha_beta_hydrolase": ["alpha/beta-hydrolase", "esterase", "cutinase",
                                     "lipase", "aes72", "tflabh", "mthabh",
                                     "ser-his-asp", "ser-his-glu"],
            "aryl_acylamidase": ["aryl acylamidase", "anilide", "acylanilide",
                                 "linuron", "propanil", "phenylurea"],
            "carbamate_pesticide": ["carbaryl", "carbofuran", "cahA", "cehA",
                                    "mcbA", "carbamate hydrolase"],
            "urease_related": ["urease", "urea amidolyase", "allophanate hydrolase",
                               "biuret hydrolase", "ethyl carbamate"],
        },
    },
    # Aliphatic carbamate: same bond, aliphatic nitrogen.  The honest ground
    # truth here is THINNER than the aromatic case -- the characterised
    # urethanases were selected on N-aryl substrates, and the only enzymes with
    # demonstrated activity on an aliphatic-PU material are UMG-SP1/UMG-SP2 on
    # Impranil DLN (a POLYESTER-PU, so that activity may be ester rather than
    # carbamate hydrolysis -- a real confound, not a technicality).  Accession
    # recall is therefore scored over the same verified set, while
    # ``aliphatic_demonstrated`` is the strict target and is deliberately small.
    "aliphatic_carbamate_hydrolase": {
        "query": ("find enzymes that hydrolyse the carbamate (urethane) linkages in "
                  "this aliphatic biscarbamate, releasing the free aliphatic diamine, "
                  "the oligo-ether alcohol and CO2"),
        "substrate_name": ("bis[2-(2-hydroxyethoxy)ethyl] "
                           "N,N'-(hexane-1,6-diyl)dicarbamate"),
        "recall_targets": {
            "aliphatic_demonstrated": {"scope": "papers",
                                       "values": ["umg-sp", "impranil", "ethyl carbamate",
                                                  "urethanase"]},
            "specific_accession": {"scope": "artifacts",
                                   "values": ["wbr49956", "wbr49957", "wbr49958",
                                              "a0abh7fg12", "a0a7h0lpj3", "a0a679eij6"]},
            "mechanism_class": {"scope": "artifacts",
                                "values": ["urethanase", "carbamate hydrolase",
                                           "amidase", "alpha/beta-hydrolase",
                                           "carbamate"]},
        },
        "primary_targets": ["aliphatic_demonstrated"],
        "family_coverage": {
            "amidase_signature": ["amidase signature", "ser-cisser-lys", "umg-sp",
                                  "ccpur", "saamd", "fatty acid amide hydrolase"],
            "alpha_beta_hydrolase": ["alpha/beta-hydrolase", "esterase", "cutinase",
                                     "lipase", "aes72", "tflabh", "mthabh"],
            "urethane_small_molecule": ["ethyl carbamate", "urethanase", "urethane",
                                        "candida parapsilosis"],
            "urease_related": ["urease", "urea amidolyase", "allophanate hydrolase",
                               "biuret hydrolase"],
            "polyamide_hydrolase": ["nylc", "nylb", "nyla", "tvgc",
                                    "6-aminohexanoate", "n-terminal nucleophile"],
        },
    },
}
