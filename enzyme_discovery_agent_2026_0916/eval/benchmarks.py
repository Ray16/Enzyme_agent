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
}
