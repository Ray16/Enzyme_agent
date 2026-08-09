"""Ground-truth benchmark campaigns for literature-exploration evaluation.

Answer identities are used ONLY to score a run after the fact (did the answer
survive?), NEVER as input to the agent's search — this preserves the leakage
firewall (the agent must reach these blind, from the target chemistry).

Each benchmark:
  query           - the campaign prompt given to the agent
  substrate_name  - optional substrate hint (as a user would provide)
  answer_accessions - UniProt accessions of the confirmed answer enzyme(s)
  answer_markers  - lowercase gene-name / trivial-name substrings that identify
                    an answer-bearing paper in a title/abstract
"""

BENCHMARKS = {
    # Amide-synthetase ligation: confirmed RECALL FAILURE under generic queries;
    # answers are ATP-grasp amide ligases reachable only via trivial-name queries.
    "amide_synthetase": {
        "query": "amide synthetases for nylon-6,6 synthesis",
        "substrate_name": "adipic acid and hexamethylenediamine (nylon-6,6 monomers)",
        "answer_accessions": ["E2JA32", "A0A2H4T912"],   # DdaG, SfaB
        "answer_markers": ["dapdiamide", "ddag", "sfab"],
    },
    # Nylon hydrolysis: recall-OK / precision-imperfect (NylC vs NylB/NylA cascade).
    "nylon_hydrolase": {
        "query": "enzymes that hydrolyze nylon-6 oligomers",
        "substrate_name": "nylon-6 oligomer (6-aminohexanoate linear oligomer)",
        "answer_accessions": [],   # add confirmed accession(s) when available
        "answer_markers": ["nylc", "nylb", "nyla", "6-aminohexanoate",
                           "nylon oligomer hydrolase"],
    },
}
