# Rhea-anchored deterministic fold coverage (validation harness)

**Status: in progress.** Standalone, out-of-tree — **not** wired into the agent.

## Question
Can a *deterministic*, reaction-anchored pass guarantee recall of every relevant enzyme **fold**
over the curated universe — recovering families the LLM agent missed (panC, gss, NIS breadth) —
without an EC anchor and without flooding the agent with off-target folds?

## Approach
1. **Derive** the target transformation from the user's substrate SMILES (generic; no hardcoded
   chemistry) — e.g. `carboxyl + amine → amide`. Also handles ester/thioester.
2. **Match** every Rhea reaction (SMILES) that *forms* the target bond from a free acid + free
   acceptor, via RDKit, with **protonation neutralized** so charge state never matters.
3. **Map** matching Rhea reactions → reviewed (Swiss-Prot) UniProt (`rhea2uniprot`), bridging the
   Rhea master↔directional reaction-ID numbering.
4. **Collapse** to one reviewed exemplar per InterPro fold (reviewed-first, to avoid the blank-
   exemplar trap).

Reaction-anchored, not EC-anchored: EC is coarse and annotation-gated, and EC-6.3 scoping misses
amide-formers outside 6.3 (e.g. acyltransferases). See `../Reaction_anchored_retrieval/` and
`../Interpro_exhaustive/` for the EC-anchored comparison.

## Files
- `rhea_coverage.py` — reaction matcher + Rhea→UniProt mapping (streaming, resumable).
- `build_folds.py` — collapse matched enzymes to InterPro folds; compare Rhea vs EC-anchored vs the live run.
- `cache/` — downloaded Rhea tables and intermediate outputs (git-ignored where large).

## Honest scope
A **recall floor over the curated (Rhea/Swiss-Prot) universe** — it cannot reach unannotated/
metagenomic/literature-only enzymes. Substructure matching carries an inherent self-amidation/NH₃
false positive (a substrate carrying both a free acid and its own amine) that only atom-mapping or
the downstream decomposed evaluation removes. It is a breadth floor, **not** a precision filter.

`REPORT.md` (verdict: bounded? recovers panC/gss/NIS? net-new vs the run? precision cost?) will be
added when the run completes.
