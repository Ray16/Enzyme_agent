# Enzyme Discovery Agent — Development Log

**Append-only.** Newest entries at the bottom. One entry per change/experiment:
what was done, why, what it validated, and any failure modes found. Do not edit
past entries — correct them by appending a new dated entry.

---

## 2026-08-16 — Substrate-fit family expansion, Stage 1 (`substrate-fit-promotion-stage1`)

**What & why.** The family survey computed rich per-member native-substrate
resemblance but only *displayed* it to the LLM — strong members were never turned
into candidates, and enumeration was reviewed-only. Implemented family-anchored
substrate-fit **selection + promotion** so the relevant enzymes in a family
actually enter evaluation.

- `select_relevant_members()` — ranks same/unknown-direction members by
  multi-dimensional native-substrate resemblance (FG, MW, volume, chain length,
  HBA/HBD, polarity), RELATIVE (no tuned cutoff), bounded by reusing
  `MAX_REPRESENTATIVES_PER_FAMILY`.
- Gated TrEMBL deepening (`_unreviewed_members_by_ec`) — after the reviewed pass
  finds the best-fit reaction, pull that fold's unreviewed members of the SAME EC
  (`reviewed:false AND ec:<EC>`), capped. Turns TrEMBL from a 10^5 noise dump into
  targeted deepening of the relevant reaction.
- Design principle enforced: **code extracts evidence, LLM only reasons**; no new
  empirical thresholds; deterministic (no LLM calls in this path).

**Validated.** Full suite green (183). Live smoke on IPR000291: 120 reviewed + 120
gated-TrEMBL, 2 promoted. Firewall grep clean.

**Failure modes found.**
- **SfaB (A0A2H4T912)** is TrEMBL with NO EC and NO catalytic-activity annotation →
  no native substrate to compare → **unreachable by substrate-fit; literature-only.**
- Exact substrate match is useless for non-natural targets: `chebi:adipate` → 0
  UniProt enzymes. Recall must come from *similar* substrates (promiscuity).

## 2026-08-16 — Ensure the agent USES it, Stage 2 (`substrate-fit-promotion-stage2`)

**What & why.** "A tool existing ≠ the agent using it." Traced promotion → final
answer and found a hole: `candidate_count` counts only {screened,evaluated,ranked}
(not `discovered`), and `auto_screen_candidates` screens a fold-relevance-BUDGETED
subset — so a promoted `discovered` candidate could stall, never evaluated.

- `auto_screen` now treats substrate-fit-promoted candidates (`source_tool`
  startswith `family_substrate_fit`) as PROTECTED (union with literature_named):
  always screened→evaluated→ranked, never dropped by fold budget/stop, never
  collapsed behind a longer ortholog.
- Promotion writes `protein_name` (not `literature_name`) — avoids corrupting
  provenance-by-name.

**Validated.** `test_promoted_candidate_screened.py` proves a promoted candidate
survives an allocator that drops everything else. Full suite green (184).
Oracle test: surveying DdaG's fold (IPR053158) and PuuA's fold (IPR027303)
retrieves + promotes both — the enzymes "mostly uncoverable via DB search."

**Failure mode / boundary.** Survey only fires on a fold it has a SEED into
(derived-family keyword → interpro, or a discovered member's most-specific fold,
which must satisfy 1<count<500 — so a superfamily-sized parent like IPR042099/500
is skipped). Cold-start reach of an obscure answer-fold is unsolved here → the
literature/mechanism-derivation axis (see TODO.md).

## 2026-08-16 — End-to-end validation run (in progress)

Launched a full monitored agent run on the amide/nylon benchmark (Run ID
`20260816_125338_...`) to confirm survey→promote→protect→rank fires in a real loop
and to compare ranking vs. the 20260812 baseline (DdaG #6, SfaB demoted, PuuA
absent). Result + observations to be appended here.
