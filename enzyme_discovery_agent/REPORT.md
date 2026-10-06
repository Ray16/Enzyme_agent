# Enzyme Discovery Agent — Progress Report

_Human-facing report. Benchmark answers named here are used only to grade runs; the
agent never sees them (ground truth lives in the held-out test oracle)._

## Summary

The Enzyme Discovery Agent finds enzymes that could catalyze a **target reaction**
without being told the answer: the LLM proposes the *strategy* (reaction type,
mechanism, search terms) and deterministic tools + gates find the *entities*
(folds, specific enzymes). This cycle improved **discovery quality** (more of the
right enzymes, ranked more honestly), cut **cost**, cleaned up the **codebase**, and
added **visualization + a slide deck** — all validated end-to-end on the benchmark.

Headline result on the benchmark (join a C6 diacid + a C6 diamine, i.e. nylon-6,6
precursors): the previously mis-ranked #1 candidate (an aromatic amino-acid
conjugator) **dropped out of the ranking**; a genuine diamine ligase (**PuuA**) is now
**#1**, the strict answer (**DdaG**) rose to **#3**, and the full analogue family
(DesD, AsbA, AsbB, sbnE, iucA/iucC) is recovered — at **lower cost** (~1.43M tokens vs
~2.18M) and the **same wall-clock** (~32 min).

## What we changed

### Discovery quality (correctness)
- **One-to-one substrate matching** (new shared `substrate_fit.py`). When a target has
  two substrates, each native can serve only one — so a candidate can't "cheat" by
  matching both target substrates onto its single best native. Scored by the worst
  pairing (the bottleneck). Used by *both* promotion and ranking, so they agree.
  This is what demoted the bad #1 (its aromatic native can't stand in for a diamine).
- **Promote every relevant member.** The family survey no longer keeps only a fold's
  top-2 by fit; every same-direction, substrate-resolved member becomes a candidate.
  This recovered the long tail of the analogue family (DesD/AsbA/AsbB, etc.).
- **Evaluate every representative.** Removed the pre-evaluation budget cap. Since
  evaluation is deterministic (database lookups, no LLM), breadth is free — every
  distinct gene is evaluated, in parallel, so nothing relevant is silently dropped.
- **Lean literature harvest.** Stopped reading ~40 pooled papers and regex-mining
  every deposited accession (which pulled in off-target enzymes). Now it reads only a
  few open-access papers and keeps just the LLM-extracted **named/cited** enzymes —
  literature's unique, irreplaceable channel (it reaches enzymes named only in a
  recent paper; it is how DdaG's biosynthetic cluster is found).

### Efficiency
- Lean literature harvest: ~90% fewer harvest LLM calls, no paywall stalls.
- Parallel evaluation (dedicated I/O worker pool).
- Net: **~2.18M → ~1.43M tokens** per run, no increase in wall-clock (~32 min).

### Code health (behavior-preserving refactor)
- Removed **~730 lines of dead code** (dead state fields; obsolete single-pass /
  calibration-scoring / batch-ranking report paths).
- De-duplicated shared logic (one reaction-direction constant, one target reader).
- **Unified the gate system**: all nine pipeline gates now share one signature and
  one intercept routine, replacing ~60 lines of copy-pasted control flow.
- Made token accounting thread-safe (groundwork for parallel report generation).

### Visualization & communication
- **Family native-substrate-fit plot**: raised to **300 DPI**, de-duplicated
  near-identical InterPro entries so distinct families show (not duplicates), and
  switched to a readable categorical palette with larger fonts.
- **Pipeline figure** and a **12-slide project deck** (`slides/`), self-contained
  (native LaTeX/TikZ).

## Validation (benchmark run, current code)

| | Before | After |
|---|---|---|
| Mis-ranked #1 (aromatic conjugator) | **#1** | **not ranked** |
| #1 candidate | aromatic conjugator | **PuuA** (a real diamine ligase) |
| DdaG (strict answer) | #4 | **#3** |
| Analogue family (DesD/AsbA/AsbB/…) | mostly missing | **recovered** |
| Tokens / run | ~2.18M | **~1.43M** |
| Wall-clock | ~32 min | ~32 min |

Every screened candidate is now evaluated (no budget drop). All changes ship with the
test suite green (215 tests), the data-leakage firewall intact, and an append-only
development log.

## Known limitations & next steps
- **Literature recall.** The field's best papers are still missed because the agent
  searches with mechanistic vocabulary while papers use the field's own terms (and
  some are paywalled). Query reformulation surfaces them but not yet reliably.
- **Family → InterPro resolution.** Some derived families resolve to the wrong or a
  too-generic fold (e.g. an ATP-grasp family mapping to "DNA ligase"), so they lack a
  computed substrate fit and don't appear in the family landscape/plot even when a
  member ranks well. Improving this resolution is the next high-value fix.
- **Speed.** The loop is ~98% LLM-inference bound (large per-iteration prompts); the
  levers are prompt pruning and parallel/fewer finalization passes (the report passes
  can be parallelized once that path has test coverage).
- **Ranking.** A couple of scoring projections still differ; unify them and keep
  validating ranking quality run-over-run.
