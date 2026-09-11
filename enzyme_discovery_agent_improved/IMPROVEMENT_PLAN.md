# enzyme_discovery_agent_improved — improvement plan

This package is a self-contained copy of `archive/enzyme_discovery_agent_archive_20260910_102425`
(the proven agent; imports rewritten `enzyme_discovery_agent` -> `enzyme_discovery_agent_improved`).
We build improvements HERE and leave both `enzyme_discovery_agent/` (retrieval-expansion,
regressed) and the archive untouched.

## Why we pivoted (evidence)
The retrieval-expansion rewrite (`../enzyme_discovery_agent`) ended up LESS effective:
- Archive good run (0830): completes, 89 ranked candidates, **recovers DdaG (E2JA32) #4 and
  PuuA #2**, ~1.24M tokens.
- Retrieval-expansion (even after convergence fixes): 21–31 ranked, NRPS-heavy, **misses
  DdaG/PuuA**, ~5M tokens.
- Root cause of the recall loss is ARCHITECTURAL: the rewrite demoted literature-named
  enzymes from direct candidates to "leads" gated behind a family survey, and DdaG's
  family (IPR048223 / ANL homologous-superfamily) is unreachable by survey. The archive
  promotes named characterized enzymes DIRECTLY as candidates (fulltext_named/cited) — that
  is HOW it recovers DdaG. We must PROTECT this.

## Discipline (how we avoid the previous wall)
1. Freeze a baseline (below) and judge every change against it.
2. One change at a time, each behind a measured gate. Keep it ONLY if it improves its
   target axis and regresses nothing. **Hard regression guard: DdaG stays in the ranked
   list and PuuA stays near the top.** If a change drops them, revert.
3. Every change maps to ONE axis (efficiency / relevance / coverage) with a mechanism and a
   pass/fail metric. No general refactors; no porting bookkeeping (ledger/event-log) for its
   own sake.

## Order (user-chosen): EFFICIENCY -> RELEVANCE -> COVERAGE
Eval mode: discovery-stage signal for fast screening; full-analysis-to-ranking at milestones.
(The archive has no `--workflow-stage` flag yet; adding a light discovery-only stop is a
small, justified port to enable cheap screening.)

## Baseline (fill from run b90l646n1)
- tokens: ___ ; wall-clock: ___ ; iterations: ___
- ranked candidates: ___ ; relevant families: ___
- recall: DdaG(E2JA32)=___  PuuA=___  SfaB(A0A2H4T912)=___

## EFFICIENCY — parallelization findings (measured)
The archive is ALREADY well-parallelized:
- 28/31 tools are `parallel_safe=True`, so multi-tool-call turns run concurrently
  (only `fetch_full_text`, `search_structural_homologs`, `set_family_focus` excluded).
- Family SURVEYS run in a ThreadPool (`pool.map(_survey, new_folds)`, reduced in order).
- Discovered-gene fold resolution already pooled; tools fan out internally (database,
  literature, fulltext).
- Tool calls are fast (~1.9s avg); the dominant wall-clock is the SERIAL LLM reasoning
  loop (~12 turns, each waits on the model) — inherently not parallelizable.
DONE: parallelized the one remaining serial network loop — derived-family InterPro
resolution in `collect_survey_folds` (`pool.map`, order-preserving; result byte-identical
to serial). Small wall-clock win; 314 tests green.
CONCLUSION: parallelization headroom is small; the 4x efficiency win (5M->1.24M tokens)
came from the pivot to the archive itself. Bigger efficiency would require fewer/cheaper
LLM turns (quality risk) — defer unless the baseline shows otherwise.

## Efficiency candidates (token-side, lower priority — measure headroom)
Archive already prunes history (`_prune_tool_history`) and caps results. Knobs in config.py:
- `HISTORY_KEEP_RECENT_TOOL_MSGS=16` -> try 8 (halves full tool-results in prompt)
- `MAX_RESULT_CHARS=20_000` -> try 10_000 (halves largest tool dumps)
- finalization makes multiple LLM passes (output/*.py) -> audit for redundant passes
Each: run, compare tokens AND recall guard. Expected: modest (archive is already ~4x
leaner than the rewrite; the pivot itself is the big efficiency win).

## Relevance candidates (Tier 2)
- Confirm/broaden DIRECT promotion of literature-named on-target enzymes (recover SfaB too).
- Ranking check: free diacid+diamine / polyamide ligases (PuuA-type, poly-ornithine-type)
  above tethered NRPS megasynthetases.

## Coverage candidates (Tier 3)
- Only if baseline shows a genuinely missed relevant family. Archive already surveys 9–12.

## NON-goals
No ledger/event-log/claim-validation port, no new gates, no full-text-harvest rework
(full text yields ~0 enzymes in both agents; abstract mining is what works).
