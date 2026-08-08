# Focused Discovery Workflow Refactor — Progress

Last updated: 2026-07-28

## Reference run

- Preserved diagnostic campaign: `enzyme_discovery_agent/runs/20260728_125515_Discover_and_critically_evaluate_enzyme_candidates_for_hydrolysis_of_amide_bonds/`
- Outcome: direct Nyl-family leads were found, but broad EC/family expansion registered 677 raw candidates and delayed evaluation and structural-homolog search.

## Work items

| Item | Status | Evidence / next action |
| --- | --- | --- |
| Repair Argo + Sonnet 5 runtime | Complete | Proxy 3.2.3, OpenAI client, Sonnet 5 completion and tool-call checks passed. |
| Install chemistry dependency | Complete | RDKit substrate analysis passed. |
| Install semantic literature reranker | Complete | Installed `sentence-transformers` and `einops`; configured Nomic reranker loads successfully on CPU. |
| Preserve diagnostic nylon campaign | Complete | Run stopped deliberately; artifacts retained. |
| Fix discovery phase state | Complete | Agent now reads candidate count from `candidate_registry.json`. |
| Limit automatic EC expansion | Complete | Cap: 12 EC queries, 5 hits/query. |
| Add focused family-level candidate triage | Complete | Group by gene or normalized protein name; screen at most 24 representatives. |
| Trigger structural-homolog search automatically | Complete | After automatic evaluation, Foldseek expands up to two qualified seeds with provenance. |
| Tighten focused-discovery prompt and stopping rules | Complete | Prompt, reflection, and coverage redirect now stop broad expansion after three distinct families. |
| Add regression tests from nylon run | Complete | Added tests for evidence tiers and invalid tool arguments; full offline suite passes. |
| Rerun a short nylon-6,6 validation campaign | Next | Compare tool calls, candidate volume, evaluation timing, and structural expansion. |

## Current code changes

- `enzyme_discovery_agent/agent.py`: registry is now the source of truth for discovery candidate count.
- `enzyme_discovery_agent/config.py`: caps added for systematic EC discovery.
- `enzyme_discovery_agent/gates/systematic_coverage.py`: honors the focused EC caps.
- `enzyme_discovery_agent/gates/ec_triage.py`: groups candidates into coarse families and caps screening at 24 representatives.
- `enzyme_discovery_agent/gates/homolog_expansion.py`: runs capped Foldseek expansion for qualified evaluated seeds.
- `enzyme_discovery_agent/agent.py`: provides a supplied target profile to the model at run start.

## Validation so far

`python -m unittest discover -s enzyme_discovery_agent/tests -q`

- 27 tests passed; 7 expected skips.
