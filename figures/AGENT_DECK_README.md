# Enzyme-discovery agent — group-meeting figure deck

Publication-quality figures (vector **PDF** + 300-dpi **PNG**) covering the agent's
architecture and results. All are generated from **real run artifacts** by one
script — edit it and re-run to refresh:

```bash
python figures/make_agent_overview_figures.py
```

Primary data sources: improved-agent run `20260914_220431` (amide synthetase /
nylon-6,6) + its regenerated `structural_modeling_handoff.json`; the older run
`20260830_125748` (cost/quality comparison); `structural_modeling/results/*.csv`
and `exp_data/…/panel_activity.csv` (geometry proxies vs assay); and the
out-of-tree `Interpro_exhaustive/` sweep (recall audit).

## Suggested presentation order

| # | File (`agent_*`) | What it shows | Key message |
|---|---|---|---|
| 1 | `pipeline_architecture` | 7-stage retrieval-expansion pipeline with real funnel numbers + cross-cutting design | How the framework works end to end |
| 2 | `handoff_schema` | Per-candidate handoff record; 3 agent-derived blocks (family, mechanism, cofactors) | The discovery → structural-modeling contract |
| 3 | `mechanism_bridge` | Deterministic activation classifier (ADP+Pi→Pγ vs AMP+PPi→Pα) + validation | Mechanism from the reaction, not the family name (ddaG fix) |
| 4 | `ranking_decomposition` | Old collapsed score vs 3 interpretable axes | Why decomposed ranking beats one fused score |
| 5 | `candidate_landscape` | 38 ranked candidates by InterPro family, colored by mechanism | The discovery result at a glance |
| 6 | `discovery_completeness` | Exhaustive EC 6.3.- sweep: 12,456 → 270 → agent coverage | Recall is complete over the reviewed universe — with the SfaB ceiling stated |
| 7 | `geometry_proxy_benchmark` | AUROC of co-fold proxies vs assay + angle–activity scatter | Which structural signal actually predicts activity |
| 8 | `structural_geometry` | Mechanism-aware step-1 in-line angle, within family | Modeling read-out, scored vs the correct phosphorus |
| 9 | `run_comparison` | Old vs improved run: candidates, leads, token cost | Cleaner ranking, same leads — plus the cost regression to fix |

## Honesty notes for the talk
- Figure 9 deliberately shows the **72% idle token regression** in the improved run — present it as a known issue to fix, not a win.
- Figure 6 states the **recall ceiling**: the ground-truth SfaB is unreviewed/no-EC, outside any reviewed-DB sweep; reaching it needs the literature/unannotated channel.
- Figure 7's angle signal is **necessary-not-sufficient** (a coarse negative filter), not a quantitative activity predictor.

Superseded drafts live in `figures/archive/`.
