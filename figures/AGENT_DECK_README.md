# Enzyme-discovery agent — group-meeting figure deck

300-dpi PNG figures covering the agent's architecture and results, generated from
**real run artifacts**. Most come from one script (edit and re-run to refresh):

```bash
python figures/make_agent_overview_figures.py
```

The proxy-predictivity slides come from the **validated** analysis:

```bash
python analysis/make_synthetase_figures.py   # angle-gate proxy, AUROC 0.95
```

Primary data sources: improved-agent run `20260914_220431` (amide synthetase /
nylon-6,6) + its regenerated `structural_modeling_handoff.json`; the older run
`20260830_125748` (cost/quality comparison); `structural_modeling/results/*.csv`;
`exp_data/amide_synthetase/…` (panel assay + angle-gate proxy) and
`exp_data/nylon_hydrolase/…`; and the out-of-tree `Interpro_exhaustive/` sweep.

## Presentation order

| # | File | What it shows | Key message |
|---|---|---|---|
| 00 | `00_overview.png` | Problem / framework / key results / limitations | Opening summary slide |
| 01 | `01_pipeline_architecture.png` | 7-stage retrieval-expansion pipeline + cross-cutting design | How the framework works end to end |
| 02 | `02_handoff_schema.png` | Per-candidate handoff record; 3 agent-derived blocks | The discovery → structural-modeling contract |
| 03 | `03_mechanism_bridge.png` | Deterministic activation classifier (ADP+Pi→Pγ / AMP+PPi→Pα) + validation | Mechanism from the reaction, not the family name (ddaG fix) |
| 05 | `05_candidate_landscape.png` | Ranked candidates by InterPro family, colored by mechanism | The discovery result at a glance |
| 06 | `06_discovery_completeness.png` | Exhaustive EC 6.3.- sweep: 12,456 → 270 → agent coverage | Recall complete over the reviewed universe — SfaB ceiling stated |
| 07 | `synthetase_angle_proxy_ranking.png` | **Validated proxy**: distance<4 Å gate → rank by in-line angle | AUROC 0.95; hits AS9/AS17/AS23 rank #1/#4/#7 |
| — | `synthetase_activity_landscape.png` | The assay: 3 hits over a flat tail (3/49) | Why the panel is a hard, imbalanced target |
| 09 | `09_hydrolase_second_target.png` | 96-enzyme nylon-hydrolase panel: activity + NAC | Same framework transfers to the cleavage direction |
| 10 | `10_run_comparison.png` | Old vs improved run: candidates, leads, token cost | Cleaner ranking, same leads — plus the cost regression to fix |

## Honesty notes for the talk
- **The validated activity proxy is the distance-gated in-line angle** (`scripts/angle_gate_proxy.py`): keep only poses with nucleophile–reactive-P distance < 4 Å, then rank by the mean in-line attack angle. AUROC 0.95, Spearman +0.30 — a coarse ranker / negative filter, **not** a quantitative predictor (only 3 actives). Raw NAC fraction is **not** the proxy.
- Fig 10 shows the **72% idle token regression** — a known issue to fix, not a win.
- Fig 06 states the **recall ceiling**: SfaB is unreviewed/no-EC, outside any reviewed-DB sweep.

## Retired (in `figures/archive/`)
- `04_ranking_decomposition` — not meaningful (a score dump: `substrates_hosted` is flat at 2/2 across the top set, so it shows no discrimination).
- `07_geometry_proxy_benchmark` — superseded: led with raw NAC fraction; replaced by the validated angle-gate figure above.
- Earlier drafts and the vector PDFs (deck is PNG-only now).
