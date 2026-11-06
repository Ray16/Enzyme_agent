# Archive Manifest

This archive is recoverable storage. Items were moved here during the July 2026 workspace reorganization; no files were deleted.

| Archive path | Previous location | Reason |
| --- | --- | --- |
| `enzyme_discovery/legacy_agent_runs/` | `prev_agent_runs/` | Historical outputs from older enzyme-agent implementations. |
| `enzyme_discovery/legacy_package_metadata/` | `enzyme_agent.egg-info/` | Generated metadata for the obsolete package name. |
| `enzyme_discovery/legacy_scripts/` | `scripts/` | Legacy reaction-retriever scripts referencing the obsolete package layout. |
| `enzyme_discovery/legacy_summary/` | `summary/` | Documentation and diagrams for the obsolete discovery implementation. |
| `design_experiments/deterministic_six_stage_workflow/` | `enzyme_design_agent_prev/` | Earlier deterministic baseline; retained for reproducibility. |
| `design_experiments/retrieval_expansion_prototype/` | `enzyme_design_agent_prev2/` | Retrieval-expansion prototype. |
| `design_experiments/failed_map_elites_rewrite/` | `enzyme_design_agent_map-elites/` | Failed MAP-Elites experiment; intentionally not an active workflow. |
| `design_experiments/legacy_monolithic_design_agent/` | `enzyme_design_agent/` | Superseded monolithic discovery/design agent. Its structural-homolog capability was ported to `enzyme_discovery_agent`; retained only as historical reference. |
| `structural_modeling/earlier_refactor/` | `experimental_feedback_refactor/` | Older structural-modeling refactor. |
| `structural_modeling/legacy_pipeline_2026/` | `structural_modeling_pipeline_prev/` | Superseded structural-modeling workflow and retained outputs; replaced by the focused `structural_modeling/` workflow. |
| `presentation_generators/candidacy_slides_gen_figs.py` | `docs/presentations/slides/gen_figs.py` | One-off generator for a retained presentation; not part of an active workflow. |
| `manual_inspection/inspect_AS9_step2.pml` | `figures/inspect_AS9_step2.pml` | One-off PyMOL inspection session retained for provenance. |
| `snapshots/` | root `*.tar` snapshots | Historical package and pipeline snapshots. |
| `legacy_root_utilities/` | assorted root scripts/results | One-off legacy utilities and logs no longer used by the active workflows. |
| `runtime_metadata/` | root bytecode cache and one-off runtime markers | Generated or transient root-level files. |
| `external_references/Biomni/` | `Biomni/` | Unused external reference repository. |
| `reaction_retrieval/horizyn_derived_retriever/` | `enzyme_discovery_agent/models/rxn_retriever*` and associated cache/checkpoints | Horizyn-derived retrieval experiments. Benchmark metrics were retained, but the bidirectional setup and lack of integration made them unsuitable as active evidence for directed non-natural reactions. |

## 2026-10-03 workspace cleanup (obsolete / one-off sweep)

Moved here on 2026-10-03; no files deleted. Firewall-protected `exp_data/` and `data/` were left untouched.

| Archive path | Previous location | Reason |
| --- | --- | --- |
| `lanthanide_binding_oneoff_20260917/` | `lanthanide_binding/` | One-off Sep-17 Ln-binding exploration (205 MB); referenced nowhere else in the repo and not in README/PROJECT_LAYOUT. |
| `cofold_20260830_125748_superseded_synthetase_panel/` | `structural_modeling/cofold/20260830_125748_find_amide_synthetases_..._for_nyl/` | Superseded Aug-30 synthetase cofold panel (~5.6 GB); the canonical run (`20260917_135330`) README calls it the "archived" panel and already merged the folds it reused. |
| `superseded_content_20261003/slides_orphan_pngs/` | `enzyme_discovery_agent_2026_0916/slides/*.png` | Orphan slide-export PNGs (ev-/pr-/rev-/review-/montage); not referenced by any `.tex` in the deck. |
| `superseded_content_20261003/NylC_build_md_scratch/NylC_build/` | `docs/notes/md/NylC_build/` | One-off NylC MD system-prep scratch workspace (pass/tleap/sqm intermediates). |
| `superseded_content_20261003/docs/` | `docs/{candidacy_staleness_audit,discovery_improvement_plan,experiment_log,integrated_discovery_structural_report_2026-09-15,WORKFLOW_REFACTOR_PROGRESS}.md` | Dated planning/progress/report snapshots; current reference docs kept in `docs/`. |
| `superseded_content_20261003/docs_history/` | `docs/history/` | De-facto archive: historical build/implementation/redesign prose, April candidacy-proposal LaTeX artifacts, and a dated citation backup. |
| `_regenerable_junk_20261003/` | `docs/presentations/**`, `enzyme_discovery_agent_2026_0916/slides/`, `logs/` | Regenerable LaTeX intermediates (`.aux/.log/.out/.nav/.toc/.snm`), slide build logs, and old run/demo `.log` files. Safe to delete outright; quarantined here only because the never-delete hook blocks `rm`. |

In-place moves (kept under each area's own `archive/` subdir per the figure-organization convention, not under `archive/`):
- `enzyme_discovery_agent_2026_0916/figures/esm_map_hydrolase_20260919_201625.{png,pdf}` → `enzyme_discovery_agent_2026_0916/figures/archive/` (superseded by the `_corrected_20260920_144401` version).
- `analysis/reports/{ec_recall_probe_6_3*.json, hydrolysis_native.html, synthesis_new_native.html}` → `analysis/reports/archive/` (Sep-15 run snapshots).

Also restored `structural_modeling/CANDIDATE_RANKING_POLICY.md` (git-tracked, missing from the working tree, still referenced by `resolve_handoff.py` and three READMEs) — recovered, not archived.

To restore an archived component, move it back to the workspace root and update its package/path references deliberately; archived code retains its original package names and may not run unchanged beside the active layout.

## 2026-10-03 — analysis/ cleanup
- `analysis/{evaluate_all_proxies,make_figures,rescore_ranking,native_reaction_grid}.py` → `archive/analysis_obsolete_20261003/` (no references anywhere in the repo; superseded by the per-area scripts under `analysis/{synthetase,hydrolase}/` and `figures/`).
- `figures/hydrolase_nac_rank_roc.png` (Sep 8, NAC-era, unreferenced) and `docs/presentations/` (Apr–Jun 2026 slide .tex + figs, unreferenced) → `archive/figures_docs_obsolete_20261003/`. Empty `docs/notes/` removed (no files).

## 2026-11-05 repository clutter cleanup

Legacy agent snapshots, completed validation workspaces, backups, historical figures and generated files were consolidated in [cleanup_20261105/MANIFEST.md](cleanup_20261105/MANIFEST.md). That manifest records every move and the local checksum inventory. Current workflows and research data remain in place.
