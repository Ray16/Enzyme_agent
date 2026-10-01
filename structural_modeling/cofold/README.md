# Cofold campaigns

One subfolder per **enzyme-discovery run**, named exactly after that run's
directory (`runs/<run-id>/`), e.g.:

```
structural_modeling/
  cofold/
    20260917_135330_canonical/                     # CANONICAL synthetase panel (run 20260917_135330): 36 cofolds (29 reused + 7 folded)
    20260830_125748_synthetase_panel_and_probes/   # Aug-30 panel (migrated) + 1mg/pocket/control/sfab synthetase probes
    20260919_201625_find_amide_hydrolases_.../     # nylon-6,6 hydrolase panel
  scripts/                 # SHARED tooling (build_step1_inputs.py, step1_nac.py, ...)
  tests/                   # SHARED tests
  CANDIDATE_RANKING_POLICY.md
  figures/                 # SHARED figure gallery
```

Each campaign folder holds everything specific to that run's cofolding:
`step1/` (Boltz YAML inputs), `stage_configs/`, `results/` (Boltz predictions +
NAC/angle-gate CSVs), `msa_cache/`, `seqs/`, `candidates.csv`,
`representatives.*`, and any probe/control subruns.

The shared `scripts/` are argument-driven (input/output paths passed explicitly),
so they operate on any campaign — point them at `cofold/<run-id>/...`. Boltz
prediction outputs are large (~1 GB/campaign) and are intentionally **untracked**
by git; only small inputs/scripts/controls are versioned.

`results/discovery_angle_gate.csv` records the frozen proxy version, mechanism
assignment, evidence scope, gated-pose frequency, conditional mean angle, and
bootstrap uncertainty. `results/candidate_prioritization.csv` carries those fields
into the discovery-ordered table and adds a within-family structural rank. See
`CANDIDATE_RANKING_POLICY.md`; transferred P-alpha/P-gamma coordinates must not be
reported as panel-validated activity models.

Convention established Sep 19 2026 (migrated the pre-existing flat synthetase
campaign into this layout).
