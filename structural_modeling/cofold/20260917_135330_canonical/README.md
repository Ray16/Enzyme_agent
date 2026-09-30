# Canonical synthetase step1 cofold — run 20260917_135330

Step-1 Boltz-2 cofolds for the candidates in the **canonical** discovery run
`20260917_135330` (adipate + HMD, nylon-6,6) that were **missing** from the older
`20260830_125748` panel. See `../CANDIDATE_RANKING_POLICY.md`.

## Why this folder exists
The pre-existing panel folder was named `20260917_135330_...` but its
`candidates.csv`/`representatives.csv` (dated Sep 8) are a byte-for-byte match to the
**archived `20260830_125748`** run, not this one (81/81 reps, 89/89 candidates). It was
the Aug-30 panel migrated under a new name on Sep 19. This folder holds the folds that
the canonical run actually needs and the older panel lacks.

- Canonical run candidates: 43.  Already folded in the old panel: 29.  **Folded here: 14.**

## Mechanism + Mg are reaction-derived, not family-name-guessed
Each YAML's catalytic Mg2+ count and the phosphorus to be scored come from the
enzyme's **UniProt/Rhea catalytic-activity equation** via
`proxy_policy.resolve_mechanism` (basis `reaction` for all 14):

  - `... ATP = ... + AMP + diphosphate`  -> P-alpha attack, **1 Mg2+**  (adenylate)
  - `... ATP = ... + ADP + phosphate`    -> P-gamma attack, **2 Mg2+**  (acyl-phosphate)

Here: 13 x (PG, 2 Mg) + 1 x (PA, 1 Mg = A1AHS4 / asnA). Per-candidate detail and MSA
depth (QC) in `panel_inputs.csv`.

### DdaG correction (E2JA32) — carries into re-analysis, not this folder
DdaG is already folded in the old panel at **1 Mg (correct)** but was **scored at Pgamma
(wrong)**: its reaction releases AMP + diphosphate, so it is a Palpha/1-Mg adenylate. The
old family-name map put "dapdiamide" in the acyl-phosphate group. Re-score its existing
50 models at Palpha (min d 3.74 A, 4/50 gated, mean 102 deg -> still inconclusive, but for a
legitimate reason). Its paralogue DdaF (E2JA31) genuinely IS Pgamma/2-Mg — same family,
opposite coordinate — which is why the family name cannot decide this.

## Layout
- `step1/step1_<acc>.yaml`  — Boltz inputs (protein + ATP + N x MG + adipate, affinity on adipate)
- `msa_cache/*.csv`         — batched ColabFold MSAs (repo-local; referenced by the YAMLs)
- `results/<acc>/`          — Boltz predictions (50 diffusion samples; untracked, ~large)
- `fleet/`                  — claim-based resumable fan-out (`items.txt`, `fold_loop.sh`, `launch_all.sh`)
- `panel_inputs.csv`        — per-candidate mechanism, Mg, MSA depth + QC

## Reproduce
```bash
# fan across free GPUs (system-wide gpu_reserve gate; claim-based, resumable)
bash fleet/launch_all.sh            # MAX defaults to 10 workers
# then score (correct phosphorus per mechanism)
python ../../scripts/angle_gate_proxy.py results/output_configs --reps <reps.csv>
```
