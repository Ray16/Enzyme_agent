# Agent Instructions

## Candidate Modeling and Ranking

- Ranked discovery candidates automatically enter rolling structural-modeling
  review, including candidates added after launch. Use the coordinated production
  workflow; do not leave a discovery run without its modeling coordinator unless
  the user explicitly requests discovery only. The modeling agent gathers and
  weighs independent sources, prioritizes informative target-substrate tests,
  validates the molecular setup and submits justified cofolds. Missing native
  reaction annotation alone neither authorizes compute nor excludes a hypothesis.
- Model the requested target substrates, including non-native partners. Reuse
  predictions only after verifying the exact sequence and complete molecular
  setup. Native controls are additional discriminating tests, not substitutes
  for the target. Preserve unresolved setups and compute-budget checkpoints.
- A request to model candidates authorizes running the established candidate modeling pipeline, with its existing quality settings, resource reservations, and validation checks. Continue through structural output and ranking without asking again about routine submission steps.
- Use the existing frozen proximity-weighted attack-angle metric for the in-silico ranking, unless the user explicitly requests another metric. Preserve its definition and calibration; do not refit it on new candidates.
- For the current candidate workflow, that metric is the inverse-distance-weighted mean attack angle, `sum(angle / distance) / sum(1 / distance)`, in degrees (larger ranks higher). Report harmonic distance alongside it; do not silently substitute an older sigmoid/Gaussian composite score.
- Match the established molecular setup when it is supported for the candidate's reaction step. Resolve sequence identity and verify the applicable mechanism, reactive atoms, assembly and cofactors before submission; preserve uncertainty when transfer is unsupported.
- Report structural results and the metric separately from experimentally established activity. Retain failed poses and incomplete ensembles.
- Distinguish phosphate inline attack from carbonyl attack. A carbonyl attack angle uses the attacking nucleophile, activated carbonyl carbon and double-bonded carbonyl oxygen, with approximately 107 degrees as a reference; the angle against a leaving-group oxygen is a different quantity. Preserve historical scores and label supplementary measurements explicitly.
- Let the structural agent decide when a separate pocket-localization test is informative. The user's broad-pocket reference is 10 angstroms with `force: true`; resolve member residue mappings under shared family rules, retain uncertain cases, and do not impose reactive-atom distance or angle restraints to obtain favorable scores. Preserve and label the unrestrained and guided arms separately.

## Manuscript Overleaf Sync

- Before making any changes to the Overleaf manuscript in `/homes/rzhu/Enzyme_agent/overleaf-candidacy` (including LaTeX, bibliography, figures, or manuscript helper scripts), first run `git pull` from its configured upstream. The user may have edited the manuscript on the Overleaf server. A fetch-only check is not a substitute for this required pull.
- Preserve all uncommitted local work before pulling. If necessary, save it safely, pull, and restore/reconcile it without dropping either local or server edits. Resolve conflicts before editing; if the pull fails or reconciliation is ambiguous, report the issue instead of continuing from a stale version.
- Use the complete current working version at that path as the editing and publication base, including its existing local revisions after reconciliation. Do not substitute an older committed snapshot or silently omit existing manuscript revisions from the published version.
- After each completed and checked manuscript update in `overleaf-candidacy`, commit the changes belonging to that update and push to the repository's configured Overleaf Git remote on the working branch. The user explicitly selected the existing Overleaf remote as the publication destination. This is a standing user preference; do not ask for confirmation each time.
- Review the diff and stage only the relevant files; do not include unrelated work or overwrite others' changes. Do not force-push.
- Verify that the push succeeded and report the commit. If a push fails or the destination is ambiguous, report the blocker explicitly rather than implying the manuscript is synced.

## CPU Resource Safety

- For large independent structure-QC and feature-extraction jobs on the 80-CPU node, use the user's approved 32-CPU budget when capacity permits. Assign the coordinator and workers separate CPU sets, and cap native-library pools to one thread per worker.
- Never launch workloads that oversubscribe a machine's CPUs.
- Before submission, check each target host's online CPU count and current load.
- Account for every concurrently launched process and its possible native-library thread pools.
- Set explicit thread limits, including `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and other applicable runtime settings.
- Pin concurrent workers to non-overlapping CPU sets with `taskset`, a scheduler CPU binding option, or an equivalent mechanism.
- Require `current load + newly reserved CPU slots <= online CPUs`; fail closed when capacity or affinity cannot be verified.
- Use at most one worker per assigned GPU unless a workload has been explicitly validated for GPU sharing.
- Avoid GPU overlap with other users regardless of who launched first. If another user's compute process appears on our GPU, stop our affected worker and preserve its partial results.
- Recheck CPU capacity and GPU occupancy immediately before launching a distributed job.
- After submission, verify worker counts, CPU affinities, thread counts, host load, and GPU assignments. Stop and correct the submitted run if these guarantees are violated.

## Fleet Computations

- Whenever the user asks to submit, launch, distribute, restart, or resume fleet computations, read and use the Codex `lambda-fleet` skill before submission actions. This is a standing user preference; explicit `$lambda-fleet` invocation is not required.
- Skill entrypoint: `/homes/rzhu/.agents/skills/lambda-fleet/SKILL.md`. Canonical NFS fallback: `/nfs/lambda_stor_01/homes/rzhu/.codex-personal/skills/lambda-fleet/SKILL.md`.
- Preserve the requested workload and any explicitly chosen scheduler/provider. If the skill cannot be read, report that blocker before submitting instead of silently bypassing it.
- Prefer Aurora for RFD4-Proteina inference and its independent Boltz evaluation. Use the isolated FP32 XPU profile under `/lus/flare/projects/FRAME-IDP/rzhu/rfd4_proteina_20261026`, with `frameworks/2026.1.0` and `/lus/flare/projects/FRAME-IDP/rzhu/Software/envs/rfd4-proteina-xpu-20261026`. Local setup, validation records and controls are under `/nfs/lambda_stor_01/homes/rzhu/Tool/RFD4-Proteina-compat/aurora`. Select Lambda for this workflow only when the user requests it; preserve model/input identity, sampling settings, explicit numerical profile and bounded resource allocation.
- Default to `batch_size=6` for RFD4-Proteina generation on Aurora, per the user's preference recorded on 2026-10-29. This is the generation batch size, not the total requested replicate count. Six completed successfully for the 516-residue AS9 input in the FP32 XPU profile; preserve explicit user overrides and check capacity for materially different inputs. This preference does not set the batch size for Boltz or other workloads.
- For future scientific runs, log protein length(s), requested/completed sample counts, batch size, accelerator and numerical profile, CPU allocation/thread caps, peak allocated/reserved device memory, startup/import overhead, warm-up to the first generation batch, generation-entrypoint duration, total process wall time, and average generation time per completed sample. Record scheduler wait separately. State timing definitions, label batch averages as throughput rather than individual latency, and leave unavailable measurements null rather than estimating them silently.
