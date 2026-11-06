# Family-based rolling structural modeling

The default handoff is a shared family/mechanism protocol with a list of member
sequences. The structural modeling agent compiles the protocol once and renders
member inputs in code. It does not start a separate LLM research session for
every enzyme. The older individual-research route remains explicit opt-in.

## Inputs and family decisions

The protocol is runtime task data: component structures and charge states,
stage-specific species/roles, mechanism-assignment rules, cofactors, metal counts
and the modeled enzyme unit. No target molecules or enzyme identities are
runtime Python defaults. Reaction coordinates and atom selectors come from the
existing modeling-reasoning library; another coordinate file can be supplied.
Directly conflicting member evidence splits groups or leaves a family pending.
Unknown chemistry is not assigned an arbitrary default cofactor.

The current task uses the user-requested two-step test. Step 1 contains the
supplied acid dianion, nucleotide and mechanism-dependent metals. Step 2 contains
the activated intermediate and the supplied amine; consumed nucleotide is not
silently retained. The current single-chain screening protocol is conditional,
not proof of native assembly. Family unit overrides are explicit. For example,
Gamma-glutamylputrescine synthetase PuuA (P78061) is annotated as a dodecamer;
its single-chain screen must not be presented as native-assembly validation.

The standard discovery CLI attaches the family-mode watcher. It waits for a
shared protocol, or adopts a registered protocol for the exact ordered target
structures. That registry contains chemical protocol data, not prior candidate
findings or cohort-specific pilot accessions. New members join their family;
new sequences are resolved and aligned in batches. Existing alignments are reused
only for the exact sequence and after fingerprint checks. A different ligand
requires no new MSA when the protein sequence is unchanged.

```bash
python -m structural_modeling_agent family-cofold \
  --run /path/to/discovery-run \
  --specification /path/to/shared-protocol.json \
  --backend /path/to/backend.json --out /path/to/results \
  --worker-id unique-worker --watch-seconds 30
```

## Execution, reuse and ownership

The Lambda fleet launcher preflights node interpreters, CUDA architecture,
checkpoints, imports, host RAM and CPU capacity. Frontends and native workers use
disjoint CPU allocations and capped native pools. GPU selection uses the actual
free-device gate and an additional dispatch claim to prevent our own workers
choosing the same device. The authoritative reservation remains `gpu_reserve run`.
Real foreign-user GPU contention stops our affected process and preserves its
partial work. Completed ensembles are never replaced as a resource retry.

GPU out-of-memory failures are recorded separately, including Boltz's zero-exit
"skipping batch" failure. The shared job state imposes a cooldown and a limit of
three distinct OOM attempts. A retry requires a larger memory tier, retains all
failed artifacts, and preserves the input, prepared MSA and 50-prediction quality
setting. The native provider rechecks device memory and reservations at dispatch.
Resource deferrals do not trip the worker's scientific-failure circuit breaker.
Full saved coordinates are held for validation before any replacement cofold.
Unavailable remote ownership probes defer the retry without authorizing a
replacement. Shared system claims are published with their owner record and
retired into the archive atomically, avoiding empty claims after interruption.

H100 workers use the exact optimized profile. V100 workers use the established
FP32 serial-sampling profile, with 50 predictions, exact-sequence MSA and no
reaction-geometry restraint. Numerical profiles are recorded separately; they
are not claimed byte-identical across hardware/software or sampling layouts.
The serial sampler patch is process-local and fails closed on unknown code.

The measured T4 test exhausted its 15 GiB device with this full-quality profile.
Those workers are held; quality was not reduced. The former provisional
400-residue guard is not a candidate-selection criterion and is no longer the
fleet default. This task's measured profile now requests at least 24 GiB VRAM.
Other protocols can declare their own resource requirement.

## Optional pocket-localization arm

The structural agent can prepare a separate broad-pocket test with
`python -m structural_modeling_agent.family_pockets --run RUN --source BASELINE
--out GUIDED --max-distance 10`, then execute it using `family-cofold
--pocket-guided-from BASELINE` and the guided queue's specification. A fleet
specification can provide `pocket_guided_from` and `pocket_max_distance_A` so new
eligible baseline members are prepared automatically. Preparation is fingerprinted
and shared across workers; no member-specific LLM session or new MSA is needed.

The current diagnostic rule selects distant second-step partners, maps a pocket
around the activated carboxyl end of an identity-verified first-step model, and
checks that the activated intermediate occupies that region in a second-step
reference. Insufficient localization or residue mapping stays pending. Residue
numbers are resolved independently for each member under the shared family rule.
These pockets are model-derived hypotheses, not experimentally validated sites.

The user-requested historical setup uses `max_distance: 10.0`, `force: true`,
and a protein-residue pocket list for the amine-bearing ligand. It imposes no
reactive-atom distance or attack-angle constraint. The original unrestrained
inputs/results are preserved. A different restraint changes physical-system
identity, preventing inappropriate reuse of the unrestrained second step.
First-step structures are reused with their original provenance. Guided
localization and reactive geometry are reported separately from enzyme activity.

Reuse compares the exact sequence/construct and copy count, ligand structures
including charge/stereochemistry, cofactor identities/counts and restraints.
Registered pool inputs and coordinate files must retain their fingerprints and
have complete sample coverage. Earlier family results are revalidated and reused
across runs even when the logical job ID is identical. Legacy exact coordinates
without sufficient atom-map metadata are marked for analysis reconciliation,
rather than automatically folded again. Source and consuming discovery runs
remain distinct. In-flight duplicates are held rather than submitted twice.

## Rolling results

Every cohort is pinned to an explicit discovery-run directory; there is no
latest-run pointer. Results, inputs and reports stay beneath its `family_cofold`:

- `cohort.json`: discovery identity and target.
- `family_handoff.json`: shared protocols, families and member lists.
- `input_index.csv` and `inputs/by_accession/<accession>/step_<n>.yaml`.
- `family_results.csv`: member/step completion and geometry summary.
- `jobs/<physical-job-id>/complete.json`: producer receipt, all pose observations
  and source fingerprints; resource stops are retained separately.
- `results/<accession>/step_<n>/`: readable structure and result views.
- `analysis/pose_geometry.csv`, `analysis/ensemble_summary.csv`, per-ensemble
  plots and separate step comparisons, refreshed after completed ensembles.
- `analysis/both_steps_comparison.png` and `.svg`: 24 paired candidates ordered
  by the larger of their two harmonic attack distances; the full paired view is
  `both_steps_all_candidates.png` and `.svg`. `both_steps_summary.csv` retains
  all paired rows, including numerical profile and restraint-arm identity.

Comparison figures give each enzyme its own named row, with aligned weighted
angle and harmonic-distance panels. Plot labels use enzyme symbols or specific
enzyme names; organism names distinguish homologs. Accessions remain in the
underlying identity records and tables but are omitted from plot labels.
The distance axis is explicitly logarithmic. Numerical values and metric
definitions are unchanged, and vector SVG comparisons accompany the PNGs.

For carbonyl attack, the paired view supplements the legacy leaving-group angle
with the N–acyl C–carbonyl O angle (approximately 107 degrees as a geometric
reference), using bond-topology-verified atom maps. The legacy angle against the
leaving oxygen is not the Bürgi–Dunitz angle and is not interpreted as one.
Saved scores remain unchanged. The paired ordering is a distance-based display,
not a calibrated combined activity score. Guided and unrestrained arms remain
separate, and raw per-pose results are retained.

The existing metric remains `sum(angle/distance)/sum(1/distance)`, with harmonic
mean distance alongside it. Every pose is retained. No sigmoid, fitted cutoff,
or assay-activity claim is introduced. Different reaction steps and numerical
profiles are reported separately. These measurements describe geometry under
the declared setup, not experimental substrate acceptance.

## Development validation

The first completed pilot belongs to discovery run `20261104_160236`.
Paper-associated ShABS / UniProt Fatty-acyl-CoA synthase (A0A1M5ABR5) produced
50 predictions for each step. Both ensembles passed sequence and molecular
inventory checks. Step 1 measured 147.6914 degrees and harmonic distance
4.6096 A; step 2 measured 90.5187 degrees and 7.5550 A. They are separate
reaction-center observations, not a common activity ranking.

The step-1 output bundle occupies 118.49 MiB: 15.13 MiB PDBs and 103.11 MiB NPZ
arrays, plus small logs/metadata. Shared sequence alignments and model weights
are not replicated per job. Larger proteins/assemblies can use substantially
more storage and memory.

The former workflow consumed 8.48 million model tokens in setup investigations
without submitting a GPU job. That historical failure remains recorded. The
family compiler/native execution path makes zero setup LLM calls; unsupported
families stay visible for one shared review instead of repeating the failure
per member. Software and live validation are recorded in the review manifest.
