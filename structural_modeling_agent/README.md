# Structural-modeling agent

The canonical directory and Python package are `structural_modeling_agent`.
The workspace retains `structural_modeling` as a filesystem compatibility link
for historical campaign paths. New imports and commands use the canonical name;
the handoff artifact remains `structural_modeling_handoff.json`.

`structural_modeling_agent` has an independent, persistent decision agent. It accepts
a standalone task or a discovery handoff, investigates scientific uncertainties,
chooses a model, checks the choices, renders inputs, inspects predictions and
revises decisions in response to failed checks. Its controller does not invoke
the discovery orchestrator. The existing modeling pipeline modules are tools
available to the agent.

Reaction-barrier calculations live in the [barrier submodule](barrier/README.md).
It provides generic reaction specifications, cluster and embedded potentials,
path/sampling operators and separate evidence gates. The controller exposes
barrier preparation, planning, reserved submission, status and assessment tools.
Use `objective: "evaluate_barriers"` and `--barrier-backend` for an explicit
energetics task; optional cofold follow-ups retain their primary-evidence gate. Run
`python -m structural_modeling_agent barrier --help`; historical
`python -m enzyme_barrier` commands remain compatible.

```bash
python -m structural_modeling_agent --task modeling_task.json --system-id my_enzyme_system
# Allocate and inspect destinations without model calls or compute submission:
python -m structural_modeling_agent --task modeling_task.json --system-id my_enzyme_system --initialize-only
python -m structural_modeling_agent --inspect-run /absolute/path/to/run
python -m structural_modeling_agent --resume /absolute/path/to/run
python -m structural_modeling_agent --check-environment
# Installed console command:
structural-modeling-agent --task modeling_task.json --output-dir modeling_run
```

New runs default to `<workspace>/results/<system-label>-<target-hash>/structural_modeling/<UTC-run-id>/`.
In a source checkout, the workspace root is fixed independently of the launch
directory. An installed package falls back to the caller's `./results`, never
to site-packages.
Set `--results-root` or `ENZYME_AGENT_RESULTS_ROOT` for a stable shared root;
`--output-dir` remains an explicit override. The system hash separates different
targets with the same label, and unique run IDs prevent accidental overwrites.
Absolute locations are pinned when the run is created, so changing the working
directory on resume does not relocate results. Existing campaigns and older run
paths are preserved; no automatic migration or `latest` alias is used.

```text
results/<system>-<target-hash>/structural_modeling/<run-id>/
  run_manifest.json          # immutable system/run/source-task identity
  task_snapshot.json
  results_index.json         # regenerable directory/status map
  checkpoint.json            # authoritative state and transcript
  report.json
  structural_modeling_evidence.json
  sources/ and artifacts/
  models/<accession>/revision_<n>/
    cofold/input.yaml
    cofold/execution.json
    cofold/predictions/       # default provider destination
    followups/dynamics/<id>/request.json
    followups/dynamics/<id>/bioemu2/job.json
    followups/dynamics/<id>/bioemu2/results/
    followups/de_novo_design/<id>/design_request.json
    followups/de_novo_design/<id>/design_runs/<native-design-run>/
```

The agent uses `inspect_result_locations` to find planned and recorded paths.
The design handoff includes `--output-dir`, so its agent-injected RFD4 job folders
stay beneath that follow-up's `design_runs` root. `register_design_run` links the
actual design manifest/report without treating the agent's report as validated
enzyme generation. BioEmu2's prepared command names its result directory explicitly.
Externally supplied cofold predictions remain at their original locations and are
listed in the index rather than copied. Planned paths and existing folders are
not completion evidence. Offline inspection rebuilds the index from the checkpoint
even if the saved index lags after an interruption.

Install the repository with the `structural-modeling` extra and configure the
shared LLM provider client. Each invocation has independent turn, token and
runtime allowances. `--model`, `--max-turns`, `--token-budget` and
`--runtime-budget-seconds` override them. Provider calls or tools already in
progress can finish beyond an allowance. CLI help and offline validators do
not need provider credentials.
The environment check reports package availability without starting a model or
contacting a provider; it does not verify credentials or resource reservations.

A standalone task is JSON with `target` and a nonempty `candidates` list. Each
candidate has a unique `accession` (also usable as a local identifier), its
`sequence`, and supplied `evidence`. An existing `modeling_input` is a proposal
for the agent to inspect. Schema-3.0 discovery handoffs are also accepted as
tasks; schema-2.0 handoff entries lacking admission evidence remain leads.
Authoritative sequence retrieval is available when a candidate has no
supplied sequence. No discovery run directory is required. `objective` is
`model_and_validate` by default; `prepare_inputs` finishes after validated input
generation. Candidate evidence can include paths to existing full/fragment MSAs;
MSA generation is not automatically invoked.

The model chooses among evidence inspection, primary-source retrieval,
literature search, molecular chemistry inspection, assembly assessment,
model proposal/revision, setup validation, input rendering, provider submission,
job reconciliation and prediction validation. Each proposal records its
rationale and rejected alternatives. Failed validation is returned to the
decision loop. A prose completion claim does not complete the task.

`agent_state.json`, `transcript.json`, `events.jsonl`, immutable source snapshots,
decision artifacts, revision-specific model inputs and `report.json` live in the
agent's own run directory. New revisions do not inherit previous input or pose
validation. Source, MSA, rendered-input and prediction fingerprints are checked
again before completion. One controller owns a run at a time. Resume retains
the task, evidence, decisions, usage and provider receipts.

`checkpoint.json` is the authoritative atomic snapshot of both controller state
and transcript. The two separate JSON files are compatibility views and can lag
after an interruption. Existing runs without a checkpoint use the legacy files;
once resumed and saved, they use the coherent checkpoint. A legacy pair already
made inconsistent before this upgrade cannot be reconstructed automatically.

Local evidence reads and source declarations are restricted to paths explicitly
supplied in the input task (a `path` or `*_path` field, or `evidence_paths` list)
and files created in the run's `sources/`, `artifacts/` or `models/` folders,
plus its own manifest, task snapshot, result index and final reports. Relative paths
are resolved against the invocation directory. Symlinks are resolved before
authorization, and changing a symlink cannot grant access to a new target. Local
inspection is limited to 16 MiB per file. Source content is untrusted input, not
an instruction to the controller. This restriction is not a sandbox for every
other tool or an assurance against prompt injection; run the process with only
the filesystem and provider access appropriate to the task.

## Production cofold execution

Set `workflow_mode` to `production_cofold` in a production task. The setup
controller retains the complete compact contract, exposes only stage-relevant
tools, and checkpoints an unproposed setup after a bounded turn allowance
(`setup_turn_limit`, default 8). Cached schemas read through another artifact
path do not count as new information. These are resource controls, not scientific
rejections or evidence of exhausted discovery.

After the agent has selected and recorded a source-linked model, use
`python -m structural_modeling_agent.cofold_execution --request REQUEST.json`.
The request names the task, saved proposal, model run, provider adapter, decision
origin and explicit `execution_intent: submit_after_checks_pass`. This executor
uses the existing agent gates to render and submit, then reconciles jobs,
checks the complete prediction set and exports measurements without provider
calls for polling or file inspection. Existing jobs are reconciled before any
submission. Scientific interpretation remains a separate agent decision.

Production workers operate per ready candidate and can run independently of
discovery. Completed evidence is imported as it arrives. `execution_status`,
pose QC and scientific status are reported separately: a full ensemble with a
contradicted or unresolved hypothesis is still a completed computation. No
structural result automatically establishes activity or changes discovery rank.
The shared catalog accepts `structural_agent_sources` and indexes prediction
artifacts in place; `campaign_catalog.lookup.lookup_predictions` supports
accession and sequence lookup before any new prediction.

`python -m structural_modeling_agent.production --task TASK --run-dir RUN
--backend CONFIG --execution-intent submit_after_checks_pass` runs the complete
per-candidate controller. The model records its scientific choices; native
execution validates, submits, reconciles and measures; the model then interprets
the returned observations. Token and decision allowances cover the entire saved
candidate task across these stages and resumes. Polling and file checks use no
provider calls. A stalled setup returns a checkpoint so other candidates can
advance. `prepare_only` preserves inputs without submitting them.

The rolling discovery-side selector is
`enzyme_discovery_agent.models.modeling_portfolio.select_modeling_portfolio`.
It reads the current registry on each invocation, schedules source-cited
investigate/prioritize hypotheses without requiring a native activity rank, and
accepts source-scoped identifiers. Its optional protected unannotated slots are
an exploration policy, not evidence of higher activity. Identity and molecular
setup must still pass the modeling gates before compute.

The Lambda adapter requires explicit `--state-dir` and `--pool` arguments in its
backend command for new submissions. It verifies current pool lookup tables and
checks exact sequence aliases at submission time. Existing prediction matches
require review; a changed scoring definition does not trigger a new prediction.
Sample counts come from the declared plan within the backend resource allowance.
Concurrent submissions share one claim lock. These operational controls do not
prescribe a reaction mechanism, participant, ion count or assembly.

When the supplied pool has predictions, `review_existing_predictions` verifies
the original producer report and its bound files and returns the saved frozen
measurements. The agent records applicability and unresolved assumptions. This
path preserves the original scientific record and needs no fresh prospective
proposal or GPU submission. Reuse reviews are indexed as reviews of the original
execution, so the pool does not count them as new prediction ensembles.

## Scientific validation tools

### Default cofolding and purpose-specific follow-ups

A discovery handoff defaults to cofolding. Evidence investigation and molecular
setup precede it; BioEmu2 and generative design do not satisfy that baseline or
serve as automatic fallbacks when the cofold provider is unavailable.
`inspect_workflow_policy` reports the roles and the current baseline gaps.

After a complete, traceable cofold ensemble has been inspected, `plan_followup`
requires a scientific question, the hypotheses involved, why cofolding cannot
answer it, expected information and consequences under supporting/refuting
outcomes. A nonproductive but complete cofold can motivate a follow-up; missing
protein atoms, stale inputs or unresolved molecular identities cannot.

| Purpose | Tool/workflow | Meaning of its outputs |
| --- | --- | --- |
| Initial structural assessment | Configured cofold predictor | Joint complex poses checked against the declared setup and physical criteria |
| Conformational variability/dynamics investigation | BioEmu2 | Optional separate ensemble study; never retroactively validates the primary cofold |
| De novo enzyme generation | RFD4-Proteina through the design agent | A design request with target, rationale and reference evidence; generated enzymes need their own validation |
| Chemical-step energetics | ML/MM, when justified | Reaction-path/energy evidence requiring its own input, numerical and reporting checks |

The BioEmu2 adapter is optional and was checked against Microsoft's
[system](https://github.com/microsoft/bioemu2/blob/06be369dd555849165a737472d42d936959e46fc/docs/system.md),
[MSA](https://github.com/microsoft/bioemu2/blob/06be369dd555849165a737472d42d936959e46fc/docs/msa.md)
and [sampling](https://github.com/microsoft/bioemu2/blob/06be369dd555849165a737472d42d936959e46fc/docs/sampling.md)
interfaces. It preserves the rendered proteins, ligands and ions. Preparation
uses local fingerprinted alignments and a local checkpoint with `config.yaml`;
it performs no MSA query, model download or GPU execution. CSV alignments are
converted explicitly rather than relying on Boltz conversion, which ignores
MSA paths. `prepare_bioemu2` only accepts a saved dynamics plan after cofold
inspection. Its prepared job can be run later in a separately provisioned
BioEmu2 environment under the user's resource scheduler.

The current adapter uses equilibrium sampling for conformational hypotheses;
time-lag propagation is not wired into the controller. Equilibrium samples
must not be interpreted as a trajectory or kinetic rates. `inspect_dynamics`
retains every frame, including engine-rejected or unchecked frames, and records
its results under the follow-up. It does not overwrite primary pose validation
or fabricate Boltz confidence metrics. Engine physicality is protein-only; the
independent full-complex checks remain necessary. RFD4 requests produce a
reviewable handoff and native design-agent command; they do not launch generation.
No BioEmu2 installation, GPU sampler run, or engine numerical benchmark was
performed in this remediation; the optional runtime is tested with a mock.

### Acceptance and saved-result integrity

Protein validation requires the standard residue heavy-atom inventories,
including elements. OXT and hydrogens are not required by this inventory check.
Nonstandard residues and permitted omissions need an explicit
`poses.protein_atom_policy` declaration with independent source records and
rationale. Declaring a required heavy atom missing records the limitation but
does not authorize a completeness pass or remove it from the missing inventory.
Distinct site IDs
cannot count identical physical reactive assignments repeatedly. Shared atoms
are possible, but identical reaction pairs/geometries cannot satisfy multiple
active sites. These checks do not establish fold accuracy or chemical connectivity.

The complete physical acceptance plan is checked before rendering/submission.
Malformed readable prediction files retain their hashes and failed checks; a
declared minimum passing count still governs the ensemble outcome. An unavailable
fingerprint blocks acceptance rather than crashing reporting. The standalone
builder also validates and fingerprints full and fragment MSAs.

Queue completions now bind input/MSA bytes, seed, producer declaration/files,
geometry, requested sample count and scoring/verification code. Reuse and ranking
recheck the actual files and recompute geometry. Legacy unbound or missing
evidence becomes unresolved; it is not silently reused or overwritten. Orphan
output directories are retained, while attempts are journaled before allocation
and numbering skips existing outputs. A prior running/unknown producer still
requires reconciliation. Reservations and actual provider execution remain
external responsibilities; environment flags do not prove ownership.

Frozen bundle application rejects identical pose contents under different paths.
Distinct hashes alone do not prove independent sampling. `approach_consistency`
has definition version 2.0: `degrees(acos(norm(mean(unit_attack_vectors))))`,
with zero for aligned vectors and 90 degrees for a vanishing resultant. Historical
v1 fits are not automatically transferable; models using it require explicit
version review/recalibration. No stored calibration or campaign score was refitted.

`audit_exports` performs a read-only exposure review of selected saved formats.
The remediation audit is retained under `analysis/review_remediation_20261030`.
Unverified provenance or unavailable current evidence does not prove a historical
campaign was affected. Source results, rankings and campaign data are preserved.

### Scientific hypotheses and feedback

The controller implements the reasoning loop requested in the supplied essay:
form competing explanations, evaluate evidence, choose a mechanism-specific
model, state falsifiable predictions, and respond when observations disagree.
`scientific_reasoning.py` checks these records separately from physical setup
and pose validation. The tools and required fields are described in
`inspect_validation_contract`; every `propose_model` call now requires
`scientific_reasoning` and an explicit `modeled_step.reaction_center`.

The compact scientific schema remains in the controller context after ordinary
history compaction. Use `inspect_validation_contract(section=...)` for focused
`scientific_reasoning`, `reasoning_rules`, `source_record`, `proposal`, `setup` or
`poses` sections. `read_candidate_section` also accepts a JSON pointer within a
section. Evidence and candidate text pages return `next_offset` and fit within
the newest tool preview, so truncated JSON does not silently skip source text.
The controller retains up to four short inspected source passages as untrusted
data, checks their current file hashes and preserves full tool artifacts.

`propose_model` accepts either the complete `modeling_input`, or an explicit
`modeling_input_from: "candidate" | "current"` with optional
`modeling_input_changes`. Changes follow JSON merge-patch semantics: nested
objects merge, arrays replace and `null` deletes a field. The controller records
the base fingerprint and changes, copies only the model data, and applies the
same source, scientific and setup checks to the resulting proposal. It never
inherits an earlier execution or validation pass. This lets the model focus on
its scientific decisions without repeatedly reproducing a large unchanged input.
`render_inputs` can use the task's explicit candidate `evidence.msa_path` when
`msa_index` is omitted. An explicit index maps candidate accessions to full
precursor MSAs, not chain IDs. Sequence and fragment-MSA validation still apply.

Each proposal declares at least two hypotheses, their assumptions, uncertainties,
source interpretations (supporting, contradictory or contextual), numerical
predictions, and a test with distinct expected outcomes and controls. Unsupported
alternatives can be recorded with an empty evidence list. Execution requires
supporting evidence for the selected hypothesis and valid source fingerprints
and passages for every citation, including contrary evidence. The checks do not
establish that a passage entails an interpretation.

The selected hypothesis must match the modeled mechanism, step index and reaction
center. ATP α attack requires `PA`/`O3A` on the same declared ATP residue at every
site; γ transfer requires `PG`/`O3B`. These checks use CCD atom names. A different
naming convention needs an adapter. Carbonyl and other centers require an
explicit interpretation and the existing atom/physical checks; the controller
does not independently infer their chemistry. Proxy plans must use the selected
hypothesis's exact mechanism and step description. ATP plans also declare
`context.reaction_center`, preventing silent reuse of an α-based plan for γ transfer.

After setup/pose checks, proxy evaluation or new observations:

1. Call `inspect_reasoning` to obtain the prediction comparisons and
   `assessment_sha256`.
   `compare_hypothesis_tests` compares the declared numerical intervals across
   hypotheses. It prioritizes readouts that distinguish unresolved hypothesis
   pairs, retains each test's declared controls, and identifies pairs needing new
   predictions. Overlapping intervals, touching bounds, and different units or
   endpoints do not establish discrimination. These priorities do not estimate
   Bayesian information gain, experiment feasibility or cost.
2. Use `record_reasoning_observation` for source-backed numerical measurements.
   Predictions and observations compare only when metric, structural/experimental
   scope, units, endpoint and conditions match exactly. Other observations remain
   visible as unmatched evidence. Numerical extraction from a quoted source is an
   asserted interpretation, not independently authenticated measurement data.
   If the source reports measurement uncertainty, include `measurement_interval`
   with finite bounds containing `value`. An interval partially overlapping the
   prediction is unresolved; an entirely disjoint interval is contradictory.
3. Call `respond_to_reasoning_feedback` with the current fingerprint, rationale,
   affected hypothesis IDs, action and controlled next test. It must address the
   selected hypothesis and every contradicted hypothesis. A stale response cannot
   authorize execution, revision or completion.

Actions `revise`, `investigate` and `abstain` withhold further execution and
completion. A new proposal can follow the response; `revise` requires an actual
change to the model or scientific record. `retain` allows a supported model to be
used while the contradiction and next test stay explicit. This records the
agent's scientific judgment; it does not prove that judgment is justified or
force a particular causal explanation. Model error, the wrong step/proxy,
assay endpoint mismatch and active-enzyme/stability differences are competing
explanations to consider where relevant. Tests are proposed, not automatically
submitted to a laboratory.

The automatic `pose_pass_fraction` readout uses scope `structural`, unit
`fraction`, endpoint `declared_pose_checks` and conditions
`{"basis": "declared_ensemble"}`. It measures the fraction passing **all** declared
physical checks, not a near-attack-only frequency or experimental activity.
Missing samples/atoms, duplicate sample contents or changed prediction files
yield an unresolved observation. Even a pose with both failing and missing
checks must not become an observed zero.

Proxy predictions use `metric: "proxy:<descriptor_id>"`, scope `structural`,
endpoint `declared_proxy_plan`, the catalog's exact units, and conditions
`{"parameters": {<resolved extractor parameters>}}`. The adapter requires verified
proxy artifact integrity, the complete declared input ensemble and computed
coverage; partial or unavailable extraction remains unresolved. Other readouts
use explicit source-backed observations; there is no general assay extractor or
unit converter.

Controller proxy extraction uses the exact nucleophile, electrophile and angle
reference declared in the pose validation site. A separate `geometry` atom map
cannot cause an α-phosphate score to be returned for a validated γ-phosphate
step. Optional `leaving_group` and `charge_centre` references also belong in that
site for descriptors that need them. For multisite assemblies the single-site
runner requires `context.site_id`; it does not pick a site automatically. The
extraction report retains `role_resolution` so the measured atoms can be checked.
The standalone compatibility runner keeps its geometry-based resolver unless
`use_validated_sites=True` is selected programmatically.

Observation IDs are immutable. A favorable result cannot overwrite a negative
one. Source-backed observations carry into subsequent revisions; earlier
model-derived geometry stays attached to its original model in history.
Source-backed structural observations default to `applicability: "model"` and
retain the originating model fingerprint. After a model change they remain
visible as inapplicable evidence rather than judging the new model. An
independent structural measurement can declare `applicability: "candidate"`
with an explicit `transfer_rationale`. Experimental observations default to
candidate applicability and still require matching endpoints and conditions.
Reports, discovery evidence exports and resumed context retain predictions,
contradictions and responses. Registration distinguishes an initial record from
a revision after feedback and lists observations already known. Registration
does not establish blinding or prospective validation. Earlier checkpoint runs
can still be inspected and jobs reconciled; they need a new proposal with the
scientific record before further execution or completion under these checks.
When older saved pose/check feedback needs a response, `inspect_reasoning`
supplies `response_hypothesis_ids: ["legacy_unrecorded"]`. Use that ID to
acknowledge the historical checks before the new proposal. This does not invent
a prior hypothesis or allow the legacy decision to complete without the new record.

Offline behavioral verification:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest structural_modeling_agent/tests/test_scientific_reasoning.py \
  structural_modeling_agent/tests/test_modeling_agent.py
```

These tests compare a prose-only proposal with a structured decision and exercise
mechanism switching, contradictory assays, endpoint/condition mismatches, missing
measurements, stale sources, actual revision, resume, reports and a scripted tool
loop. They establish software behavior; they do not measure autonomous LLM
reasoning quality, prospective enzyme hit rates or experimental improvement.

A [live synthetic integration audit](analysis/reasoning_eval_20261029T221125Z/evaluation_audit.json)
also records a configured-provider run that preserved H1's contradictory endpoint,
revised the explanation to H2 with competing H3, proposed an orthogonal measurement,
and prepared validated inputs. The run was resumed during controller debugging;
its substantial token usage is recorded. It is not a single uninterrupted trial,
an efficiency comparison, a broad reasoning benchmark or biological validation.

The agent reads [validation_contract.json](validation_contract.json) for the
machine-readable contract. Each setup choice has independently reviewed
expectations grounded in sources independent of the proposed model, with source
identity/type, exact passage,
local text snapshot, SHA-256 and the agent's interpretation. Hypotheses and
predictions cannot validate their own setup. Source-integrity checks establish
traceability, not the truth of an interpretation; conflicts and transfer gaps
must remain explicit.

Source checks distinguish verified file integrity from asserted attribution and
interpretation. `reviewed` is a submitted assertion, not proof of independent
review. Proxy plan registration records the plan hash and time, but does not
establish that the decision maker was blinded to labels. Proxy evaluation has a
separate integrity status covering the saved plan, result, predictions and
confidence sidecars; it does not establish predictive validity.

For externally supplied feature values, `feature_source` may point to a JSON
object containing `accession`, `feature_definition_id`, `n_poses` and `features`.
When `feature_source_sha256` is also supplied, bundle application checks the
file fingerprint and exact agreement of those fields before scoring. This checks
content identity, not whether the features were correctly extracted. Legacy
inputs without a declared source fingerprint remain explicitly unverified.

Setup checks cover:

- Functional assembly and complete site context, distinguishing protomers,
  processed chains and active sites.
- Full-sequence identity, maturation state and chain boundaries.
- The complete cofactor inventory for the chosen step, with identities, roles
  and an explicit per-site or per-model counting basis. There is no default metal.
- Substrate identities, molecular states, roles and inclusion.
- The declared mechanism and reaction step.

After cofolding, `validation.py` reads PDB or mmCIF predictions and compares them
with the renderer's entity map. It checks every declared active site, jointly
measuring attack distance and angle; it also checks ligand contacts/clashes,
required cofactor contacts, declared metal-donor distances and multichain
interfaces. Sampling requirements and all physical tolerances are explicit plan
inputs supported by source evidence. All poses and failed checks are retained.
Missing evidence or atoms yield `abstain`; contradictory or failing checks yield
`fail`. No pose or setup failure establishes experimental inactivity.

Metal checks assess declared modeled donors, not an omitted hydration sphere.
Substrate elemental inventory does not validate bond connectivity. Construct
identity does not establish folding, maturation rate or active enzyme fraction.
These limitations remain attached to the report. The agent currently renders
unrestrained cofold systems; other modeling methods require their own adapters.

Offline tools can also be called directly:

```bash
python -m structural_modeling_agent.validation setup \
  --record resolved_modeling_input.json --sequence FULL_PRECURSOR_SEQUENCE \
  --output setup_validation.json
python -m structural_modeling_agent.validation poses \
  --manifest campaign_manifest.json --accession CANDIDATE \
  --predictions predictions/ --output pose_validation.json
```

The legacy `build_step1_inputs.py` CLI remains available for compatibility. Add
`--require-validated-setup` to require the scientific checks there as well.
The agent always requires them. Renderer shape/readiness and scientific support
are separate judgments; a legacy `status=ready` alone does not establish support.

## Compute provider adapter

Without `--backend`, the agent can investigate, prepare inputs and validate
existing predictions, and cannot submit compute jobs. The optional backend file
selects the user's provider explicitly:

```json
{
  "provider": "YOUR_PROVIDER",
  "argv": ["/absolute/path/to/your/provider-adapter"],
  "timeout_seconds": 60,
  "resource_policy": {
    "reservation_managed": true,
    "max_jobs": 1,
    "max_predictions_per_job": 50
  }
}
```

The numeric limits above illustrate the adapter format and are not scientific
sampling defaults. The adapter receives a JSON request on stdin with operation
`submit` or `status`, an idempotency key, provider, input path/hash, prediction
count, work directory and resource policy. It returns JSON containing the same
key and a status (`queued`, `running`, `complete`, `failed`, `unknown`, or
`not_found`). Additional scheduler job identifiers remain in the saved receipt.
The configured adapter owns actual reservations, worker/GPU pinning, dependency
checks, resumability and result retrieval for that provider. Its resource behavior
must be verified independently; a configuration flag is not proof of a reservation.
The interface remains provider-neutral; actual execution belongs to the selected
deployment adapter.

For the user's configured Lambda deployment, the reusable provider is
`python -m structural_modeling_agent.lambda_boltz_backend --state-dir /absolute/NFS/provider-state`.
The state directory and canonical-pool lookup guard are explicit run inputs.
It validates the installed predictor's argument parser before GPU allocation,
uses atomic reservations, and distinguishes live foreign processes from stale
driver observations or a producer that has already exited. Other providers
remain selected through their own adapter configuration.

For a `model_and_validate` completion, predictions also need a producer manifest
binding them to the current rendered input and execution. Pass its path to
`validate_predictions`, or return `prediction_manifest` in the adapter receipt.
The JSON fields are `input_sha256`, `execution_sha256`, and `predictions` (an array
of `{path, sha256}`). Relative prediction paths resolve from that manifest's
directory. Physical checks can still run without this provenance, but completion
is withheld. Duplicate files and replacement of a previously inspected ensemble
cannot silently inflate or replace sampling support.

Submission intent is checkpointed before calling the adapter. An interrupted or
ambiguous submission is reconciled by key, not blindly submitted again. Pending
jobs occupy the run's job allowance and prevent incompatible revisions. A
`waiting_for_backend` checkpoint can be resumed after provider state changes.
On resume, pending receipts are queried before the next model decision, subject
to the invocation budget. A failed status request preserves the prior pending
state and records the error; it never implies that the job is absent.

## Existing modeling and experimental tools

The active workflow connects discovery evidence, predicted structures, experimental
proxy evaluation, and application to new candidates. Code runs from
`enzyme_discovery_agent` and `structural_modeling_agent`; the previous agent archive is
historical provenance. Experimental measurements remain in `exp_data` and are
linked by paths and SHA-256 fingerprints rather than copied into model code.

## Folder organization

| Location | Purpose |
| --- | --- |
| `modeling_reasoning/` | Molecular-state, assembly and reactive-role resolution |
| `proxies/` | Reusable descriptor extraction, calibration and execution |
| `integration.py`, `integration_examples.py` | Export and apply frozen proxy bundles |
| `developability/`, `loop/`, `mlmm/`, `exploration/` | Reusable supporting workflows |
| `scripts/` | Maintained input/ranking utilities and compatibility scorers |
| `../exp_data/amide_synthetase/`, `../exp_data/nylon_hydrolase/` | Co-located study code, proxy evidence, and figures |
| `campaigns/`, `integrated_campaigns/` | Current input/output data and frozen evidence bundles |
| `figures/candidacy/` | Authoritative manuscript figure masters |
| `tests/` and package-local `tests/` | Verification of maintained code |

Historical launchers, superseded plots and retired figure copies were consolidated
in [the cleanup archive](../0_prev_agent_archive/structural_modeling/cleanup_20261008/README.md).
Its manifest records original paths and SHA-256 hashes; no archived file was deleted.
Historical scripts retain their original campaign paths and are not current launch
instructions. The active code does not import them.

The six retained `scripts/` tools are `build_step1_inputs.py`,
`cofactor_stoichiometry.py`, `dedup_candidates.py`, `build_candidate_prioritization.py`,
`angle_gate_proxy.py`, and `step1_nac.py`. The last two remain for existing callers
and compatibility checks. The current ranking metric is
`inverse_distance_attack_angle_v1` in `proxies/attack_angle.py` and
`knowledge/attack_angle_metric.json`; harmonic distance is always reported
alongside the weighted angle. Historical Gaussian-score bundles remain available
only through their explicit definitions. Study-specific generators are retained
to reproduce their associated results and figures.
See [the ranking policy](CANDIDATE_RANKING_POLICY.md) for current and historical
definitions. Earlier synthetase proxy
screens and drafts are in [the study archive](../exp_data/amide_synthetase/archive/cleanup_20261008/MANIFEST.md).

## Workflow

1. Receive a standalone modeling task or the discovery agent's evidence handoff.
2. The modeling agent investigates and chooses the molecular state, assembly,
   reaction step, participants and reactive atoms, using `modeling_reasoning`
   and validation tools. Generate the declared cofold ensemble after setup checks.
3. Use the agent tools `get_structural_proxy_guidance` and
   `validate_structural_proxy_plan`, then `declare_proxy_plan`, to save a small panel before inspecting
   activity labels. Each hypothesis specifies its rationale, direction, evidence,
   atom assignments and parameters.
4. Use `evaluate_proxy_plan` on the current inspected PDB ensemble and report descriptor values, input requirements, missingness
   and ensemble coverage. Confidence is a baseline, not an activity filter.
5. Compare descriptors with matched enzyme-level assay measurements. Keep the
   endpoint, controls, selection procedure, expected direction and validation scope
   attached to the result. Freeze any selected definition or empirical model.
6. Apply that frozen artifact to discovery candidates without refitting. Return
   the score together with its units, calibration provenance, transfer status and
   uncertainties to the discovery agent for review.

Every controller report also writes `structural_modeling_evidence.json`, with
schema 3.0. It retains setup/pose checks, all failed samples and per-site weighted
angles and harmonic distances. Its status is separate from experimental activity.
It can be imported explicitly without changing discovery ranking:

```bash
python -m enzyme_discovery_agent.import_structural_evidence DISCOVERY_RUN MODELING_RUN/structural_modeling_evidence.json
```

Existing reports can be exported with
`python -m structural_modeling_agent.evidence MODELING_RUN/report.json`.
Both export and import check supporting artifact identity; handoffs bound to a
discovery run cannot silently transfer to a different run. Old reports without
the required execution/provenance fields remain incomplete. Proxy plans are
immutable within a revision. The current runner requires the exact inspected
`*_model_*.pdb` file set; other formats require their own descriptor adapter.

The `loop/` helpers are not an autonomous experimental driver. Calibration uses
matching family, proxy version and assay conditions, counts each candidate once,
and gives negative correlations zero directional trust. A zero-factor veto or
explicitly ineligible candidate cannot acquire an exploration bonus.

## Choosing physical hypotheses

The existing guidance tool now accepts an optional explicit `assay_context`.
It organizes questions about access, adsorption, maturation, assembly, stability,
substrate positioning, chemical support, product release and prediction
reproducibility before proposing descriptor names. Availability, relevance,
modeling fidelity and experimental validation remain separate judgments.
Priorities identify processes worth investigating; they do not assert the
rate-limiting step or infer assay conditions from a family name.

For example, explicitly declaring an insoluble polymer, a crude-lysate endpoint,
a mature modeled state and required processing surfaces access/adsorption
hypotheses and the missing precursor/active-enzyme-fraction measurements.
A mature complex cannot establish processing rate. Total-enzyme normalization
does not establish the amount of active mature enzyme either.

```python
from enzyme_discovery_agent.structural_proxy_planning import proxy_guidance

guidance = proxy_guidance(
    mechanism="Evidence-supported N-terminal Thr amide hydrolysis",
    reaction_step="Hydrolysis in a processed substrate-bound assembly",
    assay_endpoint="Product release from polymer by a crude preparation",
    available_inputs=["reactive_roles", "substrate_chains", "protein_alignment"],
    assay_context={
        "substrate_state": "insoluble_polymer",
        "measurement_basis": "crude_lysate",
        "modeled_state": "mature",
        "requires_processing": True,
        "assembly_status": "uncertain",
        "substrate_is_assayed": False,
    },
)
```

`physical_process_map`, `modeled_state_gaps` and `exploration_protocol` explain
what to investigate, what the structures omit, and how to organize the search.
The catalog adds targeted adsorption, assembly, stability, product-release and
substrate-strain hypotheses without claiming a generic implementation.

Plans may optionally declare `search_design`: objective, discovery versus
confirmatory mode, candidate budgets, enzyme-level validation, combination
policy, full-search correction, negative-result recording and QC sensitivity.
Small physically coupled pairs require a coupling rationale and decomposed
component evidence. Selection and transformations belong inside training folds
and must be repeated under the null. These fields declare a protocol; the
existing single-candidate runner does not execute a panel search or combinations.
Existing plans and tool calls remain compatible.

[The worked exploration-plan example](../docs/examples/structural_proxy_exploration_plan.json)
shows the optional declaration. Its parameters and pose counts are examples,
not universal defaults; exact atom assignments, chemical references and missing
states must be resolved before extraction. Plan validation checks declarations,
not chemical truth or predictive success.

The current worked exploration, with parallel extraction, exact scripts,
physical feature definitions, negative outcomes and selection-aware comparison,
is documented in
[the nylon-hydrolase study](../exp_data/nylon_hydrolase/structural_proxies/attempts/broad_proxy_search_20261008/README.md).
Its retrospective outcomes are not supplied as answers to upstream hypothesis
selection.

```bash
python -m structural_modeling_agent.proxies --catalog
python -m structural_modeling_agent.proxies \
  --record resolved_candidate.json --plan reviewed_proxy_plan.json \
  --predictions predictions/ --output candidate_descriptors.json

python -m structural_modeling_agent.integration export campaign_config.json --output bundle.json
python -m structural_modeling_agent.integration apply bundle.json candidate_index.json --output-dir results/
python -m enzyme_discovery_agent.import_structural_evidence \
  enzyme_discovery_agent/runs/RUN_ID results/candidate_proxy_evidence.json
```

The structural evidence tool reads `candidate_proxy_evidence.json` directly.
Registry import is explicit and preserves the discovery rank. Experimental-model
scores are kept in their original units, including log concentration, rather
than coerced into normalized mechanistic checks.

Install the repository's `structural-modeling` dependency extra for SciPy, Gemmi
and plotting support. The shared basic descriptor library uses NumPy; chemical
role resolution additionally uses RDKit.

## Five descriptor categories

The packaged catalog is
`enzyme_discovery_agent/structural_proxy_catalog.json`. It supplies hypothesis
questions, prerequisites, units, scope limitations and execution status.

| Category | Shared implementations |
| --- | --- |
| Prediction confidence | Complex pLDDT/ipTM, aggregate ligand ipTM, explicitly mapped directional protein–substrate chain-pair ipTM |
| Reactive geometry | Approach distance, mechanism-relative angle deviation, joint near-attack fraction, approach orientation |
| Substrate accommodation | Target-ligand contact fraction and residue count, explicit reactive/distal-group SASA |
| Catalytic environment | Annotated acid/base proximity, annotated donor contacts, signed protein-charge potential and projected field |
| Prediction consistency | Contact-fingerprint Jaccard, protein-aligned atom-corresponded ligand/reactive-atom dispersion, distance and field variability |

Access-channel geometry, metal coordination and maturation need targeted
extractors and reviewed references. The runner reports those requirements rather
than claiming that a generic static descriptor covers them. Reaction barriers
remain outside the static/cofold tier.

The unannotated `base_proximity` and donor-count features are exploratory
neighborhood measurements. Use mechanism-supported atom annotations for claims
about catalytic acid/base or charge-stabilizing groups. Explicitly distinguish
the atom defining an attack angle from the leaving atom; they can differ.

`proxies/workflow.py` retains actual parameters and coverage. Positional
dispersion requires valid protein alignment and consistent atom identity.
Protein, substrate and reactive-role coordinates share the same transformed
frame. Missing roles/alignment are reported as unavailable, not zero or inactivity.

## Frozen calibration and discovery application

`integration_examples.py` contains downstream configurations for the two current
studies. These configurations are not supplied to upstream enzyme discovery.

```bash
python -m structural_modeling_agent.integration examples
```

This exports bundles and application reports under `integrated_campaigns/`.
Each bundle contains the selected parameters or fitted coefficients, original
training means/scales, required sampling, assay endpoint, mechanistic/family and
assembly scope, validation status, limitations and source fingerprints. A changed
extractor cannot silently score under an old bundle. Additional measurements
should produce a reviewed new version rather than overwrite its interpretation.

The existing activation score uses exactly 50 predictions and retains its
sampling support statistics. Its strong retrospective top-three recovery does
not establish a general activity ranker. Application to another family or
reaction coordinate is labeled as transfer.

The existing mature-complex yield model reuses its frozen nine coefficients and
training standardization. It requires ten predictions of the same two-site
assembly and the same internal-bond register/contact definitions. Different
hydrolase mechanisms or assemblies are reported as inapplicable; shared
confidence baselines can still be evaluated. A new mechanism requires its own
reviewed proxy plan and calibration, not the borrowed empirical model.

The lightweight `screen` function now enforces its confidence-adjustment gate
before mechanistic promotion. That partial-correlation threshold is a descriptive
screen, not a conditional significance test. Selection-aware tests, held-out
models with training-only preprocessing, and prospective measurements remain
the basis for stronger predictive claims. Multiple cofolds are not independent
experimental observations.

## Active data and figure ownership

Imported discovery records live in
`enzyme_discovery_agent/runs/imported_amide_discovery` and
`enzyme_discovery_agent/runs/imported_hydrolase_discovery`.
Required activation cofold files live in `campaigns/amide_activation/modeling`.
`campaigns/active_artifact_provenance.json` records verified copies and their
historical origins. The archive is preserved and is not a runtime dependency.

Authoritative manuscript figure masters live in `figures/candidacy`, with active
plotting code under `../exp_data/amide_synthetase/code` and `../exp_data/nylon_hydrolase/code`.
Experimental input tables and analyses remain linked to `exp_data`.
Plotting does not mutate Overleaf. Publication copies only referenced figures:

```bash
python -m structural_modeling_agent.publish_figures --check
python -m structural_modeling_agent.publish_figures
```

The publication command preserves local manuscript work, pulls Overleaf first,
verifies copies against their masters, and commits/pushes only relevant figure
changes. Overleaf contains manuscript copies, not the only surviving originals.

## Verification

Run the focused checks with native thread pools limited to one CPU:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest structural_modeling_agent/tests/test_proxies.py \
  structural_modeling_agent/tests/test_proxy_workflow.py \
  structural_modeling_agent/tests/test_integration.py \
  structural_modeling_agent/modeling_reasoning/tests/test_reactive_site.py \
  enzyme_discovery_agent/tests/test_structural_proxy_planning.py
```

### Discovery handoff 3.0

The main controller accepts a validated schema-3.0 discovery handoff. Its
`candidates` are individually admitted; unreviewed neighbors and legacy records
without explicit eligibility stay in `unresolved_leads` and do not become
completion requirements. A valid handoff with no admitted candidates returns
`no_admitted_candidates` without calling a model. Existing schema-2.0 handoffs
are adapted conservatively to investigation leads; standalone 1.0 tasks remain
supported separately.

Source references in a 3.0 handoff are relative to the handoff's bundle root.
The controller verifies the expected bytes through no-follow path traversal and
copies valid source snapshots into its own `sources` directory. Missing or
changed snapshots are marked unavailable. `inspect_candidate` exposes source
availability and local copies; `read_candidate_section` paginates large evidence
or modeling-input sections. Discovery's modeling inputs and asserted source
interpretations require the same independent setup validation as any other task.
See `enzyme_discovery_agent/README.md` for the versioned fingerprint profile,
bundle layout, and the schema-2.0-only legacy resolver limitation.

An inline handoff sequence is unverified even when its digest matches. Before
rendering or submitting a schema-3.0 candidate, the main agent currently requires
an independently retrieved, fingerprinted UniProt record whose active primary
accession and sequence match. It rechecks that source; a conflicting sequence is
retained as a conflict and blocks progress. **Current limitation:** non-UniProt
identifiers, secondary accessions, isoforms, and fully offline identity sources
cannot yet satisfy this main-agent gate. They remain unresolved until an explicit
identity adapter is implemented; they are not silently substituted.

The standalone input builder requires a separate `--sequence-identity-index`
for 3.0. The index is a JSON object mapping each accession to `{ "path": ...,
"sha256": ... }`, where the path names a fingerprinted identity-record JSON.
Its primary accession and sequence must match both the FASTA and any sequence in
the handoff. The builder also requires scientific setup validation for 3.0.
This verifies record content and consistency; it does not authenticate the
origin of a user-supplied source. Legacy 2.0 builder behavior remains separate.
To use a reviewed 3.0 decision in the standalone builder, pass
`--reviewed-decisions MODELING_RUN/report.json`. The report must be bound to
the exact handoff task fingerprint; the builder rechecks the candidate's setup
and only reads declared setup sources under that modeling run's `sources` or
`artifacts` directories. Schema 3.0 does not bypass these checks with the legacy
exploratory override. The identity source and reviewed-report fingerprints are
recorded in the generated campaign manifest.

For a read-only overview or paginated lead inspection, without starting a model:

```bash
python -m structural_modeling_agent --inspect-handoff HANDOFF
python -m structural_modeling_agent --inspect-handoff HANDOFF --lead-id LEAD_ID --offset 0
```

The controller also exposes `inspect_investigation_lead`; inspection never
admits a lead. A run is pinned to its input snapshot. Currentness is `unknown`
unless the user supplies `--check-discovery-run TRUSTED_RUN`; the agent then
compares saved registry, target, ranking, sequence-cache and candidate-map
fingerprints and reports `unchanged_checked_artifacts` or `stale` only when the
supplied run's manifest identity and saved source state can be checked, without automatically following the handoff's
`source_run` path. A stale result is a warning about the pinned snapshot; it
does not mutate it. Resume without an explicit recheck marks the prior result
historical; supply the option again to compare current state.
Unchanged checked files do not establish overall currentness: evaluation inputs,
external evidence and coverage are not recomputed by this comparison.
