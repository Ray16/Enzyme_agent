# MD and umbrella-sampling pilot

The configured multiscale tools and their exact test scope are recorded in
[`toolchain_manifest.json`](toolchain_manifest.json) and
[`VALIDATION_20261106.md`](VALIDATION_20261106.md). The agent can read this inventory
with `inspect_processivity_toolchain`; changed or missing evidence is flagged.

This pilot asks whether catalytic poses remain plausible during motion and what
it costs to access those poses. It does not predict enzyme activity or replace
the frozen proximity-weighted attack-angle ranking. A pose-accessibility PMF is
not a chemical activation barrier.

## The agent chooses the scientific protocol

The structural-modeling agent exposes `inspect_sampling_setup_contract`,
`declare_sampling_setup`, and `inspect_sampling_setup`. Its evidence ledger
asks the agent to resolve the reaction state, assembly, conditions, participants,
protonation, metals, solvent/boundaries, potential, sampling/CVs, validation and
resources. These are investigation questions, not a case recipe. The code never
chooses a pH, histidine tautomer, metal count, cofactor setup or engine merely
because it appears in an example. Family-level decisions can be shared with
explicit exact-sequence and target applicability review.

Each decision retains its selected value, alternatives, assumptions, uncertainties
and source IDs; conditional choices require a discriminating test. Source
categories distinguish primary evidence and user conditions from computational
proposals and structural observations. Missing decisions and changed sources
remain pending. A complete declaration is only `ready_for_preparation`, never
proof of correct chemistry or automatic authorization to compute.

Scientific runs require `setup_declaration` (the agent's saved `decision.json`)
and `setup_declaration_sha256`. The preparation report carries the full candidate
`sequence`, exact requested `target`, and `agent_setup_sha256`. Execution rechecks
that ledger and its sources, then compares the actual pH, temperature, model
profile IDs, sampling parameters and resource budget with the agent's decisions.
An unsupported selected sampling method requires an execution adapter; it cannot
silently become the umbrella method implemented by this reference runner.

For processivity, `inspect_processivity_contract` and
`assess_processivity_network` connect transport, orientation, chemical extension
and committed escape using an agent-defined state graph and sourced physical
rates. Unknown rates remain pending. First committed release is absorbing for
the same-association processivity calculation, so release followed by rebinding
cannot inflate continuation probability. The rate-network solver cannot establish
that an enzyme is processive: synthetic tests are labeled software-only, and
sourced results remain conditional kinetic predictions. The physical state
definitions, chemistry, rate estimators and uncertainty still require validation.

The scientific objective is productive repeated extension before escape, not
maximum binding strength or a visually favorable tunnel. Straight, curved or
V-shaped paths are candidate-specific hypotheses. Select geometry from accessible
states and transport/chemistry evidence, preserving active-site function and
cofactor/product exchange. A tunnel or groove that retains a chain in a dead-end
pose is not a processivity success.

For execution, `conditions.choice` supplies `pH` and `temperature_K`,
`model_backend.choice.profile_ids` names calculator profiles, and
`sampling.choice` supplies `method` and the exact run settings in `parameters`.
`resources.choice` is copied exactly to `resource_budget`, including
`max_workers` and `max_wall_seconds_per_task`. The initial reference runner
implements `unbiased_md` and `umbrella_distance`; additional methods can be
investigated and recorded without being misrepresented as executable.

## Execution and tool choices

- ASE 3.29 provides the reference Langevin NVT integrator. A small conservative
  harmonic-distance wrapper adds `k/2 (r-r0)^2` in eV/Angstrom units; numerical
  gradient tests cover its force and periodic minimum-image convention.
- PyMBAR 4.0.3 handles bias removal and correlation analysis. Models are analyzed
  separately: their physical Hamiltonians do not cancel across model families.
- Persistent workers keep one model loaded and claim independent
  model/replica/window tasks using atomic NFS directories. Independent seeds and
  windows are the first level of parallelism; each worker owns one GPU and a
  disjoint 1–4 CPU set. The launcher never consumes every free GPU implicitly.
- OrbMol's native TorchSim adapter is included in the compatibility probe for
  evaluating batches within one GPU process. Batch size remains one for enzyme
  runs until a representative-system memory and numerical benchmark supports
  increasing it. No multiple-process sharing of a GPU is assumed.
- SO3LR's ASE path is a reference, not a claim of native GPU-MD throughput.
  Its native PySAGES integration currently requires the neighbor-list engine,
  refusing `engine: flash`. Benchmark that path separately before selecting it.

Native references: [SO3LR sampling](https://so3lr.github.io/advanced/enhanced-sampling/),
[Orb native adapters](https://github.com/orbital-materials/orb-models),
[TorchSim](https://github.com/TorchSim/torch-sim),
[PyMBAR](https://pymbar.readthedocs.io/).

## Preparation is a required input

Raw Boltz coordinates are insufficient. Provide one reviewed, hydrogen-complete
prepared structure in an ASE-readable format and a JSON preparation report. No
protonation, solvent, Mg count, reactive-atom mapping or ligand charge is guessed.
A preparation report declares:

```json
{
  "reviewed": true,
  "structure_sha256": "hash of exact prepared coordinate file",
  "sequence_sha256": "hash of exact enzyme sequence",
  "protein_lengths": [516],
  "components": [{"name": "each protein/ligand/ion species", "count": 1, "formal_charge": 0}],
  "protonation_evidence": "Reviewed states, pH, method and evidence location",
  "solvent_evidence": "Explicit solvent, boundary conditions, density and equilibration evidence",
  "metal_evidence": "Mechanism-specific metal stoichiometry and hydrated coordination",
  "mechanism_evidence": "Family-specific reaction and activation mode",
  "reactive_atom_mapping_evidence": "Mapping from ligand chemistry to named coordinate atoms"
}
```

The example values describe the schema, not a prepared AS9 system. Every
component's formal charge must contribute to the full system charge. Net charge
and spin must agree with electron parity. Binding equilibria, solvent preparation
and metal-shell quality require scientific review in addition to these checks.

The report also requires documented `assay_conditions` and `key_residue_review`:

```json
{
  "assay_conditions": {"pH": 7.5, "pH_source": "Replace with the actual assay evidence"},
  "key_residue_review": {
    "coverage_evidence": "How catalytic residues and nearby titratable networks were identified",
    "histidines": [{
      "residue_id": "A:123",
      "state": "HIE",
      "ND1": 1200,
      "NE2": 1204,
      "role": "Replace with the supported catalytic, binding or structural role",
      "rationale": "Explain the selected tautomer using the complete local environment",
      "evidence": ["pKa calculation, annotation, structure or independent reference"],
      "considered_states": ["HID", "HIE", "HIP"],
      "metal_bindings": []
    }]
  }
}
```

These are schema illustrations, not protonation assignments or an assay pH for
this project. Key histidines require explicit HID (ND1-H), HIE (NE2-H), or HIP
(both nitrogens protonated). Prepared coordinates must retain atom names through
the `atomnames` or `atomtypes` ASE array. The gate checks their nitrogen identities
and starting H positions. Declared metal bindings add `nitrogen`, `metal_index`
and `max_distance_A`; a protonated donor nitrogen or absent coordination blocks
the run. An empty histidine list requires a scientific coverage explanation.
Unusual states outside these three variants require an explicit extension and
validation rather than silently mapping to a standard state.

Use protein pKa predictors to propose assignments, then review catalytic
histidines, metal ligands, substrate-facing acids/bases and coupled waters with
the actual ligands and cofactors present. Apo predictions alone cannot establish
the bound active-site state. Keep ambiguous initial microstates as separate
prepared inputs, and use each exact input across all three models. Their
pH-dependent populations and protonation free energies require separate treatment;
raw total energies or standard MBAR pooling across different proton inventories
cannot select a preferred state. Proton motion under an MLIP is not constant-pH
sampling. The gate verifies bookkeeping and starting geometry, not pKa accuracy.

The pilot JSON has `schema_version: 1`, `purpose: "pose_accessibility"`, paths
and SHA256 values for `structure` and `preparation_report`, explicit `charge` and
`spin_multiplicity`, plus these simulation fields:

```json
{
  "temperature_K": 300,
  "timestep_fs": 0.5,
  "friction_per_fs": 0.01,
  "max_wall_seconds": 300,
  "equilibration_steps": 2000,
  "production_steps": 10000,
  "sample_interval": 20,
  "checkpoint_interval": 2000,
  "seeds": [17, 31, 47],
  "distance_atoms": [{"index": 0, "element": "O"}, {"index": 1, "element": "P"}],
  "angle_atoms": [{"index": 0, "element": "O"}, {"index": 1, "element": "P"}, {"index": 2, "element": "O"}],
  "umbrella": {"enabled": true, "centers_A": [3.0, 3.2, 3.4], "spring_eV_per_A2": 0.1},
  "bond_guards": [{"atoms": [0, 1], "min_A": 2.7, "max_A": 8.0}],
  "metal_sites": [{"index": 3, "element": "Mg", "donor_indices": [4, 5, 6, 7, 8, 9], "cutoff_A": 2.6}],
  "analysis": {"min_effective_samples_per_window": 50, "min_adjacent_overlap": 0.03, "bin_edges_A": [2.8, 3.0, 3.2, 3.4, 3.6]}
}
```

Indices are zero-based **in the prepared file**, checked against declared elements.
The illustration is not a scientifically selected CV or window schedule. Specify
reactant-state bond guards for relevant forming/breaking bonds; a breached guard
stops sampling rather than silently mixing chemical states. Distance-only bias
leaves orientation free, but slow angle, solvent or coordination modes can require
additional coordinates. The cutoff-based metal counts are diagnostics, not
definitions of chemical bonding. Equilibration and production durations above
are software-pilot settings, not convergence claims.
For an agent-selected ordinary-MD study, set `umbrella.enabled` to false and
provide empty centers with zero spring strength. That generates only unbiased
replicas and does not add an enhanced-sampling campaign.

Each entry in `models` declares a unique `id`, a factory from `backends.py`, its
`kwargs`, exact `source_revision`, `precision`, local checkpoint `artifacts`
(`path` and `sha256` for every weight/config file), and
`force_check_tolerance_eV_per_A`. Use absolute checkpoint paths. Model identity,
units and tolerances must be reviewed before submission. Separate numerical
profiles require separate model IDs and independent validation.

## Plan, run and distribute

Run from the parent `Enzyme_agent` directory, or set `PYTHONPATH` to that directory.
The isolated validation environment is
`/nfs/lambda_stor_01/homes/rzhu/Tool/envs/amide-sampling-20261105`.
Set the five native thread caps listed in `pilot.THREAD_VARS` to one. Bind the
launcher/coordinator to its own CPU set with `taskset`, disjoint from worker CPUs.

```bash
python -m structural_modeling_agent.sampling.pilot plan pilot.json --output /absolute/run
python -m structural_modeling_agent.sampling.fleet pilot.json assignments.json --output /absolute/run --dry-run
python -m structural_modeling_agent.sampling.fleet pilot.json assignments.json --output /absolute/run
```

An assignment file explicitly limits resources:

```json
{
  "max_workers": 2,
  "workers": [
    {"host": "lambda2", "gpu": 1, "cpus": [76, 77], "python": "/absolute/model/environment/bin/python", "model": "orbmol-v2-fp32", "max_tasks": 2, "min_available_ram_mib": 8192},
    {"host": "lambda2", "gpu": 2, "cpus": [78, 79], "python": "/absolute/model/environment/bin/python", "model": "orbmol-v2-fp32", "max_tasks": 2, "min_available_ram_mib": 8192}
  ]
}
```

These are example assignments, not reserved resources. The launcher validates
disjoint CPU/GPU sets, preflights each actual interpreter, checks aggregate CPU
capacity and shared GPU policy, then runs detached workers through `gpu_reserve`.
It writes per-worker logs and a launch manifest. Submission is not verification:
inspect actual worker processes, inherited affinity, GPU occupants and checkpoint
growth. The GPU runner rechecks ownership/contention during execution.

`max_tasks` bounds each worker's work. A partial task at the wall limit stops that
worker; rerun with the same inputs to continue. Completed outputs are checked
against checkpoints. Claims are never cleared by age; inspect host, PID and
process-start identity before manually archiving a stale claim. A changed code
revision or dependency stack cannot silently resume an old trajectory.

## Interpretation and efficiency gates

Each task includes an unbiased control or one umbrella window. State checkpoints
contain coordinates, momenta, RNG state and sampled CVs, allowing deterministic
continuation under the same numerical stack. Full-system coordinates are stored
at checkpoints, rather than every CV sample, to reduce NFS traffic. Samples retain
step, distance, angle, temperature, potential energy and metal coordination.

```bash
python -m structural_modeling_agent.sampling.pilot analyze pilot.json --output /absolute/run --model orbmol-v2-fp32
```

Analysis requires every requested umbrella window/replica to have matching complete
outputs, discards equilibration, estimates correlation using distance, angle and
coordination, subsamples, and checks adjacent-window overlap. Histograms use
MBAR-derived unbiased weights; raw biased hit counts are not ranking features.
The marginal distance PMF has **no radial Jacobian correction** and is not a
standard-state binding free energy.

The current PMF output is explicitly exploratory. Bootstrap uncertainty,
independent-replica agreement and split-time convergence review are still needed
before reporting a scientific free-energy difference. The output keeps chemical
barriers and activity predictions null. Hidden slow coordinates and incorrect
model chemistry cannot be repaired by collecting more biased frames.

For timing, construction, the first force call and force validation are separate
from sustained MD. Sustained timing includes monitoring and checkpoint I/O.
Compare the same prepared system, numerical profile, sampling budget and outputs.
Tiny-water compatibility timings cannot predict enzyme throughput. Benchmark
native batches on a representative prepared system and check forces against the
reference path before enabling production batching.

## Transport, chemistry and release tools

OpenMM 8.6.1 with the CUDA 12 plugin supplies the solvated classical MD engine.
PLUMED 2.10.2 and its locally built OpenMM adapter supply collective variables
and biases. WESTPA supplies weighted-ensemble transport/release sampling;
`westpa_bound` assigns disjoint local CPU workers, and `westpa_gpu_worker` requires
one reserved GPU for each ZeroMQ worker node. The sole master owns HDF5 writes.
The GPU worker monitors contention, and completed segments survive restart.
Agent-selected state boundaries, independent stochastic streams, convergence,
uncertainty and concentration conventions remain required for physical rates.
The tutorial's short segments and seed scheme are software fixtures, not a
production protocol.

CAVER and pyKVFinder supply complementary tunnel and open-pocket measurements.
Choose origins, probe sizes, path definitions and snapshots from the modeled
substrate and state. Static length, curvature and bottleneck radius do not
establish transport rates or favor a straight or V-shaped design by themselves.

ML/MM uses mlmm-toolkit 0.6.0's public custom ASE interface. The
`persistent_calculator` bridge keeps one model loaded in its validated environment,
while OpenMM evaluates the subtractive MM terms. Profiles require explicit
charge/multiplicity, precision, interpreter and hashed artifacts. The same bridge
supports ordinary UMA-S-1.2.1, OrbMol-v2 and SO3LR Tauri-M without combining their
dependency stacks. SO3LR reads `multiplicity`; UMA/Orb use `spin` as multiplicity.
Conflicting metadata is rejected. `toolchain_validation` checks forces on both
sides of covalent boundaries, a short MD integration and reaction-path forces.
Give each GPU worker one core/bridge instance. Reuse it for sequential image or
window evaluations; constructing a model for every image defeats this allocation.
The reference band uses ASE's explicit shared-calculator mode. Distribute
independent workers across separately reserved GPUs, and do not call a bridge
concurrently from multiple threads.

The validated hybrid benchmark is **finite, nonperiodic mechanical ONIOM**.
Native PySCF electrostatic embedding and the optional xTB correction have separate
interface tests. Those tests do not validate electrostatic embedding for an enzyme
reaction or full periodic QM/MM. Keep changing bonds and transferring protons
inside the high-level region, check boundary/region convergence, and compare ML
energetics with quantum references. Static path energies and QRRHO estimates are
not condensed-phase activation free energies.

Piece the stages together through chemically defined states and mapped atoms:
solvated access/retention/release ensembles, a representative reactive region,
validated chemical kinetics, then a newly prepared product/intermediate state for
continued MD. A reacted ligand cannot continue under its old fixed reactant
topology. Generate and review new parameters, protonation and solvent/cofactor
states; preserve atom correspondence and any exchanged species. These per-enzyme
handoffs require investigation and have not been demonstrated by the software
fixtures. The kinetic-network interface preserves missing rates as gaps.

## Multistate design and sampler selection

The design objective covers a common protein sequence across the required
reaction, cofactor-exchange and representative chain-length states, together
with access, retention and release. The agent chooses those states from the
mechanism. Each has its own full chemical graph, protonation/charge, cofactors
and atom mapping. A large growing ligand can require an open groove and flexible
regions; its full volume need not become a rigidly enclosed cavity.

RFD4's standard enzyme tutorial scaffolds a conditioned catalytic constellation.
The tutorial leaves some ligand coordinates flexible while preserving ligand
identity. Its YAML input list combines structures into one molecular context;
it is not an alternative-state ensemble. The published ligase demonstration uses
multistate AF3 filtering. Accordingly the current proposed handoff is generation
from a justified state, followed by shared-sequence evaluation/refinement across
the required states. Joint multistate conditioning is not claimed as an
implemented feature. Preserve original conditioning targets during refinement.
See the official [enzyme tutorial](https://rfd4-proteina.org/latest/examples/enzyme_design_tutorial.html),
[input specification](https://rfd4-proteina.org/latest/how-to/input_prep/yaml-json.html)
and [ligase demonstration](https://rfd4-proteina.org/).

WESTPA is the configured initial weighted-ensemble implementation for competing
productive motion and escape. Its [methods paper](https://doi.org/10.1021/acs.jctc.1c01154)
describes kinetic/pathway estimation, parallel trajectories and adaptive
resampling. Suitability depends on the actual states, progress descriptors,
segment duration, dynamics and convergence. The software benchmarks do not
establish an advantage on this enzyme. A representative pilot should compare
ordinary MD and an appropriate PLUMED free-energy calculation with WE, preserving
the Hamiltonian, starting ensembles, physical definitions and comparable budgets.
Compare distinct observables separately: a PMF is not a rate estimate.

Use effective transition counts, flux/rate stability, uncertainty, independent
replica/lineage support and total GPU-hours including startup and I/O. Repeated
descendants of one successful walker are correlated observations. If WE is
useful but its resampling is inefficient, compare a justified WESTPA resampler
or [wepy's REVO/WExplore alternatives](https://adicksonlab.github.io/wepy/_source/api.html).
wepy has been reviewed as an alternative; it is not installed or benchmarked in
this configuration pass. The agent retains sampler choice as an explicit
decision rather than embedding WESTPA in every scientific protocol.

## Measured worker overhead

The efficiency pass adds `efficiency` for bounded reference benchmarks and
`westpa_controller` for completion validation. The controller checks committed
HDF5 iterations, segment status, finite coordinates and the protocol's declared
probability total; a zero CLI exit code alone is insufficient. Set
`WESTPA_EXPECTED_TOTAL_PROBABILITY` explicitly for a compatible ensemble protocol.
Heartbeat tolerance must cover measured segment/startup time and scheduling
jitter; the one-second tutorial setting expired on longer benchmark segments.

The experimental `openmm_worker.PersistentOpenMMPropagator` reuses a context in
each GPU worker and computes the progress coordinate in-process. Separate,
recorded integrator/barostat streams belong to worker contexts; cloned walkers
receive ordinary portable states while the RNG streams continue. A shared atomic
seed registry prevents reuse on worker restart. Completed segment caches bind
parent state, profile, implementation, runtime and output hashes, so recovery
cannot silently accept changed artifacts. The reserved-node watchdog remains
active; redundant `nvidia-smi` checks are rate-limited to ten seconds.

This adapter accepts **software-performance fixtures only**, currently with one
distance coordinate and endpoint resampling. It has not been qualified for
enzyme first-passage kinetics, continuous boundary detection, multiple CVs or
ligand parameter correctness. Its measured speedup is a worker implementation
improvement; it does not establish WESTPA as the most efficient scientific
sampler. Results, input identities and limitations are summarized in
[`VALIDATION_20261106.md`](VALIDATION_20261106.md).

## Agent-owned preparation studies

A standalone task with `objective: "prepare_sampling"` enters a preparation
completion gate rather than the cofold completion gate. Optional
`preparation_instructions` and explicitly authorized `evidence_paths` stay in
context across turns. The agent uses `inspect_preparation_contract`,
`inspect_preparation_structure`, `declare_preparation_study` and
`inspect_preparation_study` to investigate and revise alternative preparations.
Each variant binds an exact sampling setup and records chemical state,
parameters, solvation, minimization, equilibration and pilot review. Missing
measurements, failed alternatives, changed sources/artifacts or changed baseline
model context prevent readiness. Conditional chemistry can be investigated;
there is no default protonation, metal count or force-field recipe.

`run_preparation_operator` executes an explicitly supplied Python script in an
explicit environment for at most 60 seconds on one CPU. Inputs are authorized
and hashed; expected output paths are relative to a fresh attempt directory.
The script reads `request.json` and writes those outputs in its working directory.
The tool retains source, logs, input/output identities and failure records,
limits native thread pools, separates controller/worker affinity, hides GPUs,
and terminates the process group at completion or timeout. It is a code-execution
tool, not a security sandbox. Use reserved external execution for larger work.
Successful execution is only `operator_complete_requires_review`; the agent must
inspect numerical evidence and explicitly assess scientific criteria afterward.

Preparation studies and operator histories survive checkpoints. The result index
links study artifacts; live inspection rechecks their dependencies. A study can
be `ready_for_sampling_review` only after every declared alternative has completed
its required stages. This is a traceability/agent-review status, not independent
proof of the chemistry or production convergence. Partial studies retain concrete
next actions. Separate proton inventories remain separate studies/variants; their
raw energies must not be used to infer pH-dependent populations.

Preparation context is limited to 24,000 serialized characters and uses a focused
prompt/tool set. `record_preparation_findings` preserves up to 6,000 characters of
scientific working notes with source hashes and a concrete next action; source
changes are flagged on reuse. Identical notes and unchanged declarations do not
count as progress. After two turns without a preparation-state change, the agent
must update findings/decisions or return an explicit inconclusive result before
more investigation. Full sources and inspection history remain on disk. This
limits repeated reading without treating incomplete evidence as resolved chemistry.
