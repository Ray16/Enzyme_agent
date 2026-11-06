# MD and umbrella-sampling pilot

This pilot asks whether catalytic poses remain plausible during motion and what
it costs to access those poses. It does not predict enzyme activity or replace
the frozen proximity-weighted attack-angle ranking. A pose-accessibility PMF is
not a chemical activation barrier.

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
  "umbrella": {"centers_A": [3.0, 3.2, 3.4], "spring_eV_per_A2": 0.1},
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
