# Sampling setup validation — 2026-11-06

## Current review status

The essential tools are configured and their bounded software interfaces have
been exercised. The agent has a source-backed setup ledger and a read-only
`inspect_processivity_toolchain` tool, so scientific choices remain specific to
the enzyme, chemical state and question. This is readiness for reviewed pilots;
it does not establish a correct catalytic protonation state, amide reaction
barrier, physical processivity rate or processive enzyme.

The [machine-readable inventory](toolchain_manifest.json) pins source commits,
environment records, check scopes and SHA256 values. Compact reports are committed
in [validation_20261106](validation_20261106/); their full local runs remain under
`runs/amide_sampling_pilot_20261105/`. The inventory checks immutable report bytes,
not live hardware availability or scientific accuracy.

| Component | Completed validation |
| --- | --- |
| OpenMM 8.6.1 | CPU and V100 MD on the 4,010-atom official WESTPA NaCl/water fixture; exact checkpoint coordinate restoration and numerical replay |
| PLUMED 2.10.2/OpenMM adapter | Two native tests; analytical restraint energy/forces on CPU and V100, including 1-based atom selection and units |
| WESTPA | Two local CPU workers over three iterations, then two independently reserved V100 workers over four iterations; separate master, restart, finite trajectories and normalized weights |
| CAVER 3.02 + pyKVFinder 0.9.6 | Static tunnel length, bottleneck, curvature, cavity volume and depth; explicit example origin/probe scope retained |
| UMA-S-1.2.1, OrbMol-v2, SO3LR Tauri-M | Same finite ML/MM fixture, conservative force checks across two covalent boundaries, short MD integration and reaction-path force evaluation |
| Native PySCF/xTB embedding interfaces | 79 upstream tests passed, one skipped; the skipped xTB test then passed separately with its configured executable |
| Preparation | Acetamide AM1-BCC/GAFF2 → tleap → OpenMM; atom/total-charge checks, with the small charge correction explicitly recorded |
| Agent/pilot regression | Final full suite: 409 passed. Earlier targeted inventory/process-adapter and worker-lifecycle checks also passed |

The V100 hybrid comparison used 122 real atoms, 46 ML-region atoms and two link
hydrogens, FP64 high-level inference and explicit OpenMM MM. It checked selected
Cartesian derivatives on both sides of the boundaries. Three 0.05-fs NVE steps
check the integration interface; one three-image NEB force evaluation checks the
path interface. Neither is a stability or transition-state convergence study.

| High-level model | Maximum force error (eV/Å) | Three-step NVE energy drift (eV) |
| --- | ---: | ---: |
| Ordinary UMA-S-1.2.1 | 0.0007263 | 0.00000295 |
| OrbMol-v2 | 0.0007255 | 0.00000362 |
| SO3LR Tauri-M | 0.0007261 | 0.00000174 |

The shared force-error scale is consistent with the MM numerical path; these
values do not rank the ML models' chemical accuracy. The finite-difference gate
was 0.02 eV/Å. All models remained in separate validated environments, with one
loaded model per worker and a persistent ASE bridge to the public toolkit API.
Model profiles pin their artifacts and explicit charge, multiplicity and precision.
Conflicting metadata is rejected, including the different SO3LR spin convention.

The two-GPU WESTPA check used lambda4 GPUs 4/5, CPU sets 60–61/62–63, and a sole
HDF5-writing master on lambda2 CPU 76. Saved segment records confirm both physical
GPUs performed MD. These were 20-step software segments; no transport rate or
convergence claim follows. Production stochastic streams, state definitions,
sampling length and uncertainty require their own review.

Shutdown validation found lingering WESTPA ZMQ helpers. The reserved worker now
waits for the ZMQ communication thread, adopts orphaned descendants with a Linux
subreaper, and terminates/reaps its remaining children before releasing the
reservation. A subsequent GPU restart completed and an independent process check
found no remaining validation workers/helpers. The lifecycle test is included in
the final full suite.

## Material runtime findings

* OpenMM's default CUDA-12 extra selected NVRTC 12.9, which imported but failed
  during V100 execution. The tested MD stack pins NVRTC/runtime **12.4.127**.
* The original Orb stack needed the official CUDA-12 Warp wheel. A 7,690-atom
  isolated-protein Orb probe exceeded the 32-GB V100 memory budget; the validated
  hybrid region is much smaller.
* Ordinary UMA runs on a V100. On the isolated protein, FP32 failed the declared
  force tolerance (0.939 eV/Å); FP64 passed (0.0000576 eV/Å). FP64 took about
  10.42 seconds per steady force call and peaked at 29.62 GB allocated memory.
  These are isolated-protein numerical/resource probes, not solvated-enzyme MD.
* The optional hessian_ff path encountered a stale compilation lock. A dedicated
  serial prebuild succeeded using its serial bonded fallback; OpenMP bonded
  recipes failed compilation. The validated hybrid MD explicitly selects OpenMM
  MM. The optional analytical-Hessian binary loaded but was not numerically
  validated here, and its host-native build has CPU portability limits.

The environment/build details and exact pins are in
[environments/README.md](environments/README.md). No global modeling environment
was upgraded. The small acetamide charge normalization is retained with its raw
charges and rationale; it is not a universal correction policy.

## Scientific gates before an enzyme study

The validated hybrid reference is **finite, nonperiodic mechanical ONIOM**.
Native CPU electrostatic-embedding interfaces have separate tests; periodic
electrostatic QM/MM and GPU4PySCF execution are not validated by this work.
Changing bonds and transferring protons must remain inside a justified reactive
region, with quantum-reference, boundary and region-convergence checks.

For each enzyme the agent still needs to resolve functional assembly, reaction
state, ligand/oligomer parameters, key-residue/ligand protonation, metal hydration,
solvent and ionic conditions. Protein-only PROPKA and PyPka diagnostics for AS9
disagree at several sites and omit bound-ligand effects; the AS9 chemical setup
remains pending. The assay pH is confirmed, but a pH alone does not uniquely fix
active-site histidine tautomers.

The full cycle needs mapped chemical states and newly prepared product/intermediate
topologies between reactive calculations and continued classical MD. Its transport,
retention, chemistry and release rates need validated estimators, common conditions,
uncertainty and convergence. The new kinetic-network solver keeps unknown rates
null and first release absorbing, so rebinding cannot count as continuous growth.
No full-cycle enzyme trajectory or new molecular demonstration of processivity
has been produced. No shorter/wider tunnel designs were launched in this phase.

## Efficiency benchmark follow-up

All new raw outputs are grouped under
`runs/amide_sampling_pilot_20261105/efficiency/`, with `INDEX.json` and a nested
archive of initial attempts. Compact, hashed reports are in
[`validation_20261106/efficiency`](validation_20261106/efficiency/).
The main comparisons used two V100s with disjoint CPU sets and one coordinator;
comparison cases ran sequentially on that allocation. Independent workers ran
concurrently. The exact 4,010-atom reference Hamiltonian was checked across the
direct and subprocess paths.

| MD steps per segment | Original WESTPA subprocess, mean full segment (s) | Persistent worker, mean warm segment (s) | Warm propagation improvement |
| ---: | ---: | ---: | ---: |
| 100 | 3.031 | 0.167 | 18.2× |
| 1,000 | 3.525 | 0.554 | 6.36× |
| 4,000 | 5.781 | 1.849 | 3.13× |

These are measured propagation costs, including each implementation's monitoring
and output handling. They are **not whole-pipeline or scientific sampling
speedups**. One-time initialization, coordination, different realized walker
counts, resampling and failed attempts are recorded separately. Short runs and
different visited configurations limit the precision of timing comparisons.

The improvements came from keeping one OpenMM context per worker, moving CV
evaluation into the worker, and removing repeated process/import overhead and
redundant GPU-status queries. Three-iteration two-GPU tests verified context
reuse, finite trajectories, normalized weights and independent streams for
cloned initial conditions. A fourth iteration resumed successfully on available
H100s after the original V100 slots became occupied. H100 timing is excluded from
the V100 comparison. Reservation checks refused overlap, and final worker/helper
cleanup passed.

The paired direct-MD/PLUMED test reverses mode order across the two workers,
records distinct thermostat/barostat seeds, and separates monitoring time from
dynamics/reporting. A zero-strength restraint evaluates the PLUMED CV while an
explicit energy/force check confirms zero added potential. On this reference,
mean dynamics/reporting times for native versus PLUMED were 0.089/0.094 s at
100 steps, 0.482/0.555 s at 1,000, and 1.767/2.089 s at 4,000. Real biases and
larger systems require their own tests; these numbers do not measure PMF quality.

The matched hybrid test confirms **50 real model requests**, with a perturbed
ML-region atom preventing cache-only timing. All three retain their gradient
checks. Full process time includes imports, construction, reference-force checks,
MD, force benchmarking and cleanup.

| High-level model | Mean full hybrid force (ms) | Benchmark process wall (s) | Peak allocated/in-use device memory |
| --- | ---: | ---: | ---: |
| Ordinary UMA-S-1.2.1 FP64 | 94.8 | 48.84 | 2.39 GiB, Torch allocated |
| OrbMol-v2 FP64 | 44.4 | 31.65 | 0.387 GiB, Torch allocated |
| SO3LR Tauri-M FP64 | 10.6 | 19.25 | 0.106 GiB, JAX in-use |

These small-region costs favor SO3LR for throughput in this benchmark; they do
not establish chemical accuracy or justify ranking enzyme activity. The cost
breakdown supports reusing loaded models rather than reloading them for every
window/image. GPU allocator metrics differ between Torch and JAX.

The final project suite passed **428 tests**, with one optional WESTPA runtime
module skipped in the lightweight environment. The dedicated MD environment
passed all **9 runtime checks**, including seed collisions, changed-artifact cache
rejection, lifecycle cleanup and false-success detection. RFD4 generation,
multistate refolding, reaction reference accuracy and a representative enzyme
access/release efficiency comparison remain separate scientific work; prior
Aurora RFD4 qualification is retained in the inventory.

## Earlier ASE pilot baseline (historical)

The ordinary newest small base UMA checkpoint (`uma-s-1p2p1`) is configured and
has completed an energy/force compatibility probe on a Tesla V100-SXM2-32GB.
Its actual settings were FP32, TF32 disabled, general execution, compilation off,
and molecular task `omol`. This was ordinary UMA, without Nitro.

124 tests passed with 0 failures. The tests include finite-difference umbrella
forces, deterministic Langevin restart, atomic claim preservation, allocation
guards, contention handling and recovery of a known harmonic PMF. A separate
two-worker software run completed all eight tasks with disjoint worker CPU sets.

| Model | Force finite-difference maximum error (eV/Angstrom) | Evidence scope |
| --- | --- | --- |
| UMA-S-1.2.1 | 0.000114322 | Single water molecule, energy/forces only |
| OrbMol-v2 | 0.000324488 | Single water molecule, energy/forces only |
| SO3LR Tauri-M | 5.13117e-05 | Single water molecule, energy/forces only |

OrbMol's four-system native TorchSim batch matched its ASE energy to 2.38419e-07 eV
and forces to 7.15256e-07 eV/Angstrom.
The working Orb stack requires the official CUDA 12 Warp wheel; the initial
default CUDA 13 wheel failed on the cluster driver. Initial failures and the
corrected probe outputs are retained.

These checks establish software compatibility, not enzyme accuracy or useful
enzyme throughput. Tiny-water timings must not determine the production model
choice. No enzyme MD or production umbrella campaign has been submitted.

Next scientific prerequisites are a reviewed prepared enzyme/ligand/solvent/metal
system, same-system force/reference checks, timestep and stability checks, and
a representative-system runtime/memory comparison. Native PySAGES and full
TorchSim production MD still need integration validation. The current MBAR
profile remains exploratory until uncertainty and replica/split-time convergence
are assessed. No chemical barrier or activity prediction is produced.

Local run evidence:
`/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/structural_modeling_agent/runs/amide_sampling_pilot_20261105/`.

The exact ordinary UMA checkpoint and references are pinned in
[`profiles/uma-s-1p2p1-fp32.json`](profiles/uma-s-1p2p1-fp32.json).
Environment records and inheritance limits are in
[`environments/README.md`](environments/README.md).

The subsequent protonation gate requires the documented assay pH and a key-residue
review before scientific MD. Key histidines have explicit HID/HIE/HIP states,
traceable role/rationale/evidence, atom-name/index checks and starting N-H geometry
checks. Declared metal-donor nitrogens carrying a proton are rejected. The
21 targeted sampling/protonation tests pass, including a check that the scientific
input loader invokes this gate. These tests do not determine enzyme pKa values or
validate any particular enzyme's protonation assignment.
