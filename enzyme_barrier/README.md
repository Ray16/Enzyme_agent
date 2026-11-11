# Enzyme reaction barriers

This package separates chemical setup, path exploration, reference sampling,
target-level electronic structure, and validation. **No validated AS9 or GlnA3
activation free energy is available yet.** The Nov-10 campaign is in
`runs/barrier_maturation_20261110`; earlier results remain unchanged.

## Implemented operators

| Operator | Purpose | Scientific scope |
|---|---|---|
| `prepare` | Amber/GAFF2 cofold preparation with explicit C4 installation | Input preparation; protonation/assembly still require review |
| `ligands` / `components` modules | Named molecular graphs, explicit charges, SDF export, and checked multi-ligand Amber setup | Keeps source chemistry and proton inventory; graph/pose construction is a hypothesis |
| `assembly` | Exact target subunit copies fitted to an experimental assembly | Structural hypothesis, not measured oligomeric state |
| `md` | Independently seeded, unrestrained classical pilots with checkpoints | Pose, hydration and metal-shell investigation |
| `qmmm snapshot` | Fixed, hashed QM/MM partition with complete Mg shells | Same partition must be retained along a path |
| `qmmm smoke` | CPU/GPU energy–force finite differences | Implementation validation on a small system |
| `qmmm evaluate` | Embedded DFT/MM energies and forces | Electronic structure at stated geometries; not a free energy |
| `sander_backend` | Native electrostatically embedded periodic AM1/D or DFTB3/MM | AM1/D real-enzyme force checks pass; DFTB3 GlnA3 SCC/initialization remains unresolved |
| `path` | Alternative constrained reference-potential paths, with mobile MM atoms | Input generation; actual optimization convergence is recorded |
| `stationary` module | Chemistry-based endpoint identification, unrestrained endpoint relaxation, dimer saddle refinement | A converged dimer candidate still needs Hessian index and connectivity checks |
| `qmmm geometry-diagnostics` | Hashed reference geometries evaluated with DFT/MM energies and forces | Does not subtract mismatched periodic/finite Hamiltonians or claim a barrier |
| `umbrella` | Conservative bond/proton CVs and restartable sampling chunks | Reference ensembles; never a final DFT barrier by themselves |
| `analyze` | OPES reweighting with all declared biases and sampling checks | Projected PMF estimates with explicit failures |
| `free_energy` Python API | MBAR overlap and sparse target-level reweighting | Requires decorrelated data and known inclusion probabilities |
| `validation` Python API | Check the evidence needed to report a barrier | Fails closed on missing chemistry, ensembles or TS evidence |
| `worker` | One reserved GPU, disjoint CPUs, bounded tasks and durable ownership | Launch/completion is distinct from scientific validation |

The legacy `carve`, `scan`, and `run` operators are retained for compatibility.
Their isolated-cluster gaps are explicitly exploratory, including the optional
qRRHO correction. Do not use them as an enzyme activation-free-energy ranking.
Keep historical vacuum results on a relative-only reporting basis. A matched
enzyme/substrate pair may reduce systematic errors, but cancellation is a
hypothesis to test with embedding, not an accuracy certificate. The present
GlnA3 native glutamate model is -1 and glutarate is -2; their common-protein QM
regions have charges -4 and -5 respectively. Check historical charge assignments
before transferring any claim of equal-charge error cancellation. Cross-enzyme
comparisons require a consistent embedding, electronic-structure protocol,
converged sampling and explicitly defined bound-state references. Different
implicit-solvent responses demonstrate sensitivity, not calibrated corrections.

## Hamiltonians

The DFT/MM backend uses an additive expression:

```
E = E_DFT(QM + link H; redistributed MM point charges)
  + E_MM(MM/MM Coulomb + retained LJ/C4 + remaining bonded terms)
```

QM-internal classical terms and QM classical Coulomb interactions (including
exceptions) are removed. MM charges polarize the electronic density. Forces on
QM atoms, MM point charges and both link-atom parents are included. Water-pair
finite differences pass on CPU and GPU; the GPU wB97M-V/def2-SVP maximum tested
error was 2.8e-6 eV/Å. Larger partitions and boundary cases need further checks.

The present model has a **finite, fixed embedding sphere**, nonperiodic
reaction-field MM interactions, and a fixed membership mobile region. It does
not include periodic QM electrostatics. Check embedding radius, QM selection,
mobile-region size and boundary-charge treatment before interpreting barriers.
Sidechains are cut at CA–CB; ambiguous histidine, proline, disulfide and backbone
metal-ligand boundaries fail for review. Atoms and proton inventory are fixed
within each pathway. Small library charge-rounding residuals are recorded.

The candidate fast reference potential is:

```
E_ref = E_UMA(vacuum) + E_GFN2(embedded) - E_GFN2(vacuum) + E_MM_remaining
```

Both GFN2 calculations run serially within the reserved CPU pool. UMA uses
explicit IEEE FP32, general execution, no TF32 and no compilation. The installed
fairchem `default` now selects a different numerical profile, so it is not used.
The GFN2 response is an approximation. An initial diagnostic on legacy AS9 geometries
differed from DFT/SVP by about 13 kcal/mol. Those geometries are not a validated path,
and the comparison must be repeated on cell-verified ensembles. This potential is
**not validated as a quantitative barrier potential**.
It can generate exploratory ensembles, followed by DFT correction only when
target/reference overlap is adequate. Poor overlap requires bridging potentials,
additional target sampling, or a trained correction; do not hide it in an error bar.

## Run and resume

Use the existing managed `processivity-qmmm-20261106` environment for DFT/UMA and
`processivity-md-20261106` for CUDA classical MD. Do not silently replace these
with a system Python. Environment inventories and preflights belong in the run
directory. The quantum environment's MM backend currently uses CPU; its DFT
backend uses the reserved GPU. The classical environment provides CUDA OpenMM.

The installed Amber environment also supplies native `sander`, `sqm`, and 3OB
parameters. The native adapter uses a separate Amber Python process, explicit
method settings and positive electronic-convergence evidence on every accepted force call.
AM1/D uses Amber's AM1/d-PhoT H/O/P and AM1/d Mg parameters. Actual AS9, GlnA3
target and native-control force derivatives pass. This validates numerical
derivatives, not the method's accuracy for their activation barriers. AM1/D is
the current reference path/saddle potential; DFT benchmarking remains necessary.
The optional `dftb_3rd_order` field selects a diagonal DFTB2 correction/file; it is
not the switch for full DFTB3. The AS9 real-enzyme derivative check passes at the
nucleophile, MM link parent and Mg (maximum error about 1.3e-5 eV/Å). GlnA3 solver
stability and initialization remain unresolved. Do not combine a periodic SANDER
reference with nonperiodic DFT/MM target energies by naive subtraction: the MM
and embedding decomposition must be matched and verified first.

Use full-precision `final.npz` or a trajectory frame with its own cell. Earlier
PDB writers retained the initial topology cell after NPT; those headers are not
authoritative. New writers update the cell explicitly, and coordinate loading
does not infer bonds from OpenMM's extended PDB atom serials.

```bash
python -m enzyme_barrier qmmm snapshot --prmtop reviewed.prmtop \
  --coords equilibrated.pdb --spec reaction.json --out snapshot
python -m enzyme_barrier qmmm evaluate --snapshot snapshot \
  --out embedded.json --basis def2-tzvpd --platform CPU
python -m enzyme_barrier.umbrella reviewed_window.json
```

GPU commands must run through `gpu_reserve run`, with a preflight, explicit
native thread caps, a disjoint `taskset`, and a manifest. See the Nov-10 manifests
for actual commands. Any resident foreign GPU process stops our affected worker.
No age-only claim reclaim or force release is allowed. Check manifest identity,
PID start time, logs, actual outputs and scientific status when resuming.

Umbrella checkpoints save positions, velocities, RNG state, completed chunks,
configuration identity and producer hashes. Resume the identical configuration;
changed chemistry, model or implementation requires a versioned window. Orphaned
chunks are archived. Classical pilots retain OpenMM checkpoints, but automatic
in-place classical MD continuation is not implemented; use a reviewed restart
operator and preserve trajectory segments.

## Reporting and validation

Report these quantities separately:

1. Constrained electronic-energy gap, static saddle barrier, or projected PMF.
2. Activation free energy conditional on the specified bound state/microstate.
3. Reactive-pose population and the cost of amine activation at the assay pH.
4. Occupancy-weighted kinetics if multiple Mg/protonation states matter.

Use integrated basin populations for reaction free energies, not minimum-bin
differences. Remove every bias that is intended to be removed; no reweighting
recovers unsampled states. MBAR assumes a common underlying Hamiltonian across
umbrellas. Sparse DFT subsets require known sampling inclusion probabilities.
Assess overlap and target ESS by window and critical region, not only globally.

At least three independent equilibrated starting structures, block stability,
replicate agreement, chemical connectivity, and committor evidence are required
before reporting a validated PMF barrier. A static one-imaginary-mode saddle and
a finite-temperature transition-state ensemble are different checks. High barriers
or multiple maxima are not rejected solely for being inconvenient. Conditional
2-Mg/3-Mg barriers can be compared as conditional quantities; observable rates
also require Mg binding thermodynamics and solution activity.

GlnA3 aminolysis retains **cadaverine**, AS9 retains **1,6-hexanediamine**. The saved
GlnA3 step-2 cofold uses mono-protonated cadaverine (one neutral reactive N).
That is a conditional microstate; it does not include the pH-dependent cost of
creating that reactive amine. Mapped aminolysis systems are now parameterized,
including ADP3- explicitly and an L-glutamyl-phosphate native-control pose.
The primary pH-8 setup hypothesis uses diprotonated cadaverine/HMDA and therefore
requires two N-H transfers to reach an amide NH. Its classical pilots have
unrestrained production; they do not validate this protonation hypothesis.
AS9's supplied aminolysis pose has released PPi: this conditional intermediate
must not be combined with activation energies without release thermodynamics.
Competing acceptors and relay pathways still require chemical review before
reactive production. A CE structural alignment now
matches the two existing GlnA3 Mg positions to 2BVC within 0.86 Å and supplies a
third-site hypothesis. Its initial ATP overlap was explicitly recorded and had
to resolve during minimization before pilot dynamics. Reactant occupancy and
stability remain to be established. Native controls supplement requested substrates.

Distance CVs retain their original list syntax. A hydrogen-permutation-invariant
proton coordinate can instead sum donor and acceptor coordination terms:

```json
{"kind":"coordination","terms":[[0,2,1],[0,3,1],[1,2,-1],[1,3,-1]],"r0_A":1.25,"power":12}
```

Here 0/1 are donor/acceptor and 2/3 are the entire declared exchangeable-H group.
The smooth switch is `1/(1+(r/r0)^power)`. Such CVs are dimensionless: use
`springs_kcal_per_cv2`, with one center/spring per CV. An atom inventory, fixed
QM partition and chemically justified acceptors remain required. Never compare
raw energies across different hydrogen counts as if they shared a reference.

## Checks

```bash
python -m pytest enzyme_barrier/tests -q
```

Tests include analytic harmonic-distribution MBAR recovery, disconnected windows,
inclusion-probability corrections, QM/MM force derivatives, link projection,
proton-CV gradients, rejection of unconverged profiles, and exact chunk restart.
Passing these tests establishes implementation properties, not enzyme accuracy.

References: [PySCF electrostatic embedding](https://pyscf.org/user/qmmm.html),
[ML embedding](https://doi.org/10.1021/acs.jctc.2c00914),
[dual-level QM/MM correction](https://doi.org/10.1039/D2CP02820D),
[GlnA3 study](https://doi.org/10.3389/fmicb.2017.00726),
[4HPP assembly](https://www.rcsb.org/structure/4HPP),
[OPES reweighting](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s__m_e_t_a_d.html).
