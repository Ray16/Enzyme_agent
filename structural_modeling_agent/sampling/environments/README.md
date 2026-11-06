# Model stacks

Installed on the shared NFS tree under `/nfs/lambda_stor_01/homes/rzhu/Tool/envs/`:

| Environment | Role | Numerical/hardware selection |
| --- | --- | --- |
| `amide-sampling-20261105` | CPU software validation and PyMBAR analysis | ASE 3.29.0, PyMBAR 4.0.3 |
| `amide-so3lr-20261105` | SO3LR Tauri | CUDA 12 JAX 0.10.2; independent of the UMA JAX stack |
| `amide-orb-20261105` | OrbMol-v2 and native TorchSim probe | PyTorch 2.8.0+cu126, Warp 1.19.0+cu12 |
| `amide-uma-20261105` | Ordinary UMA-S-1.2.1 | Inherits the existing `uma_enzyme_ts/env_py312` packages read-only; FAIR-Chem 2.21.0 |
| `processivity-md-20261106` | Solvated MD, PLUMED, WESTPA, cavity and trajectory analysis | OpenMM 8.6.1/CUDA 12; PLUMED adapter built against this exact SDK |
| `processivity-qmmm-20261106` | mlmm-toolkit 0.6.0, PySCF reference and hybrid MM coupling | Torch 2.13.0+cu126, FAIR-Chem 2.25.0; model inference stays in the separate environments above |
| `processivity-plumed-20261106` | PLUMED kernel/headers and xTB | PLUMED 2.10.2 non-MPI, xTB 6.7.1; explicit conda package lock |

The ordinary UMA adapter loads the exact cached `uma-s-1p2p1.pt` checkpoint,
because the installed registry lists the older aliases. Its configuration is in
[`../profiles/uma-s-1p2p1-fp32.json`](../profiles/uma-s-1p2p1-fp32.json).
That profile declares the checkpoint and isolated-atom reference SHA256 values.
Charge zero and singlet multiplicity are for the water probe only; replace them
with the reviewed full enzyme-system values before an MD run.

UMA uses an explicit `InferenceSettings` object: IEEE FP32, TF32 off,
`execution_mode="general"`, compilation off, and one Torch CPU thread. It does
not rely on the meaning of `default` or a `high-precision` alias across versions.
If an installed version has a Nitro field, the adapter explicitly disables it.
The existing global UMA helper and environment were not modified.

The adapter also supports explicit FP64. The 7,690-atom isolated-protein
numerical test failed its force tolerance in FP32 and passed in FP64. Therefore
FP32 water compatibility is not a general precision recommendation. The
common 122-atom hybrid fixture uses FP64 for all three high-level models.

The ordinary UMA environment inherits packages rather than copying them.
Changes to its base environment therefore change the available stack; the pilot
records versions and refuses a resume across version changes. Preserve the
base environment or construct a fully independent environment before a long
production campaign.

The Orb stack needs the CUDA 12 compatibility build of Warp. The default
Warp 1.19 PyPI wheel uses CUDA 13 and failed on these V100 nodes. The working
wheel is pinned by its official release URL in `orb-v100.lock`. PyTorch is
explicitly the CUDA 12.6 build, whose compiled architecture list includes sm_70.
Do not replace either with a generic latest wheel during installation.

The lock files record the installed versions and VCS revisions. They are stack
records, not evidence of enzyme accuracy. Native PySAGES and full TorchSim MD
production integration have not been validated by the water energy/force probes.

## Processivity toolchain details

The `processivity-*.lock` files record the installed MD, hybrid and PLUMED stacks.
The hybrid source checkout is pinned in `../toolchain_manifest.json`. The
`[dft-cuda12,openmm,dev]` extras were installed with the official PyTorch CUDA 12.6
wheel. GPU4PySCF is installed, but its GPU execution has not been validated here;
the native PySCF embedding checks use CPU reference calculations.

OpenMM's generic CUDA 12 extra initially selected NVRTC 12.9, which failed on the
V100 driver with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The tested MD environment
pins both `nvidia-cuda-nvrtc-cu12` and `nvidia-cuda-runtime-cu12` to **12.4.127**.
Preserve those pins when rebuilding the environment. Successful import alone
does not test the runtime compiler or CUDA forces.

The PLUMED adapter was built from the official `openmm/openmm-plumed` source
against the installed OpenMM 8.6.1 wheel's `OpenMM.libs` headers/libraries, using
one build thread. Its extension has explicit library rpaths. Set
`PLUMED_KERNEL=/nfs/lambda_stor_01/homes/rzhu/Tool/envs/processivity-plumed-20261106/lib/libplumedKernel.so`
when using PLUMED. Source commits and benchmark evidence are in the inventory.
Independent MD/WE workers require the five native thread caps set to one,
`OPENMM_CPU_THREADS=1`, disjoint CPU bindings and a separate coordinator CPU.
Every GPU worker runs through `gpu_reserve`; use local device index zero inside
the reservation. Never set `CUDA_VISIBLE_DEVICES` manually.

For model/MLMM interoperability set `PYTHONPATH` to the parent `Enzyme_agent`
directory and `SAMPLING_HIGH_LEVEL_PROFILE` to the exact reviewed model-process
JSON. The toolkit's direct Python API needs **backend="custom"** as well as
`calc_file`; its CLI resolves that selector itself. Use explicit `mm_backend`,
`mm_device`, `mm_threads` and `xtb_ncores` rather than the upstream thread defaults.
The validated hybrid benchmark selects `mm_backend="openmm"`, CPU MM, one thread,
and mechanical ONIOM. It is a nonperiodic reference model, not a periodic PME
hybrid implementation.

The optional hessian_ff extension was prebuilt serially in a dedicated build
cache after a stale `/tmp` compilation lock blocked workers. Its OpenMP bonded
recipes failed compilation; the portable serial recipe loaded successfully.
The analytical-Hessian recipe also loaded, but its numerical behavior was not
tested in this pass and its host-native binary must not be reused on an
incompatible CPU. Neither native extension is selected by the validated hybrid
MD path. Preserve the explicit OpenMM selection rather than silently changing
the MM Hamiltonian or compiling in every worker.

AmberTools and PDBFixer remain available in the existing
`/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mlmm-toolkit` environment.
Use its absolute binaries and explicit `AMBERHOME` when preparing inputs; avoid
mixing its Python package stack into the new simulation environments. A complete
neutral acetamide AM1-BCC/GAFF2 → tleap → OpenMM test is recorded. Its SQM charge
rounding required a documented 0.000222333 e per-atom normalization to preserve
the declared total charge. This is a benchmark-specific correction, not a
general ligand-charge policy or validation of ATP/Mg/product parameters.
