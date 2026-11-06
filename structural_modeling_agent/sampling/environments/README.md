# Model stacks

Installed on the shared NFS tree under `/nfs/lambda_stor_01/homes/rzhu/Tool/envs/`:

| Environment | Role | Numerical/hardware selection |
| --- | --- | --- |
| `amide-sampling-20261105` | CPU software validation and PyMBAR analysis | ASE 3.29.0, PyMBAR 4.0.3 |
| `amide-so3lr-20261105` | SO3LR Tauri | CUDA 12 JAX 0.10.2; independent of the UMA JAX stack |
| `amide-orb-20261105` | OrbMol-v2 and native TorchSim probe | PyTorch 2.8.0+cu126, Warp 1.19.0+cu12 |
| `amide-uma-20261105` | Ordinary UMA-S-1.2.1 | Inherits the existing `uma_enzyme_ts/env_py312` packages read-only; FAIR-Chem 2.21.0 |

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
