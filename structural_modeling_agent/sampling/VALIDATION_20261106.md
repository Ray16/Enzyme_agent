# Sampling setup validation — 2026-11-06

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
