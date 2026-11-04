# Enzyme design agent

Design, not discovery. The discovery agent searches for a protein that already
does the chemistry; this one decides what to **build** and builds it.

The current CphA2 redesign target is glutarate/cadaverine preformed-unit extension,
corresponding to nylon-5,5 chemistry:
[campaigns/cpha2_nylon55_20261102](campaigns/cpha2_nylon55_20261102/README.md).
Earlier nylon-6,6 processivity work remains in
[campaigns/nylon66_processive_20261023](campaigns/nylon66_processive_20261023/README.md),
and its predecessors are under [campaigns/archive](campaigns/README.md).

It is an agent rather than a pipeline because the first real decision of every
campaign — generate a scaffold, or redesign one that already binds — depends on
the chemistry in front of it, and so do the inputs that decision implies (which
motif, which positions, which conditions). Hard-coding that choice would make
the tools a script and the agent unnecessary.

```
enzyme-design-agent "<target chemistry>" [--scaffold holo.pdb] [--motif notes]
enzyme-design-agent --list-tools
```

## The strategies

Two families — *generative* (build a scaffold) and *redesign* (edit one that
binds) — and the generative family offers backends that differ in what they
require you to state up front.

| | generative · motif | generative · ligand | generative · reaction | redesign |
|---|---|---|---|---|
| tool | `design_de_novo_enzyme` | `design_enzyme_with_disco` | `design_enzyme_with_genzyme` | `scan_design_pocket` → `build_saturation_library` |
| backend | AlphaProtein Novo | DISCO | GENzyme | LigandMPNN scan |
| assumes | you can state the catalytic motif geometrically | you can name the ligand(s) the site must hold | you can specify the intended transformation | a holo structure exists in a pose you believe |
| input | a 3D arrangement of catalytic sidechain atoms + ligand (TS or covalent intermediate) | the ligand(s) as SMILES / CCD / SDF — no catalytic residues fixed | substrate and product SMILES, chain length and sample count | a scaffold + its bound complex |
| cost | GPU-days | GPU-hours per length×seed | scales with chain length and candidate count | seconds to build, GPU-weeks to fold |
| finds | a backbone that holds the geometry | a sequence+backbone whose pocket complements the ligand | reaction-conditioned pockets, protein backbones and sequences | which single substitutions shift the chemistry |

The generative backends are complementary: motif-scaffolding when you *can*
state the catalytic geometry, ligand-conditioned co-design when the chemistry is
known but a trustworthy geometry is not, and reaction-conditioned generation when
the substrate-to-product transformation can be specified. Neither family is the default. The
prompt makes the agent state its choice and its reason before calling anything,
and the run manifest records which it took.

## Layout

```
agent.py              the loop: choose a strategy, specify it, run it, report
prompts/sections.py   the standing instructions, incl. how to choose
bootstrap.py          tool registration; design tools only
structure.py          residue-level PDB parsing (numbering kept, unlike parse_pose)
saturation/pocket.py    complex-centred pocket definition
saturation/variants.py  saturation library + the merge contract
saturation/results.py   campaign bookkeeping; metric delegated, see below
tools/alphaprotein_novo.py  generative design via AlphaProtein Novo (motif-scaffolding)  [GPU]
tools/disco_design.py       generative design via DISCO (ligand-conditioned, motif-free)  [GPU]
tools/genzyme_design.py     registered GENzyme readiness, generation and status tools    [CPU wrapper]
tools/_genzyme_worker.py    isolated reaction-conditioned GENzyme generation             [GPU]
tools/rfd4_design.py        registered RFD4-Proteina readiness, generation and status      [CPU wrapper]
tools/_rfd4_worker.py       isolated, reservation-managed RFD4-Proteina generation        [GPU]
tools/saturation_scan.py    the two redesign tools                                        [CPU]
```

Infrastructure (`config`, `tool_registry`, `llm_client`, run-dir conventions,
the event log) is reused from `enzyme_discovery_agent` rather than forked, so
both agents share one backend chain and one run layout.

## GENzyme installation and invocation

`check_genzyme_design_readiness` checks the external checkout, dedicated conda
interpreter, checkpoint, cached ESM3 assets and local GPU availability.
`design_enzyme_with_genzyme` validates the request in that interpreter and starts
one bounded, asynchronous generation job through `gpu_reserve run`, with explicit
CPU affinity and thread limits. `get_genzyme_design_status` verifies FASTA and
structure outputs before reporting completion. Independent cofolding and catalytic
geometry evaluation follow generation; confidence values are diagnostics.

Defaults use `external/GENzyme` and the `genzyme` conda environment at
`/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/genzyme`. Override installation paths
with `ENZYME_AGENT_GENZYME_DIR`, `ENZYME_AGENT_GENZYME_PYTHON`,
`ENZYME_AGENT_GENZYME_CHECKPOINT`, `ENZYME_AGENT_GENZYME_GPU_RESERVE` and
`ENZYME_AGENT_GENZYME_HF_HOME`. Heavy model imports stay in the worker. The wrapper
exposes upstream de novo generation; scaffold-preserving experiments use their
explicitly labeled campaign adapter.

## RFD4-Proteina installation and invocation

RFD4-Proteina adds all-atom motif/ligand-conditioned sequence–structure co-design
from explicitly conditioned inputs. The checkout is
`/nfs/lambda_stor_01/homes/rzhu/Tool/RFD4-Proteina`, with an isolated Python 3.12
environment in `Tool/envs/rfd4-proteina` (also linked as the checkout's `.venv`).
It uses PyTorch 2.14.1/cu126, upstream `all-cuda12` dependencies and a cached,
revision-pinned flow/autoencoder bundle. Exact installed versions are recorded
in the checkout's `requirements-installed.txt`.

`check_rfd4_design_readiness` checks the configured worker host, packages, model
cache, compatible GPU hardware, free GPUs and CPU/RAM capacity.
`design_enzyme_with_rfd4(input_path, num_replicates=1, seed=0,
recycle_schedule="midpeak")` accepts an annotated CIF (including gzip/zstd) or a
YAML/JSON condition spec. It launches one asynchronous job through
`gpu_reserve run`, pins two CPU slots, caps native thread pools, and uses batch
size one with no data-loader subprocesses. Relative structure paths in specs
resolve against the spec directory. Use `front_loaded` for protein/DNA binders;
`midpeak` is the medium-budget enzyme/small-molecule schedule.

The default worker host is `lambda13` (H100); V100 GPUs are rejected. Override
paths/host using `ENZYME_AGENT_RFD4_DIR`, `ENZYME_AGENT_RFD4_PYTHON`,
`ENZYME_AGENT_RFD4_HOST`, `ENZYME_AGENT_RFD4_HF_HOME` and
`ENZYME_AGENT_RFD4_GPU_RESERVE`. Shared NFS paths and noninteractive SSH access
are required for remote execution. No shell startup files are changed.

### V100 compatibility profile

The registered RFD4 worker above targets H100 and rejects V100. A separate
FP32, eager compatibility profile supports V100 runs through the shared GPU
reservation system. Its reusable batch wrapper, queue tools, and configurable
attack-geometry reports are in [`tools/rfd4_campaign`](tools/rfd4_campaign/README.md).
The installed [`rfd4-proteina-v100` skill](/homes/rzhu/.agents/skills/rfd4-proteina-v100/SKILL.md)
records the runtime, reservation, preflight, and verification procedure. Follow
project provider preferences and the `lambda-fleet` skill when submitting a fleet.

On 2026-10-28, the compatibility profile generated three 516-residue AS9
structures simultaneously on a 32 GB V100 with batch size 3. All three passed
the campaign's technical checks, and no out-of-memory error occurred. The
model entrypoint took **715.4 seconds total (238.5 seconds per structure)**;
peak PyTorch-allocated GPU memory was **16.84 GiB**, while `nvidia-smi` showed
about **25.9 GB** resident during generation. For the same input and profile,
single runs at batch sizes 1 and 2 took 272.2 and 484.5 seconds, respectively
(272.2 and 242.2 seconds per structure). These are single-run throughput
measurements, not a guarantee for other input lengths or conditions. Test the
actual input and allow GPU-memory headroom before increasing the batch size.
Technical QC and conditioned geometry do not establish catalytic activity.

For the same input, seeds, checkpoint and FP32 sampling settings, an Aurora
Intel Data Center GPU Max 1550 XPU completed batch 1 in **193.3 seconds** and
batch 3 in **456.4 seconds total (152.1 seconds per structure)**. That batch-3
entrypoint throughput was about **1.57×** the V100 result above. These are
single runs; they exclude queue wait and initial Python/module imports. Aurora
used a different PyTorch build and logged an XPU-to-CPU fallback for an
alignment operation. All four Aurora output structures passed basic finite
coordinate and backbone checks. The complete benchmark records are in
[`runs/rfd4_benchmarks/aurora_as9_activation_20261028`](runs/rfd4_benchmarks/aurora_as9_activation_20261028/summary.json).

A subsequent batch-8 test on the same input failed with a Level Zero
`UR_RESULT_ERROR_OUT_OF_RESOURCES` before producing structures. At failure,
PyTorch reported **42.17 GiB peak allocated** and **63.38 GiB peak reserved**
on a device reporting 63.98 GiB total. These allocator counters are not a
measurement of total device utilization, but the high reserved value is
consistent with memory pressure. The failure record is in
[`runs/rfd4_benchmarks/aurora_as9_batch8_20261028`](runs/rfd4_benchmarks/aurora_as9_batch8_20261028/summary.json).

A subsequent **batch-6 test succeeded**, producing six structures that passed
basic coordinate and backbone checks. Peak PyTorch memory was **32.04 GiB
allocated and 47.95 GiB reserved**, on the same 63.98 GiB device. Generation
entrypoint time was **947.7 seconds total (157.9 seconds per structure)**,
compared with 152.1 seconds per structure for batch 3; the larger batch did not
improve throughput in these single runs. Batch 6 is the largest successful
batch tested for this input; batches 4, 5 and 7 remain untested. Records are in
[`runs/rfd4_benchmarks/aurora_as9_batch6_20261028`](runs/rfd4_benchmarks/aurora_as9_batch6_20261028/summary.json).

**User default:** use `batch_size=6` when submitting RFD4-Proteina generation
on Aurora. Keep the total requested replicate count separate from batch size.
The capacity result above applies to the tested 516-residue input and FP32
profile; check capacity for materially different inputs.

`get_rfd4_design_status(job_dir)` verifies the expected generated CIF count,
parseable structures, finite coordinates, protein backbones and file hashes.
Requests, effective Hydra config, code/model revisions, input identity, resource
assignments and logs are saved with each job. The tool uses eager native
inference with scoring disabled; partial diffusion uses the upstream CLI, while
refolding, reward-guided search and catalytic evaluation require separately
configured workflows. Generation does not
establish catalytic activity.

From the `Enzyme_agent` directory:

```python
from enzyme_design_agent.tools.rfd4_design import (
    check_rfd4_design_readiness, design_enzyme_with_rfd4, get_rfd4_design_status,
)

print(check_rfd4_design_readiness())
job = design_enzyme_with_rfd4("/shared/path/design.yaml", seed=42)
print(job)
# print(get_rfd4_design_status(job["data"]["job_dir"]))
```

## Three decisions worth knowing about

**The pocket is centred on the whole complex.** Substrate or intermediate *plus*
every cofactor and catalytic metal, taken together — not one ligand at a time.
A mutable set defined as "first shell of the substrate" cannot contain a
cofactor-coordinating residue, so a scan built that way cannot find one, and
the result ("only the residues touching both ligands matter") restates the
selection rule rather than reporting a finding.

Components join the site by *connectivity* from an anchor, so a metal parked on
a surface patch tens of angstroms away does not import a shell defined by a
misplaced atom. Excluded components are reported with their distance, never
dropped silently — a cofactor excluded as off-site means the pose is wrong, not
that the cofactor is irrelevant.

**Saturation includes the positions believed essential.** They are expected to
be dead, which is exactly why they are built: a geometric or confidence score
with no measured negatives is uncalibrated, and a catalytic knockout measured
at the same sample count as everything else is the cleanest internal negative
control available. Constraints are an `annotation` column, never an exclusion.
Masking them means the data never exists.

**The metric is not owned here.** `saturation/results.py` does bookkeeping —
discovery, `(variant, mode, sample)` keys recovered from filenames, dedup
across layouts, long format as the only stored truth, `n_models` on every
aggregate. What a variant's structures *mean* comes from
`structural_modeling_agent.proxies`, which carries the proxy catalogue and the
statistics that decide whether a proxy earned belief. The metric evolves as a
campaign runs; swapping one must not require editing this package.

Raw predictor confidences are read through as provenance, not as the metric.
There is deliberately no default chain pair: which chain is the substrate is
campaign knowledge.

## Tests

```
python -m pytest enzyme_design_agent/tests -q
```

Offline and GPU-free: a synthetic holo structure (`tests/_synthetic.py`) poses
the cofactor-only position and the stray ion, and the agent loop runs against a
scripted LLM.

## Reproducible discovery-to-design campaigns

`prepare_saturation_campaign` is registered in both agents. It accepts a
`saturation-1` specification as `spec_json` or `spec_path` and an `output_dir`.
Candidate selection remains an explicit evidence-based decision: the tool does
not substitute structural rankings for discovery evidence or guess chemistry
from a protein identifier.

Each candidate supplies `id`, `sequence`, `structure_path`, `protein_chain`,
`components`, `anchor`, `selection_reason`, `assembly_status`, and `boltz_input`.
The top-level specification supplies `selection_evidence`, separate `metrics`,
and `n_samples`. For interface-only design, also provide `ensemble_paths` and
`interface` with two atom selectors (`fragment_a`, `fragment_b`, each containing
`chain` and optional `atom_names`) and `min_contact_poses`. `cutoff_A` defaults
to 5. A residue must contact both groups in the same pose; the output records
support counts across the ensemble. No fragment identities are built into the
selector. Required components must be connected in the reference structure.
The current builder accepts a single protein chain and fails explicitly on
unsupported multi-chain inputs.

Preparation verifies sequence numbering, writes variant FASTA/TSV files, includes
WT controls, and fingerprints the source structures. It does not launch jobs.
`python -m enzyme_design_agent.saturation.msa MANIFEST` obtains fresh alignments
in batches of at most 101, with sequence and batch identities checked on resume.
Ready inputs are written atomically so folding can start before all MSAs finish.

For Lambda execution, `python -m enzyme_design_agent.saturation.fleet CONFIG`
uses explicit host/preflight settings and only `gpu_reserve run`. It allocates
disjoint CPU sets, caps native thread pools, preserves partial attempts, bounds
retries, and checks the requested model slots. Workers additionally verify the
protein sequence, component heavy-atom composition, and configured reactive atom
selectors before recording completion. A driver, claim or dependency failure
is reported rather than silently discarded. The FP32 predictor records its
process-local sample-batching correction; rerun matched WT controls when changing
prediction implementations. The coordinator finishes at cofold completion;
scientific analysis is a separate step.
