# Reusable RFD4 campaign tools

Canonical location: `enzyme_design_agent/runs/rfd4_campaign/`. The old
`tools/rfd4_campaign` path is a compatibility symlink for recorded commands
and existing campaign wrappers. Generated campaign data remain in
`runs/rfd4_campaigns/<campaign>/`.

## Run statistics

`run_logging.py` supplies common timing and size fields. Deploy it beside
`aurora_generate.py` or `queue_worker.py`. Logs include protein lengths,
requested/completed samples, batch size, startup/import overhead, generation
entrypoint time, total process wall time, and average time per completed sample.
The Aurora generator also measures warm-up through the first generation batch
callback, which includes model loading. The legacy CUDA queue cannot measure that
callback and leaves its warm-up field null. Batch averages describe throughput,
not individual latency. Keep scheduler queue wait separate and retain the
existing hardware, CPU/thread and peak-memory records. Frozen historical runs
are preserved; their unmeasured stages are not filled with estimates.

`batch_profile.py` selects a process-local V100 batch size through
`RFD4_BATCH_SIZE` and delegates to the existing `legacy_gpu.py`. Use
only inside an established GPU reservation. It does not select inputs, GPUs,
seeds, or work queues, and should be capacity-tested on the actual input.

`geometry.py` measures a distance and an angle from atom selectors in a JSON
file. `center` is the angle vertex, `angle_reference` is its other fixed arm,
and the closest atom from `attack_candidates` defines the attacking arm. Each
selector specifies `chain`, optional `res_id`, and either `atom`, `atoms`, or
`element`. Specify `expected_attack_candidates` where the chemical graph gives
an exact count. The caller is responsible for validating atom identity and the
reaction mechanism. Example:

```json
{
  "center": {"chain": "B", "res_id": 1, "atom": "PA"},
  "angle_reference": {"chain": "B", "res_id": 1, "atom": "O3A"},
  "attack_candidates": {"chain": "D", "res_id": 1, "element": "O"},
  "expected_attack_candidates": 4
}
```

Run with the RFD4 Python environment:

```bash
python runs/rfd4_campaign/geometry.py generated.cif --geometry geometry.json
```

The geometry describes whatever structure is supplied. In particular, measuring
a coordinate-conditioned generation does not independently establish ligand
binding, productive attack, or activity. Preserve the project's existing ranking
metric and distinguish it from any exploratory geometry filter.

`rolling_report.py` watches completed batch JSON records (`rows` entries with a
structure `path` and optional `qc.technical_pass`, plus optional
`generation_wall_seconds`) and atomically updates a CSV. It reports a batch's
wall time divided by its structure count as an amortized throughput measure, not
an individual latency. Use `--once` to produce a snapshot; omit it to poll.

```bash
python runs/rfd4_campaign/rolling_report.py \
  --records '/shared/run/complete/*.json' --geometry geometry.json \
  --output /shared/run/rolling_metrics.csv --interval 30
```

Only publish complete batch records after all expected structure files are
written. Keep resource reservation, queue ownership, model-specific validation,
and independent refolding in the appropriate established workflow.

`prepare_queue.py` and `queue_worker.py` provide a task-configured generation
queue using the same V100 wrapper. Prepare a fresh output directory with JSON
like this (replace the input and paths for the task):

```json
{
  "python": "/nfs/lambda_stor_01/homes/rzhu/Tool/RFD4-Proteina-compat/.venv/bin/python",
  "inputs": [{"id": "motif_a", "path": "/shared/path/annotated.cif"}],
  "batch_size": 3,
  "samples_per_chunk": 6,
  "replicates_per_input": 32,
  "seed_start": 1234,
  "overrides": ["prevalidate_inputs=true", "recycle_strategy=medium/midpeak", "compress_outputs=false", "hf_ckpt_path=nvidia/RFD4-Proteina"]
}
```

```bash
python runs/rfd4_campaign/prepare_queue.py campaign.json --output /shared/new-run
```

After task-specific preflight and capacity testing, launch each worker inside a
`gpu_reserve run` process tree with separate CPU affinity and explicit thread
caps, following the `lambda-fleet` skill. Its command is
`python runs/rfd4_campaign/queue_worker.py /shared/new-run --cwd /path/to/RFD4-Proteina`.
The generated complete records represent the profile's basic output checks;
task-specific chemical QC and independent validation remain separate. Claims
are atomic, attempts are bounded to one per item, and partial results require
inspection before restart. `rolling_report.py` can read those complete records.

## Rolling Aurora copies and local geometry

`rolling_aurora_download.py CONFIG.json` watches an Aurora campaign's complete
structure records, downloads up to three files per control request, verifies
SHA-256 values, and invokes `analyze_downloads.py`. This is a fallback for when
the standard Globus transfer endpoint is unavailable; use the staging skill's
transfer script when it works. The watcher stops after the PBS job finishes,
a bounded maximum duration, or three consecutive errors.

The config supplies `campaign_dir`, `remote_root`, `pbs_job`, `endpoint`,
`analysis_python`, `geometry` (the selectors documented above), `expected`,
`interval_seconds`, and `max_seconds`. Run the control process with SDK Python,
separate CPU affinity and native thread caps of one. The analysis subprocess
uses the configured scientific interpreter. No GPU inference is launched by
these scripts. Downloads preserve remote shard paths beneath `downloads/`.

`local_analysis/reactive_geometry.csv` contains local measurements per verified
structure. `summary.json` records counts, ranges, duplicate sequence groups and
analysis errors; `sequences.fasta` contains unique sequences. Generated geometry
is explicitly labeled conditioned, and AME remains pending until independently
computed. Timing fields remain null until the corresponding shard finishes.

## Reusable motif visualization and geometry plots

Use the same scripts for each new motif campaign. They read input names,
residues and coordinate-conditioned atoms from the campaign files; AS9-specific
residue lists, sequence lengths and sample counts are not embedded in the tools.
Run rendering first, then plotting to also produce a comparison panel:

```bash
# From Enzyme_agent; replace CAMPAIGN with the actual campaign directory.
env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/pymol/bin/python \
  enzyme_design_agent/runs/rfd4_campaign/render_theozymes.py --campaign CAMPAIGN

env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /nfs/lambda_stor_01/homes/rzhu/miniforge3/bin/python \
  enzyme_design_agent/runs/rfd4_campaign/plot_geometry.py --campaign CAMPAIGN
```

Assign separate free CPU affinities with `taskset` when running these concurrently.
The renderer uses one CPU thread. Both commands accept `--outdir` and `--help`.

`render_theozymes.py` uses `inputs/*_view.cif` and matching `*.metadata.json`
with a `motif_fixed_atoms` map of residue IDs to residue names and fixed atom names.
It displays full retained residues as sticks and marks coordinate-conditioned
protein atoms with spheres. Input coordinates and chemistry remain unchanged.
Default chains are A (protein), B (ATP), D (substrate), and C (metal); override
them with the corresponding `--*-chain` options. Use `--poses` to select inputs
and `--label-residues` to control residue labels. Inputs must share a coordinate
frame for the common comparison camera. PNGs and interactive PyMOL `.pse`
sessions are saved, with source hashes and camera settings in a provenance JSON.
Styling is imported directly from
`slides-candidacy/enzyme_visualization/style.py` and `style.json`: element
colors, ligand ball-and-stick sizes, transparency, lighting and label settings
stay consistent with the candidacy figures. Protein anchor atoms are opaque;
the other retained atoms use the shared translucent side-chain context.
`--style` accepts an alternative style JSON; `--style-dir` locates the helper.
The default `--camera attack-plane` follows the candidacy attack-angle renderer,
facing the closest substrate oxygen–ATP PA–ATP O3A plane. Explicitly change
`--center-atom` and `--leaving-atom` for other reaction steps, or use
`--camera adenine-plane`. The first input defines the shared comparison camera.
Each render verifies that all source coordinates remain unchanged, and records
the style/helper hashes with the input hashes.

`plot_geometry.py` reads `local_analysis/reactive_geometry.csv`: unique `id`,
`input_id`, `angle_deg` and positive `distance_A` values are required. Optional
`motif_integrity_audit.json` supplies per-design integrity flags; missing audit
results are labeled unknown. `--csv`, `--audit`, `--group-column`, `--bin-width`
and `--title` allow other compatible tables and presentation settings.
The script writes PNG/SVG angle distributions and distance-versus-angle plots,
plus a summary with source hashes, `sum(angle/d)/sum(1/d)` and harmonic distance
`N/sum(1/d)` for each input group. Flagged designs are retained and marked.

The histogram weights each generated structure by `1/d`, normalizes each group
separately, and shows that group's proximity-weighted mean angle. This is a
distribution across generated designs. It does not supply independent ensemble
scores for individual sequences. Scripts do not launch refolding, change the
frozen ranking metric, or infer activity from the conditioned geometry.
