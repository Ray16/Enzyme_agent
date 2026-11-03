# BioEmu2 substrate-retention pilot

This tests whether predicted substrate retention adds information about measured
N66L1 product concentration across AS1–AS48. It reuses the historical one-Mg
step-1 complexes (protein, ATP, Mg and adipate), without generating sequences or
changing the established static ranking metric.

The default protocol measures attack geometry in all 50 existing Boltz cofolds
per enzyme. It selects the shortest substrate carboxylate O–ATP PA distance among
poses with an O–PA–O3A angle of at least 150 degrees and a distance at most 4.5 Å,
subject to protein physicality and absence of severe substrate/protein clashes.
Larger angles and lower pose indices break distance ties. These are operational
starting-pose criteria, not proof of catalysis. Selection never uses activity.
If no pose qualifies, the enzyme is excluded from BioEmu2 submission. Its best
available cofold is retained only in the starting-geometry audit and is not
sampled. Fallback audit ordering prefers angle at least 150 degrees and then
shortest distance. Every pose's measured geometry and selection outcome is
recorded in `inputs/<enzyme>/selection.json`.

Each qualifying enzyme's selected starting structure is propagated independently
to 1 ns and 10 ns, two seeds per lag. The current panel has 45 qualifying enzymes
and 180 endpoints; AS13, AS28 and AS36 have no qualifying pose and are not submitted.
The runtime is BioEmu2 preview at the recorded local checkpoint, FP32, batch size
one, default EDM propagation and up to 128 rows of the existing MSA. No contact
conditioning or steering forces the substrate to remain bound.

The primary measurement is mean retention of starting substrate-contacting
protein residues at 10 ns among enzymes with qualifying starts. A contact means any substrate/protein heavy-atom pair
within 4.5 Å. Starting residues within 6 Å define the pocket for C-alpha rigid
alignment; a recorded 10 Å fallback is used only when fewer than three C-alpha
atoms are available. Centroid displacement and contact retention are separate
measurements. The descriptive pocket-retained flag uses at least half the
starting contacts and at most 5 Å displacement; this flag is not optimized on
activity. A pocket defined around the starting ligand can be a misplaced pocket;
starting substrate–ATP distances and the static baseline are therefore retained.

Every endpoint and failure remains saved. Geometry-eligible analyses require
both endpoints at a lag to pass the engine's protein physicality checks and have
no substrate/protein heavy-atom pair below 1.5 Å. This is not a complete ligand
physicality or chemistry validation. A separate sensitivity analysis includes all
saved endpoints. Failures and missing frames are never counted as dissociation.

The statistical unit is the enzyme. Primary assessment is Spearman correlation
with continuous measured product concentration, with enzyme-level bootstrap and
permutation uncertainty. The existing three strong-producer annotations are a
secondary comparison, not an invented active/inactive detection threshold.
Incremental predictive assessment uses fixed-alpha leave-one-enzyme-out ridge
regression against the starting contact count, selected-pose substrate–ATP
distance, and established 50-pose weighted attack angle. Scaling is fitted only
on training enzymes. This is retrospective internal assessment; related homologs
may appear in both training and test sets. Sample frequencies are not calibrated
binding probabilities, residence times, or rate constants. Four endpoint samples
from one best starting pose per enzyme provide a pilot, not a converged dynamic
ensemble. Selecting the best among 50 poses is an optimistic starting condition;
the analysis tests persistence after that selection and does not establish how
frequently the protein reaches that condition naturally.

Use the dedicated interpreter:

```bash
BIOEMU_PYTHON=/nfs/lambda_stor_01/homes/rzhu/Tool/envs/bioemu2/bin/python
```

Run `prepare.py --selection best_geometry --out <new NFS result directory>` once. It verifies sequences,
molecular composition, local MSAs and substrate atom mapping, hashes dependencies,
and writes immutable starting inputs and a manifest. `audit_existing.py --out
<JSON path>` measures the earlier AS9 run without changing its outputs.

`launch.py --root <result directory> --count 1 --only <task ID from manifest>` is the technical
GPU check. After verifying its saved output and memory footprint, use
`launch.py --root <result directory> --count 12` for the bounded panel fleet.
The launcher preflights dependencies and CPU capacity, selects available GPUs
through the shared reservation gate, snapshots source files, pins each worker to
two disjoint CPUs, and caps native library pools at one thread. The default host
is lambda5; override `--host` only with a policy-allowed, preflighted node.

Workers keep an enzyme's four tasks together to reuse its embedding, using an
atomic enzyme lease and independently resumable per-task claims with
host/PID/start-time ownership records. They
retain partial attempts, verify saved frame coverage and hashes, and apply two
attempts per item plus a three-consecutive-failure circuit breaker. They stop
their own process if another GPU process overlaps their reservation. Claims are
never reclaimed automatically by age.

Run `analyze.py --root <result directory> --wait` on a pinned CPU with native
threads capped at one. It writes progress, waits up to six hours, verifies saved
output hashes, and generates JSON statistics, endpoint and enzyme CSVs, plots,
and `report.html`. Partial reports explicitly state that the panel is incomplete.
The final analysis can be rerun safely after additional tasks finish.

`python -m unittest discover -s <this directory> -p test_core.py` verifies that
rigid protein/ligand motion does not look like substrate departure and that an
actual ligand displacement is detected. It also verifies selection of the
shortest aligned pose rather than a closer, wrongly oriented substrate.

The original fixed-index run in `bioemu2_retention_20261103` was stopped at the
user's request and its 24 completed endpoints preserved. Its outputs must not be
pooled with the corrected `bioemu2_best_geometry_20261103` experiment. The legacy
preparation mode remains available as `--selection fixed` for reproduction only.

The corrected run was narrowed from 48 to 45 enzymes before any excluded enzyme
was sampled. `scope_amendment.json` records the archived original manifest and
removed task IDs. `scope_guard.py` holds explicit exclusion leases while already
running worker versions drain their in-memory queues; it does not fabricate task
completion or consume a GPU. This lets useful predictions continue without
recomputing their embeddings. The leases are released after those exact legacy
worker identities have exited. New workers read the filtered manifest directly.
