# Project Layout

## Active workflows

`enzyme_discovery_agent` is the maintained discovery source directory. The older
`enzyme_discovery_agent_2026_0916` implementation is preserved under
`archive/cleanup_20261105/legacy_agents/`. The former
`0_prev_agent_archive/` path is a local compatibility symlink for historical
provenance references. Run the maintained checks with:

```bash
pytest -q
```

Its current entry points and ownership are documented in
[`enzyme_discovery_agent/README.md`](../enzyme_discovery_agent/README.md).

`structural_modeling_agent` is an independent decision agent with its own CLI, task,
conversation, state and tool loop: `python -m structural_modeling_agent --task TASK.json`.
It can consume a discovery handoff or a standalone target/candidate task. It owns
modeling choices, setup validation, prediction validation and evidence-led
revision; its existing computational modules are tools rather than a fixed
controller workflow. See `structural_modeling_agent/README.md` for the agent contract.
It prepares and evaluates structures,
pockets, and co-folds; reserve it for a shortlisted set of candidates rather than a full
discovery pool. Reusable descriptors and proxy-plan execution live in `proxies/`;
reaction-state and atom selection live in `modeling_reasoning/`. Study-specific
calibration and plotting code lives in `exp_data/<study>/code/`, with frozen evidence bundles
under `integrated_campaigns/`. Current candidate ranking uses the frozen
inverse-distance-weighted attack angle plus harmonic distance, defined in
`structural_modeling_agent/knowledge/attack_angle_metric.json`. Historical
Gaussian-score bundles and angle-gate utilities remain explicit compatibility tools.
The [synthetase study README](../exp_data/amide_synthetase/structural_proxies/README.md)
is its active entry point. Superseded synthetase screens, tables, and drafts are
preserved in [the dated archive](../exp_data/amide_synthetase/archive/cleanup_20261008/MANIFEST.md).

Required activation cofolds live in `structural_modeling_agent/campaigns/amide_activation/`.
Nylon hydrolase cofolds and experimental analyses remain under
`exp_data/nylon_hydrolase/`. Current manuscript figure masters live in
`exp_data/<study>/figures/candidacy/`; Overleaf receives published copies.
Historical fixed-campaign launchers, one-off plots and retired figures are under
`archive/cleanup_20261105/legacy_agents/structural_modeling/cleanup_20261008/`,
with a path/hash manifest.
See [`structural_modeling_agent/README.md`](../structural_modeling_agent/README.md) for the active
workflow and ownership map. Large model outputs are not copied into reusable code.

Discovery finalization writes a schema-2.0 `structural_modeling_handoff.json` when evaluations are
available. Each candidate separates retrieved `evidence`, agent-derived `modeling_input`, and panel
`selection` provenance. The evidence layer includes identity, native reactions and substrates,
InterPro family and domain assignments, available structures, cofactor annotations, and
literature-derived mechanism evidence. The modeling layer contains the reviewed enzyme construct,
modeled molecular representations and roles, cofactor identities and counts, activation assignment,
geometry hypothesis, and explicit unresolved items.

`structural_modeling_agent/scripts/build_step1_inputs.py` accepts only this layered schema and fails closed
unless the modeling input is ready. An explicit exploratory override may bypass evidence-review
items, but never malformed execution inputs. It compiles the modeling layer into Boltz YAML and
records every execution or rejection in `campaign_manifest.json`. Structure-only cofolding is the
default; `--include-affinity` explicitly enables the optional Boltz-2 affinity property.

After structural evaluation, export and import decomposed evidence rather than a single opaque
activity score:

```bash
python -m enzyme_discovery_agent.import_structural_evidence \
  <discovery-run> <structural-evidence-manifest.json>
```

The resulting `structural_evidence_review.json` is an evidence matrix, not a ranking. It reports each
candidate's raw structural checks, whether required checks are present, and the remaining review
questions. Final candidate selection must retain the discovery, literature, substrate-scope, and
reaction-direction evidence.

## New-system output locations

New structural-modeling results default to
`results/<system-label>-<target-hash>/structural_modeling/<run-id>/`, outside the
source package. `--results-root`/`ENZYME_AGENT_RESULTS_ROOT` configures the base;
`--output-dir` explicitly overrides it. Each run has an immutable identity
manifest and task snapshot, an authoritative checkpoint, and `results_index.json`
with planned and registered output locations. Cofold inputs/predictions live in
`models/<accession>/revision_<n>/cofold/`; later dynamics and de novo requests
live under that revision's `followups/<purpose>/<id>/`. Design handoffs explicitly
set their native design-agent output root. The structural README documents the
full layout. Historical study/campaign folders are preserved in place.

## Candidate ESM-embedding map (agent-invoked)

The agent can map its discovered candidates in an ESM2 protein-language-model space and
retrieve the nearest neighbors of its most-promising candidates — as new leads to expand
and as the cofold panel. It is exposed as the LLM-callable `map_candidate_neighbors` tool
(the agent decides *when* to call it and *how many* neighbors / candidates to use via tool
params; `config.CANDIDATE_MAP_*` only supplies fallback defaults), gated behind
`ENABLE_CANDIDATE_MAP` (env `ENZYME_AGENT_ENABLE_CANDIDATE_MAP=1`) so a normal run never
imports torch. The heavy ESM forward pass runs in a **separate GPU subprocess**
(`tools/esm_embed_worker.py`, fair-esm) launched through the cluster reservation gate
`gpu_reserve` — the main run stays GPU-free. State persists per run under `embedding_store/`
(`vectors.npy` + `index.json` + `sequences.json`) and `candidate_map.json`; neighbors are
folded into `structural_modeling_handoff.json`.

**Embedding-driven expansion (reach outside the pool)** — where `map_candidate_neighbors`
finds neighbors *within* the discovered pool, the `expand_candidates_esm` tool reaches
*out*: for the top candidates (or one you name) it fetches the wider UniProt membership of
the InterPro families they belong to (Swiss-Prot first, widening to TrEMBL for sparse
families), embeds that background in the same space, and registers as new `discovered`
leads the members sitting NEAREST the top candidates — the embedding is the decision layer
that picks which broadly retrieved sequences to pursue. The agent chooses `n_seeds` / `k` /
`per_family` (family membership provides clean search provenance — the families come from
its own candidates, never a seeded answer); `config.CANDIDATE_MAP_EXPANSION_*` are fallback
defaults. Selected leads' vectors are appended to the store (so they appear on the map) and
written into `candidate_map.json`, so they flow into the cofold panel unchanged. Same
`ENABLE_CANDIDATE_MAP` gate + `gpu_reserve` subprocess embedding as the map tool.

Setup — install the `embedding-map` extra and point the worker at a fair-esm env (palm's
`transformers` is version-broken; fair-esm loads ESM2 via torch.hub, needing neither
transformers nor huggingface_hub):

```bash
python -m venv --system-site-packages /nfs/lambda_stor_01/homes/rzhu/esm_embed_env
/nfs/lambda_stor_01/homes/rzhu/esm_embed_env/bin/python -m pip install --no-deps fair-esm
# config.CANDIDATE_MAP_EMBED_PYTHON points here by default
```

Publication figure of a run's explored candidates — embeds every candidate across all free
GPUs (parallel, sharded via `gpu_reserve`), then UMAP with numbered top-ranked candidates:

```bash
python enzyme_discovery_agent/figures/plot_esm_embedding_map.py <run_dir> \
  --title "..." --top 10        # add --no-embed to re-plot from a cached store
                                # add --show-expansion to mark esm_expansion leads
                                #   with a diamond and a leader back to their seed
```

## Standalone validation harnesses

Out-of-tree studies that validate a design idea before (if ever) wiring it into the agent — keep
them here rather than embedding experimental logic in the package:

- `archive/Interpro_exhaustive/` — exhaustive reviewed EC 6.3.- recall sweep.
- `archive/Reaction_anchored_retrieval/` — reaction- (Rhea/RDKit) vs EC/InterPro-anchored retrieval.
- `archive/Rhea_anchored_coverage/` — Rhea-anchored deterministic fold-coverage (recall floor).
- `archive/Literature_selection_validation/` — full-text paper-selection precision.

Completed November validation workspaces and their copied source trees live in
`archive/cleanup_20261105/discovery_validation/`. A local `discovery_validation/`
compatibility symlink preserves the paths recorded in existing validation reports.

## Target profiles

Use `--target-profile` to provide a JSON description of the reaction, substrate state, enzyme
requirements, and required structural checks. The workflow applies the same schema to any enzyme
class; it contains no target-specific rules and produces no composite score. See
[`enzyme_discovery_agent/target_profile.example.json`](../enzyme_discovery_agent/target_profile.example.json).

## Archived clutter

The [November cleanup manifest](../archive/cleanup_20261105/MANIFEST.md) records
the original and archived locations of legacy agents, manuscript backup copies,
completed validation workspaces, historical decks, retired discovery figures,
old local configuration backups, and generated build/cache files. File contents,
permissions, directory membership and symlink targets were checked after each
move; the local archive contains the detailed SHA-256 inventory.

The active source packages, tests, campaign and run directories, shared data,
model assets, current figure masters and separate Overleaf checkout remain in
place. Generated `build/`, bytecode and test caches may reappear during normal use.

## Naming rules

- Use descriptive role-based names for active components; avoid date-only or generic names such as `refactor`, `prev`, and `feedback`.
- Put superseded work in `archive/` and document it in `archive/MANIFEST.md`. Nothing is deleted.
- Keep shared checkpoints, databases, caches, and run outputs at the workspace root until their consumers have configurable paths.
- Add new experiments as a standalone harness folder (see above) or beneath `archive/experiments/` until they become an active, documented workflow.
- Keep READMEs and this layout current when the structure changes.
