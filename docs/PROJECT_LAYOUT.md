# Project Layout

## Active workflows

`enzyme_discovery_agent_improved` is the **maintained** discovery package (the earlier
`enzyme_discovery_agent` is retired). Run its offline checks with:

```bash
python -m unittest discover -s enzyme_discovery_agent_improved/tests -v
```

Its internal ownership and dependency rules are documented in
[`enzyme_discovery_agent_improved/ARCHITECTURE.md`](../enzyme_discovery_agent_improved/ARCHITECTURE.md).
In particular, LLM-facing operations live in `tools/`, deterministic workflow policy in
`gates/`, cross-stage orchestration in `services/`, persisted state in `models/`, and
rendering/export code in `output/`.

`structural_modeling` is the dedicated precision layer. It prepares and evaluates structures,
pockets, and co-folds; reserve it for a shortlisted set of candidates rather than a full
discovery pool. Geometry uses the validated **distance-gated in-line angle** proxy
(`scripts/angle_gate_proxy.py`), not raw NAC.

Cofold work is organized **one folder per discovery run** under
`structural_modeling/cofold/<run-id>/` (folder named exactly after `runs/<run-id>/`); each
holds that campaign's `step1/`, `stage_configs/`, `results/`, `msa_cache/`, `candidates.csv`,
etc. Shared tooling stays at the `structural_modeling/` root: `scripts/` (argument-driven, so
they run on any campaign — point them at `cofold/<run-id>/...`), `tests/`,
`CANDIDATE_RANKING_POLICY.md`, and `figures/`. Large Boltz outputs (~1 GB/campaign) are
untracked by git. See `structural_modeling/cofold/README.md`.

Discovery finalization writes a schema-2.0 `structural_modeling_handoff.json` when evaluations are
available. Each candidate separates retrieved `evidence`, agent-derived `modeling_input`, and panel
`selection` provenance. The evidence layer includes identity, native reactions and substrates,
InterPro family and domain assignments, available structures, cofactor annotations, and
literature-derived mechanism evidence. The modeling layer contains the reviewed enzyme construct,
modeled molecular representations and roles, cofactor identities and counts, activation assignment,
geometry hypothesis, and explicit unresolved items.

`structural_modeling/scripts/build_step1_inputs.py` accepts only this layered schema and fails closed
unless the modeling input is ready. An explicit exploratory override may bypass evidence-review
items, but never malformed execution inputs. It compiles the modeling layer into Boltz YAML and
records every execution or rejection in `campaign_manifest.json`. Structure-only cofolding is the
default; `--include-affinity` explicitly enables the optional Boltz-2 affinity property.

After structural evaluation, export and import decomposed evidence rather than a single opaque
activity score:

```bash
python -m enzyme_discovery_agent_improved.import_structural_evidence \
  <discovery-run> <structural-evidence-manifest.json>
```

The resulting `structural_evidence_review.json` is an evidence matrix, not a ranking. It reports each
candidate's raw structural checks, whether required checks are present, and the remaining review
questions. Final candidate selection must retain the discovery, literature, substrate-scope, and
reaction-direction evidence.

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
python enzyme_discovery_agent_2026_0916/figures/plot_esm_embedding_map.py <run_dir> \
  --title "..." --top 10        # add --no-embed to re-plot from a cached store
```

## Standalone validation harnesses

Out-of-tree studies that validate a design idea before (if ever) wiring it into the agent — keep
them here rather than embedding experimental logic in the package:

- `Interpro_exhaustive/` — exhaustive reviewed EC 6.3.- recall sweep.
- `Reaction_anchored_retrieval/` — reaction- (Rhea/RDKit) vs EC/InterPro-anchored retrieval.
- `Rhea_anchored_coverage/` — Rhea-anchored deterministic fold-coverage (recall floor).
- `Literature_selection_validation/` — full-text paper-selection precision.

## Target profiles

Use `--target-profile` to provide a JSON description of the reaction, substrate state, enzyme
requirements, and required structural checks. The workflow applies the same schema to any enzyme
class; it contains no target-specific rules and produces no composite score. See
[`enzyme_discovery_agent_improved/target_profile.example.json`](../enzyme_discovery_agent_improved/target_profile.example.json).

## Naming rules

- Use descriptive role-based names for active components; avoid date-only or generic names such as `refactor`, `prev`, and `feedback`.
- Put superseded work in `archive/` and document it in `archive/MANIFEST.md`. Nothing is deleted.
- Keep shared checkpoints, databases, caches, and run outputs at the workspace root until their consumers have configurable paths.
- Add new experiments as a standalone harness folder (see above) or beneath `archive/experiments/` until they become an active, documented workflow.
- Keep READMEs and this layout current when the structure changes.
