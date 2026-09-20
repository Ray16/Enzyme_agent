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

Discovery finalization writes `structural_modeling_handoff.json` when evaluations are available —
now carrying, per candidate, the InterPro family, the deterministically-derived activation
mechanism (reactive phosphate), and a cofactor spec. The active structural workflow keeps reviewed
candidates in `structural_modeling/candidates.csv` and prepares staged inputs with the scripts under
`structural_modeling/scripts/`.

Provide verified substrate SMILES when preparing stage inputs; the discovery handoff does not infer
unverified ligand chemistry.

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
