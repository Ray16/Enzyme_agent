# Project Layout

## Active workflows

`enzyme_discovery_agent` is the maintained discovery package. Run its offline checks with:

```bash
python -m unittest discover -s enzyme_discovery_agent/tests -v
```

`structural_modeling_pipeline` is the dedicated precision layer. It prepares and evaluates structures, pockets, and co-folds; reserve it for a shortlisted set of candidates rather than a full discovery pool.

Discovery finalization writes `structural_modeling_handoff.json` when evaluations are available. Import a reviewed shortlist with:

```bash
python structural_modeling_pipeline/import_discovery_shortlist.py \
  <discovery-run>/structural_modeling_handoff.json --project <new-project>
```

Then provide verified substrate SMILES to `run_structural_modeling.py`; the handoff does not infer unverified ligand chemistry.

After structural evaluation, export and import decomposed evidence rather than a single opaque activity score:

```bash
python structural_modeling_pipeline/export_structural_evidence.py <project> \
  --cofold-metrics structural_modeling_pipeline/analysis_results/cofolding_metrics.csv
python -m enzyme_discovery_agent.import_structural_evidence \
  <discovery-run> <project>/structural_evidence_manifest.json
```

The resulting `structural_evidence_review.json` is an evidence matrix, not a ranking. It reports each candidate's raw structural checks, whether required checks are present, and the remaining review questions. Final candidate selection must retain the discovery, literature, substrate-scope, and reaction-direction evidence.

## Target profiles

Use `--target-profile` to provide a JSON description of the reaction, substrate state, enzyme requirements, and required structural checks. The workflow applies the same schema to any enzyme class; it contains no target-specific rules and produces no composite score. [`examples/nylon_66_hydrolase_profile.json`](examples/nylon_66_hydrolase_profile.json) is a configured stress-test example, not embedded logic.

## Naming rules

- Use descriptive role-based names for active components; avoid date-only or generic names such as `refactor`, `prev`, and `feedback`.
- Put superseded work in `archive/` and document it in `archive/MANIFEST.md`.
- Keep shared checkpoints, databases, caches, and run outputs at the workspace root until their consumers have configurable paths.
- Add new experiments beneath `archive/experiments/` until they become an active, documented workflow.
