# Project Layout

## Active workflows

`enzyme_discovery_agent_improved` is the **maintained** discovery package (the earlier
`enzyme_discovery_agent` is retired). Run its offline checks with:

```bash
python -m unittest discover -s enzyme_discovery_agent_improved/tests -v
```

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
```
