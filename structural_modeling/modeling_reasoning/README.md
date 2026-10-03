# modeling_reasoning

Bridges the **reasoning module** (PDF-extracted computational-enzyme-design
knowledge, in `kg-memory-system/reasoning`) and this **structural-modeling
pipeline**. It resolves the fields a human previously hand-engineered in the
discovery handoff, with full provenance, and hands a renderer-ready handoff to
`scripts/build_step1_inputs.py`.

## Where it sits

```text
discovery agent
  -> structural_modeling_handoff.json   (schema 2.0; setup fields "unresolved")
  -> modeling_reasoning/resolve_handoff.py
       - literature knowledge  (PDF rules, vendored in knowledge/)
       - per-target facts      (UniProt metal count, RDKit chemistry)
       - scoring-policy guard   (unrestrained; geometry-filter-first)
  -> resolved handoff.json      (schema 2.0; "ready")
  -> scripts/build_step1_inputs.py
  -> Boltz YAML
```

It **consumes and produces the schema-2.0 handoff** — the same contract the
renderer already reads. The renderer is unchanged; it just receives a handoff
whose `unresolved` fields are now filled.

## What it resolves (and from what)

| handoff field (was `unresolved`) | resolved from |
| --- | --- |
| `enzyme_state.chains` | manifest directive (intact / mature-chain split, copies) |
| `enzyme_state` oligomeric unit | **ligand-site reasoning** (`modeling_unit.py` + `knowledge/oligomeric_state.json`): assembly vs functional unit, decided by where the ligand sits, not by UniProt SUBUNIT |
| `target_substrates[].modeling_smiles`/role/`affinity_target`/`include_in_model` | manifest directive + **RDKit re-verification** (`chemistry.py`) |
| `cofactors` catalytic metal | **family/mechanism reasoning** (`knowledge/family_metal.json`) x modeled active sites; UniProt tally is a cross-check reference only |
| `geometry` | **derived from the declared reaction step** (nucleophile/electrophile atoms resolved onto the candidate), recorded **measurement-only** (never restrained) |
| `status` | re-derived against the renderer's own `readiness_errors` gate |

Every resolved field carries a `resolution` block: `origin`
(`source`/`database`/`experience`/`inference`), `rationale`, `evidence_ids`, and
`literature_basis` — the matched PDF rules with `paper_id` + `claim_id` + verbatim
quote.

## Catalytic metal: reason, don't count

A catalytic-metal count is **reasoned, not tallied**. A raw UniProt number
conflates per-subunit vs per-assembly and binding-site-label artifacts (a naive
tally produced a spurious "4 Mg per subunit"). The quantity actually placed is:

```text
per-active-site metal (from the candidate's mechanism/family)
  x number of active sites modeled (from the oligomeric state)
```

`knowledge/family_metal.json` supplies the per-site value keyed on mechanism
(acyl_adenylate -> 1 Mg, acyl_phosphate -> 2 Mg, extensible by family), and the
resolver multiplies by the active sites implied by `enzyme_state.chains` (one per
catalytic chain unless the manifest overrides `active_sites`, since some ligases
have inter-subunit sites). Any UniProt tally is attached as
`uniprot_reference` with an `agrees_with_per_site` flag — a cross-check, never
the placement count. Runs are therefore deterministic and need no network
(`--offline`); the cross-check is optional.

The discovery agent extracts metal *names* sometimes (nothing for TrEMBL) but
never stoichiometry. Extracting the raw per-subunit metal *fact* upstream in the
agent's evidence layer would let the resolver consume it as one more reference;
the reasoning (per-site x sites) stays here.

## Reaction steps (step-general, fail-closed)

An agent must be able to model **any step** of a reaction and never silently model
the wrong one. The step is a **required, explicit** parameter — `manifest.step.index`
or `--step` — with no default.

`knowledge/reaction_coordinates.json` holds mechanism reaction coordinates. Each step
declares, as **roles** (never target-specific atoms), the species present, the species
consumed/produced, and the nucleophile/electrophile of its near-attack.

**Mechanism is a family-level property.** Each coordinate lists the InterPro `families`
that use it, so it **transfers to any homolog of that family — including unannotated
ones**. The resolver selects a candidate's coordinate by family first (from the
handoff's `family_assignment` / `domain_architecture`), then falls back to activation
mode, then abstains; the basis is recorded in `modeled_step.mechanism_selected_by`.

The adenylate amide-ligase coordinate is **literature-derived**: `reasoning/extract_reaction_steps.py`
harvests a pinned, provenance-stamped mechanism corpus (`reasoning/data/mechanism_corpus.json`),
extracts quote-grounded step-claims (validation level 1), and synthesizes coordinates
**grouped by (paper, mechanism_class)** so each enzyme's mechanism is assembled
independently. Each step carries its `PMID` + verbatim quote. Coordinates not yet
extracted (e.g. the acyl-phosphate two-metal mechanism) are marked `seed_unverified`.

Given a declared step, the resolver:
- **drops consumed cofactors** — ATP (NTP) is present at step 1 but consumed into the
  intermediate by step 2, so step 2 emits no free ATP (recorded in `dropped_species`);
- **derives the geometry from that step** — step 1 → `carboxylate → NTP:Pα`; step 2 →
  `amine-N → acyl-carbonyl-C` — resolving the atoms onto the candidate (CCD atom names
  for cofactors, SMARTS for ligands), abstaining if an atom can't be located;
- **runs a fail-closed step-consistency gate**: the modeled non-protein species must
  match the step's `species_present`, and the geometry must describe that step's
  attack. A mismatch (e.g. stale ATP for step 2, or step-1 participants declared as
  step 2) blocks — the model cannot be emitted.

The code contains no reaction-specific branches; step physics comes entirely from the
coordinate + atom selectors. `examples/amide_adipate_hmda.json` (step 1) and
`amide_adipate_hmda_step2.json` (step 2) are worked examples. Note a real acyl-adenylate
intermediate has an unassigned phosphorus stereocenter, so step 2 correctly returns
`needs_review` until that is resolved — the fail-closed check surfacing a real ambiguity,
not a defect.

## Scoring-policy guard

Literature-faithful setups are not always what scores. The reasoning module
distills what authors *did* (including pocket restraints), but
`CANDIDATE_RANKING_POLICY.md` and `angle_gate_proxy.py` score on **unrestrained**
cofolds because restraints invert the activity signal. The compiler therefore
emits **no `constraints` block**; it records `scoring_policy: unrestrained` and
keeps geometry as a post-hoc `measurement_only` target. Pipeline-learned policy
overrides literature-faithful reproduction.

## Self-evolution

The reasoning is not frozen. As the reasoning module extracts more PDFs and
re-synthesizes its candidate rules, `evolve.py` pulls the richer synthesis into
`knowledge/cofolding_rules.json`, versions it in `knowledge/VERSIONS.jsonl`, and
reports how reasoning changed (new decision points, new corroborating papers,
rules promoted single->cross-paper). `resolve_handoff.py` always reads whatever
is in `knowledge/` and matches rules by decision point + feature guard, so more
and better-corroborated papers automatically yield richer, better-grounded
resolutions **with no code change**.

```bash
# Pull the latest synthesis, version it, report the improvement
CONDA_NO_PLUGINS=true ~/miniforge3/bin/python modeling_reasoning/evolve.py --dry-run
CONDA_NO_PLUGINS=true ~/miniforge3/bin/python modeling_reasoning/evolve.py

# Re-extract from newly queried PDFs first (needs the LLM endpoint), then pull
CONDA_NO_PLUGINS=true ~/miniforge3/bin/python modeling_reasoning/evolve.py \
  --run-pipeline --pipeline-cmd "<reasoning harvest+extract+synthesize command>"
```

## Run

Use the base env (RDKit lives there, not in `ragmosis`):

```bash
# Resolve a discovery handoff -> renderer-ready handoff
CONDA_NO_PLUGINS=true ~/miniforge3/bin/python modeling_reasoning/resolve_handoff.py \
  --handoff <discovery>/structural_modeling_handoff.json \
  --manifest <your_target_manifest.json> \
  --out resolved_handoff.json --report resolution_report.json
# add --offline to skip the UniProt metal cross-check (placement is unaffected)
# a worked example manifest lives at modeling_reasoning/examples/amide_adipate_hmda.json

# Then render as usual
CONDA_NO_PLUGINS=true ~/miniforge3/bin/python scripts/build_step1_inputs.py \
  --handoff resolved_handoff.json --fasta <fasta> --msa-index <tsv> --out step1

# Tests
CONDA_NO_PLUGINS=true ~/miniforge3/bin/python -m unittest discover \
  -s modeling_reasoning/tests -p 'test_*.py' -v
```

## Layout

- `resolve_handoff.py` — the compiler (handoff in -> resolved handoff out) + CLI.
- `literature.py` — load/match vendored PDF rules by decision point + feature guard.
- `chemistry.py` — RDKit re-verification of changed ligand SMILES.
- `reaction.py` — reaction-coordinate loader, mechanism/step selection (family-first), species-role mapping.
- `refresh_coordinates.py` — vendor the discovery agent's family-keyed mechanism cache into
  `knowledge/reaction_coordinates.json` (enriching merge: add new mechanisms, union families,
  never downgrade structured steps; empty cache = no-op). The discovery agent OWNS extraction +
  the cache; this pulls its output into the consumer snapshot.
- `evolve.py` — self-evolution: pull, version, and diff the literature knowledge.
- `knowledge/` — vendored literature snapshot, family/mechanism metal reasoning,
  reaction coordinates, atom selectors, provenance, and version history.
- `examples/` — worked, target-specific manifests (`amide_adipate_hmda.json` step 1,
  `amide_adipate_hmda_step2.json` step 2). Examples are inputs, not part of the pipeline.
- `tests/` — generic unit tests (renderer-ready output, family-reasoned metal, oligomer
  scaling, reference-vs-placement, ligand re-verification, exclusion, no-restraints,
  literature provenance, step-required, step-1/step-2 geometry, wrong-step-blocked). No
  specific target is baked in.

## Generic by construction

The module is domain-agnostic. All target-specific facts (sequences, ligand
SMILES, roles, oligomeric state) live in the per-target **resolution manifest**
supplied at run time; the code, the literature knowledge, and the mechanism metal
reasoning are generic. A resolution manifest has:

```text
{ target_id, task_type, scoring_policy, features,
  resolutions: { enzyme_state, substrates[], catalytic_metal, geometry } }
```

Each resolution carries `origin`, `rationale`, `evidence_ids`, and a
`decision_point` (which both routes literature guidance and lets new decision
points flow through as the knowledge evolves). It is a structured, machine-
actionable projection of a reasoning decision manifest.

## Status / limits

Behavioral bridge, exercised on one discovery batch (amide synthetases) as an
example: step 1 → 19/19 `ready`; step 2 correctly drops ATP, includes the amine, and
derives the amine-attack geometry; a wrong-step declaration is blocked by the
consistency gate. Family-reasoned per-site metal scaled by oligomeric state; no
restraints; PDF-grounded provenance on every decision. It does not establish physical
validity or cross-family generalization.

Two things are still hand-supplied / seeded and are the next steps:
- the per-target **resolution manifest** (hand-authored; should be generated by the
  reasoning pipeline);
- the **reaction coordinates** and the **intermediate chemistry** for step ≥2 (seeded /
  manifest-supplied; the intermediate SMILES and its stereochemistry should be
  constructed and grounded, and the reaction coordinates populated by exact-grounded
  `reaction-step` extraction).
