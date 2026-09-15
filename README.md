# Biochemical Agent Workspace

Discovery and structural validation of enzymes for target (often non-natural) reactions,
e.g. nylon-6,6 amide synthesis and hydrolysis.

## Active components

| Directory | Purpose |
| --- | --- |
| `enzyme_discovery_agent_improved/` | **Canonical discovery agent.** LLM-driven, reaction-anchored enzyme search + decomposed ranking over database, literature, and structural-homology evidence. Emits a reviewable structural-modeling handoff per candidate (family, mechanism, cofactor spec). |
| `structural_modeling/` | Precision layer for a shortlisted set: candidate prep, Boltz-2 co-folding, and mechanism-aware geometry (distance-gated in-line angle). |
| `figures/` | Group-meeting figure deck (see `figures/AGENT_DECK_README.md`) + validated proxy figures. |
| `exp_data/` | Assay panels (amide-synthetase, nylon-hydrolase) used to benchmark structural proxies against measured activity. |
| `analysis/` | Analysis scripts, incl. the validated angle-gate proxy figures (`make_synthetase_figures.py`). |

The intended progression for non-natural reactions:

```text
enzyme discovery → candidate shortlist → structural modeling → evidence-informed ranking
```

The discovery agent writes a reviewable `structural_modeling_handoff.json` after candidate
evaluation. Structural validation is a deliberate downstream step, not an automatic claim of
catalytic activity.

> **Note:** the earlier monolithic `enzyme_discovery_agent/` package is **retired**; the
> maintained package is `enzyme_discovery_agent_improved/`.

## Standalone validation harnesses (out-of-tree, not wired into the agent)

| Directory | What it validates |
| --- | --- |
| `Interpro_exhaustive/` | Exhaustive reviewed EC 6.3.- sweep — proves discovery recall vs the annotated universe. |
| `Reaction_anchored_retrieval/` | Anchoring discovery on the target reaction (RDKit over Rhea SMILES) rather than EC/InterPro. |
| `Rhea_anchored_coverage/` | Rhea-anchored deterministic fold-coverage (recall floor over curated reactions); see its README. |
| `Literature_selection_validation/` | Full-text paper-selection precision for literature-guided discovery. |

## Supporting directories

- `docs/` — planning history, notes, presentations, and [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md) (naming + workflow guidance).
- `paper/` — manuscript and proposal material.
- `data/` — panel inputs and cofold input sets.
- `archive/` — recoverable legacy implementations, superseded runs, and snapshots. **Nothing is deleted**; see [`archive/MANIFEST.md`](archive/MANIFEST.md).

## Usage

```bash
# discovery run
python -m enzyme_discovery_agent_improved "find amide synthetases that ligate adipic acid and hexamethylenediamine" \
  --substrate "O=C(O)CCCCC(=O)O.NCCCCCCN"

# resume an interrupted run in place
python -m enzyme_discovery_agent_improved --resume-run enzyme_discovery_agent_improved/runs/<run-id>

# optional generic target profile (evidence requirements; no target-specific rules, no composite score)
python -m enzyme_discovery_agent_improved --target-profile target_profile.example.json "<query>"
```

Key flags: `--substrate` / `--structure-file` / `--sketch` (substrate input), `--model`,
`--output-dir`, `--max-iterations` (default 100), `--llm-timeout`, `--resume-run`, `--target-profile`.

## Run provenance

Every run writes a self-describing record into its run directory so a completed run is
reproducible and an interrupted one stays identifiable:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Machine-readable record: git commit + dirty flag, resolved config (secrets redacted), Python/OS/package versions, model, args, timing, outcome (iterations, candidates, tokens). |
| `RUN_SUMMARY.md` | Human-readable digest of the manifest. |
| `agent_state/transcript.json` | Full conversation trace, checkpointed each iteration (survives a crash). |
| `candidate_registry.json` | Per-candidate provenance: source tool, status transitions, mechanism/cofactor evidence. |
| `events.jsonl` | Structured event stream (tool calls, gate transitions, registrations). |
| `structural_modeling_handoff.json` | Per-candidate handoff to structural modeling (family, mechanism, cofactor spec). |

## Development

- **Version control:** git repository on `master`. Run outputs, large data trees, and caches are excluded via `.gitignore`. A PostToolUse hook auto-commits and pushes each edited file to `origin/master`.
- **Tests** (offline, stdlib `unittest`; live-API tests skip when offline):

  ```bash
  PYTHONPATH=. python -m unittest discover -s enzyme_discovery_agent_improved/tests -p "test_*.py"
  ```
