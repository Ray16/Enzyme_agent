# Biochemical Agent Workspace

This workspace is organized around two active components:

| Directory | Purpose |
| --- | --- |
| `enzyme_discovery_agent/` | Discovers and ranks enzymes for target reactions using database, literature, and structural-homology evidence. |
| `structural_modeling_pipeline/` | Structural validation workflow: catalytic-site and pocket analysis, Boltz-2 co-folding, and geometry-based assessment. |

The intended progression for non-natural reactions is:

```text
enzyme discovery → candidate shortlist → structural modeling → evidence-informed ranking
```

`enzyme_discovery_agent` writes a reviewable structural-modeling handoff after candidate evaluation. Structural validation remains a deliberate downstream step rather than an automatic claim of catalytic activity.

## Supporting directories

- `checkpoints/`, `foldseek_db/`: shared model assets and databases.
- `runs/`: retained run outputs.
- `benchmark/`, `LigandMPNN/`, `ligandmpnn_demo/`: external/reference projects. Biomni and the superseded monolithic design agent are archived.
- `docs/`: planning history, notes, and presentations.
- `paper/`: manuscript and proposal material.
- `archive/`: recoverable legacy implementations, failed experiments, historical runs, and snapshots. Nothing has been deleted.

See [`archive/MANIFEST.md`](archive/MANIFEST.md) for the archive map and [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md) for naming and workflow guidance.
