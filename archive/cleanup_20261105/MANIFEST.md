# Workspace cleanup — 2026-11-05

Files were moved within the same filesystem. No archived file contents were edited or deleted.

| Previous path | Archive path | Compatibility link | Reason |
| --- | --- | --- | --- |
| `0_prev_agent_archive` | `archive/cleanup_20261105/legacy_agents` | yes | Superseded source snapshots and historical modeling material. |
| `manuscript_backups` | `archive/cleanup_20261105/manuscript_backups` | no | October manuscript backup copies; current Overleaf repository stays in place. |
| `discovery_validation` | `archive/cleanup_20261105/discovery_validation` | yes | Completed dated validation workspaces and copied source trees. |
| `figures` | `archive/cleanup_20261105/historical_deck` | no | Deck documented as historical in AGENT_DECK_README.md, with retired figure variants. |
| `enzyme_discovery_agent/figures/archive` | `archive/cleanup_20261105/retired_discovery_figures` | no | Figures already designated as archived. |
| `build` | `archive/cleanup_20261105/generated/build` | no | Generated package copies and dated development audit checkpoints. |
| `.claude.bak` | `archive/cleanup_20261105/local_backups/claude` | no | Old local assistant configuration backup. |
| `__pycache__` | `archive/cleanup_20261105/generated/root_bytecode` | no | Stale root test bytecode. |
| `.pytest_cache` | `archive/cleanup_20261105/generated/pytest_cache` | no | Regenerable test-run cache. |
| `texput.log` | `archive/cleanup_20261105/generated/texput.log` | no | Stray LaTeX build log. |

Preserved 10,493 regular files (544,047,893 bytes).
The local `inventory.json` records each file SHA-256, inode, permissions, and original symlink target.
Verification compared the complete inventory before and after every move.

The legacy-agent and validation paths remain local compatibility symlinks so historical provenance paths still resolve.
Current source, tests, campaigns, run histories, model assets, shared data, figure masters, and the Overleaf checkout were left in place.

To restore an item, ensure the previous location has not been reused, remove its compatibility symlink if present,
then rename the archive path back to the previous path. Do not overwrite newly generated build or cache files.

Large archive payloads and detailed inventories remain local and ignored by Git. Human-readable manifests and the four previously tracked retired figure assets remain versioned.
