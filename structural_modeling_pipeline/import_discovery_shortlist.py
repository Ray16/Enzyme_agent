#!/usr/bin/env python3
"""Import a discovery-agent structural-modeling handoff into a pipeline project.

This intentionally imports only reviewed candidate identifiers and preserves the
original manifest. Substrate SMILES still require user verification before a
co-fold run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def import_shortlist(manifest_path: Path, project_dir: Path, limit: int | None = None) -> Path:
    """Copy a handoff manifest and emit a deduplicated UniProt-ID input file."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported handoff schema; expected version 1.0.")
    candidates = manifest.get("candidates", [])
    accessions = list(dict.fromkeys(candidate.get("accession", "") for candidate in candidates if candidate.get("accession")))
    if limit is not None:
        accessions = accessions[:limit]
    if not accessions:
        raise ValueError("The handoff contains no candidate accessions.")

    project_dir.mkdir(parents=True, exist_ok=True)
    copied_manifest = project_dir / "discovery_handoff.json"
    shutil.copy2(manifest_path, copied_manifest)
    accession_file = project_dir / "discovery_candidate_accessions.txt"
    accession_file.write_text("\n".join(accessions) + "\n")
    return accession_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a reviewed discovery shortlist for structural modeling.")
    parser.add_argument("handoff", type=Path, help="structural_modeling_handoff.json from enzyme_discovery_agent")
    parser.add_argument("--project", type=Path, required=True, help="Target structural-modeling project directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of candidate accessions")
    args = parser.parse_args()

    accession_file = import_shortlist(args.handoff, args.project, args.limit)
    print(f"Imported shortlist to {args.project}")
    print(f"Candidate file: {accession_file}")
    print("Next: review candidate evidence and run run_structural_modeling.py with --uniprots-file plus verified --substrates.")


if __name__ == "__main__":
    main()
