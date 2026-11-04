"""Publish referenced figure copies from authoritative modeling outputs.

--check is read-only. Publication preserves local manuscript work, pulls the
configured upstream, copies only referenced figures, and commits/pushes only
changed figure files. Plot generation is separate from publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(manuscript, *args):
    return subprocess.run(["git", "-C", str(manuscript), *args], check=True,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def pull_preserving_work(manuscript: Path):
    dirty = bool(git(manuscript, "status", "--porcelain"))
    if dirty:
        git(manuscript, "stash", "push", "--include-untracked", "-m", "Preserve local work before figure publication")
    try:
        git(manuscript, "pull", "--ff-only")
    finally:
        if dirty:
            git(manuscript, "stash", "pop", "--index")


def figure_manifest(source: Path | list[Path], manuscript: Path) -> dict:
    sources = [source] if isinstance(source, Path) else list(source)
    text = "\n".join(p.read_text() for p in manuscript.glob("*.tex"))
    refs = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)))
    rows = []
    for ref in refs:
        path = Path(ref)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"figure reference must stay inside the manuscript: {ref}")
        if not ref.startswith("figure/"):
            continue
        matches = [directory / path.name for directory in sources
                   if (directory / path.name).is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one authoritative figure for {path.name}, found {len(matches)}")
        master, target = matches[0], manuscript / path
        data = master.read_bytes()
        if path.suffix.lower() == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"invalid PNG master: {master}")
        checksum = hashlib.sha256(data).hexdigest()
        same = target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum
        rows.append({"reference": ref, "source": str(master), "sha256": checksum,
                     "copy_matches": same})
    return {"source_directory": str(sources[0]) if len(sources) == 1 else None,
            "source_directories": [str(directory) for directory in sources],
            "manuscript_directory": str(manuscript),
            "figures": rows, "all_copies_match": all(r["copy_matches"] for r in rows)}


def publish(source: Path | list[Path], manuscript: Path) -> dict:
    pull_preserving_work(manuscript)
    manifest = figure_manifest(source, manuscript)
    # Validate all sources before mutating any manuscript files.
    changed = []
    for row in manifest["figures"]:
        if not row["copy_matches"]:
            target = manuscript / row["reference"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["source"], target)
        result = subprocess.run(["git", "-C", str(manuscript), "status", "--porcelain", "--", row["reference"]],
                                check=True, text=True, stdout=subprocess.PIPE)
        if result.stdout.strip():
            changed.append(row["reference"])
    checked = figure_manifest(source, manuscript)
    if not checked["all_copies_match"]:
        raise ValueError("published copies differ from authoritative outputs")
    if changed:
        git(manuscript, "add", "--", *changed)
        git(manuscript, "commit", "--only", "-m", "Publish structural modeling figure copies", "--", *changed)
        git(manuscript, "push")
        checked["commit"] = git(manuscript, "rev-parse", "HEAD")
    first = source if isinstance(source, Path) else source[0]
    (first / "publication_manifest.json").write_text(json.dumps(checked, indent=2)+"\n")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source", type=Path, action="append",
                        help="Authoritative figure folder; may be repeated for multiple studies")
    parser.add_argument("--manuscript", type=Path, default=ROOT / "overleaf-candidacy")
    args = parser.parse_args()
    if args.source is None:
        args.source = [ROOT / f"exp_data/{study}/figures/candidacy"
                       for study in ("amide_synthetase", "nylon_hydrolase")]
    result = figure_manifest(args.source, args.manuscript) if args.check else publish(args.source, args.manuscript)
    print(json.dumps(result, indent=2))
    if args.check and not result["all_copies_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
