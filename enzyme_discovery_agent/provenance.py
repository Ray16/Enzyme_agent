"""Run provenance: capture *what code, config, and environment produced a run*.

The agent already records rich per-candidate provenance (source tool, structural
homology, status transitions).  What was missing was run-level provenance: the
code revision, the resolved configuration, and the execution environment.  Without
those a completed run cannot be reproduced or trusted as evidence for a future
iteration.

This module writes two human- and machine-readable artifacts into every run dir:

* ``manifest.json`` -- the canonical, machine-readable record.  Written once at
  run start with ``status="running"`` and rewritten at finalize with the outcome,
  so an interrupted run is still identifiable (it keeps ``status="running"``).
* ``RUN_SUMMARY.md`` -- a human-readable digest of the same facts, written at
  finalize.

Design constraints:
* stdlib only (no import cost on a network-bound run);
* never records secrets (API keys are redacted from the config snapshot);
* never raises into the agent loop -- provenance capture is best-effort.
"""

from __future__ import annotations

import json
import logging
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("enzyme_agent")

MANIFEST_SCHEMA_VERSION = 1

# Package root is the directory containing this file; repo root is its parent.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent

# Config constants whose names contain any of these are redacted from snapshots.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

# Packages worth pinning in provenance: their versions change agent behaviour.
_TRACKED_PACKAGES = ("openai", "rdkit", "pandas", "numpy", "requests", "Bio")


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo root, returning stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_code_provenance() -> dict[str, Any]:
    """Capture the git revision and working-tree cleanliness of the code.

    ``dirty`` is True when uncommitted changes exist -- an essential caveat when
    a run is used as evidence, because the committed SHA alone would not describe
    the code that actually executed.
    """
    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return {"vcs": None, "note": "not a git repository or git unavailable"}
    status = _run_git("status", "--porcelain")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "vcs": "git",
        "commit": commit,
        "commit_short": commit[:12],
        "branch": branch,
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in (status or "").splitlines()][:50],
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def capture_environment() -> dict[str, Any]:
    """Capture the Python/OS environment and key package versions."""
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {
            name: _package_version(name) for name in _TRACKED_PACKAGES
        },
    }


def snapshot_config(config_module) -> dict[str, Any]:
    """Snapshot the resolved UPPER_CASE constants of the config module.

    Secrets are redacted by name so the manifest is safe to share as evidence.
    """
    snapshot: dict[str, Any] = {}
    for name in dir(config_module):
        if not name.isupper():
            continue
        value = getattr(config_module, name)
        if not isinstance(value, (str, int, float, bool, type(None), list, tuple)):
            continue
        if any(marker in name for marker in _SECRET_MARKERS):
            snapshot[name] = "<redacted>" if value else None
            continue
        snapshot[name] = list(value) if isinstance(value, tuple) else value
    return snapshot


def _now_iso() -> str:
    return datetime.now().isoformat()


def write_manifest_start(
    run_dir: Path,
    *,
    query: str,
    model: str,
    config_module,
    args: dict[str, Any] | None = None,
    substrate: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the initial ``manifest.json`` at run start and return it.

    Best-effort: any failure is logged, never raised.
    """
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": "running",
        "query": query,
        "model": model,
        "substrate": substrate or {},
        "args": args or {},
        "code": capture_code_provenance(),
        "environment": capture_environment(),
        "capabilities": capabilities or {},
        "config": snapshot_config(config_module),
        "timing": {"started_at": _now_iso(), "finished_at": None, "wall_clock_s": None},
        "result": None,
    }
    _write_manifest(run_dir, manifest)
    return manifest


def finalize_manifest(
    run_dir: Path,
    *,
    status: str = "completed",
    iterations_used: int | None = None,
    candidates_evaluated: int | None = None,
    tokens: dict[str, Any] | None = None,
    outcome: str | None = None,
    report_path: str | None = None,
) -> None:
    """Update ``manifest.json`` with the outcome and write ``RUN_SUMMARY.md``.

    Reads the start manifest so it can compute wall-clock time and preserve the
    captured code/environment/config.  Best-effort throughout.
    """
    manifest = _read_manifest(run_dir) or {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "timing": {"started_at": None},
    }
    finished = _now_iso()
    started = (manifest.get("timing") or {}).get("started_at")
    wall = _wall_clock_seconds(started, finished)
    manifest["status"] = status
    manifest.setdefault("timing", {})
    manifest["timing"]["finished_at"] = finished
    manifest["timing"]["wall_clock_s"] = wall
    manifest["result"] = {
        "iterations_used": iterations_used,
        "candidates_evaluated": candidates_evaluated,
        "tokens": tokens or {},
        "outcome": outcome,
        "report_path": report_path,
    }
    _write_manifest(run_dir, manifest)
    try:
        _write_run_summary(run_dir, manifest)
    except Exception as e:  # noqa: BLE001 - provenance must never break finalize
        logger.warning("Failed to write RUN_SUMMARY.md: %s", e)


def _wall_clock_seconds(started: str | None, finished: str) -> float | None:
    if not started:
        return None
    try:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(finished)
        return round((t1 - t0).total_seconds(), 2)
    except ValueError:
        return None


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    try:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
    except OSError as e:
        logger.warning("Failed to write manifest.json: %s", e)


def _read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_run_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Render the manifest as a human-readable Markdown digest."""
    code = manifest.get("code") or {}
    env = manifest.get("environment") or {}
    timing = manifest.get("timing") or {}
    result = manifest.get("result") or {}
    tokens = result.get("tokens") or {}
    substrate = manifest.get("substrate") or {}

    dirty = code.get("dirty")
    code_line = "not version-controlled"
    if code.get("commit_short"):
        flag = " (uncommitted changes present)" if dirty else ""
        code_line = f"`{code.get('commit_short')}` on `{code.get('branch')}`{flag}"

    lines = [
        f"# Run Summary — {manifest.get('run_id', '')}",
        "",
        f"- **Status:** {manifest.get('status', 'unknown')}",
        f"- **Query:** {manifest.get('query', '')}",
        f"- **Model:** {manifest.get('model', '')}",
    ]
    if substrate.get("smiles"):
        name = substrate.get("name") or "unnamed"
        lines.append(f"- **Substrate:** {name} (`{substrate['smiles']}`)")
    lines += [
        "",
        "## Provenance",
        f"- **Code:** {code_line}",
        f"- **Environment:** Python {env.get('python', '?')} on {env.get('hostname', '?')}",
        f"- **Started:** {timing.get('started_at', '?')}",
        f"- **Finished:** {timing.get('finished_at', '?')}",
        f"- **Wall clock:** {timing.get('wall_clock_s', '?')} s",
        "",
        "## Result",
        f"- **Outcome:** {result.get('outcome', '?')}",
        f"- **Iterations used:** {result.get('iterations_used', '?')}",
        f"- **Candidates evaluated:** {result.get('candidates_evaluated', '?')}",
        f"- **Total tokens:** {tokens.get('total_tokens', '?'):,}"
        if isinstance(tokens.get("total_tokens"), int)
        else f"- **Total tokens:** {tokens.get('total_tokens', '?')}",
    ]
    if result.get("report_path"):
        lines.append(f"- **Report:** `{result['report_path']}`")

    pkgs = env.get("packages") or {}
    pinned = ", ".join(f"{k}={v}" for k, v in pkgs.items() if v)
    if pinned:
        lines += ["", "## Key package versions", f"- {pinned}"]

    lines += [
        "",
        "---",
        "*Full machine-readable provenance in `manifest.json`; "
        "conversation transcript in `agent_state/transcript.json`; "
        "per-tool artifacts under `artifacts/`.*",
        "",
    ]
    (run_dir / "RUN_SUMMARY.md").write_text("\n".join(lines))
