"""Runtime capability detection and run directory creation."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


def check_pymol_available() -> bool:
    """Check if PyMOL is available for pocket analysis."""
    try:
        import pymol  # noqa: F401
        return True
    except ImportError:
        return False


def check_boltz2_available() -> bool:
    """Check if Boltz-2 conda env exists."""
    return shutil.which("boltz") is not None


def check_rdkit_available() -> bool:
    """Check if RDKit is available."""
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        return False


OPTIONAL_CAPABILITIES: dict[str, bool] = {}


def init_capabilities() -> dict[str, bool]:
    """Detect optional capabilities at startup."""
    global OPTIONAL_CAPABILITIES
    OPTIONAL_CAPABILITIES = {
        "rdkit": check_rdkit_available(),
        "pocket_analysis": check_pymol_available(),
        "boltz2": check_boltz2_available(),
    }
    return OPTIONAL_CAPABILITIES


def _sanitize_query_for_dirname(query: str, max_len: int = 80) -> str:
    """Sanitize a query string for use in a directory name."""
    name = re.sub(r'\s+', '_', query.strip())
    name = re.sub(r'[/\\?*:|"<>,;!@#$%^&()\'`~\[\]{}+=]', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > max_len:
        name = name[:max_len].rstrip('_')
    return name


def create_run_dir(base_dir: Path, query: str = "") -> Path:
    """Create a timestamped run directory, optionally with query slug."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if query:
        slug = _sanitize_query_for_dirname(query)
        if slug:
            run_id = f"{run_id}_{slug}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("artifacts/literature", "artifacts/database",
                    "artifacts/candidates", "agent_state"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir
