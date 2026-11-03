"""Legacy family-indexed mechanism records for explicit offline inspection.

These records are not candidate-specific evidence. Production extraction never
uses them to fill missing reaction steps. A caller must choose a store explicitly.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CACHE_SCHEMA_VERSION = "reaction-coordinate-cache-v1"
_CACHE_LOCK = threading.RLock()


class MechanismCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if self.path.exists():
            self.doc = json.loads(self.path.read_text())
        else:
            self.doc = {"schema_version": CACHE_SCHEMA_VERSION, "families": {}}

    @contextmanager
    def _locked(self):
        """Serialize read-modify-write across threads and local processes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with _CACHE_LOCK, lock_path.open("a+") as lock_handle:
            try:
                import fcntl
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                fcntl = None
            try:
                if self.path.exists():
                    try:
                        self.doc = json.loads(self.path.read_text())
                    except (OSError, json.JSONDecodeError):
                        self.doc = {"schema_version": CACHE_SCHEMA_VERSION, "families": {}}
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    # ---- retrieve ----
    def get(self, family_ids) -> dict | None:
        """Return the cached coordinate for the first matching family, or None."""
        with self._locked():
            families = self.doc.get("families") or {}
            for fam in family_ids or []:
                if fam in families:
                    return families[fam]
        return None

    def has(self, family_ids) -> bool:
        return self.get(family_ids) is not None

    # ---- update ----
    def put(
        self,
        family_ids,
        *,
        mechanism_class: str | None,
        steps: list,
        provenance: dict | None = None,
        grounding_score: float | None = None,
        force: bool = False,
    ) -> bool:
        """Write/refresh the coordinate for each family id. Returns True if written.

        Skips (keeps the existing entry) when a comparably or better grounded
        coordinate already exists, unless ``force``."""
        if not family_ids or not steps:
            return False
        wrote = False
        with self._locked():
            families = self.doc.setdefault("families", {})
            record = {
                "mechanism_class": mechanism_class,
                "steps": steps,
                "provenance": provenance or {},
                "grounding_score": grounding_score,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            for fam in family_ids:
                existing = families.get(fam)
                if existing and not force:
                    prev = existing.get("grounding_score")
                    if (isinstance(prev, (int, float))
                            and isinstance(grounding_score, (int, float))
                            and prev > grounding_score):
                        continue
                new = dict(record)
                new["version"] = (existing.get("version", 0) + 1) if existing else 1
                if existing:
                    history = list(existing.get("history") or [])
                    history.append({k: v for k, v in existing.items() if k != "history"})
                    new["history"] = history[-5:]
                families[fam] = new
                wrote = True
            if wrote:
                self._save_unlocked()
        return wrote

    def _save_unlocked(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(self.doc, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def save(self) -> None:
        """Persist the latest on-disk document.

        Normal updates should use :meth:`put`; retaining a public save method is
        useful for compatibility, but it must not overwrite another instance's
        more recent update with stale in-memory state.
        """
        with self._locked():
            self._save_unlocked()

    # ---- export for the modeling_reasoning consumer snapshot ----
    def export_mechanisms(self) -> dict:
        """Emit the family-keyed store in the modeling_reasoning coordinate shape.

        Groups families by mechanism_class so the consumer's ``reaction_coordinates.json``
        (mechanisms keyed by class, each listing the InterPro ``families`` that use it)
        can be vendored directly from the cache."""
        mechanisms: dict = {}
        with self._locked():
            for fam, rec in (self.doc.get("families") or {}).items():
                mech = rec.get("mechanism_class") or "unknown_mechanism"
                entry = mechanisms.setdefault(mech, {
                    "keys": {"families": []},
                    "steps": rec.get("steps"),
                    "provenance": {
                        "status": "literature_extracted",
                        **(rec.get("provenance") or {}),
                    },
                })
                if fam not in entry["keys"]["families"]:
                    entry["keys"]["families"].append(fam)
        return {
            "schema_version": "reaction-coordinate-v2",
            "source": "mechanism_coordinate_cache",
            "mechanisms": mechanisms,
        }
