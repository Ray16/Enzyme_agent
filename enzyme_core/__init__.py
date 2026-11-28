"""Shared, agent-agnostic infrastructure for discovery, structural modeling, and design.

Provides atomic persistence and POSIX locking (:mod:`enzyme_core.state_store`),
canonical run-directory layout (:mod:`enzyme_core.run_layout`), and versioned
handoff integrity contracts (:mod:`enzyme_core.handoff_contract`) without
introducing circular dependencies between the three agent packages.
"""

from __future__ import annotations

from .state_store import (
    atomic_write_json,
    atomic_write_text,
    clear_storage_fault,
    controller_lock,
    file_sha256,
    json_sha256,
    read_json,
    state_lock,
    storage_fault,
    write_tool_artifact,
)

__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "clear_storage_fault",
    "controller_lock",
    "file_sha256",
    "json_sha256",
    "read_json",
    "state_lock",
    "storage_fault",
    "write_tool_artifact",
]
