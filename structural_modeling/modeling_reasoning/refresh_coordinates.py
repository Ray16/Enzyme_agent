#!/usr/bin/env python3
"""Vendor the discovery agent's family-keyed reaction-coordinate cache into the
consumer snapshot `knowledge/reaction_coordinates.json`.

The discovery agent OWNS extraction and writes a family-keyed cache
(`tools/mechanism_coordinate_cache.json`). This tool pulls that cache across the
repo boundary as a plain JSON file (no cross-repo import) and MERGES it into the
consumer coordinate store, enriching without downgrading:

  - a mechanism class absent from the store is ADDED (its extracted steps, flagged
    `needs_structuring` because LLM-extracted species/roles must be normalized to
    the compiler's role vocabulary + atom selectors before geometry can resolve);
  - a mechanism class already present keeps its structured steps; only its
    `keys.families` are UNIONED (so newly-cached families widen transfer) and its
    provenance is refreshed.

Empty cache => no-op (safe to run any time). A version line is appended to
`knowledge/COORDINATE_VERSIONS.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWLEDGE = HERE / "knowledge" / "reaction_coordinates.json"
VERSIONS = HERE / "knowledge" / "COORDINATE_VERSIONS.jsonl"
DEFAULT_CACHE = Path(
    "/homes/rzhu/Enzyme_agent/enzyme_discovery_agent/data/mechanism_coordinate_cache.json"
)


def cache_to_mechanisms(cache: dict) -> dict:
    """Group a family-keyed cache into mechanism-class coordinates (mirrors the
    discovery agent's MechanismCache.export_mechanisms, done here as a pure transform)."""
    mechanisms: dict = {}
    for fam, rec in (cache.get("families") or {}).items():
        mech = rec.get("mechanism_class") or "unknown_mechanism"
        entry = mechanisms.setdefault(mech, {
            "keys": {"families": []},
            "steps": rec.get("steps"),
            "provenance": {"status": "literature_extracted", **(rec.get("provenance") or {})},
        })
        if fam not in entry["keys"]["families"]:
            entry["keys"]["families"].append(fam)
    return mechanisms


def merge(current: dict, exported: dict) -> tuple[dict, dict]:
    """Enriching merge; returns (merged_document, change_report)."""
    mechanisms = current.setdefault("mechanisms", {})
    added, families_added = [], {}
    for mech_key, ex in exported.items():
        ex_families = (ex.get("keys") or {}).get("families") or []
        if mech_key not in mechanisms:
            mechanisms[mech_key] = {
                "keys": {"families": list(ex_families)},
                "steps": ex.get("steps"),
                "provenance": {**(ex.get("provenance") or {}), "needs_structuring": True},
            }
            added.append(mech_key)
            continue
        # exists: union families only, preserve structured steps
        keys = mechanisms[mech_key].setdefault("keys", {})
        fams = keys.setdefault("families", [])
        new = [f for f in ex_families if f not in fams]
        if new:
            fams.extend(new)
            families_added[mech_key] = new
        prov = mechanisms[mech_key].setdefault("provenance", {})
        prov.setdefault("cache_refreshed_at", datetime.now(timezone.utc).isoformat())
    return current, {"added_mechanisms": added, "families_added": families_added}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="discovery-agent cache JSON")
    ap.add_argument("--knowledge", type=Path, default=KNOWLEDGE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.cache.exists():
        print(f"cache not present ({args.cache}); nothing to vendor (no-op)")
        return 0
    cache = json.loads(args.cache.read_text())
    exported = cache_to_mechanisms(cache)
    current = json.loads(args.knowledge.read_text())
    merged, report = merge(current, exported)

    print(f"cache families: {len(cache.get('families') or {})} | exported mechanisms: {len(exported)}")
    print(f"added mechanisms: {report['added_mechanisms']}")
    print(f"families unioned: {report['families_added']}")
    if args.dry_run:
        print("dry-run (not written)")
        return 0
    if not exported:
        print("empty cache -> no-op")
        return 0
    args.knowledge.write_text(json.dumps(merged, indent=2) + "\n")
    with open(VERSIONS, "a") as handle:
        handle.write(json.dumps({
            "vendored_at": datetime.now(timezone.utc).isoformat(),
            "cache": str(args.cache),
            "cache_families": len(cache.get("families") or {}),
            **report,
        }) + "\n")
    print(f"wrote {args.knowledge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
