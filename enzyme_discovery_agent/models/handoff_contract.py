"""Compatibility re-export of :mod:`enzyme_core.handoff_contract`."""

from enzyme_core.handoff_contract import (  # noqa: F401
    HANDOFF_KIND,
    HANDOFF_VERSION,
    HASH_PROFILE,
    _reject_constant,
    _unique_object,
    adapt_legacy_handoff,
    canonical_bytes,
    digest,
    load_document,
    loads_strict_json,
    parse_unique_json_object,
    read_bundle_source,
    reject_json_constant,
    snapshot_payload,
    validate_handoff,
)
