#!/usr/bin/env bash
# Re-vendor the literature knowledge snapshot from the reasoning repo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-/homes/rzhu/kg-memory-system/reasoning/results/v2/cofolding_batch/reasoning_synthesis.json}"
if [[ ! -f "$SRC" ]]; then
  echo "source synthesis not found: $SRC" >&2
  exit 1
fi
cp "$SRC" "$HERE/cofolding_rules.json"
sha256sum "$SRC" | awk '{print $1"  reasoning_synthesis.json"}' > "$HERE/SOURCE.sha256"
echo "re-vendored from $SRC"
cat "$HERE/SOURCE.sha256"
