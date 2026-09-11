"""Resolve every ChEBI id referenced in reviewed_63.tsv to SMILES + name.

Only ~405 distinct ids appear across all 12,456 reactions, so one parallel pass
against the EBI OLS4 API (the same source the agent uses) builds a complete local
cache (chebi_cache.json).  Idempotent: re-runs only fetch ids not already cached.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "chebi_cache.json"


def _fetch(chebi_id: str) -> tuple[str, dict]:
    numeric = chebi_id.replace("CHEBI:", "")
    iri = urllib.parse.quote_plus(f"http://purl.obolibrary.org/obo/CHEBI_{numeric}")
    url = f"https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms?iri={iri}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    out = {"smiles": None, "name": None}
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        for term in data.get("_embedded", {}).get("terms", []):
            ann = term.get("annotation", {})
            sm = ann.get("smiles_string", [])
            if sm and out["smiles"] is None:
                out["smiles"] = sm[0].strip()
            if term.get("label") and out["name"] is None:
                out["name"] = term.get("label")
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return chebi_id, out


def main() -> None:
    tsv = HERE / "reviewed_63.tsv"
    ids = sorted(set(re.findall(r"CHEBI:\d+", tsv.read_text())))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [c for c in ids if c not in cache]
    print(f"{len(ids)} distinct ChEBI ids; {len(todo)} to fetch")
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (cid, rec) in enumerate(pool.map(_fetch, todo), 1):
            cache[cid] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
    CACHE.write_text(json.dumps(cache, indent=0))
    n_smiles = sum(1 for v in cache.values() if v.get("smiles"))
    print(f"cached {len(cache)} ids; {n_smiles} have SMILES "
          f"({len(cache) - n_smiles} without)")


if __name__ == "__main__":
    sys.exit(main())
