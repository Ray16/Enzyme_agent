"""Review expression-host sources for an explicitly supplied candidate selection."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

from enzyme_discovery_agent.models.expression import assess_expression
from .catalog import Sources, encoded, write, identity


def collect(campaign_dir):
    root = Path(campaign_dir).resolve(); config = json.loads((root / "sources.json").read_text())
    source = config["host_review_selection"]
    files = Sources(config["project_root"])
    candidates = [r["accession"] for r in files.rows(source["ranking"]) if r.get("rank")]
    saved = root / "catalog/host_source_snapshots"; saved.mkdir(parents=True, exist_ok=True)
    receipts, cache, lock = [], {}, threading.Lock()
    def fetch(url):
        with lock:
            if url in cache: return cache[url]
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=20) as response:
                raw = response.read(); status = response.status
        except Exception as exc:
            with lock:
                receipts.append({"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                 "error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)})
            raise
        parsed = json.loads(raw); checksum = hashlib.sha256(raw).hexdigest(); path = saved / (checksum + ".json")
        write(path, raw)
        with lock:
            cache[url] = parsed
            receipts.append({"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                             "snapshot": str(path), "sha256": checksum, "http_status": status})
        return parsed
    def inspect(accession):
        try:
            entry = fetch(f"https://rest.uniprot.org/uniprotkb/{accession}.json")
            pdbs = [r["id"] for r in entry.get("uniProtKBCrossReferences", []) if r.get("database") == "PDB"]
            result = assess_expression(accession, entry, pdbs, fetch)
            return {"protein_id": identity(accession, "host_review", accession), "accession": accession,
                    "assessment": result, "sequence_sha256": hashlib.sha256(entry.get("sequence", {}).get("value", "").encode()).hexdigest(),
                    "entry_version": entry.get("entryAudit", {}).get("entryVersion")}
        except Exception as exc:
            return {"protein_id": identity(accession, "host_review", accession), "accession": accession,
                    "assessment": {"preferred_host_assessment": {"level": "unknown", "lookup_failures": [{"error_type": type(exc).__name__}]}}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        reviews = list(pool.map(inspect, candidates))
    result = {"reviewed_at": datetime.now(timezone.utc).isoformat(), "selection_source": files.ref(source["ranking"]),
              "reviews": reviews, "retrievals": sorted(receipts, key=lambda r: r["url"]),
              "limitations": ["PDB expression evidence applies to the deposited construct.",
                             "No source establishes activity on the campaign's target substrates merely through expression."]}
    path = files.path(config["host_review_file"]); write(path, result)
    return {"reviewed": len(reviews), "requests": len(receipts), "output": str(path)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir")
    print(json.dumps(collect(parser.parse_args().campaign_dir), indent=2))
