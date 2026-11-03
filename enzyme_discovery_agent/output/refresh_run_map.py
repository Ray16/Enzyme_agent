"""Refresh a run's ESM map; embedding mode must run under the GPU reservation gate."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

from enzyme_discovery_agent.models import embedding_store as es
from enzyme_discovery_agent.models.candidate_registry import load_candidate_registry, UNIPROT_ACC_RE
from enzyme_discovery_agent.output.evolving_map import update_candidate_map
from enzyme_discovery_agent.state_store import atomic_write_json
from enzyme_discovery_agent.tools._http import request_with_retry


def fetch_sequences(run_dir):
    root = Path(run_dir)
    candidates = load_candidate_registry(root, strict=True).get("candidates", {})
    cache = es.load_sequences(root)
    missing = [acc for acc in candidates if acc not in cache and UNIPROT_ACC_RE.fullmatch(acc)]
    chunks = [missing[i:i + 100] for i in range(0, len(missing), 100)]
    def fetch(chunk):
        url = "https://rest.uniprot.org/uniprotkb/accessions?" + urllib.parse.urlencode({"accessions": ",".join(chunk), "format": "fasta"})
        text = request_with_retry(urllib.request.Request(url), timeout=120).decode()
        result, acc, sequence = {}, None, []
        for line in text.splitlines() + [">"]:
            if line.startswith(">"):
                if acc and sequence:
                    result[acc] = "".join(sequence)
                fields = line[1:].split("|")
                acc = fields[1] if len(fields) > 1 else None
                sequence = []
            else:
                sequence.append(line.strip())
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, chunks):
            es.cache_sequences(root, result)
            cache.update(result)
    atomic_write_json(root / "artifacts" / "candidate_map" / "sequence_coverage.json", {
        "registered": len(candidates), "resolved": len(cache),
        "unresolved": [acc for acc in candidates if acc not in cache],
    })
    return cache


def refresh(run_dir, embed=False, model="facebook/esm2_t33_650M_UR50D"):
    root = Path(run_dir)
    if embed:
        from enzyme_discovery_agent.tools.esm_embed_worker import embed_sequences
        cache = fetch_sequences(root)
        candidates = load_candidate_registry(root, strict=True).get("candidates", {})
        pending = [acc for acc in es.pending_accessions(root, candidates) if acc in cache]
        if pending:
            print(f"Embedding {len(pending)} new sequences with {model}", flush=True)
            vectors = embed_sequences(model, [cache[acc] for acc in pending])
            es.append_vectors(root, model, [(acc, es.seq_sha1(cache[acc]), vectors[i]) for i, acc in enumerate(pending)])
    return update_candidate_map(root, log_fn=lambda message: print(message, flush=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--watch", action="store_true", help="Refresh existing embeddings as candidate state changes")
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.embed and args.watch:
        parser.error("Embedding batches must finish and release their GPU; use a CPU-only watcher for live rendering")
    while True:
        result = refresh(args.run_dir, args.embed, args.model)
        print(result or "No candidates available", flush=True)
        if not args.watch:
            return
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        if manifest.get("status") != "running":
            return
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
