#!/usr/bin/env python3
"""Seed-and-expand candidate discovery over the ESM Metagenomic Atlas.

Strategy (not a bulk download): take a SEED structure (a top candidate's model),
find its structural neighbors in the Atlas via the public Foldseek server, filter,
and return them as new candidates. Exploration only ever touches the guided
neighborhood of good candidates, so it stays a small, storage-light, on-demand
subspace that grows as seeds improve.

Route: the ESM Atlas REST API (api.esmatlas.com) is gated (HTTP 403 for search and
fetch), so queries go through the Foldseek public server, which hosts the Atlas as
target DB ``ESM30``. This is a shared public service: keep query volume modest and
poll politely.

The returned neighbors are unannotated metagenomic sequences -> hand them to the
family-assignment + reaction-coordinate-transfer + developability + structural-proxy
evaluator; that is what makes dark-matter hits actionable.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

# The Foldseek public server hosts the Atlas as the standard (non-motif) structure-search
# index MGnify-ESM30; database[] must carry its PATH, not the display name.
FOLDSEEK_API = "https://search.foldseek.com/api"
ATLAS_DB = "mgnify_esm30"


def _post_multipart(url: str, fields: list[tuple[str, str]], files: dict) -> dict:
    """POST multipart/form-data. ``fields`` is a list of (name, value) tuples so a
    repeated key like ``database[]`` can appear more than once."""
    boundary = "----esmatlasexpand"
    parts: list[bytes] = []
    for name, value in fields:
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode())
    for name, (fname, content) in files.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n"
                      "Content-Type: application/octet-stream\r\n\r\n").encode())
        parts.append(content if isinstance(content, bytes) else content.encode())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def submit(pdb_text: str, databases=(ATLAS_DB,), mode: str = "3diaa") -> str:
    fields = [("mode", mode)] + [("database[]", d) for d in databases]
    resp = _post_multipart(f"{FOLDSEEK_API}/ticket", fields, {"q": ("seed.pdb", pdb_text)})
    ticket = resp.get("id")
    if not ticket:
        raise RuntimeError(f"no ticket id in response: {resp}")
    return ticket


def ticket_status(ticket: str) -> str:
    """One status probe for a ticket (COMPLETE / ERROR / PENDING / RUNNING)."""
    return _get(f"{FOLDSEEK_API}/ticket/{ticket}").get("status")


def wait(ticket: str, poll_s: float = 3.0, timeout_s: float = 300.0) -> str:
    start = time.time()
    while time.time() - start < timeout_s:
        status = ticket_status(ticket)
        if status in ("COMPLETE", "ERROR"):
            return status
        time.sleep(poll_s)
    return "TIMEOUT"


def results(ticket: str) -> list[dict]:
    out = _get(f"{FOLDSEEK_API}/result/{ticket}/0")
    hits: list[dict] = []
    for block in out.get("results", []):
        for aln in block.get("alignments", []) or []:
            # foldseek server nests alignments one level deeper in some versions
            records = aln if isinstance(aln, list) else [aln]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                tseq = rec.get("tSeq") or rec.get("tseq") or (rec.get("taln") or "").replace("-", "")
                hits.append({
                    "db": block.get("db"),
                    "target": rec.get("target"),
                    "prob": rec.get("prob"),
                    "eval": rec.get("eval"),
                    "seqId": rec.get("seqId"),
                    "alnLength": rec.get("alnLength"),
                    "tseq": tseq,
                })
    return hits


def filter_hits(hits: list[dict], *, max_hits: int = 25, min_prob: float = 0.5) -> list[dict]:
    """Sort by probability, drop below-threshold + duplicate targets, cap at max_hits."""
    seen, kept = set(), []
    for h in sorted(hits, key=lambda x: -(x.get("prob") or 0)):
        t = h.get("target")
        if not t or t in seen:
            continue
        if (h.get("prob") or 0) < min_prob:
            continue
        seen.add(t)
        kept.append(h)
        if len(kept) >= max_hits:
            break
    return kept


def expand(pdb_text: str, *, max_hits: int = 25, min_prob: float = 0.5,
           timeout_s: float = 300.0) -> list[dict]:
    ticket = submit(pdb_text)
    status = wait(ticket, timeout_s=timeout_s)
    if status != "COMPLETE":
        raise RuntimeError(f"foldseek ticket {ticket} status={status}")
    return filter_hits(results(ticket), max_hits=max_hits, min_prob=min_prob)


# --------------------------------------------------------------------------------
# Async batch expansion: many seeds against one shared public server.
#
# Seeds are embarrassingly parallel, but the Foldseek server is a shared public
# resource, so we do NOT fire every ticket at once. A bounded-inflight scheduler
# keeps at most ``max_inflight`` tickets outstanding, polls the whole live batch in
# one round per interval (one probe per ticket, then one polite sleep), and refills
# from the pending queue as tickets finish. Transport calls are injected so the
# scheduler is unit-testable without the network.
# --------------------------------------------------------------------------------
def expand_many(
    seeds: list[tuple[str, str]],
    *,
    max_hits: int = 25,
    min_prob: float = 0.5,
    max_inflight: int = 4,
    poll_s: float = 3.0,
    timeout_s: float = 300.0,
    _submit=submit,
    _status=ticket_status,
    _results=results,
) -> dict[str, dict]:
    """Expand many seeds concurrently under a polite in-flight cap.

    ``seeds`` is a list of ``(seed_id, pdb_text)``. Returns ``{seed_id: record}``
    where record is ``{"status", "ticket", "neighbors"|"error"}``. Never raises for
    a single failed seed -- failures are captured per seed so one bad seed does not
    sink the batch.
    """
    pending = list(seeds)
    inflight: dict[str, dict] = {}   # ticket -> {seed_id, deadline}
    out: dict[str, dict] = {}
    now = time.monotonic

    while pending or inflight:
        # Fill the in-flight window from the pending queue.
        while pending and len(inflight) < max_inflight:
            seed_id, pdb_text = pending.pop(0)
            try:
                ticket = _submit(pdb_text)
                inflight[ticket] = {"seed_id": seed_id, "deadline": now() + timeout_s}
            except Exception as exc:  # noqa: BLE001 - isolate per-seed submit failures
                out[seed_id] = {"status": "SUBMIT_ERROR", "error": str(exc)}

        if not inflight:
            continue

        # One poll round over the whole live batch.
        for ticket, meta in list(inflight.items()):
            seed_id = meta["seed_id"]
            try:
                status = _status(ticket)
            except Exception as exc:  # noqa: BLE001
                out[seed_id] = {"status": "POLL_ERROR", "ticket": ticket, "error": str(exc)}
                inflight.pop(ticket)
                continue
            if status == "COMPLETE":
                try:
                    hits = filter_hits(_results(ticket), max_hits=max_hits, min_prob=min_prob)
                    out[seed_id] = {"status": "COMPLETE", "ticket": ticket, "neighbors": hits}
                except Exception as exc:  # noqa: BLE001
                    out[seed_id] = {"status": "RESULT_ERROR", "ticket": ticket, "error": str(exc)}
                inflight.pop(ticket)
            elif status == "ERROR":
                out[seed_id] = {"status": "ERROR", "ticket": ticket, "error": "server ERROR"}
                inflight.pop(ticket)
            elif now() > meta["deadline"]:
                out[seed_id] = {"status": "TIMEOUT", "ticket": ticket, "error": "poll timeout"}
                inflight.pop(ticket)

        if inflight:
            time.sleep(poll_s)  # polite: one sleep per round, not per ticket

    return out


def _print_neighbors(label: str, neighbors: list[dict]) -> None:
    print(f"ESM Atlas neighbors (ESM30) for {label}: {len(neighbors)}")
    for h in neighbors[:15]:
        seqlen = len(h.get("tseq") or "")
        print(f"  {h['target']:28} prob={h.get('prob')} seqId={h.get('seqId')} "
              f"alnLen={h.get('alnLength')} seqlen={seqlen}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seed-pdb", help="single seed structure (.pdb) to expand from")
    src.add_argument("--seeds-dir", help="directory of seed .pdb files to expand as a batch")
    ap.add_argument("--max-hits", type=int, default=25)
    ap.add_argument("--min-prob", type=float, default=0.5)
    ap.add_argument("--max-inflight", type=int, default=4,
                    help="max tickets outstanding at once (polite cap for the shared server)")
    ap.add_argument("--out", help="write neighbors JSON here")
    args = ap.parse_args(argv)

    if args.seed_pdb:
        neighbors = expand(open(args.seed_pdb).read(),
                           max_hits=args.max_hits, min_prob=args.min_prob)
        _print_neighbors(args.seed_pdb, neighbors)
        payload = neighbors
    else:
        paths = sorted(p for p in os.listdir(args.seeds_dir) if p.endswith(".pdb"))
        seeds = [(p, open(os.path.join(args.seeds_dir, p)).read()) for p in paths]
        print(f"batch: expanding {len(seeds)} seeds (max_inflight={args.max_inflight})")
        results_by_seed = expand_many(seeds, max_hits=args.max_hits, min_prob=args.min_prob,
                                      max_inflight=args.max_inflight)
        for seed_id, rec in results_by_seed.items():
            if rec["status"] == "COMPLETE":
                _print_neighbors(seed_id, rec["neighbors"])
            else:
                print(f"  {seed_id}: {rec['status']} ({rec.get('error')})")
        payload = {sid: rec for sid, rec in results_by_seed.items()}

    if args.out:
        json.dump(payload, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
