#!/usr/bin/env python3
"""
Fetch protein sequences and structures for a list of UniProt IDs.

For each candidate:
  1. Fetch protein sequence from UniProt (with UniParc fallback).
  2. Search RCSB PDB for experimental structures.
  3. Download AlphaFold predicted structures from EBI.
  4. Write all_sequences.fasta and structure_sequence_summary.csv.

Usage:
    python fetch_structures.py <project_dir> --uniprots P31663 P0A6W9 P39641
    python fetch_structures.py <project_dir> --uniprots-file candidates.txt
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_url(url, timeout=30):
    """GET request, return bytes or None on error."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "StructureFetcher/1.0")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError):
            return None
    return None


def post_json(url, data, timeout=30):
    """POST JSON, return parsed response or None."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "StructureFetcher/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw or not raw.strip():
                return None
            return json.loads(raw)
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, json.JSONDecodeError):
        return None


def fetch_uniprot_sequence(uniprot_id):
    """Fetch protein sequence from UniProtKB, with UniParc fallback.
    Returns (header, sequence, source) or (None, None, None)."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    data = fetch_url(url)
    if data is not None:
        text = data.decode("utf-8", errors="replace").strip()
        if text and text.startswith(">"):
            lines = text.split("\n")
            header = lines[0]
            sequence = "".join(line.strip() for line in lines[1:])
            if sequence:
                return header, sequence, "uniprotkb"

    url = f"https://rest.uniprot.org/uniparc/search?query={uniprot_id}&format=fasta&size=1"
    data = fetch_url(url)
    if data is not None:
        text = data.decode("utf-8", errors="replace").strip()
        if text and text.startswith(">"):
            lines = text.split("\n")
            header = lines[0]
            sequence = "".join(line.strip() for line in lines[1:])
            if sequence:
                return header, sequence, "uniparc"

    return None, None, None


def search_pdb_by_uniprot(uniprot_id):
    """Search RCSB PDB for experimental structures by UniProt accession."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers."
                             "reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry",
    }
    result = post_json(url, query)
    if result and "result_set" in result:
        return [hit["identifier"] for hit in result["result_set"]]
    return []


def download_pdb(pdb_id, dest_dir):
    """Download a PDB file from RCSB."""
    path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    if os.path.exists(path):
        return path
    data = fetch_url(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    if data:
        with open(path, "wb") as f:
            f.write(data)
        return path
    # mmCIF fallback
    path_cif = os.path.join(dest_dir, f"{pdb_id}.cif")
    data = fetch_url(f"https://files.rcsb.org/download/{pdb_id}.cif")
    if data:
        with open(path_cif, "wb") as f:
            f.write(data)
        return path_cif
    return None


def _download_alphafold_by_accession(accession, dest_dir, save_as_id=None):
    """Download AlphaFold structure for a given accession.
    Returns (pdb_path, cif_path) or (None, None)."""
    save_id = save_as_id or accession
    data = fetch_url(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
    if data is None:
        return None, None
    try:
        entries = json.loads(data)
    except json.JSONDecodeError:
        return None, None
    if not entries:
        return None, None

    entry = entries[0] if isinstance(entries, list) else entries
    pdb_path = cif_path = None

    pdb_url = entry.get("pdbUrl")
    if pdb_url:
        fname = f"AF-{save_id}-F1-model_v4.pdb"
        pdb_path = os.path.join(dest_dir, fname)
        if not os.path.exists(pdb_path):
            pdb_data = fetch_url(pdb_url)
            if pdb_data:
                with open(pdb_path, "wb") as f:
                    f.write(pdb_data)
            else:
                pdb_path = None

    cif_url = entry.get("cifUrl")
    if cif_url:
        fname = f"AF-{save_id}-F1-model_v4.cif"
        cif_path = os.path.join(dest_dir, fname)
        if not os.path.exists(cif_path):
            cif_data = fetch_url(cif_url)
            if cif_data:
                with open(cif_path, "wb") as f:
                    f.write(cif_data)
            else:
                cif_path = None

    return pdb_path, cif_path


def get_alphafold_structure(uniprot_id, dest_dir):
    """Download AlphaFold structure, trying alternate accessions if needed."""
    pdb_path, cif_path = _download_alphafold_by_accession(uniprot_id, dest_dir)
    if pdb_path or cif_path:
        return pdb_path, cif_path

    # Try UniParc cross-references
    url = f"https://rest.uniprot.org/uniparc/search?query={uniprot_id}&format=json&size=1"
    data = fetch_url(url)
    if data:
        try:
            result = json.loads(data)
            for entry in result.get("results", []):
                for xref in entry.get("uniParcCrossReferences", []):
                    db = xref.get("database", "")
                    acc = xref.get("id", "")
                    active = xref.get("active", False)
                    if db in ("UniProtKB/Swiss-Prot", "UniProtKB/TrEMBL") \
                            and active and acc != uniprot_id:
                        pdb_path, cif_path = _download_alphafold_by_accession(
                            acc, dest_dir, save_as_id=uniprot_id)
                        if pdb_path or cif_path:
                            return pdb_path, cif_path
                        time.sleep(0.2)
        except json.JSONDecodeError:
            pass

    return None, None


_print_lock = threading.Lock()
_counter = [0]


def process_one(uid, pdb_dir, af_dir, total):
    """Fetch sequence, PDB structures, and AlphaFold prediction for one UID."""
    header, sequence, seq_source = fetch_uniprot_sequence(uid)
    if not (header and sequence):
        seq_source = "none"

    pdb_ids = search_pdb_by_uniprot(uid)
    pdb_files = []
    for pdb_id in pdb_ids:
        path = download_pdb(pdb_id, pdb_dir)
        if path:
            pdb_files.append(os.path.basename(path))

    af_pdb, af_cif = get_alphafold_structure(uid, af_dir)

    organism = ""
    if header:
        m = re.search(r"OS=(.+?)\s+OX=", header)
        if m:
            organism = m.group(1).strip()

    # Extract gene name from FASTA header: "|GENE_NAME "
    gene_name = uid
    if header:
        m = re.search(r"\|(\S+)\s", header)
        if m:
            gene_name = m.group(1).split("_")[0]

    result = {
        "gene_name": gene_name,
        "uniprot_id": uid,
        "organism": organism,
        "seq_source": seq_source or "none",
        "sequence_length": len(sequence) if sequence else 0,
        "sequence": sequence or "none",
        "header": header,
        "pdb_ids": ";".join(pdb_ids) if pdb_ids else "none",
        "pdb_files": ";".join(pdb_files) if pdb_files else "none",
        "alphafold_pdb": os.path.basename(af_pdb) if af_pdb else "none",
        "alphafold_cif": os.path.basename(af_cif) if af_cif else "none",
    }

    with _print_lock:
        _counter[0] += 1
        seq_st = f"{result['sequence_length']} aa" if sequence else "FAILED"
        pdb_st = f"PDB: {len(pdb_ids)}" if pdb_ids else "PDB: 0"
        af_st = "AF: yes" if (af_pdb or af_cif) else "AF: no"
        print(f"  [{_counter[0]}/{total}] {uid}: {seq_st} | {pdb_st} | {af_st}",
              flush=True)

    return result


def fetch_all(project_dir, uniprots, workers=8):
    """Fetch structures and sequences for all UniProt IDs.

    Creates project_dir with:
      - alphafold_structures/
      - pdb_structures/
      - all_sequences.fasta
      - structure_sequence_summary.csv
    """
    os.makedirs(project_dir, exist_ok=True)
    pdb_dir = os.path.join(project_dir, "pdb_structures")
    af_dir = os.path.join(project_dir, "alphafold_structures")
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(af_dir, exist_ok=True)

    total = len(uniprots)
    print(f"Fetching structures for {total} candidates ({workers} threads)...")

    _counter[0] = 0
    results_map = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, uid, pdb_dir, af_dir, total): uid
            for uid in uniprots
        }
        for future in as_completed(futures):
            uid = futures[future]
            try:
                results_map[uid] = future.result()
            except Exception as e:
                print(f"  ERROR fetching {uid}: {e}", flush=True)
                results_map[uid] = {
                    "gene_name": uid, "uniprot_id": uid, "organism": "",
                    "seq_source": "none", "sequence_length": 0,
                    "sequence": "none", "header": None,
                    "pdb_ids": "none", "pdb_files": "none",
                    "alphafold_pdb": "none", "alphafold_cif": "none",
                }

    # Preserve input order
    results = [results_map[uid] for uid in uniprots]

    # Write FASTA
    fasta_path = os.path.join(project_dir, "all_sequences.fasta")
    with open(fasta_path, "w") as f:
        for r in results:
            if r["sequence"] != "none" and r["header"]:
                uid = r["uniprot_id"]
                gene = r["gene_name"]
                f.write(f">{uid} {gene} {r['header'][1:]}\n")
                seq = r["sequence"]
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j + 80] + "\n")

    # Write summary CSV
    csv_path = os.path.join(project_dir, "structure_sequence_summary.csv")
    fields = [
        "gene_name", "uniprot_id", "organism", "seq_source",
        "sequence_length", "sequence", "pdb_ids", "pdb_files",
        "alphafold_pdb", "alphafold_cif",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Summary
    n_seq = sum(1 for r in results if r["sequence"] != "none")
    n_pdb = sum(1 for r in results if r["pdb_ids"] != "none")
    n_af = sum(1 for r in results if r["alphafold_pdb"] != "none")
    print(f"\nFetch complete: {n_seq}/{total} sequences, "
          f"{n_pdb} with PDB, {n_af} with AlphaFold")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch structures and sequences for UniProt IDs")
    parser.add_argument("project_dir",
                        help="Project directory to create/populate")
    parser.add_argument("--uniprots", nargs="+",
                        help="UniProt IDs to fetch")
    parser.add_argument("--uniprots-file",
                        help="File with one UniProt ID per line")
    parser.add_argument("--workers", type=int, default=8,
                        help="Download threads (default: 8)")
    args = parser.parse_args()

    uniprots = []
    if args.uniprots:
        uniprots = args.uniprots
    elif args.uniprots_file:
        with open(args.uniprots_file) as f:
            uniprots = [line.strip() for line in f if line.strip()]
    else:
        sys.exit("ERROR: Provide --uniprots or --uniprots-file")

    uniprots = list(dict.fromkeys(uniprots))  # deduplicate
    fetch_all(args.project_dir, uniprots, args.workers)


if __name__ == "__main__":
    main()
