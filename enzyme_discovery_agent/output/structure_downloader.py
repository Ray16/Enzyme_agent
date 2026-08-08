"""Download PDB/AlphaFold structures for evaluated candidates."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from enzyme_discovery_agent.models.candidate_registry import UNIPROT_ACC_RE
from enzyme_discovery_agent.tools._http import _get_session

logger = logging.getLogger("enzyme_agent")


def download_candidate_structures(
    run_dir: Path,
    report_text: str,
    artifacts_dir: Path,
    log_fn=None,
) -> None:
    """Download PDB/AlphaFold structures for evaluated candidates."""
    evals_path = artifacts_dir / "candidates" / "all_evaluations.json"
    if not evals_path.exists():
        return

    try:
        with open(evals_path) as f:
            all_evals = json.load(f)
    except Exception:
        return

    eval_lookup: dict[str, dict] = {}
    for ev in all_evals:
        data = ev.get("data") or ev.get("result", {}).get("data", {})
        if not data:
            continue
        acc = data.get("accession", "")
        if acc:
            eval_lookup[acc] = data

    if not eval_lookup:
        return

    seen = set()
    ranked_accessions = []
    for match in UNIPROT_ACC_RE.finditer(report_text):
        acc = match.group(1)
        if acc in eval_lookup and acc not in seen:
            seen.add(acc)
            ranked_accessions.append(acc)
    for acc in eval_lookup:
        if acc not in seen:
            ranked_accessions.append(acc)

    out_dir = run_dir / "top_candidates"
    out_dir.mkdir(exist_ok=True)

    session = _get_session()
    source_rows: list[dict] = []

    for rank, acc in enumerate(ranked_accessions, 1):
        data = eval_lookup[acc]
        gene_names = data.get("gene_names", [])
        protein_name = data.get("protein_name", "")
        if gene_names:
            filename = gene_names[0]
        elif protein_name:
            filename = protein_name.split()[0].rstrip(",;:")
        else:
            filename = acc
        filename = re.sub(r'[^\w\-.]', '_', filename)

        struct = data.get("structural_data", {})
        pdb_ids = struct.get("pdb_ids", [])
        has_alphafold = struct.get("has_alphafold", False)

        source = "none"
        downloaded = False

        if pdb_ids:
            pdb_id = pdb_ids[0]
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code == 200:
                    out_path = out_dir / f"{filename}.pdb"
                    out_path.write_bytes(resp.content)
                    source = f"PDB:{pdb_id}"
                    downloaded = True
                    if log_fn:
                        log_fn(f"  Downloaded {out_path.name} (PDB: {pdb_id})")
            except Exception as e:
                logger.warning("Failed to download PDB %s: %s", pdb_id, e)

        if not downloaded and has_alphafold:
            try:
                api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
                api_resp = session.get(
                    api_url, timeout=10,
                    headers={"Accept": "application/json"},
                )
                if api_resp.status_code == 200:
                    af_data = json.loads(api_resp.content)
                    if isinstance(af_data, list) and af_data:
                        pdb_url = af_data[0].get("pdbUrl", "")
                        if pdb_url:
                            resp = session.get(pdb_url, timeout=15)
                            if resp.status_code == 200:
                                out_path = out_dir / f"{filename}.pdb"
                                out_path.write_bytes(resp.content)
                                af_id = pdb_url.split("/")[-1].replace(".pdb", "")
                                source = f"AlphaFold:{af_id}"
                                downloaded = True
                                if log_fn:
                                    log_fn(f"  Downloaded {out_path.name} (AlphaFold: {acc})")
            except Exception as e:
                logger.warning("Failed to download AlphaFold %s: %s", acc, e)

        source_rows.append({
            "rank": rank,
            "enzyme_name": filename,
            "accession": acc,
            "source": source,
            "pdb_ids": ";".join(pdb_ids) if pdb_ids else "",
            "alphafold": "Y" if has_alphafold else "N",
        })

    csv_path = out_dir / "source.csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "enzyme_name", "accession", "source",
                            "pdb_ids", "alphafold"],
            )
            writer.writeheader()
            writer.writerows(source_rows)
        if log_fn:
            log_fn(f"  Structure sources: {csv_path}")
    except Exception as e:
        logger.warning("Failed to write source.csv: %s", e)

    n_downloaded = sum(1 for r in source_rows if r["source"] != "none")
    if log_fn:
        log_fn(f"  Structures: {n_downloaded}/{len(source_rows)} downloaded to {out_dir}")
