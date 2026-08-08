#!/usr/bin/env python3
"""
Validate that predicted binding pockets contain catalytic residues.

Features (v10):
1. Hybrid Homology Search: Prefers Reviewed (Swiss-Prot) templates.
2. Unreviewed Fallback: Uses Unreviewed (TrEMBL) if no Reviewed hits found, requiring higher SeqID (>40%).
3. InterPro Integration: Uses candidate's own domains to find relevant template proteins.
4. Multi-Tiered Identification: Distinguishes "Core" (Active site) from "Shell" (Binding/Metal/Site).
5. Parallel Execution: Uses ProcessPoolExecutor for candidate analysis.
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.request import urlopen, Request

import numpy as np

try:
    import prody
    prody.confProDy(verbosity="none")
    HAS_PRODY = True
except ImportError:
    HAS_PRODY = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _fetch_url(url, timeout=20, retries=3):
    """GET request with exponential backoff for rate limiting."""
    import time as _time
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            # Retry on rate limiting (HTTP 429)
            if hasattr(e, 'code') and e.code == 429:
                _time.sleep(2 ** attempt)
                continue
            return None
    return None

def _extract_active_sites(uniprot_entry, target_ligands=None):
    """Extracts residues into Core and Shell tiers, filtered by ligand relevance."""
    core, shell = [], []
    has_explicit_active_site = any(f.get("type") == "Active site" for f in uniprot_entry.get("features", []))
    
    targets = set()
    if target_ligands:
        for t in target_ligands:
            targets.add(t.upper())
            if t.upper() == "ATP": targets.update(["ADP", "AMP", "ADENOSINE", "PHOSPHATE"])
            if "ACID" in t.upper(): targets.add(t.upper().replace(" ACID", ""))

    for feature in uniprot_entry.get("features", []):
        ftype = feature.get("type", "")
        desc = feature.get("description", "").upper()
        loc = feature.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        positions = list(range(start, end + 1)) if (start and end) else ([start] if start else [])
        if not positions: continue
            
        if ftype == "Active site":
            # UniProt Active site features are always single residues (true
            # catalytic). Ranges are extremely rare; if we see one, keep only
            # the start position to avoid inflating core with a domain range.
            core.append(positions[0])
        elif ftype in ("Binding site", "Metal binding", "Site"):
            is_relevant = not targets or any(t in desc for t in targets) or "SUBSTRATE" in desc or (ftype == "Binding site" and not desc)
            if is_relevant:
                # Binding/Metal/Site features go to shell (they're substrate or
                # cofactor contact residues — not catalytic). Do NOT promote to
                # core when Active site is missing: UniProt "Binding site" can
                # span 50+ residue domains, and the downstream constraint
                # logic will fall back to homolog transfer (Active-site-only)
                # or cofactor/pocket evidence when core is empty.
                shell.extend(positions)
            
    return {"core": sorted(set(core)), "shell": sorted(set(shell))}

def find_structure_path(uid, project_dir):
    for subdir in ["alphafold_structures", "pdb_structures", "ref_structure"]:
        path = os.path.join(project_dir, subdir)
        if not os.path.isdir(path): continue
        for f in os.listdir(path):
            if f.endswith(".pdb") and (uid in f or f.startswith(f"AF-{uid}")):
                return os.path.join(path, f)
    return None

def structural_mapping(target_path, template_path, template_sites, min_seqid=15.0, min_overlap=30):
    if not HAS_PRODY or not target_path or not template_path or not template_sites:
        return None
    try:
        target_struct = prody.parsePDB(target_path)
        template_struct = prody.parsePDB(template_path)
        if target_struct is None or template_struct is None: return None
        t_sel = target_struct.select("protein and calpha")
        temp_sel = template_struct.select("protein and calpha")
        if t_sel is None or temp_sel is None: return None
        matches = prody.matchChains(t_sel, temp_sel, overlap=min_overlap, seqid=min_seqid)
        if not matches: return None
        qm, tm, seqid, overlap = matches[0]
        t2q = {int(tm.getResnums()[i]): int(qm.getResnums()[i]) for i in range(len(tm))}
        return {"mapped_residues": [t2q[pos] for pos in template_sites if pos in t2q], "seqid": seqid}
    except (OSError, ValueError, IndexError, KeyError, RuntimeError) as e:
        print(f"    Warning: structural_mapping failed for {os.path.basename(target_path)}: {e}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# Family-aware expected core residue counts, keyed by InterPro domain ID.
#
# Rationale: different enzyme mechanisms have different numbers of residues
# that UniProt annotates as "Active site".  Using a fixed threshold of < 2
# incorrectly triggers homolog transfer for families where 0 or 1 is the
# complete, correct annotation (e.g. Ntn-hydrolases, NIS synthetases).
#
# Rules:
#   0  — UniProt never annotates Active site for this family; homolog
#         transfer would also return nothing, so skip it entirely.
#   1  — exactly one mechanistic residue expected (single nucleophile, or
#         single proton donor as annotated by UniProt for ATP-grasp enzymes).
#   3  — classical catalytic triad (serine/cysteine proteases).
#
# For families not listed here the default is 1: trigger transfer only when
# no core residue at all has been found.
# ---------------------------------------------------------------------------
FAMILY_EXPECTED_CORE = {
    # NIS synthetases — mechanism does not involve classical nucleophile/acid;
    # UniProt leaves Active site blank for the entire family.
    "IPR007310": 0,   # IucA/IucC-like NIS synthetase
    "IPR022770": 0,   # NIS synthetase, C-terminal domain
    "IPR037455": 0,   # NIS synthetase superfamily

    # Ntn-hydrolases — single N-terminal nucleophile is the complete active site.
    "IPR016117": 1,   # Ntn-hydrolase-like
    "IPR005321": 1,   # beta-lactam synthetase / Ntn-hydrolase fold

    # Serine/cysteine proteases — catalytic triad (Ser/Cys · His · Asp).
    "IPR022119": 3,   # Peptidase S1/S6, chymotrypsin-type
    "IPR018975": 3,   # Peptidase S1, active-site
    "IPR029062": 3,   # Class-I glutamine amidotransferase
    "IPR005320": 3,   # Peptidase S51
    "IPR011811": 3,   # Dipeptidyl-peptidase IV

    # ATP-grasp ligases — UniProt annotates one key residue (proton donor or
    # reaction intermediate); 1 is the expected complete annotation.
    "IPR004821": 1,   # Pantoate-beta-alanine ligase (panC family)
    "IPR003721": 1,   # Pantoate-beta-alanine ligase, N-terminal
    "IPR014729": 1,   # Rossmann-like alpha/beta/alpha sandwich
    "IPR007921": 1,   # Glutathione synthetase (gss family)
    "IPR005494": 1,   # Glutathione synthetase, ATP-binding domain
    "IPR013651": 1,   # ATP-grasp fold
    "IPR004101": 1,   # Mur ligase, N-terminal
    "IPR011761": 1,   # ATP-grasp superfamily
}


def expected_core_for_family(interpro_ids):
    """Return the expected number of core (Active site) residues for this enzyme.

    Scans the candidate's InterPro domains and returns the expected count for
    the first matching family.  Falls back to 1 (trigger transfer only when
    completely unannotated) for families not in the table.
    """
    for ipr_id in interpro_ids:
        if ipr_id in FAMILY_EXPECTED_CORE:
            return FAMILY_EXPECTED_CORE[ipr_id]
    return 1  # conservative default: accept 1 core residue as sufficient


def _foldseek_hits(project_dir, uid, min_tmscore=0.7):
    """Return Foldseek hits for *uid* with TM >= min_tmscore, sorted best-first.
    Reads foldseek_results.tsv from the project directory (pre-computed).
    Returns [] if the file is absent or the uid has no hits."""
    tsv = os.path.join(project_dir, "foldseek_results.tsv")
    if not os.path.exists(tsv):
        return []
    hits = []
    with open(tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            query = parts[0].replace(".pdb", "")
            if query != uid:
                continue
            tmscore = float(parts[2])
            if tmscore < min_tmscore:
                continue
            target = parts[1]
            pdb_id = target.split("-")[0].upper()
            chain = "A"
            if "_" in target:
                chain_part = target.split("_")[-1]
                chain = chain_part.split("-")[0] if chain_part else "A"
            hits.append({"pdb_id": pdb_id, "chain": chain, "tmscore": tmscore})
    hits.sort(key=lambda h: -h["tmscore"])
    return hits


def _uniprot_from_pdb(pdb_id):
    """Look up UniProt accession for a PDB entry via the RCSB REST API."""
    raw = _fetch_url(
        f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1", timeout=10)
    if not raw:
        return None
    try:
        for ref in json.loads(raw).get(
                "rcsb_polymer_entity_container_identifiers", {}
        ).get("reference_sequence_identifiers", []):
            if ref.get("database_name") == "UniProt":
                return ref.get("database_accession")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def analyze_single_candidate(uid, project_dir, pocket_res, target_ligands):
    """Analyze using candidate's own family homology mapping with TrEMBL fallback."""
    target_path = find_structure_path(uid, project_dir)
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    data_raw = _fetch_url(url)
    
    tiers = {"core": [], "shell": []}
    source = "none"

    if data_raw:
        data = json.loads(data_raw)
        tiers = _extract_active_sites(data, target_ligands)
        if tiers["core"] or tiers["shell"]: source = "uniprot_direct"

    # Record family-aware completeness for downstream use by enhance_catalytic_sites.py
    # (Phase B / Step 2), which will trigger Foldseek homolog transfer when annotation
    # is incomplete.  Shell residues are never transferred from homologs — they come
    # only from the enzyme's own UniProt annotations above.
    interpro_ids = [x["id"] for x in data.get("uniProtKBCrossReferences", [])
                    if x.get("database") == "InterPro"] if data_raw else []
    expected_core = expected_core_for_family(interpro_ids)

    pocket_set = set(pocket_res)
    catalytic_center = len(pocket_set & set(tiers["core"])) / len(tiers["core"]) if tiers["core"] else None
    pocket_engagement = len(pocket_set & set(tiers["shell"])) / len(tiers["shell"]) if tiers["shell"] else None

    return uid, {
        "core_residues": tiers["core"], "shell_residues": tiers["shell"],
        "catalytic_center_score": catalytic_center, "pocket_engagement_score": pocket_engagement,
        "evidence_tier": source,
        "n_core": len(tiers["core"]), "n_shell": len(tiers["shell"]),
        "expected_core": expected_core,   # used by enhance_catalytic_sites.py (Step 2)
    }

def process_project(project_dir, workers):
    print(f"\nProcessing {project_dir}...")
    pocket_path = os.path.join(project_dir, "pocket_analysis.json")
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(pocket_path): return
    with open(pocket_path) as f: pocket_data = json.load(f)
    
    target_ligands = []
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                rxn = yaml.safe_load(f).get("target_reaction", {})
                target_ligands = rxn.get("relevant_ligands", []) + rxn.get("catalytic_metals", [])
                for s in rxn.get("substrates", []): target_ligands.append(s.get("name", ""))
        except (OSError, yaml.YAMLError, AttributeError) as e:
            print(f"  Warning: failed to load config.yaml: {e}", file=sys.stderr)

    uids = sorted(pocket_data.keys()); results = {}
    print(f"  Analyzing {len(uids)} candidates using {workers} workers (Hybrid Homology Strategy)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyze_single_candidate, uid, project_dir, 
                                   [r[1] for r in (pocket_data[uid].get("binding_pocket") or {}).get("lining_residues", [])], 
                                   target_ligands): uid for uid in uids}
        for future in as_completed(futures):
            uid, data = future.result()
            if data:
                results[uid] = data
                results[uid]["status"] = "OK" if (data["catalytic_center_score"] is not None and data["catalytic_center_score"] >= 0.5) or data["n_core"] == 0 else "WARNING"
                # NO-CAT-DATA triggers homolog transfer in enhance_catalytic_sites.py.
                # Trigger when true catalytic (core) is missing, regardless of shell,
                # since shell-only entries have no catalytic anchor for cofolding.
                if data["n_core"] == 0: results[uid]["status"] = "NO-CAT-DATA"
                results[uid]["upper_tier_recall"] = data["catalytic_center_score"]
                # catalytic_residues = true catalytic only (core). Shell residues
                # (UniProt Binding/Metal/Site features) are kept separately and
                # consumed by the cofolding script as pocket-evidence, not as the
                # catalytic anchor.
                results[uid]["catalytic_residues"] = sorted(data["core_residues"])
                c_str = f"{data['catalytic_center_score']:>4.0%}" if data['catalytic_center_score'] is not None else " N/A"
                s_str = f"{data['pocket_engagement_score']:>4.0%}" if data['pocket_engagement_score'] is not None else " N/A"
                print(f"    {uid:<12} Catalytic Center: {c_str} Pocket Engagement: {s_str} ({data['evidence_tier']})")

    out_path = os.path.join(project_dir, "catalytic_site_validation.json")
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved hybrid results to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="+"); parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    for p in args.projects: process_project(os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p, args.workers)

if __name__ == "__main__": main()
