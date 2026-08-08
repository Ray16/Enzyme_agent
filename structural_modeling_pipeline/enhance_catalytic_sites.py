#!/usr/bin/env python3
"""
Enhance catalytic site coverage using FoldSeek structural homologs.

Augments existing catalytic_site_validation.json by finding UniProt "Active site"
annotations from FoldSeek homologs and mapping them back to query enzymes via
structural alignment.

This fills gaps left by the InterPro-based approach in validate_catalytic_sites.py,
which only covers ~20/178 enzymes. FoldSeek structural homologs at TM >= 0.7
can add ~30 more.

IMPORTANT: Only transfers UniProt "Active site" features (true catalytic residues).
Does NOT transfer "Binding site" or "Metal binding" features, which are
substrate-specific and may not apply to our target reaction.

Usage:
    python enhance_catalytic_sites.py amide_synthetase amide_hydrolase
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
from openai import OpenAI

try:
    import prody
    prody.confProDy(verbosity="none")
except ImportError:
    sys.exit("ERROR: ProDy is required. Install with: pip install prody")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_url(url, timeout=20, retries=3):
    """GET with exponential backoff."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "CatalyticSiteEnhancer/1.0")
    req.add_header("Accept", "application/json")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 ** attempt)
            continue
    return None


def parse_foldseek_tsv(tsv_path):
    """Parse FoldSeek results into {query_uid: [{pdb_id, target_chain, tmscore}, ...]}."""
    hits = {}
    if not os.path.exists(tsv_path):
        return hits
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            query = parts[0].replace(".pdb", "")
            target = parts[1]
            pdb_id = target.split("-")[0].upper()
            tmscore = float(parts[6])
            target_chain = "A"
            if "_" in target:
                chain_part = target.split("_")[-1]
                target_chain = chain_part.split("-")[0] if chain_part else "A"
            hits.setdefault(query, []).append({
                "pdb_id": pdb_id,
                "target_chain": target_chain,
                "tmscore": tmscore,
            })
    return hits


def get_uniprot_id_from_pdb(pdb_id):
    """Get UniProt accession from a PDB entry via RCSB API."""
    raw = fetch_url(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1", timeout=10)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        for ref in data.get("rcsb_polymer_entity_container_identifiers", {}).get(
                "reference_sequence_identifiers", []):
            if ref.get("database_name") == "UniProt":
                return ref.get("database_accession")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def get_uniprot_sites(uid):
    """Fetch Active site (core) and Binding/Metal binding (shell) features from UniProt.

    Returns (core_sites, shell_sites) where each is a list of
    {"position": int, "description": str} dicts.
    """
    raw = fetch_url(f"https://rest.uniprot.org/uniprotkb/{uid}.json")
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None

    core, shell = [], []
    for feat in data.get("features", []):
        ftype = feat.get("type", "")
        start = feat.get("location", {}).get("start", {}).get("value")
        end   = feat.get("location", {}).get("end",   {}).get("value")
        desc  = feat.get("description", "")
        if not start:
            continue
        positions = list(range(start, (end or start) + 1))
        if ftype == "Active site":
            for pos in positions:
                core.append({"position": pos, "description": desc})
        elif ftype in ("Binding site", "Metal binding", "Site"):
            for pos in positions:
                shell.append({"position": pos, "description": desc})
    return core or None, shell or None


# Keep old name as alias so existing call sites don't break
def get_uniprot_active_sites(uid):
    core, _ = get_uniprot_sites(uid)
    return core


ARGO_BASE_URL = os.environ.get("ARGO_BASE_URL",
                               "https://apps-dev.inside.anl.gov/argoapi/v1")
ARGO_API_KEY = os.environ.get("ARGO_API_KEY", os.environ.get("USER", "rzhu"))
VALIDATION_MODEL = os.environ.get("VALIDATION_MODEL", "claudesonnet46")


def get_pdb_title(pdb_id: str) -> str:
    """Get the title/description of a PDB entry from RCSB."""
    raw = fetch_url(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=10)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        return data.get("struct", {}).get("title", "")
    except (json.JSONDecodeError, KeyError):
        return ""


def validate_catalytic_transfer_llm(
    uid: str,
    mapped_residues: list[tuple[int, str, str]],
    homolog_pdb: str,
    pdb_title: str,
    tmscore: float,
) -> list[tuple[int, str, str]]:
    """Use LLM to validate whether transferred catalytic residues are biochemically plausible.

    Args:
        uid: query protein UniProt ID
        mapped_residues: list of (resid, 3-letter AA code, annotation description)
        homolog_pdb: PDB ID of the structural homolog
        pdb_title: title of the homolog PDB entry (contains family/mechanism info)
        tmscore: TM-score of the structural alignment

    Returns:
        Filtered list of (resid, aa, description) that passed validation.
    """
    if not mapped_residues:
        return []

    residue_lines = "\n".join(
        f"  {resid} {aa} ({desc})" if desc else f"  {resid} {aa}"
        for resid, aa, desc in mapped_residues
    )

    prompt = f"""You are validating catalytic site annotation transfers from a structural homolog to a query protein.

Homolog: PDB {homolog_pdb} (TM-score={tmscore:.2f})
Homolog description: {pdb_title}

The following catalytic residue positions were transferred from the homolog to query protein {uid} via structural alignment. The amino acid shown is what the QUERY protein has at each mapped position.

Mapped residues:
{residue_lines}

For each residue, decide KEEP or REJECT based on whether the amino acid can plausibly perform the catalytic function expected for this enzyme family's mechanism.

Output EXACTLY one line per residue:
RESID KEEP|REJECT"""

    try:
        client = OpenAI(base_url=ARGO_BASE_URL, api_key=ARGO_API_KEY)
        response = client.chat.completions.create(
            model=VALIDATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        llm_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    LLM validation failed ({e}), keeping all residues", file=sys.stderr)
        return mapped_residues

    # Parse LLM response
    keep_resids = set()
    for line in llm_text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                resid = int(parts[0])
                decision = parts[1].upper().strip("|:,")
                if decision == "KEEP":
                    keep_resids.add(resid)
            except ValueError:
                continue

    # Filter
    kept = [(r, aa, desc) for r, aa, desc in mapped_residues if r in keep_resids]
    n_rejected = len(mapped_residues) - len(kept)
    if n_rejected > 0:
        rejected = [(r, aa, desc) for r, aa, desc in mapped_residues if r not in keep_resids]
        rej_str = ", ".join(f"{aa}{r}" for r, aa, _ in rejected)
        print(f"    LLM filtered {n_rejected} incompatible residue(s): {rej_str}")

    return kept


def structural_alignment_map(query_pdb_path, homolog_pdb_path, homolog_chain="A"):
    """Structurally align query and homolog, return residue mapping (homolog→query)."""
    try:
        query_struct = prody.parsePDB(query_pdb_path)
        homolog_struct = prody.parsePDB(homolog_pdb_path, chain=homolog_chain)
        if homolog_struct is None:
            homolog_struct = prody.parsePDB(homolog_pdb_path)
        if query_struct is None or homolog_struct is None:
            return None

        query_ca = query_struct.select("protein and calpha")
        homolog_ca = homolog_struct.select("protein and calpha")
        if query_ca is None or homolog_ca is None:
            return None

        matches = prody.matchChains(query_ca, homolog_ca, overlap=30, seqid=10)
        if not matches:
            return None

        qm, hm, seqid, overlap = matches[0]
        h2q = {int(hm.getResnums()[i]): int(qm.getResnums()[i])
               for i in range(len(hm))}
        return h2q
    except Exception as e:
        print(f"    Alignment error: {e}", file=sys.stderr)
        return None


def find_structure_path(uid, project_dir):
    """Find the PDB structure file for a candidate."""
    for subdir in ["alphafold_structures", "pdb_structures"]:
        dirpath = os.path.join(project_dir, subdir)
        if not os.path.isdir(dirpath):
            continue
        for f in os.listdir(dirpath):
            if f.endswith(".pdb") and uid in f:
                return os.path.join(dirpath, f)
    return None


def process_project(project_dir, min_tmscore=0.7):
    """Enhance catalytic site data for one project directory."""
    project_name = os.path.basename(project_dir)
    print(f"\n{'='*60}")
    print(f"Processing: {project_name}")
    print(f"{'='*60}")

    # Load existing catalytic site validation
    cat_val_path = os.path.join(project_dir, "catalytic_site_validation.json")
    if not os.path.exists(cat_val_path):
        print(f"  WARNING: {cat_val_path} not found. Run validate_catalytic_sites.py first.")
        return

    with open(cat_val_path) as f:
        cat_data = json.load(f)

    # Back up before modifying
    bak_path = cat_val_path + ".bak"
    with open(bak_path, "w") as f:
        json.dump(cat_data, f, indent=2)

    # Find candidates where core annotation is incomplete for their enzyme family.
    # Uses expected_core written by validate_catalytic_sites.py (Step 1); falls
    # back to the old "NO-CAT-DATA" sentinel for entries predating this field.
    def _needs_enhancement(v):
        if not isinstance(v, dict):
            return False
        expected = v.get("expected_core")
        if expected is not None:
            return len(v.get("core_residues") or []) < expected
        return v.get("status") == "NO-CAT-DATA"   # legacy fallback

    no_data_uids = [uid for uid, v in cat_data.items() if _needs_enhancement(v)]
    existing_uids = [uid for uid, v in cat_data.items() if v.get("core_residues")]
    print(f"  Annotation complete: {len(existing_uids)} enzymes")
    print(f"  Needs enhancement:   {len(no_data_uids)} enzymes")

    if not no_data_uids:
        print("  Nothing to enhance.")
        return

    # Parse FoldSeek results
    foldseek_path = os.path.join(project_dir, "foldseek_results.tsv")
    fs_hits = parse_foldseek_tsv(foldseek_path)
    print(f"  FoldSeek results loaded: {len(fs_hits)} queries")

    # Load pocket_analysis for binding pocket overlap check
    pa_path = os.path.join(project_dir, "pocket_analysis.json")
    pocket_data = {}
    if os.path.exists(pa_path):
        with open(pa_path) as f:
            pocket_data = json.load(f)

    # Cache for UniProt lookups
    uniprot_cache = {}
    pdb_uniprot_cache = {}
    pdb_title_cache = {}

    enhanced = 0
    skipped_no_homolog = 0
    skipped_no_active_site = 0
    skipped_no_alignment = 0

    for uid in sorted(no_data_uids):
        # Skip PDB IDs (not our candidates)
        if len(uid) <= 5 and uid[0].isdigit():
            continue

        hits = fs_hits.get(uid, [])
        good_hits = [h for h in hits if h["tmscore"] >= min_tmscore]
        if not good_hits:
            skipped_no_homolog += 1
            continue

        query_pdb = find_structure_path(uid, project_dir)
        if not query_pdb:
            skipped_no_homolog += 1
            continue

        # Try each homolog (best TM-score first)
        good_hits.sort(key=lambda h: h["tmscore"], reverse=True)

        found = False
        all_shell_mapped = []   # accumulate shell across all tried homologs
        n_homologs_tried = 0
        for hit in good_hits[:3]:  # Try up to 3 homologs
            pdb_id = hit["pdb_id"]
            chain = hit["target_chain"]
            tm = hit["tmscore"]

            # Get homolog's UniProt ID
            if pdb_id not in pdb_uniprot_cache:
                pdb_uniprot_cache[pdb_id] = get_uniprot_id_from_pdb(pdb_id)
                time.sleep(0.15)
            homolog_uid = pdb_uniprot_cache[pdb_id]
            if not homolog_uid:
                continue

            # Get UniProt Active site (core) AND Binding/Metal binding (shell) for homolog
            if homolog_uid not in uniprot_cache:
                core_sites, shell_sites = get_uniprot_sites(homolog_uid)
                uniprot_cache[homolog_uid] = (core_sites, shell_sites)
                time.sleep(0.15)
            sites, shell_sites = uniprot_cache[homolog_uid]
            if not sites:
                continue

            # Download homolog PDB and do structural alignment
            pdb_content = fetch_url(
                f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
            if not pdb_content:
                continue

            tmp_pdb = tempfile.NamedTemporaryFile(
                suffix=".pdb", delete=False, prefix=f"{pdb_id}_")
            try:
                tmp_pdb.write(pdb_content.encode())
                tmp_pdb.close()

                h2q = structural_alignment_map(query_pdb, tmp_pdb.name, chain)
                if not h2q:
                    continue

                # Map catalytic positions from homolog to query
                # Build residue type lookup from query PDB
                query_res_types = {}
                with open(query_pdb) as qf:
                    for line in qf:
                        if line.startswith("ATOM") and line[12:16].strip() == "CA":
                            query_res_types[int(line[22:26])] = line[17:20].strip()

                raw_mapped = []
                for site in sites:
                    hpos = site["position"]
                    qpos = h2q.get(hpos)
                    if qpos is not None:
                        aa = query_res_types.get(qpos, "UNK")
                        raw_mapped.append((qpos, aa, site.get("description", "")))

                # Map shell (Binding + Metal binding) — collect across all homologs
                for site in (shell_sites or []):
                    hpos = site["position"]
                    qpos = h2q.get(hpos)
                    if qpos is not None:
                        all_shell_mapped.append(qpos)
                n_homologs_tried += 1

                if not raw_mapped:
                    continue

                # LLM-based validation: check biochemical compatibility
                if os.environ.get("SKIP_LLM_VALIDATION"):
                    validated = raw_mapped
                else:
                    if pdb_id not in pdb_title_cache:
                        pdb_title_cache[pdb_id] = get_pdb_title(pdb_id)
                        time.sleep(0.1)

                    validated = validate_catalytic_transfer_llm(
                        uid, raw_mapped, pdb_id,
                        pdb_title_cache[pdb_id], tm)

                mapped_core = [r for r, _, _ in validated]
                descriptions = [d for _, _, d in validated]

                if not mapped_core:
                    continue

                # Check overlap with binding pocket
                pocket_set = set()
                pa_entry = pocket_data.get(uid, {})
                if isinstance(pa_entry, dict):
                    bp = pa_entry.get("binding_pocket") or {}
                    pocket_set = set(r[1] for r in bp.get("lining_residues", []))

                core_set = set(mapped_core)
                pocket_overlap = len(core_set & pocket_set) if pocket_set else None
                cat_center_score = (pocket_overlap / len(core_set)
                                    if pocket_overlap is not None and core_set
                                    else None)

                # Write core — shell will be filled in after the loop
                cat_data[uid] = {
                    "core_residues": sorted(set(mapped_core)),
                    "core_descriptions": descriptions,
                    "shell_residues": [],   # filled below if Step 1 gave no shell
                    "catalytic_center_score": cat_center_score,
                    "pocket_engagement_score": None,
                    "evidence_tier": "foldseek_homology",
                    "homolog_pdb": pdb_id,
                    "homolog_uniprot": homolog_uid,
                    "homolog_chain": chain,
                    "tmscore": tm,
                    "n_core": len(set(mapped_core)),
                    "n_shell": 0,
                    "status": "OK",
                    "upper_tier_recall": cat_center_score,
                    "catalytic_residues": sorted(set(mapped_core)),
                }

                enhanced += 1
                found = True

                overlap_str = (f"{cat_center_score:.0%}"
                               if cat_center_score is not None else "N/A")
                desc_str = "; ".join(d for d in descriptions[:3] if d)
                print(f"  + {uid:<18} {len(set(mapped_core))} core residues from "
                      f"{homolog_uid} (PDB {pdb_id}, TM={tm:.2f}, "
                      f"pocket overlap={overlap_str})"
                      + (f"  [{desc_str}]" if desc_str else ""))
                # Don't break — continue iterating to collect shell from remaining homologs

            finally:
                os.remove(tmp_pdb.name)

            time.sleep(0.1)

        # After trying all homologs: add transferred shell only when Step 1
        # gave no shell at all (fallback). Consensus filter across all tried homologs.
        if found and all_shell_mapped:
            existing_shell = cat_data[uid].get("shell_residues") or []
            if len(existing_shell) == 0 and n_homologs_tried > 0:
                from collections import Counter as _Counter
                counts = _Counter(all_shell_mapped)
                threshold = max(1, (n_homologs_tried + 1) // 2)
                core_set = set(cat_data[uid].get("core_residues") or [])
                shell_consensus = sorted(
                    r for r, c in counts.items()
                    if c >= threshold and r not in core_set)
                cat_data[uid]["shell_residues"] = shell_consensus
                cat_data[uid]["n_shell"] = len(shell_consensus)
                if shell_consensus:
                    print(f"    shell: {len(shell_consensus)} residues transferred "
                          f"(consensus {threshold}/{n_homologs_tried} homologs)")

        if not found:
            # Determine why
            had_homolog = bool(good_hits)
            if not had_homolog:
                skipped_no_homolog += 1
            else:
                # Had homologs but none had Active site data or alignment failed
                any_had_sites = any(
                    uniprot_cache.get(pdb_uniprot_cache.get(h["pdb_id"]))
                    for h in good_hits[:3]
                    if pdb_uniprot_cache.get(h["pdb_id"])
                )
                if any_had_sites:
                    skipped_no_alignment += 1
                else:
                    skipped_no_active_site += 1

    # Write updated results
    with open(cat_val_path, "w") as f:
        json.dump(cat_data, f, indent=2)

    total_with_data = sum(1 for v in cat_data.values() if v.get("core_residues"))
    print(f"\n  Summary for {project_name}:")
    print(f"    Enhanced:             {enhanced}")
    print(f"    No high-TM homolog:   {skipped_no_homolog}")
    print(f"    Homolog no Active site: {skipped_no_active_site}")
    print(f"    Alignment failed:     {skipped_no_alignment}")
    print(f"    Total with catalytic data: {total_with_data}/{len(cat_data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance catalytic site coverage using FoldSeek homologs")
    parser.add_argument("projects", nargs="+",
                        help="Project directories (e.g., amide_synthetase amide_hydrolase)")
    parser.add_argument("--min-tmscore", type=float, default=0.7,
                        help="Minimum TM-score for catalytic residue transfer (default: 0.7)")
    args = parser.parse_args()

    for p in args.projects:
        project_dir = os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p
        if not os.path.isdir(project_dir):
            print(f"WARNING: {project_dir} not found, skipping.")
            continue
        process_project(project_dir, min_tmscore=args.min_tmscore)


if __name__ == "__main__":
    main()
