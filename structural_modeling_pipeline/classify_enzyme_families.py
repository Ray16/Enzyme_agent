#!/usr/bin/env python3
"""Classify enzyme candidates into families using InterPro/Pfam domain
annotations from the UniProt API.

Family assignment is fully automatic: the script discovers which Pfam
domains are shared across enzymes, then groups enzymes by their most
specific shared domain.  No project-specific domain IDs are hard-coded.

Usage:
    python classify_enzyme_families.py [--no-cache] [--output-dir DIR]

Outputs:
    analysis_results/enzyme_family_classification.csv
    analysis_results/enzyme_family_summary.md
    analysis_results/uniprot_domain_cache.json  (API response cache)
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict

import yaml


# ═══════════════════════════════════════════════════════════════════════
# UNIPROT API
# ═══════════════════════════════════════════════════════════════════════

_COFACTOR_TERMS = {
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "UTP",
    "phosphate", "diphosphate", "H(+)", "H2O", "water",
    "NAD(+)", "NADH", "NADP(+)", "NADPH", "FAD", "FADH2",
    "CoA", "coenzyme A",
}


def _parse_reaction_substrates(reaction_name):
    """Parse a UniProt reaction string into substrates, products, cofactors.

    Reaction format: 'substrate1 + substrate2 + ATP = product1 + ADP + ...'
    """
    if not reaction_name or "=" not in reaction_name:
        return [], [], []

    lhs, rhs = reaction_name.split("=", 1)
    lhs_parts = [s.strip() for s in lhs.split("+")]
    rhs_parts = [s.strip() for s in rhs.split("+")]

    def _classify(parts):
        substrates = []
        cofactors = []
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            # Strip leading stoichiometry like "2 D-alanine"
            p_check = p_clean
            if p_check and p_check[0].isdigit() and " " in p_check:
                p_check = p_check.split(" ", 1)[1]
            if p_check in _COFACTOR_TERMS:
                cofactors.append(p_clean)
            else:
                substrates.append(p_clean)
        return substrates, cofactors

    substrates, lhs_cofactors = _classify(lhs_parts)
    products, rhs_cofactors = _classify(rhs_parts)
    cofactors = lhs_cofactors + rhs_cofactors

    return substrates, products, cofactors


def fetch_uniprot_domains(uniprot_id):
    """Query UniProt REST API for InterPro/Pfam cross-references and
    catalytic activity.

    Returns dict with keys: interpro, pfam, protein_name, organism,
    ec_numbers, catalytic_activity — or None on failure.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        req = urllib.request.Request(url,
                                    headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError,
            json.JSONDecodeError, TimeoutError) as exc:
        print(f"  WARNING: {uniprot_id} API error: {exc}", file=sys.stderr)
        return None

    xrefs = data.get("uniProtKBCrossReferences", [])

    def _extract(database):
        results = []
        for x in xrefs:
            if x.get("database") == database:
                props = {p["key"]: p["value"]
                         for p in x.get("properties", [])}
                results.append({
                    "id": x["id"],
                    "name": props.get("EntryName", ""),
                    "match_status": props.get("MatchStatus", ""),
                })
        return results

    # Protein name
    prot_desc = data.get("proteinDescription", {})
    rec_name = prot_desc.get("recommendedName", {})
    if rec_name:
        protein_name = rec_name.get("fullName", {}).get("value", "")
    else:
        sub_names = prot_desc.get("submissionNames", [])
        protein_name = (sub_names[0].get("fullName", {}).get("value", "")
                        if sub_names else "")

    # EC numbers
    ec_numbers = []
    for name_block in [rec_name] + prot_desc.get("alternativeNames", []):
        for ec in name_block.get("ecNumbers", []):
            ec_numbers.append(ec.get("value", ""))

    organism = data.get("organism", {}).get("scientificName", "")

    # Catalytic activity
    catalytic_activity = []
    for comment in data.get("comments", []):
        if comment.get("commentType") != "CATALYTIC ACTIVITY":
            continue
        rxn = comment.get("reaction", {})
        rxn_name = rxn.get("name", "")
        rxn_ec = rxn.get("ecNumber", "")
        substrates, products, cofactors = _parse_reaction_substrates(rxn_name)
        catalytic_activity.append({
            "reaction": rxn_name,
            "ec": rxn_ec,
            "substrates": substrates,
            "products": products,
            "cofactors": cofactors,
        })

    return {
        "interpro": _extract("InterPro"),
        "pfam": _extract("Pfam"),
        "protein_name": protein_name,
        "organism": organism,
        "ec_numbers": ec_numbers,
        "catalytic_activity": catalytic_activity,
    }


# ═══════════════════════════════════════════════════════════════════════
# AUTO-DISCOVERY FAMILY ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════

# InterPro type prefixes that indicate broad superfamilies (less specific)
_SUPERFAMILY_KEYWORDS = {"_sf", "superfamily", "Superfamily", "-like_dom_sf",
                         "P-loop", "Rossmann"}


def _is_superfamily(name):
    """Heuristic: InterPro entries ending in _sf or containing
    'superfamily' are broad structural groupings, not specific families."""
    return any(kw in name for kw in _SUPERFAMILY_KEYWORDS)


def build_family_map(cache, enzymes):
    """Auto-discover enzyme families from shared Pfam/InterPro domains.

    Strategy:
      1. Collect all Pfam domains across all enzymes.
      2. For each Pfam domain, count how many enzymes carry it.
      3. A domain shared by >=2 enzymes but not by >80% of all enzymes
         is a candidate "family-defining" domain.
      4. Assign each enzyme to the most specific family-defining domain
         it carries (prefer Pfam over InterPro superfamilies, prefer
         domains shared by fewer enzymes = more specific).
      5. Enzymes with no family-defining domain: use their most specific
         InterPro entry (the one shared by fewest enzymes).
    """
    n_enzymes = len(enzymes)
    ubiquity_threshold = 0.8 * n_enzymes  # domains in >80% are too generic

    # Step 1: map each enzyme to its domain set
    enzyme_domains = {}  # uid -> set of (db, id, name)
    for enzyme in enzymes:
        uid = enzyme["uniprot_id"]
        info = cache.get(uid)
        domains = set()
        if info:
            for entry in info.get("pfam", []):
                domains.add(("Pfam", entry["id"], entry["name"]))
            for entry in info.get("interpro", []):
                if not _is_superfamily(entry["name"]):
                    domains.add(("InterPro", entry["id"], entry["name"]))
        enzyme_domains[uid] = domains

    # Step 2: count domain prevalence
    domain_counts = Counter()
    for uid, domains in enzyme_domains.items():
        for d in domains:
            domain_counts[d] += 1

    # Step 3: identify family-defining domains
    #   - present in >=2 enzymes
    #   - not ubiquitous (present in >80% of enzymes)
    family_domains = {}
    for d, count in domain_counts.items():
        if 2 <= count <= ubiquity_threshold:
            family_domains[d] = count

    # Step 4: assign each enzyme to its best family-defining domain
    #   Priority: Pfam > InterPro; fewer members = more specific
    def _domain_sort_key(d):
        db, did, name = d
        count = family_domains.get(d, n_enzymes)
        db_priority = 0 if db == "Pfam" else 1
        return (db_priority, count)

    assignments = {}  # uid -> (family_name, description, evidence_id)
    for enzyme in enzymes:
        uid = enzyme["uniprot_id"]
        info = cache.get(uid)
        domains = enzyme_domains[uid]

        if info is None:
            assignments[uid] = ("api_error",
                                "UniProt API query failed", "")
            continue

        if not domains:
            # No domains at all — try protein name
            protein_name = info.get("protein_name", "")
            if protein_name:
                assignments[uid] = (
                    _sanitize_family_name(protein_name),
                    protein_name + " (name-based, no domain annotation)",
                    f"protein_name:{protein_name}")
            else:
                assignments[uid] = ("no_domain_annotation",
                                    "No InterPro/Pfam domains in UniProt", "")
            continue

        # Find the best family-defining domain this enzyme carries
        candidate_domains = [d for d in domains if d in family_domains]

        if candidate_domains:
            best = min(candidate_domains, key=_domain_sort_key)
            db, did, name = best
            assignments[uid] = (name, f"{name} ({db}: {did})", did)
        else:
            # No family-defining domain — use most specific domain
            all_sorted = sorted(domains, key=_domain_sort_key)
            if all_sorted:
                db, did, name = all_sorted[0]
                assignments[uid] = (name,
                                    f"{name} ({db}: {did}, singleton)",
                                    did)
            else:
                protein_name = info.get("protein_name", "")
                assignments[uid] = ("other_annotated",
                                    protein_name or "Has domains, no family",
                                    "")

    return assignments


def _sanitize_family_name(name):
    """Convert a protein name to a clean family label."""
    name = name.strip()
    # Truncate at 60 chars
    if len(name) > 60:
        name = name[:57] + "..."
    return name


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_enzymes(base_dir):
    """Load enzyme UniProt IDs from all projects in pipeline_config.yaml."""
    config_path = os.path.join(base_dir, "pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    enzymes = []
    for project in config["projects"]:
        proj_path = os.path.join(base_dir, project["path"])
        proj_name = project["name"]
        csv_path = os.path.join(proj_path, "structure_sequence_summary.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping",
                  file=sys.stderr)
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uniprot_id", "").strip()
                name = row.get("gene_name", "").strip()
                if uid:
                    enzymes.append({
                        "project": proj_name,
                        "uniprot_id": uid,
                        "enzyme_name": name,
                    })
    return enzymes


# ═══════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_path):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY WRITER
# ═══════════════════════════════════════════════════════════════════════

def write_summary(rows, output_path):
    """Write a manuscript-ready markdown summary."""
    sections = []
    sections.append("# Enzyme Family Classification (InterPro/Pfam)\n")
    sections.append(
        "*Domain annotations retrieved from UniProt REST API "
        "(uniProtKBCrossReferences). Families auto-discovered by "
        "identifying Pfam/InterPro domains shared across multiple "
        "enzymes in each project.*\n")

    # Detect projects dynamically
    projects = []
    seen = set()
    for r in rows:
        if r["project"] not in seen:
            projects.append(r["project"])
            seen.add(r["project"])

    for proj_key in projects:
        proj_rows = [r for r in rows if r["project"] == proj_key]
        if not proj_rows:
            continue

        sections.append(f"\n## {proj_key} (n={len(proj_rows)})\n")
        fam_counts = Counter(r["assigned_family"] for r in proj_rows)

        sections.append("| Family | Count | Evidence |")
        sections.append("|---|---|---|")
        for fam, count in fam_counts.most_common():
            desc = ""
            for r in proj_rows:
                if r["assigned_family"] == fam and r["family_description"]:
                    desc = r["family_description"]
                    break
            sections.append(f"| {fam} | {count} | {desc} |")

        # List members per family
        sections.append(f"\n### Per-Family Members\n")
        families = defaultdict(list)
        for r in proj_rows:
            families[r["assigned_family"]].append(r)

        for fam, count in fam_counts.most_common():
            members = families[fam]
            sections.append(f"\n**{fam}** ({count}):\n")
            for m in sorted(members, key=lambda x: x["uniprot_id"]):
                protein_name = m.get("protein_name", "")
                label = protein_name if protein_name else (
                    m["enzyme_name"] or m["uniprot_id"])
                evidence = []
                if m.get("interpro_ids"):
                    evidence.append(f"InterPro: {m['interpro_ids']}")
                if m.get("pfam_ids"):
                    evidence.append(f"Pfam: {m['pfam_ids']}")
                ev_str = "; ".join(evidence) if evidence else "(no domains)"
                sections.append(
                    f"- {m['uniprot_id']} ({label}) -- {ev_str}")

    with open(output_path, "w") as f:
        f.write("\n".join(sections) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify enzymes by InterPro/Pfam domains "
                    "(auto-discovered families)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached API responses")
    parser.add_argument("--output-dir", default="analysis_results",
                        help="Output directory (default: analysis_results)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "uniprot_domain_cache.json")

    # Load enzymes
    enzymes = load_enzymes(base_dir)
    print(f"Loaded {len(enzymes)} enzymes from pipeline_config.yaml")

    # Load cache
    cache = {} if args.no_cache else load_cache(cache_path)
    cached_count = sum(1 for e in enzymes if e["uniprot_id"] in cache)
    print(f"Cache: {cached_count}/{len(enzymes)} already cached")

    # Query UniProt for missing entries or entries without catalytic_activity
    to_query = [e for e in enzymes
                if e["uniprot_id"] not in cache
                or (cache.get(e["uniprot_id"]) is not None
                    and "catalytic_activity" not in cache[e["uniprot_id"]])]
    if to_query:
        print(f"Querying UniProt API for {len(to_query)} entries...")
        for i, enzyme in enumerate(to_query):
            uid = enzyme["uniprot_id"]
            print(f"  [{i+1}/{len(to_query)}] {uid}...",
                  end="", flush=True)
            result = fetch_uniprot_domains(uid)
            if result is not None:
                cache[uid] = result
                print(f" OK ({len(result.get('interpro', []))} InterPro, "
                      f"{len(result.get('pfam', []))} Pfam)")
            else:
                cache[uid] = None
                print(" FAILED")
            time.sleep(0.4)
        save_cache(cache, cache_path)
        print(f"Cache saved to {cache_path}")

    # Auto-discover families
    print("\nAuto-discovering families from shared domains...")
    assignments = build_family_map(cache, enzymes)

    # Build output rows
    rows = []
    for enzyme in enzymes:
        uid = enzyme["uniprot_id"]
        domain_info = cache.get(uid)
        fam_name, fam_desc, evidence_id = assignments[uid]

        interpro_ids = ""
        interpro_names = ""
        pfam_ids = ""
        pfam_names = ""
        protein_name = ""
        organism = ""
        ec_numbers = ""
        if domain_info:
            interpro_ids = "; ".join(
                e["id"] for e in domain_info.get("interpro", []))
            interpro_names = "; ".join(
                e["name"] for e in domain_info.get("interpro", []))
            pfam_ids = "; ".join(
                e["id"] for e in domain_info.get("pfam", []))
            pfam_names = "; ".join(
                e["name"] for e in domain_info.get("pfam", []))
            protein_name = domain_info.get("protein_name", "")
            organism = domain_info.get("organism", "")
            ec_numbers = "; ".join(domain_info.get("ec_numbers", []))

        # Native substrates from catalytic activity
        native_reaction = ""
        native_substrates = ""
        native_cofactors = ""
        if domain_info:
            cat_acts = domain_info.get("catalytic_activity", [])
            if cat_acts:
                primary = cat_acts[0]
                native_reaction = primary.get("reaction", "")
                native_substrates = "; ".join(
                    primary.get("substrates", []))
                native_cofactors = "; ".join(
                    primary.get("cofactors", []))

        rows.append({
            "project": enzyme["project"],
            "uniprot_id": uid,
            "enzyme_name": enzyme["enzyme_name"],
            "protein_name": protein_name,
            "organism": organism,
            "ec_numbers": ec_numbers,
            "native_reaction": native_reaction,
            "native_substrates": native_substrates,
            "native_cofactors": native_cofactors,
            "interpro_ids": interpro_ids,
            "interpro_names": interpro_names,
            "pfam_ids": pfam_ids,
            "pfam_names": pfam_names,
            "assigned_family": fam_name,
            "family_description": fam_desc,
            "family_evidence": evidence_id,
        })

    # Write CSV
    csv_path = os.path.join(output_dir, "enzyme_family_classification.csv")
    fieldnames = [
        "project", "uniprot_id", "enzyme_name", "protein_name", "organism",
        "ec_numbers", "native_reaction", "native_substrates",
        "native_cofactors", "interpro_ids", "interpro_names", "pfam_ids",
        "pfam_names", "assigned_family", "family_description",
        "family_evidence",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {csv_path} ({len(rows)} rows)")

    # Write summary
    summary_path = os.path.join(output_dir, "enzyme_family_summary.md")
    write_summary(rows, summary_path)
    print(f"Summary written: {summary_path}")

    # Print overview
    print("\n" + "=" * 60)
    print("FAMILY CLASSIFICATION OVERVIEW")
    print("=" * 60)
    for proj_key in dict.fromkeys(r["project"] for r in rows):
        proj_rows = [r for r in rows if r["project"] == proj_key]
        print(f"\n{proj_key} (n={len(proj_rows)}):")
        fam_counts = Counter(r["assigned_family"] for r in proj_rows)
        for fam, count in fam_counts.most_common():
            print(f"  {fam:50s} {count:3d}")


if __name__ == "__main__":
    main()
