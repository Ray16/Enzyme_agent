#!/usr/bin/env python3
"""Build the mechanism-aware structural ranking for completed candidates.

Discovery rank remains the primary evidence rank. Boltz-2 geometry supplies a
separate support tier and a rank that is meaningful only within a family.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxy_policy import (  # noqa: E402
    PROXY_VERSION,
    proxy_mechanism_for_family,
    structural_support_tier,
)

def mechanism_for_family(family):
    """Return mechanism, reactive phosphate, electrophile, and leaving axis."""
    spec = proxy_mechanism_for_family(family)
    if spec:
        return spec.mechanism, spec.phosphate, spec.electrophile, spec.leaving_axis
    return "unresolved", "", "", ""


def family_group(family):
    spec = proxy_mechanism_for_family(family)
    if spec:
        return spec.family_group
    return (family or "unresolved family").strip()


def structural_tier(angle, n_gated, mechanism_status):
    return structural_support_tier(angle, n_gated, mechanism_status)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    parser = argparse.ArgumentParser()
    parser.add_argument("--representatives", default=os.path.join(root, "representatives.csv"))
    parser.add_argument("--angles", default=os.path.join(root, "results", "discovery_angle_gate.csv"))
    parser.add_argument("--out", default=os.path.join(root, "results", "candidate_prioritization.csv"))
    args = parser.parse_args()

    representatives = {row["accession"]: row for row in read_rows(args.representatives)}
    geometries = read_rows(args.angles)
    output = []

    for geometry in geometries:
        accession = geometry["id"]
        candidate = representatives[accession]
        mechanism, phosphate, electrophile, leaving_axis = mechanism_for_family(
            candidate.get("interpro_family")
        )
        proxy_spec = proxy_mechanism_for_family(candidate.get("interpro_family"))
        mechanism_status = "assigned" if mechanism != "unresolved" else "unresolved"
        if phosphate and geometry["phosphate"] != phosphate:
            mechanism_status = "phosphate mismatch"
        n_gated = int(geometry.get("n_gated") or 0)
        angle = float(geometry["mean_inline_angle"]) if geometry.get("mean_inline_angle") else None
        output.append({
            "agent_rank": candidate["rank"],
            "agent_tier": candidate["tier_label"],
            "accession": accession,
            "gene": candidate.get("gene", ""),
            "protein_name": candidate.get("protein_name", ""),
            "family": family_group(candidate.get("interpro_family")),
            "interpro_family": candidate.get("interpro_family", ""),
            "mechanism": mechanism,
            "mechanism_status": mechanism_status,
            "nucleophile": "adipate carboxylate O" if mechanism_status == "assigned" else "",
            "electrophile": electrophile,
            "leaving_group_axis": leaving_axis,
            "reactive_phosphate": phosphate,
            "proxy_version": geometry.get("proxy_version") or PROXY_VERSION,
            "proxy_evidence_scope": (
                proxy_spec.evidence_scope if proxy_spec else "unresolved"
            ),
            "n_distance_gated": n_gated,
            "n_models": geometry.get("n_models", ""),
            "distance_gated_fraction": geometry.get("gated_fraction", ""),
            "mean_inline_angle": "" if angle is None else f"{angle:.1f}",
            "angle_std": geometry.get("angle_std", ""),
            "angle_ci90_low": geometry.get("angle_ci90_low", ""),
            "angle_ci90_high": geometry.get("angle_ci90_high", ""),
            "prob_mean_ge_150": geometry.get("prob_mean_ge_150", ""),
            "prob_mean_ge_165": geometry.get("prob_mean_ge_165", ""),
            "inline_frac_gt150": geometry.get("inline_frac_gated", ""),
            "mean_ligand_iptm": geometry.get("mean_iptm", ""),
            "structural_tier": structural_tier(angle, n_gated, mechanism_status),
            "struct_rank_within_family": "",
        })

    rankable = defaultdict(list)
    for row in output:
        if row["structural_tier"] != "inconclusive":
            rankable[row["family"]].append(row)
    for rows in rankable.values():
        rows.sort(key=lambda row: (-float(row["mean_inline_angle"]), int(row["agent_rank"])))
        for rank, row in enumerate(rows, 1):
            row["struct_rank_within_family"] = rank

    # Keep discovery evidence as the primary table order.
    output.sort(key=lambda row: (int(row["agent_rank"]), row["accession"]))
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    excluded = sorted(set(representatives) - {row["id"] for row in geometries})
    counts = defaultdict(int)
    for row in output:
        counts[row["structural_tier"]] += 1
    print(f"Wrote {len(output)} completed candidates to {args.out}")
    for tier in ("strong geometric support", "compatible", "weak geometric support", "inconclusive"):
        print(f"  {tier}: {counts[tier]}")
    print("Excluded without structural results: " + ", ".join(excluded))


if __name__ == "__main__":
    main()
