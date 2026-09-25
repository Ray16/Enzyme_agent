#!/usr/bin/env python3
"""Build the mechanism-aware structural ranking for completed candidates.

Discovery rank remains the primary evidence rank. Boltz-2 geometry supplies a
separate support tier and a rank that is meaningful only within a family.
"""

import argparse
import csv
import os
from collections import defaultdict


PA_KEYS = ("iuca/iucc", "aerobactin")
PG_KEYS = (
    "atp-grasp",
    "carbamoyl-phosphate synthase",
    "dapdiamide",
    "fumarate--(s)-2,3- diaminopropanoate ligase",
    "gamma-glutamylputrescine synthetase",
    "glutamine synthetase",
    "l-propargylglycine--l-glutamate ligase",
    "phosphoribosylformylglycinamidine cyclo-ligase",
    "phosphoribosylglycinamide formyltransferase",
    "phosphoribosylglycinamide synthetase",
    "pyrrolysine biosynthesis protein pylc",
    "ribosomal bs6 modification enzyme rimk/lysine biosynthesis enzyme lysx",
    "tyramine--l-glutamate ligase",
)


def mechanism_for_family(family):
    """Return mechanism, reactive phosphate, electrophile, and leaving axis."""
    family = (family or "").strip().lower()
    if any(key in family for key in PA_KEYS):
        return "acyl-adenylate", "PA", "ATP P-alpha", "P-alpha--O3A"
    if any(key in family for key in PG_KEYS):
        return "acyl-phosphate", "PG", "ATP P-gamma", "P-gamma--O3B"
    return "unresolved", "", "", ""


def family_group(family):
    family = (family or "").lower()
    if "iuca/iucc" in family or "aerobactin" in family:
        return "Aerobactin NIS"
    if "glutamine synthetase" in family or "gamma-glutamylputrescine" in family:
        return "Glutamine synthetase"
    if "rimk" in family or "tyramine--l-glutamate" in family:
        return "RimK/bS6"
    if "dapdiamide" in family or "fumarate--(s)-2,3- diaminopropanoate" in family:
        return "Dapdiamide"
    if "phosphoribosylglycinamide synthetase" in family or "formyltransferase" in family:
        return "PurD/GAR"
    if "atp-grasp" in family or "pyrrolysine" in family or "l-propargylglycine" in family:
        return "ATP-grasp"
    # Keep residual classes separate rather than creating a heterogeneous
    # cross-family ranking bucket.
    return (family or "unresolved family").strip()


def structural_tier(angle, n_gated, mechanism_status):
    if mechanism_status != "assigned" or n_gated < 5 or angle is None:
        return "inconclusive"
    if angle >= 165.0:
        return "strong geometric support"
    if angle >= 150.0:
        return "compatible"
    return "weak geometric support"


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
            "n_distance_gated": n_gated,
            "mean_inline_angle": "" if angle is None else f"{angle:.1f}",
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
