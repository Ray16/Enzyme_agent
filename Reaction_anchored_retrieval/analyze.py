"""Characterize the new reaction-anchored candidates for eyeball verification."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "candidates_new_reviewed.csv").open()))
print(f"new reviewed candidates: {len(rows)}\n")

# EC class distribution (first two levels)
ec2 = Counter()
for r in rows:
    for ec in filter(None, r["ec"].split(";")):
        parts = ec.split(".")
        ec2[".".join(parts[:2]) if len(parts) >= 2 else ec] += 1
print("=== EC class (top 12) ===")
for ec, n in ec2.most_common(12):
    print(f"  {ec:8s} {n}")
no_ec = sum(1 for r in rows if not r["ec"].strip())
print(f"  (no EC): {no_ec}")

# Off-target amide-ligase families we do NOT want (cofactor/tRNA/peptide, not
# the diacid+diamine target chemistry). Keyword-classify by protein name + rxn.
OFFTARGET = re.compile(
    r"tRNA|glutathione|NAD\+? synthetase|NAD synthase|biotin|lipo-|lipoyl|"
    r"pantothen|folyl|folate|dihydrofolate|argininosuccinate|"
    r"asparagine synthetase|glutamine synthetase|carbamoyl|ubiquitin|"
    r"cobyric|cobyrinic|coenzyme|urea|formyl|glycinamide|phosphoribosyl",
    re.I)
ONTARGET_HINT = re.compile(
    r"diamine|polyamine|putrescine|spermidine|spermine|cadaverine|"
    r"citrate|siderophore|aerobactin|staphyloferrin|ligase|synthetase", re.I)

off = [r for r in rows if OFFTARGET.search(r["protein_name"] + " " + r["catalytic_activity"])]
print(f"\n=== likely OFF-target (cofactor/tRNA/peptide amide ligases): {len(off)} ===")
print("  e.g.:", ", ".join(sorted({r["protein_name"][:40] for r in off})[:8]))

rest = [r for r in rows if r not in off]
print(f"\n=== remaining (candidate on-target pool): {len(rest)} ===")
# Show the most promising: those whose reaction involves an amine + acid chain
name_counts = Counter(r["protein_name"].split(" (")[0][:45] for r in rest)
print("  most common remaining protein names:")
for name, n in name_counts.most_common(20):
    print(f"    {n:3d}  {name}")

# Spotlight: entries whose catalytic activity mentions a diamine/polyamine/diacid
spot = [r for r in rest if re.search(r"diamin|putrescin|spermidin|spermin|cadaverin|"
                                     r"glutar|adip|succinyl|citr", r["catalytic_activity"], re.I)]
print(f"\n=== SPOTLIGHT: remaining with di-amine/di-acid-like native chemistry: {len(spot)} ===")
for r in spot[:25]:
    ca = r["catalytic_activity"][:90]
    print(f"  {r['accession']} {r['gene'][:10]:10s} EC={r['ec'] or '-':12s} {r['protein_name'][:38]}")
    print(f"      rxn: {ca}")
