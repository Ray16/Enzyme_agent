"""Isolate genuinely on-target NEW candidates: small-molecule carboxylate + amine
-> amide ATP ligases (the target's chemistry), excluding protein/tRNA/cofactor
amide chemistry that shares the ATP-amide mechanism but not the substrate class."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "candidates_new_reviewed.csv").open()))

# Exclude macromolecular / cofactor acceptors: the amide is formed onto a protein,
# tRNA, or a cofactor scaffold, not between two free small molecules.
EXCLUDE = re.compile(
    r"\[protein\]|tRNA|--\[|ribosomal|tubulin|elongation factor|"
    r"glutathione|biotin|lipoyl|lipo-|dethiobiotin|NAD\b|NAD\+|"
    r"pantothenate|phosphopanteth|folate|folyl|dihydrofolate|"
    r"cobyrinic|cobyric|coenzyme|thiamine|molybdopterin|"
    r"argininosuccinate|carbamoyl phosphate|CoA\b", re.I)

ontarget = []
for r in rows:
    text = f"{r['protein_name']} {r['catalytic_activity']}"
    if EXCLUDE.search(text):
        continue
    if not r["catalytic_activity"].strip():
        continue
    ontarget.append(r)

# collapse ortholog groups by protein-name base for a clean inspection list
by_name: dict[str, list] = {}
for r in ontarget:
    base = r["protein_name"].split(" (")[0].strip()
    by_name.setdefault(base, []).append(r)

print(f"on-target NEW candidate rows: {len(ontarget)}  |  distinct enzyme types: {len(by_name)}\n")
print("=== distinct on-target enzyme types NEW to the pipeline (with a representative reaction) ===")
for base, group in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
    rep = group[0]
    ca = rep["catalytic_activity"].split(" = ")[0][:75]
    print(f"[{len(group):3d}x] {base[:46]:46s} EC={rep['ec'] or '-':10s}")
    print(f"        e.g. {rep['accession']} {rep['gene'][:12]:12s} | {ca} + ATP ...")

out = HERE / "candidates_ontarget_new.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(ontarget)
print(f"\nwrote {out}")
