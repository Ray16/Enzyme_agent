#!/usr/bin/env python3
"""Build Boltz-2 cofold YAMLs for the nylon-hydrolase panel.

Models the autocleaved (catalytically competent) enzyme as a heterodimer:
  chain A = alpha subunit (residues 1 .. split-1)
  chain B = beta  subunit (residues split .. end; residue 1 = catalytic Thr)
  chain C = tetrad substrate (PA66 linear dimer, charged)
with affinity prediction on the tetrad.

MSAs are the per-fragment batched MSAs (frag_msa/index.tsv), keyed <id>_a / <id>_b.
"""
import csv, os, glob, yaml

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/data/cofold_hydrolase_panel"
SEQS = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/data/amide_hydrolase_panel_sequences.csv"
SPLIT = os.path.join(ROOT, "split_sites.csv")
FRAG_INDEX = os.path.join(ROOT, "frag_msa", "index.tsv")
INP = os.path.join(ROOT, "inputs"); os.makedirs(INP, exist_ok=True)

# PA66 tetrad (linear dimer): HMDA-adipate-HMDA-adipate, charged termini
TETRAD = "[NH3+]CCCCCCNC(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)[O-]"

seqs = {r["candidate_id"]: r["sequence"] for r in csv.DictReader(open(SEQS))}
split = {r["id"]: int(r["split_resnum_1based"]) for r in csv.DictReader(open(SPLIT))}
msa = {}
with open(FRAG_INDEX) as f:
    rd = csv.reader(f, delimiter="\t"); next(rd)
    for row in rd:
        if len(row) >= 3:
            msa[row[0]] = os.path.abspath(row[2])

def y(name):
    s = seqs[name]; k = split[name]
    alpha, beta = s[:k-1], s[k-1:]
    ma, mb = msa[f"{name}_a"], msa[f"{name}_b"]
    return (
        "version: 1\n"
        "sequences:\n"
        "- protein:\n    id: A\n"
        f"    sequence: {alpha}\n    msa: {ma}\n"
        "- protein:\n    id: B\n"
        f"    sequence: {beta}\n    msa: {mb}\n"
        "- ligand:\n    id: C\n"
        f"    smiles: '{TETRAD}'\n"
        "properties:\n- affinity:\n    binder: C\n"
    )

n = 0; bad = []
for name in seqs:
    if f"{name}_a" not in msa or f"{name}_b" not in msa:
        bad.append(name); continue
    open(os.path.join(INP, f"{name}.yaml"), "w").write(y(name))
    n += 1

# validate
mal = 0
for fn in glob.glob(os.path.join(INP, "*.yaml")):
    d = yaml.safe_load(open(fn))
    prot = [x for x in d["sequences"] if "protein" in x]
    lig = [x["ligand"] for x in d["sequences"] if "ligand" in x]
    aff = d.get("properties", [{}])[0].get("affinity", {}).get("binder")
    if not (len(prot) == 2 and prot[1]["protein"]["sequence"][0] == "T"
            and len(lig) == 1 and lig[0]["smiles"] == TETRAD and aff == "C"):
        mal += 1; print("BAD", os.path.basename(fn))
print(f"Wrote {n} heterodimer YAMLs (missing MSA: {bad}); malformed: {mal}")
