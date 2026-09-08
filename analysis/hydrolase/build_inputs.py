#!/usr/bin/env python3
"""Build Boltz-2 cofold YAMLs for the nylon-hydrolase panel.

Models the autocleaved (catalytically competent) enzyme as a heterodimer:
  chain A = alpha subunit (residues 1 .. split-1)
  chain B = beta  subunit (residues split .. end; residue 1 = catalytic Thr)
  chain C = tetrad substrate (PA66 linear dimer, charged)
with affinity prediction on the tetrad.

Per-fragment MSAs (alpha/beta) live in frag_msa/, named by sha1(fragment_seq)[:16].csv.
"""
import csv, os, glob, hashlib, yaml

EROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
ROOT = f"{EROOT}/data/cofold_hydrolase_panel"
SEQS = f"{EROOT}/exp_data/annotation/amide_hydrolase_panel_sequences.csv"
SPLIT = f"{EROOT}/exp_data/cofold_hydrolase_panel/split_sites.csv"
MSA_DIR = os.path.join(ROOT, "frag_msa")   # per-fragment MSAs, named sha1(seq)[:16].csv
INP = os.path.join(ROOT, "inputs"); os.makedirs(INP, exist_ok=True)

# PA66 tetrad (linear dimer): HMDA-adipate-HMDA-adipate, charged termini
TETRAD = "[NH3+]CCCCCCNC(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)[O-]"

seqs = {r["candidate_id"]: r["sequence"] for r in csv.DictReader(open(SEQS))}
split = {r["id"]: int(r["split_resnum_1based"]) for r in csv.DictReader(open(SPLIT))}

def frag_msa(seq):
    return os.path.join(MSA_DIR, hashlib.sha1(seq.encode()).hexdigest()[:16] + ".csv")

def y(name):
    s = seqs[name]; k = split[name]
    alpha, beta = s[:k-1], s[k-1:]
    ma, mb = frag_msa(alpha), frag_msa(beta)
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
    k = split.get(name)
    if k is None:
        bad.append(name); continue
    s = seqs[name]; alpha, beta = s[:k-1], s[k-1:]
    if not (os.path.exists(frag_msa(alpha)) and os.path.exists(frag_msa(beta))):
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
