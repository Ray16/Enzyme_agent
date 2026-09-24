#!/usr/bin/env python3
"""Build IMPROVED Boltz-2 cofold YAMLs for the nylon-hydrolase panel.

Fixes two modeling errors in the original single-heterodimer build (build_inputs.py):
  1. ASSEMBLY: model the enzyme as the biological **tetramer of heterodimers** (4 alpha +
     4 beta), matching the ORNL regioselectivity paper — the substrate groove is formed at
     subunit interfaces, so a single heterodimer has an incomplete active site.
  2. SUBSTRATE: use a **neutrally capped** nylon-6,6 tetrad (acetyl / N-methylamide ends),
     representing an INTERNAL polymer segment. The old build used charged termini
     ([NH3+]...[O-]) which distort the near-active-site pose/energetics.

Autocleaved/mature form kept: chain*n = alpha (1..split-1), chain*n = beta (split..end,
residue 1 = catalytic Thr). Per-fragment MSAs reused from the original panel frag_msa/.
Unrestrained (like the original panel) + affinity on the ligand, so this is an
apples-to-apples re-test of whether the CORRECT assembly lets a mechanistic proxy predict.
"""
import csv, os, glob, hashlib, sys
from rdkit import Chem

EROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
ROOT = f"{EROOT}/data/cofold_hydrolase_panel"
SEQS = f"{EROOT}/exp_data/nylon_hydrolase/annotation/amide_hydrolase_panel_sequences.csv"
SPLIT = f"{EROOT}/exp_data/nylon_hydrolase/cofold_hydrolase_panel/split_sites.csv"
MSA_DIR = os.path.join(ROOT, "frag_msa")
INP = os.path.join(ROOT, "inputs_tetramer"); os.makedirs(INP, exist_ok=True)

NCOPIES = 4
ALPHA = ["A", "C", "E", "G"]; BETA = ["B", "D", "F", "H"]; LIG = "I"
# neutrally-capped nylon-6,6 tetrad (internal segment): Ac-[HMDA-adipoyl-HMDA-adipoyl]-NHMe
TETRAD = "CC(=O)NCCCCCCNC(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)NC"
assert Chem.MolFromSmiles(TETRAD) is not None, "bad capped tetrad SMILES"
assert Chem.GetFormalCharge(Chem.MolFromSmiles(TETRAD)) == 0, "tetrad not neutral"

seqs = {r["candidate_id"]: r["sequence"] for r in csv.DictReader(open(SEQS))}
split = {r["id"]: int(r["split_resnum_1based"]) for r in csv.DictReader(open(SPLIT))}


def frag_msa(seq):
    return os.path.join(MSA_DIR, hashlib.sha1(seq.encode()).hexdigest()[:16] + ".csv")


def yaml_for(name):
    s = seqs[name]; k = split[name]
    alpha, beta = s[:k - 1], s[k - 1:]
    ma, mb = frag_msa(alpha), frag_msa(beta)
    L = ["version: 1", "sequences:"]
    for ch in ALPHA[:NCOPIES]:
        L.append(f"- protein:\n    id: {ch}\n    sequence: {alpha}\n    msa: {ma}")
    for ch in BETA[:NCOPIES]:
        L.append(f"- protein:\n    id: {ch}\n    sequence: {beta}\n    msa: {mb}")
    L.append(f"- ligand:\n    id: {LIG}\n    smiles: '{TETRAD}'")
    L.append(f"properties:\n- affinity:\n    binder: {LIG}")
    return "\n".join(L) + "\n"


def main():
    subset = sys.argv[1:] if len(sys.argv) > 1 else list(seqs)
    n = 0; bad = []
    for name in subset:
        k = split.get(name)
        if k is None:
            bad.append(f"{name}(no split)"); continue
        s = seqs[name]; alpha, beta = s[:k - 1], s[k - 1:]
        if beta[:1] != "T":
            bad.append(f"{name}(beta[0]={beta[:1]}!=T)")
        if not (os.path.exists(frag_msa(alpha)) and os.path.exists(frag_msa(beta))):
            bad.append(f"{name}(missing MSA)"); continue
        open(os.path.join(INP, f"{name}.yaml"), "w").write(yaml_for(name))
        n += 1
    print(f"wrote {n} tetramer YAMLs -> {INP}")
    if bad:
        print(f"skipped/flagged {len(bad)}: {bad[:10]}{'...' if len(bad) > 10 else ''}")


if __name__ == "__main__":
    main()
