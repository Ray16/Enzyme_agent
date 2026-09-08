#!/usr/bin/env python3
"""Build step-1 (adenylation) cofold YAMLs for the discovery-agent candidate panel.

Per enzyme:  A=protein(+MSA)  B=ATP  C=MG  D=MG  E=adipate (dianion); affinity on E.
Mirrors the validated SfaB step-1 setup. 50 diffusion samples set at run time.
Sequences from representatives.fasta; MSA matched via msa_cache/index.tsv (id->csv).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # structural_modeling/
FASTA = f"{ROOT}/representatives.fasta"
INDEX = f"{ROOT}/msa_cache/index.tsv"
OUT = f"{ROOT}/step1"; os.makedirs(OUT, exist_ok=True)
ADIPATE = "[O-]C(=O)CCCCC(=O)[O-]"               # adipate dianion


def read_fasta(path):
    seqs = {}; acc = None; buf = []
    for line in open(path):
        if line.startswith(">"):
            if acc: seqs[acc] = "".join(buf)
            acc = line[1:].strip().split()[0]; buf = []
        else:
            buf.append(line.strip())
    if acc: seqs[acc] = "".join(buf)
    return seqs


def yaml_for(seq, msa):
    return (
        "version: 1\nsequences:\n- protein:\n    id: A\n"
        f"    sequence: {seq}\n    msa: {msa}\n"
        "- ligand:\n    id: B\n    ccd: ATP\n"
        "- ligand:\n    id: C\n    ccd: MG\n"
        "- ligand:\n    id: D\n    ccd: MG\n"
        f"- ligand:\n    id: E\n    smiles: '{ADIPATE}'\n"
        "properties:\n- affinity:\n    binder: E\n"
    )


def main():
    seqs = read_fasta(FASTA)
    msa = {}
    for line in open(INDEX):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            msa[parts[0]] = parts[2]
    n, miss = 0, []
    for acc, seq in seqs.items():
        mp = msa.get(acc)
        if not mp or not os.path.exists(mp):
            miss.append(acc); continue
        open(f"{OUT}/step1_{acc}.yaml", "w").write(yaml_for(seq, os.path.abspath(mp)))
        n += 1
    print(f"wrote {n} step-1 YAMLs to {OUT}; missing MSA: {miss}")


if __name__ == "__main__":
    main()
