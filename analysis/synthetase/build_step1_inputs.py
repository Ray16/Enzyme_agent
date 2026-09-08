#!/usr/bin/env python3
"""Rebuild step-1 (adenylation) cofold YAMLs for the SfaB synthetase panel.

Per enzyme:  A=protein(+MSA)  B=ATP  C=MG  D=MG  E=adipate (dianion); affinity on E.
50 diffusion samples are set at run time. Sequences: AS1-48 from the panel CSV,
SfaB parent from UniProt. MSA matched by sha1(seq)[:16] to msa_cache/<hash>.csv.
"""
import csv, hashlib, os, urllib.request

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
SEQS = f"{ROOT}/exp_data/SfaB_diversity_panel.csv"
MSA = f"{ROOT}/data/cofold_adipic_panel/msa_cache"
OUT = f"{ROOT}/data/cofold_adipic_panel/step1"; os.makedirs(OUT, exist_ok=True)
ADIPATE = "[O-]C(=O)CCCCC(=O)[O-]"   # adipate dianion


def seqhash(s): return hashlib.sha1(s.encode()).hexdigest()[:16]


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
    seqs = {}
    for r in csv.DictReader(open(SEQS)):
        if r["id"].startswith("AS") and r["sequence"]:
            seqs[r["id"]] = r["sequence"]
    fa = urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/A0A2H4T912.fasta").read().decode()
    seqs["SfaB"] = "".join(l.strip() for l in fa.splitlines() if not l.startswith(">"))
    n, miss = 0, []
    for name, seq in seqs.items():
        h = seqhash(seq); mp = f"{MSA}/{h}.csv"
        if not os.path.exists(mp):
            miss.append(name); continue
        open(f"{OUT}/step1_{name}.yaml", "w").write(yaml_for(seq, mp))
        n += 1
    print(f"wrote {n} step-1 YAMLs to {OUT}; missing MSA: {miss}")


if __name__ == "__main__":
    main()
