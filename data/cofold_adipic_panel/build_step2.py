#!/usr/bin/env python3
"""Build pocket-restrained STEP-2 (aminolysis) cofold inputs for the SfaB panel.

Per enzyme:  A=protein(+MSA)  B=acyl-AMP(adipyl-AMP)  C=diamine(HMDA)  D=Mg
Pocket restraint (soft) on the diamine (binder C): the first 15 protein residues
that contact the ligand+cofactor in the enzyme's STEP-1 cofold (ATP+2Mg+adipate),
max_distance 10 A, force true  -- exactly the AS9 reference recipe.

Sequences: AS1-48 from SfaB_diversity_panel.csv; SfaB parent from UniProt.
MSA: matched by sha1(seq)[:16] to msa_cache/<hash>.csv (reused from step-1).
Pocket: read from exp_data/sfab_step1_results (model_0 of each enzyme).
"""
import csv, glob, hashlib, os, urllib.request
import numpy as np

ROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
PANEL = f"{ROOT}/data/cofold_adipic_panel"
MSA = f"{PANEL}/msa_cache"
S1 = f"{ROOT}/exp_data/sfab_step1_results/output_configs"
OUT = f"{PANEL}/step2"; os.makedirs(OUT, exist_ok=True)

ACYLAMP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OC(=O)CCCCC(=O)[O-])[C@@H](O)[C@H]1O"
HMDA = "NCCCCCC[NH3+]"                       # one free -NH2 nucleophile, one -NH3+
LIG_CHAINS = ["B", "C", "D", "E"]           # step-1: ATP, MG, MG, adipate
CUTOFF, NRES = 5.0, 15

def seqhash(s): return hashlib.sha1(s.encode()).hexdigest()[:16]

def pocket_residues(name):
    """First NRES protein residues within CUTOFF of any lig/cofactor atom (step-1 model_0)."""
    pdb = f"{S1}/step1_{name}/boltz_results_step1_{name}/predictions/step1_{name}/step1_{name}_model_0.pdb"
    if not os.path.exists(pdb):
        return None
    prot = {}          # resnum -> list of heavy-atom xyz (chain A)
    lig = []           # ligand/cofactor heavy-atom xyz (chains B..E)
    for line in open(pdb):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        chain = line[21].strip()
        elem = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if elem == "H":
            continue
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if chain == "A" and line.startswith("ATOM"):
            prot.setdefault(int(line[22:26]), []).append(xyz)
        elif chain in LIG_CHAINS:
            lig.append(xyz)
    if not lig or not prot:
        return None
    lig = np.array(lig)
    hits = [rn for rn, coords in prot.items()
            if np.linalg.norm(np.array(coords)[:, None, :] - lig[None, :, :], axis=2).min() <= CUTOFF]
    return sorted(hits)[:NRES]

def yaml_for(seq, msa_path, contacts):
    c = "\n".join(f"        - [A, {r}]" for r in contacts)
    return (
        "version: 1\n"
        "# STEP 2 aminolysis: A=prot B=adipyl-AMP C=HMDA(nucleophile) D=Mg\n"
        "sequences:\n"
        "- protein:\n    id: A\n"
        f"    sequence: {seq}\n    msa: {msa_path}\n"
        "- ligand:\n    id: B\n"
        f"    smiles: '{ACYLAMP}'\n"
        "- ligand:\n    id: C\n"
        f"    smiles: '{HMDA}'\n"
        "- ligand:\n    id: D\n    ccd: MG\n"
        "constraints:\n  - pocket:\n      binder: C\n"
        f"      contacts:\n{c}\n      max_distance: 10.0\n      force: true\n"
    )

def main():
    seqs = {}
    for r in csv.DictReader(open(f"{ROOT}/exp_data/SfaB_diversity_panel.csv")):
        if r["id"].startswith("AS") and r["sequence"]:
            seqs[r["id"]] = r["sequence"]
    # SfaB parent from UniProt
    fa = urllib.request.urlopen("https://rest.uniprot.org/uniprotkb/A0A2H4T912.fasta").read().decode()
    seqs["SfaB"] = "".join(l.strip() for l in fa.splitlines() if not l.startswith(">"))

    n, miss = 0, []
    for name, seq in seqs.items():
        h = seqhash(seq); mp = f"{MSA}/{h}.csv"
        contacts = pocket_residues(name)
        if not os.path.exists(mp) or not contacts:
            miss.append((name, os.path.exists(mp), bool(contacts))); continue
        open(f"{OUT}/step2_{name}.yaml", "w").write(yaml_for(seq, mp, contacts))
        n += 1
    print(f"Wrote {n} step-2 YAMLs to {OUT}; missing: {miss}")

if __name__ == "__main__":
    main()
