#!/usr/bin/env python3
"""Build 1-Mg step-1 cofold inputs for the ANL-mechanism control experiment.

ANL / adenylate-forming enzymes (SfaB and its family) use a SINGLE Mg2+ in the
adenylation half-reaction (UniProt: firefly luciferase P08659, acetyl-CoA synthetase
P27550 both annotate one Mg2+). The original panel cofold used a blanket 2 Mg (the
agent's mechanism-agnostic fallback, because SfaB=A0A2H4T912 is TrEMBL with no cofactor
note). This rebuilds SfaB + the 3 measured actives with ATP + 1 Mg + adipate, identical
otherwise, to test whether the extra Mg was why Boltz parked SfaB's diacid 10 A from Pa.
Layout: A=protein(+MSA), B=ATP, C=MG, D=adipate; affinity on D.
"""
import os, re

SRC = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/data/amide_synthetase/step1"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/inputs"
ADIPATE = "[O-]C(=O)CCCCC(=O)[O-]"
ENZYMES = ["SfaB", "AS9", "AS17", "AS23"]


def field(text, key):
    m = re.search(rf"^\s*{key}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else None


for e in ENZYMES:
    txt = open(f"{SRC}/step1_{e}.yaml").read()
    seq = field(txt, "sequence")
    msa = field(txt, "msa")
    assert seq and msa, e
    yaml = (
        "version: 1\n"
        "sequences:\n"
        "- protein:\n"
        "    id: A\n"
        f"    sequence: {seq}\n"
        f"    msa: {msa}\n"
        "- ligand:\n"
        "    id: B\n"
        "    ccd: ATP\n"
        "- ligand:\n"
        "    id: C\n"
        "    ccd: MG\n"
        "- ligand:\n"
        "    id: D\n"
        f"    smiles: '{ADIPATE}'\n"
        "properties:\n"
        "- affinity:\n"
        "    binder: D\n"
    )
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/step1_{e}.yaml", "w") as f:
        f.write(yaml)
    print("wrote", f"{OUT}/step1_{e}.yaml", "| 1 Mg")
