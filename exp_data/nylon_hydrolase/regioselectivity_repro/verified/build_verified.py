#!/usr/bin/env python3
"""ISOLATED, self-contained rebuild of the 6 regioselectivity inputs.

Written because the shared build_substrates.py / substrates.json in the parent dir
were concurrently modified by another session and became internally inconsistent
(scissile atom names did not match their SMILES; SMARTS matched 2 sites). This
script depends on NOTHING in the parent dir except the panel sequences/MSAs, marks
the scissile carbonyl UNAMBIGUOUSLY with atom-map labels [C:1](=[O:2]) in the SMILES
itself, verifies it, strips the labels for the Boltz ligand, and reproduces Boltz's
exact atom-naming to pin the restraint. Outputs -> verified/inputs, verified/results.
"""
import csv, hashlib, json, os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

EROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
HERE = os.path.dirname(os.path.abspath(__file__))
SEQS = f"{EROOT}/exp_data/nylon_hydrolase/annotation/amide_hydrolase_panel_sequences.csv"
SPLIT = f"{EROOT}/exp_data/nylon_hydrolase/cofold_hydrolase_panel/split_sites.csv"
MSA_DIR = f"{EROOT}/data/cofold_hydrolase_panel/frag_msa"
INP = f"{HERE}/inputs"; os.makedirs(INP, exist_ok=True)

# neutrally capped tetrads; the scissile carbonyl (the one pinned to the catalytic
# Thr) is marked in-SMILES with [C:1](=[O:2]) so its identity is unambiguous.
#   MAXA precursor: M-A-X-A, scissile = A-M amide (carbonyl bonded to the M nitrogen)
#   XAMA precursor: X-A-M-A, scissile = A-X amide (carbonyl bonded to the X benzylic N)
MARKED = {
 "MAXA_precursor": "CC(=O)NCCCCCCN[C:1](=[O:2])CCCCC(=O)NCc1ccc(CNC(=O)CCCCC(=O)NC)cc1",
 "XAMA_precursor": "CC(=O)NCc1ccc(CN[C:1](=[O:2])CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)NC)cc1",
}
ASN = {"Nyl10": 180, "Nyl12": 186, "Nyl50": 180}
FY = {"Nyl10": 106, "Nyl12": 113, "Nyl50": 105}
NCOPIES = 4
ALPHA = ["A", "C", "E", "G"][:NCOPIES]
BETA = ["B", "D", "F", "H"][:NCOPIES]
LIG = "I"
MAXD = 4.0

seqs = {r["candidate_id"]: r["sequence"] for r in csv.DictReader(open(SEQS))}
split = {r["id"]: int(r["split_resnum_1based"]) for r in csv.DictReader(open(SPLIT))}


def resolve(marked):
    """Return (plain_smiles, scissile_C_name, scissile_O_name) from a map-marked SMILES."""
    m = Chem.MolFromSmiles(marked)
    assert m is not None, f"bad SMILES: {marked}"
    c_idx = o_idx = None
    for a in m.GetAtoms():
        if a.GetAtomMapNum() == 1: c_idx = a.GetIdx()
        elif a.GetAtomMapNum() == 2: o_idx = a.GetIdx()
    assert c_idx is not None and o_idx is not None, "scissile map labels missing"
    # scissile C must be a carbonyl carbon bonded to exactly one amide N (the cleaved bond)
    cat = m.GetAtomWithIdx(c_idx)
    nbrs = {n.GetSymbol(): n for n in cat.GetNeighbors()}
    assert "N" in nbrs and "O" in nbrs, "scissile C is not an amide carbonyl"
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)                       # strip labels for the Boltz ligand
    plain = Chem.MolToSmiles(m)
    # reproduce Boltz naming on a fresh parse of the plain SMILES (graph-invariant)
    mp = Chem.MolFromSmiles(plain); mH = Chem.AddHs(mp)
    ranks = list(Chem.CanonicalRankAtoms(mH))
    # map scissile atoms from m -> mp by canonical rank match (unique heavy atoms)
    rk_m = list(Chem.CanonicalRankAtoms(Chem.AddHs(m)))
    name_by_rank = {ranks[i]: mp.GetAtomWithIdx(i).GetSymbol().upper() + str(ranks[i] + 1)
                    for i in range(mp.GetNumAtoms())}
    cC = name_by_rank[rk_m[c_idx]]; cO = name_by_rank[rk_m[o_idx]]
    return plain, cC, cO, rdMolDescriptors.CalcMolFormula(mp), round(Descriptors.MolWt(mp), 1)


def frag_msa(seq):
    return os.path.join(MSA_DIR, hashlib.sha1(seq.encode()).hexdigest()[:16] + ".csv")


def main():
    subs = {}
    for label, marked in MARKED.items():
        plain, cC, cO, formula, mw = resolve(marked)
        subs[label] = {"smiles": plain, "scissile_C_name": cC, "scissile_O_name": cO,
                       "formula": formula, "mw": mw}
        print(f"[{label}] {formula} MW {mw}  smiles={plain}\n    scissile C={cC} O={cO}")
    json.dump(subs, open(f"{HERE}/substrates_verified.json", "w"), indent=2)

    n = 0
    for enz in ["Nyl10", "Nyl12", "Nyl50"]:
        s = seqs[enz]; k = split[enz]; alpha, beta = s[:k - 1], s[k - 1:]
        assert beta[0] == "T"
        ma, mb = frag_msa(alpha), frag_msa(beta)
        assert os.path.exists(ma) and os.path.exists(mb), f"missing MSA for {enz}"
        for label, sub in subs.items():
            cn, on = sub["scissile_C_name"], sub["scissile_O_name"]
            L = ["version: 1", "sequences:"]
            for ch in ALPHA:
                L.append(f"- protein:\n    id: {ch}\n    sequence: {alpha}\n    msa: {ma}")
            for ch in BETA:
                L.append(f"- protein:\n    id: {ch}\n    sequence: {beta}\n    msa: {mb}")
            L.append(f"- ligand:\n    id: {LIG}\n    smiles: '{sub['smiles']}'")
            L += ["constraints:",
                  f"  - pocket:\n      binder: {LIG}\n      contacts: [[B, 1], [A, {ASN[enz]}], [A, {FY[enz]}]]\n"
                  f"      max_distance: 6.0\n      force: true",
                  f"  - contact:\n      token1: [B, 1]\n      token2: [{LIG}, {cn}]\n      max_distance: {MAXD}\n      force: true",
                  f"  - contact:\n      token1: [A, {ASN[enz]}]\n      token2: [{LIG}, {on}]\n      max_distance: {MAXD}\n      force: true"]
            open(f"{INP}/{enz}__{label}.yaml", "w").write("\n".join(L) + "\n")
            n += 1
    print(f"\nwrote {n} verified YAMLs -> {INP}")


if __name__ == "__main__":
    main()
