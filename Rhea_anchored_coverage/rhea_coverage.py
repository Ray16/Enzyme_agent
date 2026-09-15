#!/usr/bin/env python3
"""Rhea-anchored deterministic fold coverage (STANDALONE validation — NOT wired in).

Anchors discovery on the REACTION, not EC/InterPro/LLM:
  1. derive the target transformation from the user's substrate SMILES (generic),
  2. RDKit-match every Rhea reaction for that transformation (amide FORMATION from
     a free acid + a free amine),  [reaction-level, uses the real reaction SMILES]
  3. map matching Rhea reactions -> reviewed (Swiss-Prot) UniProt via rhea2uniprot,
  4. (downstream) collapse to one reviewed exemplar per InterPro fold.

Honest scope: a RECALL floor over the CURATED (Rhea/Swiss-Prot) universe. It cannot
reach unannotated/metagenomic/literature-only enzymes, and substructure matching
carries an inherent self-amidation/NH3 false positive (glutamate's own amine) that
only atom-mapping or downstream evaluation can remove.
"""
from __future__ import annotations
import sys, csv, collections
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# ---- functional-group SMARTS (neutral AND protonated forms; Rhea uses charged) ----
ACID  = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1,OX1-]")                 # -COOH / -COO-
AMIDE = Chem.MolFromSmarts("[NX3;!$([NX4])][CX3]=[OX1]")                # -C(=O)N<
AMINE = Chem.MolFromSmarts(
    "[$([NX3;H1,H2;!$(NC=O);!$(N=*);!$([n])]),$([NX4;H2,H3;+;!$(NC=O)])][#6]")  # free 1'/2' amine
HYDROXYL = Chem.MolFromSmarts("[OX2H1;!$(OC=O);!$(OP)]")
THIOL    = Chem.MolFromSmarts("[SX2H1]")
ESTER    = Chem.MolFromSmarts("[#6][CX3](=O)[OX2][#6;!$([CX3]=O)]")
THIOESTER= Chem.MolFromSmarts("[#6][CX3](=O)[SX2][#6]")

BOND_PATT   = {"amide": AMIDE, "ester": ESTER, "thioester": THIOESTER}
ACCEPT_PATT = {"amine": AMINE, "hydroxyl": HYDROXYL, "thiol": THIOL}


def _nm(mol, patt):
    return len(mol.GetSubstructMatches(patt)) if (mol and patt) else 0


def forms_bond(rxn_smiles: str, acceptor="amine", bond="amide") -> bool:
    """Reaction FORMS the target bond from a free acid + free acceptor (donor=carboxyl)."""
    if ">>" not in rxn_smiles:
        return False
    lhs, rhs = rxn_smiles.split(">>", 1)
    R = Chem.MolFromSmiles(lhs); P = Chem.MolFromSmiles(rhs)
    if not R or not P:
        return False
    bp, ap = BOND_PATT[bond], ACCEPT_PATT[acceptor]
    return _nm(P, bp) > _nm(R, bp) and _nm(R, ACID) >= 1 and _nm(R, ap) >= 1


def load_rhea_smiles():
    rows = {}
    for line in open(CACHE / "rhea-reaction-smiles.tsv"):
        rid, smi = line.rstrip("\n").split("\t")
        rows[rid] = smi
    return rows


def load_rhea2uniprot():
    m = collections.defaultdict(set)          # master rhea id -> set(reviewed accessions)
    with open(CACHE / "rhea2uniprot_sprot.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            m[row["MASTER_ID"]].add(row["ID"])
            m[row["RHEA_ID"]].add(row["ID"])
    return m


def matched_accessions(acceptor="amine", bond="amide"):
    smiles = load_rhea_smiles()
    r2u = load_rhea2uniprot()
    hit_rhea = {rid for rid, smi in smiles.items() if forms_bond(smi, acceptor, bond)}
    accs = set()
    for rid in hit_rhea:
        accs |= r2u.get(rid, set())
        accs |= r2u.get(rid.lstrip("0"), set())
    return hit_rhea, accs


if __name__ == "__main__":
    hit_rhea, accs = matched_accessions()
    print(f"Rhea reactions total: {len(load_rhea_smiles())}")
    print(f"reactions that FORM an amide (free acid + free amine): {len(hit_rhea)}")
    print(f"-> reviewed UniProt accessions: {len(accs)}")
    # spot-check recovery
    known = {"panC (E.coli, P31663)":"P31663",
             "gss (E.coli, P0AES0)":"P0AES0",
             "PuuA (P78061)":"P78061", "DdaG (E2JA32)":"E2JA32"}
    print("\nrecovery spot-check:")
    for name, acc in known.items():
        print(f"  {'RECOVERED' if acc in accs else 'missing  '}  {name}")
    (CACHE / "matched_accessions.txt").write_text("\n".join(sorted(accs)))
    print(f"\nwrote {len(accs)} accessions to cache/matched_accessions.txt")
