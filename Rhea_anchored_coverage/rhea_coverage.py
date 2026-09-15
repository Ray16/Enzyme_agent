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
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

_UNCHARGER = rdMolStandardize.Uncharger()


def _neutralize(mol):
    """Normalize protonation so charge state never affects matching (COO-/COOH,
    NH3+/NH2, etc. all collapse to a canonical neutral form)."""
    if mol is None:
        return None
    try:
        return _UNCHARGER.uncharge(mol)
    except Exception:
        return mol

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
    R = _neutralize(Chem.MolFromSmiles(lhs)); P = _neutralize(Chem.MolFromSmiles(rhs))
    if not R or not P:
        return False
    bp, ap = BOND_PATT[bond], ACCEPT_PATT[acceptor]
    return _nm(P, bp) > _nm(R, bp) and _nm(R, ACID) >= 1 and _nm(R, ap) >= 1


def iter_rhea_smiles():
    """Stream (rid, smiles) to keep memory flat over 36k reactions."""
    with open(CACHE / "rhea-reaction-smiles.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                yield parts[0], parts[1]


def match_reactions(acceptor="amine", bond="amide", cache_file="matched_rhea_ids.txt"):
    """Stream all Rhea reactions, write matched ids incrementally (resumable)."""
    out = CACHE / cache_file
    hits = []
    n = 0
    with open(out, "w") as fh:
        for rid, smi in iter_rhea_smiles():
            n += 1
            try:
                if forms_bond(smi, acceptor, bond):
                    hits.append(rid); fh.write(rid + "\n"); fh.flush()
            except Exception:
                continue
    print(f"scanned {n} reactions -> {len(hits)} matches (cached in {out.name})")
    return hits


def load_master2acc():
    """MASTER Rhea id -> reviewed accessions. Rhea numbers a reaction as a master
    (lowest id) plus 3 consecutive directional variants (master+1..+3); the
    smiles file holds the directional ids, rhea2uniprot maps the master."""
    m = collections.defaultdict(set)
    with open(CACHE / "rhea2uniprot_sprot.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m[row["MASTER_ID"]].add(row["ID"])
    return m


def matched_accessions(acceptor="amine", bond="amide"):
    cache = CACHE / "matched_rhea_ids.txt"
    if cache.exists():
        hit_rhea = set(cache.read_text().split())
    else:
        hit_rhea = set(match_reactions(acceptor, bond))
    master2acc = load_master2acc()
    accs = set()
    for rid in hit_rhea:
        try:
            d = int(rid)
        except ValueError:
            continue
        # a directional id maps back to its master at d, d-1, d-2 or d-3;
        # take the NEAREST master only (blocks are master,+1,+2,+3) to avoid
        # spilling into an adjacent reaction block.
        for cand in (d, d - 1, d - 2, d - 3):
            if str(cand) in master2acc:
                accs |= master2acc[str(cand)]
                break
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
