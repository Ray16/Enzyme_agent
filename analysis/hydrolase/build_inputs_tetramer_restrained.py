#!/usr/bin/env python3
"""Build the RESTRAINED arm (B) of the nylon-hydrolase tetramer cofold panel.

Arm A (unrestrained) is written by build_inputs_tetramer.py. This script writes the
matched restrained arm, using the same 4x(alpha+beta) assembly and the same neutrally
capped tetrad, plus the two atom-level contacts + pocket restraint from the
regioselectivity protocol (regioselectivity_repro/build_inputs.py, itself following
Dishner/Bourgery JACS 2026):

    contact  beta1 Thr (chain B res 1)      <-> scissile carbonyl C   force, 4.0 A
    contact  alpha1 oxyanion Asn (chain A)  <-> scissile carbonyl O   force, 4.0 A
    pocket   ligand seated on [[B,1],[A,Asn],[A,FY]]                  soft,  6.0 A

WHY BOTH ARMS. Restraints manufacture near-attack geometry, so the restrained poses
cannot be used to rank enzymes by near-attack geometry -- that would be circular. The
pair is what is informative:
  * Arm A measures whether productive geometry forms SPONTANEOUSLY (the discriminator),
    and stays apples-to-apples with the existing 96-enzyme heterodimer null.
  * Arm B guarantees a productive register, giving a pose on which energy /
    electrostatics / subsite-packing terms are meaningful.
  * A-vs-B contrast = RESTRAINT STRAIN: how much confidence the enzyme gives up when
    forced into productive geometry. A competent site accommodates it cheaply.

OXYANION Asn / aromatic FY POSITIONS are curated in the source study for only 3 of 96
enzymes (Nyl10 180/106, Nyl12 186/113, Nyl50 180/105). Here they are transferred to all
96 through the panel MSA (panel_aln.fasta). Every transferred position is VERIFIED to
carry the expected residue identity; enzymes failing verification are reported and
skipped rather than silently restrained to the wrong atom.

Usage: python analysis/hydrolase/build_inputs_tetramer_restrained.py [Nyl01 ...]
"""
import csv, os, sys, hashlib
from collections import Counter
from rdkit import Chem

EROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
ROOT = f"{EROOT}/data/cofold_hydrolase_panel"
SEQS = f"{EROOT}/exp_data/nylon_hydrolase/annotation/amide_hydrolase_panel_sequences.csv"
SPLIT = f"{EROOT}/exp_data/nylon_hydrolase/cofold_hydrolase_panel/split_sites.csv"
ALN = f"{EROOT}/exp_data/nylon_hydrolase/cofold_hydrolase_panel/panel_aln.fasta"
MSA_DIR = os.path.join(ROOT, "frag_msa")
INP = os.path.join(ROOT, "inputs_tetramer_restr"); os.makedirs(INP, exist_ok=True)

NCOPIES = 4
ALPHA = ["A", "C", "E", "G"]; BETA = ["B", "D", "F", "H"]; LIG = "I"
MAXD = 4.0          # atom-contact pull, matches regioselectivity_repro
POCKET_D = 6.0

# neutrally-capped nylon-6,6 tetrad (internal segment) -- identical to arm A
TETRAD = "CC(=O)NCCCCCCNC(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)NC"

# curated anchors from the regioselectivity study (full-length == alpha-chain numbering)
ANCHOR_ASN = {"Nyl10": 180, "Nyl12": 186, "Nyl50": 180}
ANCHOR_FY = {"Nyl10": 106, "Nyl12": 113, "Nyl50": 105}


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            if name: seqs[name] = "".join(buf)
            name = ln[1:].split()[0]; buf = []
        else:
            buf.append(ln.strip())
    if name: seqs[name] = "".join(buf)
    return seqs


def res_to_col(aln_seq):
    """1-based ungapped residue index -> 0-based alignment column."""
    m, r = {}, 0
    for c, ch in enumerate(aln_seq):
        if ch != "-":
            r += 1; m[r] = c
    return m


def col_to_res(aln_seq):
    """0-based alignment column -> 1-based ungapped residue index (None if gap)."""
    m, r = {}, 0
    for c, ch in enumerate(aln_seq):
        if ch != "-":
            r += 1; m[c] = r
        else:
            m[c] = None
    return m


def scissile_atom_names(smiles):
    """Boltz ligand atom names = <ELEM><CanonicalRankAtoms(AddHs)+1>.

    Pick the most CENTRAL amide C(=O)N of the capped tetrad as the scissile bond --
    an internal polymer amide, not a capping amide.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "bad tetrad SMILES"
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]"))
    assert matches, "no amide found"
    dmat = Chem.GetDistanceMatrix(mol)
    n = mol.GetNumAtoms()
    # centrality = lowest max graph distance to any other atom (graph centre)
    best = min(matches, key=lambda m: max(dmat[m[0]][j] for j in range(n)))
    c_idx, o_idx = best[0], best[1]
    ranks = list(Chem.CanonicalRankAtoms(Chem.AddHs(mol)))
    return f"C{ranks[c_idx] + 1}", f"O{ranks[o_idx] + 1}", c_idx, o_idx


def frag_msa(seq):
    return os.path.join(MSA_DIR, hashlib.sha1(seq.encode()).hexdigest()[:16] + ".csv")


def yaml_for(name, seqs, split, asn, fy, cn, on):
    s = seqs[name]; k = split[name]
    alpha, beta = s[:k - 1], s[k - 1:]
    ma, mb = frag_msa(alpha), frag_msa(beta)
    L = ["version: 1", "sequences:"]
    for ch in ALPHA[:NCOPIES]:
        L.append(f"- protein:\n    id: {ch}\n    sequence: {alpha}\n    msa: {ma}")
    for ch in BETA[:NCOPIES]:
        L.append(f"- protein:\n    id: {ch}\n    sequence: {beta}\n    msa: {mb}")
    L.append(f"- ligand:\n    id: {LIG}\n    smiles: '{TETRAD}'")
    # restraints target heterodimer-1 only (alpha=A, beta=B)
    L.append("constraints:")
    L.append(f"- contact:\n    token1: [B, 1]\n    token2: [{LIG}, {cn}]\n"
             f"    max_distance: {MAXD}\n    force: true")
    L.append(f"- contact:\n    token1: [A, {asn}]\n    token2: [{LIG}, {on}]\n"
             f"    max_distance: {MAXD}\n    force: true")
    L.append(f"- pocket:\n    binder: {LIG}\n    contacts: [[B, 1], [A, {asn}], [A, {fy}]]\n"
             f"    max_distance: {POCKET_D}\n    force: false")
    L.append(f"properties:\n- affinity:\n    binder: {LIG}")
    return "\n".join(L) + "\n"


def main():
    seqs = {r["candidate_id"]: r["sequence"] for r in csv.DictReader(open(SEQS))}
    split = {r["id"]: int(r["split_resnum_1based"]) for r in csv.DictReader(open(SPLIT))}
    aln = read_fasta(ALN)
    # alignment names may carry an accession suffix; key on the first token
    aln = {k.split()[0]: v for k, v in aln.items()}

    cn, on, ci, oi = scissile_atom_names(TETRAD)
    print(f"scissile amide: C idx {ci} -> {cn} , O idx {oi} -> {on}")

    # --- transfer the oxyanion anchors through the MSA ---------------------------
    def consensus_col(anchor):
        cols = []
        for enz, pos in anchor.items():
            if enz not in aln:
                print(f"  WARN anchor {enz} absent from alignment"); continue
            cols.append(res_to_col(aln[enz])[pos])
        c = Counter(cols)
        col, k = c.most_common(1)[0]
        print(f"  anchor cols {cols} -> consensus {col} ({k}/{len(cols)} agree)")
        return col

    print("Asn anchor:"); asn_col = consensus_col(ANCHOR_ASN)
    print("FY anchor:");  fy_col = consensus_col(ANCHOR_FY)

    subset = sys.argv[1:] or list(seqs)
    n, skipped = 0, []
    for name in subset:
        if name not in aln:
            skipped.append(f"{name}(not in aln)"); continue
        k = split.get(name)
        if k is None:
            skipped.append(f"{name}(no split)"); continue
        s = seqs[name]
        c2r = col_to_res(aln[name])
        asn, fy = c2r.get(asn_col), c2r.get(fy_col)
        if asn is None or fy is None:
            skipped.append(f"{name}(gap at anchor)"); continue
        if s[asn - 1] != "N":
            skipped.append(f"{name}(col{asn_col}->{s[asn-1]}{asn}!=N)"); continue
        if s[fy - 1] not in "FYW":
            skipped.append(f"{name}(col{fy_col}->{s[fy-1]}{fy} not FYW)"); continue
        if asn >= k or fy >= k:
            skipped.append(f"{name}(anchor not in alpha chain)"); continue
        alpha, beta = s[:k - 1], s[k - 1:]
        if beta[:1] != "T":
            skipped.append(f"{name}(beta[0]={beta[:1]}!=T)"); continue
        if not (os.path.exists(frag_msa(alpha)) and os.path.exists(frag_msa(beta))):
            skipped.append(f"{name}(missing MSA)"); continue
        open(os.path.join(INP, f"{name}.yaml"), "w").write(
            yaml_for(name, seqs, split, asn, fy, cn, on))
        n += 1
    print(f"wrote {n} restrained tetramer YAMLs -> {INP}")
    if skipped:
        print(f"SKIPPED {len(skipped)}:")
        for x in skipped: print(f"   {x}")


if __name__ == "__main__":
    main()
