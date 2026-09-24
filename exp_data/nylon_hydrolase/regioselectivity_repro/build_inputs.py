#!/usr/bin/env python3
"""Step 1 (cont.): build the 6 restrained Boltz-2 YAMLs for the regioselectivity
reproduction — Nyl10/Nyl12/Nyl50 x {MAXA_precursor, XAMA_precursor}.

Faithful to Dishner/Bourgery et al. (JACS 2026, 148, 18285) Methods:
  * enzyme = TETRAMER OF HETERODIMERS: 4 alpha + 4 beta chains
    (autocleaved/mature form; beta chain residue 1 = catalytic N-terminal Thr)
    chains A,C,E,G = alpha (share alpha MSA);  B,D,F,H = beta (share beta MSA)
  * substrate = neutrally-capped poly(X,AMA) tetrad (ligand chain I), one framing each
  * two CONTACT restraints guiding the scissile amide into the catalytic centre of
    heterodimer-1:  scissile carbonyl C  -> beta1 Thr (chain B, residue 1)
                    scissile carbonyl O  -> alpha1 Asn (chain A, oxyanion-hole Asn)
Run folding with --diffusion_samples 10 (paper protocol), then geometry-filter + pick
top model, MM-minimise (OpenMM FF19SB/GAFF), score interface_delta (PyRosetta).
"""
import csv, hashlib, json, os

EROOT = "/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent"
HERE = os.path.dirname(os.path.abspath(__file__))
SEQS = f"{EROOT}/exp_data/nylon_hydrolase/annotation/amide_hydrolase_panel_sequences.csv"
SPLIT = f"{EROOT}/exp_data/nylon_hydrolase/cofold_hydrolase_panel/split_sites.csv"
MSA_DIR = f"{EROOT}/data/cofold_hydrolase_panel/frag_msa"
SUBS = json.load(open(f"{HERE}/substrates/substrates.json"))
INP = f"{HERE}/inputs"; os.makedirs(INP, exist_ok=True)

# oxyanion-hole Asn residue (full-length numbering == alpha-chain numbering; all < split)
ASN = {"Nyl10": 180, "Nyl12": 186, "Nyl50": 180}
FY = {"Nyl10": 106, "Nyl12": 113, "Nyl50": 105}   # oxyanion Phe/Tyr (backbone N)
NCOPIES = 4                                        # tetramer of heterodimers
ALPHA_CHAINS = ["A", "C", "E", "G"][:NCOPIES]
BETA_CHAINS = ["B", "D", "F", "H"][:NCOPIES]
LIG = "I"
MAXD = 5.0                                         # soft placement bias (deg filter later)

seqs = {r["candidate_id"]: r["sequence"] for r in csv.DictReader(open(SEQS))}
split = {r["id"]: int(r["split_resnum_1based"]) for r in csv.DictReader(open(SPLIT))}


def frag_msa(seq):
    return os.path.join(MSA_DIR, hashlib.sha1(seq.encode()).hexdigest()[:16] + ".csv")


def build(enz, sub):
    s = seqs[enz]; k = split[enz]
    alpha, beta = s[:k - 1], s[k - 1:]
    assert beta[0] == "T", f"{enz}: beta[0] != Thr"
    ma, mb = frag_msa(alpha), frag_msa(beta)
    smiles = sub["smiles"]; cn = sub["scissile_C_name"]; on = sub["scissile_O_name"]
    L = ["version: 1", "sequences:"]
    for ch in ALPHA_CHAINS:
        L += [f"- protein:\n    id: {ch}\n    sequence: {alpha}\n    msa: {ma}"]
    for ch in BETA_CHAINS:
        L += [f"- protein:\n    id: {ch}\n    sequence: {beta}\n    msa: {mb}"]
    L += [f"- ligand:\n    id: {LIG}\n    smiles: '{smiles}'"]
    # restraints into heterodimer-1 (alpha=A, beta=B):
    #  (i) a pocket constraint keeps the whole ligand seated in the catalytic
    #      machinery (Thr nucleophile + oxyanion Asn + Phe/Tyr) so it cannot wander
    #      out of the site; (ii) two atom-level contacts pin the SPECIFIC scissile
    #      carbonyl (defines the MAXA vs XAMA framing).
    fy = FY[enz]
    L += ["constraints:",
          f"  - pocket:\n      binder: {LIG}\n      contacts: [[B, 1], [A, {ASN[enz]}], [A, {fy}]]\n"
          f"      max_distance: 6.0\n      force: true",
          f"  - contact:\n      token1: [B, 1]\n      token2: [{LIG}, {cn}]\n"
          f"      max_distance: {MAXD}\n      force: true",
          f"  - contact:\n      token1: [A, {ASN[enz]}]\n      token2: [{LIG}, {on}]\n"
          f"      max_distance: {MAXD}\n      force: true"]
    return "\n".join(L) + "\n"


def main():
    n = 0
    for enz in ["Nyl10", "Nyl12", "Nyl50"]:
        for label, sub in SUBS.items():
            name = f"{enz}__{label}"
            open(f"{INP}/{name}.yaml", "w").write(build(enz, sub))
            n += 1
            print(f"wrote {name}.yaml  (scissile C={sub['scissile_C_name']} "
                  f"O={sub['scissile_O_name']}; Asn={ASN[enz]})")
    print(f"\n{n} YAMLs -> {INP}")


if __name__ == "__main__":
    main()
