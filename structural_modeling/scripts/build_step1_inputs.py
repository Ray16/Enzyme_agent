#!/usr/bin/env python3
"""Build step-1 (adenylation) cofold YAMLs for the discovery-agent candidate panel.

Per enzyme:  A=protein(+MSA)  +  the nucleotide & metal COFACTORS derived per-enzyme by
the agent (structural_modeling_handoff.json -> candidate.modeling_input.cofactors)  +  the
TARGET substrate adipate (dianion); affinity on adipate.

The cofactor spec (metal identity + COUNT) comes from the agent's derive_modeling_cofactors,
which reads the UniProt cofactor comment/note (basis "uniprot_cofactor_note") and otherwise
falls back to the two-metal-ion default (basis "mechanistic_default"). So the metal
stoichiometry now VARIES per enzyme/family from annotation instead of a blanket 2 Mg.
Without --handoff (or for an accession absent from it) it falls back to ATP + 2 Mg and says so.

Sequences from representatives.fasta; MSA matched via msa_cache/index.tsv (id->csv).
"""
import argparse, json, os, string

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # structural_modeling/
FASTA = f"{ROOT}/representatives.fasta"
INDEX = f"{ROOT}/msa_cache/index.tsv"
OUT = f"{ROOT}/step1"
ADIPATE = "[O-]C(=O)CCCCC(=O)[O-]"               # adipate dianion (target substrate)
DEFAULT_COFACTORS = [                             # fallback when no handoff spec exists
    {"ccd": "ATP", "count": 1, "role": "nucleotide_substrate", "basis": "default"},
    {"ccd": "MG", "count": 2, "role": "catalytic_metal", "basis": "default: two-metal-ion"},
]


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


def load_handoff_modeling(path):
    """accession -> modeling_input dict (cofactors, ligation_frame, ...) from the handoff."""
    spec = {}
    if not path or not os.path.exists(path):
        return spec
    d = json.load(open(path))
    for c in d.get("candidates", []):
        acc = c.get("accession")
        mi = c.get("modeling_input") or {}
        if acc and mi.get("cofactors"):
            spec[acc] = mi
    return spec


def yaml_for(seq, msa, cofactors):
    """A=protein, then each cofactor expanded by its count, then adipate; affinity on adipate."""
    lines = ["version: 1", "sequences:",
             "- protein:", "    id: A", f"    sequence: {seq}", f"    msa: {msa}"]
    ids = iter(string.ascii_uppercase[1:])          # B, C, D, ...
    for cof in cofactors:
        ccd = cof.get("ccd") or cof.get("name")
        for _ in range(int(cof.get("count", 1) or 1)):
            lines += ["- ligand:", f"    id: {next(ids)}", f"    ccd: {ccd}"]
    adip_id = next(ids)
    lines += ["- ligand:", f"    id: {adip_id}", f"    smiles: '{ADIPATE}'"]
    lines += ["properties:", "- affinity:", f"    binder: {adip_id}"]
    return "\n".join(lines) + "\n"


def metal_summary(cofactors):
    return ", ".join(f"{c.get('ccd')}x{c.get('count')}" for c in cofactors if c.get("role") == "catalytic_metal") or "(no metal)"


def basis_of(cofactors):
    for c in cofactors:
        if c.get("role") == "catalytic_metal":
            return c.get("basis", "?")
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", default=None, help="structural_modeling_handoff.json (per-enzyme modeling_input.cofactors)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="report per-enzyme cofactor spec; do not write YAMLs")
    args = ap.parse_args()

    seqs = read_fasta(FASTA)
    msa = {}
    for line in open(INDEX):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            msa[parts[0]] = parts[2]
    handoff = load_handoff_modeling(args.handoff)
    if args.handoff:
        print(f"loaded modeling specs for {len(handoff)} accessions from {args.handoff}")

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    n, miss, fell_back, nondefault = 0, [], [], []
    for acc, seq in seqs.items():
        mp = msa.get(acc)
        if not mp or not os.path.exists(mp):
            miss.append(acc); continue
        mi = handoff.get(acc)
        cofactors = (mi or {}).get("cofactors")
        if not cofactors:
            cofactors = DEFAULT_COFACTORS; fell_back.append(acc)
        summ = metal_summary(cofactors); basis = basis_of(cofactors)
        if summ != "MGx2":
            nondefault.append(f"{acc}:{summ}")
        if args.dry_run:
            print(f"  {acc:12} metals={summ:10} basis={basis}")
        else:
            open(f"{args.out}/step1_{acc}.yaml", "w").write(yaml_for(seq, os.path.abspath(mp), cofactors))
        n += 1
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {n} step-1 YAMLs to {args.out}")
    print(f"  from-handoff: {n - len(fell_back)}  fell-back-to-default(ATP+2MG): {len(fell_back)}")
    if nondefault:
        print(f"  NON-(2 Mg) metal specs: {nondefault}")
    else:
        print("  all written enzymes resolved to 2 Mg (handoff + default agree for this set)")
    if miss:
        print(f"  missing MSA (skipped): {miss}")


if __name__ == "__main__":
    main()
