#!/usr/bin/env python3
"""Build Boltz-2 cofold YAMLs for a protein panel against one ligand.

Each job is a single protein chain plus one ligand: these enzymes are monomeric
and use a Ser/Thr nucleophile, so there is no cofactor and no second chain to
model. The MSA is taken per sequence from the batched fetch (never transplanted
between sequences).

Refuses to emit a job whose inputs are incomplete rather than silently folding a
protein with no MSA or an unparseable ligand, because a cofold that ran on the
wrong inputs is indistinguishable downstream from one that ran on the right ones.

Usage:
  build_cofold_inputs.py --fasta panel_mature.fasta --msa-index msa/index.tsv \
      --ligand-smiles '<SMILES>' --ligand-name PURp1 --out cofold_inputs/
"""
from __future__ import annotations

import argparse
from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    seqs, name = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split("|")[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line
    return seqs


def read_msa_index(path: Path) -> dict[str, str]:
    """id -> absolute MSA csv path, from the batched fetch's index.tsv."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[0].lower() in ("id", "name"):
            continue
        out[parts[0].split("|")[0]] = str(Path(parts[2]).resolve())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--msa-index", required=True)
    ap.add_argument("--ligand-smiles", required=True)
    ap.add_argument("--ligand-name", default="LIG")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from rdkit import Chem
    mol = Chem.MolFromSmiles(args.ligand_smiles)
    if mol is None:
        raise SystemExit(f"ligand SMILES does not parse: {args.ligand_smiles}")
    canonical = Chem.MolToSmiles(mol)

    seqs = read_fasta(Path(args.fasta))
    msa = read_msa_index(Path(args.msa_index))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, []
    for name, seq in sorted(seqs.items()):
        if not seq or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            skipped.append((name, "sequence missing or non-standard residues"))
            continue
        path = msa.get(name)
        if not path or not Path(path).exists():
            skipped.append((name, "no MSA for this sequence"))
            continue
        (out / f"{name}.yaml").write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {seq}\n"
            f"      msa: {path}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{canonical}'\n"
        )
        written += 1

    print(f"ligand {args.ligand_name}: {canonical}")
    print(f"wrote {written} cofold job(s) to {out}")
    for name, why in skipped:
        print(f"  SKIPPED {name}: {why}")
    if skipped:
        print(f"{len(skipped)} job(s) skipped; these are NOT folded rather than "
              f"folded on incomplete input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
