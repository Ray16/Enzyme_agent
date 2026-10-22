#!/usr/bin/env python3
"""Build the 18 STEP-1 (acid-side) Boltz-2 cofold inputs for the regio-paper Fig-2 test.

6 enzymes x 3 unique acid donors (A = adipic, Z = azelaic, H = 6-aminohexanoic).
See PREREGISTRATION.md -- composition is held identical to the existing
literature_cofolding campaign so scores are comparable across campaigns:

  A = protein (+ precomputed batched MSA csv)
  B = ligand ccd: ATP
  C = ligand ccd: MG          # ONE Mg2+ -- adenylate half-reaction (Palpha attack, AMP + PPi)
  D = ligand smiles: <acid donor, deprotonated at pH 7>
  properties: affinity: binder D

SMILES are read from the canonical substrate_reference.csv, never retyped.
"""
import csv, json, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
REF = ROOT / "exp_data/amide_synthetase/literature_activity/substrate_reference.csv"
SHARED_MSA = ROOT / "exp_data/amide_synthetase/literature_cofolding/_shared_msa"

ENZYMES = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA", "DesD"]
# acid donor -> the Fig-2 panel(s) that activate it
ACIDS = {"A": "2A (X+A->XA)", "Z": "2B (M+Z->MZ)", "H": "2C (H+H->HH); 2D (M+H->MH)"}

# DdaG/SfaB reuse the MSAs already fetched for the 49-enzyme panel (identical sequences).
PRIOR_MSA = {"DdaG": SHARED_MSA / "DdaG_E2JA32.csv", "SfaB": SHARED_MSA / "SfaB_A0A2H4T912.csv"}

YAML = """version: 1
# STEP-1 acid-side | {enz} ({acc}) + ATP + 1 MG + {aname} ({code}) | affinity on acid donor (D)
# Fig-2 panel(s): {panel}
# chains (PRODUCTION order): A=protein  B=ATP  C=MG(1, adenylate half-reaction)  D=acid donor (deprotonated, pH 7)
sequences:
- protein:
    id: A
    sequence: {seq}
    msa: {msa}
- ligand:
    id: B
    ccd: ATP
- ligand:
    id: C
    ccd: MG
- ligand:
    id: D
    smiles: '{smiles}'
properties:
- affinity:
    binder: D
"""


def main():
    ent = json.loads((CAMP / "sequences/enzymes_provenance.json").read_text())

    acids = {}
    for row in csv.DictReader(REF.open()):
        if row["letter_code"] in ACIDS and "regio" in row["paper"]:
            acids[row["letter_code"]] = row
    missing = set(ACIDS) - set(acids)
    if missing:
        sys.exit(f"acid codes absent from substrate_reference.csv for the regio paper: {missing}")

    # resolve each enzyme's MSA: prior shared csv, else the freshly batched one via index.tsv
    msa = dict(PRIOR_MSA)
    index = CAMP / "msa/index.tsv"
    if index.exists():
        for line in index.read_text().splitlines():
            if line.startswith("id\t") or not line.strip():
                continue
            name, _hash, path = line.split("\t")[:3]
            msa[name] = pathlib.Path(path)

    absent = [e for e in ENZYMES if e not in msa or not msa[e].exists()]
    if absent:
        sys.exit(f"no MSA yet for {absent} -- wait for the batched MSA job, then re-run")

    outdir = CAMP / "inputs"
    outdir.mkdir(parents=True, exist_ok=True)
    items = []
    for enz in ENZYMES:
        for code, panel in ACIDS.items():
            stem = f"step1_{enz}__{code}"
            (outdir / f"{stem}.yaml").write_text(YAML.format(
                enz=enz, acc=ent[enz]["acc"], aname=acids[code]["name"], code=code, panel=panel,
                seq=ent[enz]["seq"], msa=msa[enz], smiles=acids[code]["smiles"]))
            items.append(f"inputs/{stem}.yaml")

    (CAMP / "fleet").mkdir(exist_ok=True)
    (CAMP / "fleet/items.txt").write_text("\n".join(items) + "\n")
    print(f"wrote {len(items)} YAMLs -> {outdir}")
    for e in ENZYMES:
        print(f"  {e:6s} len={len(ent[e]['seq']):4d}  msa={msa[e].name}")


if __name__ == "__main__":
    main()
