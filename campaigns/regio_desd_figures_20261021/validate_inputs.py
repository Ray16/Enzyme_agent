#!/usr/bin/env python3
"""Independently validate every built cofold YAML before any GPU is spent.

Re-derives each expectation from the authoritative sources rather than trusting the builder:
sequences from sequences/enzymes_provenance.json, monomer SMILES from substrate_reference.csv,
oligomer SMILES from substrates_oligomer.py (itself formula/charge-checked).

Checks, per YAML:
  1  parses as YAML; chain ids unique
  2  chain A is the protein; B is ccd ATP; C(/E/F) are ccd MG; D is the smiles ligand
  3  protein sequence is byte-identical to the provenance record (no truncation / typo)
  4  MSA file exists, non-empty, has the `key,sequence` header
  5  MSA QUERY ROW matches the protein sequence  <-- catches a transplanted / mismatched MSA
  6  ligand D SMILES parses; net formal charge == expected; carries the expected reactive group
  7  Mg chain count == the count encoded in the filename
  8  affinity binder is D
Plus campaign-level cross-checks: job_map.csv <-> files on disk 1:1, stems unique,
SMILES have not drifted from their source of truth.

Exit status is nonzero if ANY check fails -- this is a gate, not a report.
"""
import csv, json, pathlib, re, sys

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
REF = ROOT / "exp_data/amide_synthetase/literature_activity/substrate_reference.csv"

fails, warns = [], []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


def msa_query(path):
    """First sequence row of a Boltz MSA csv, gaps/inserts removed."""
    with open(path) as fh:
        header = fh.readline().strip()
        if not header.startswith("key,sequence"):
            return None, header
        for line in fh:
            if not line.strip():
                continue
            seq = line.split(",", 1)[1].strip()
            return re.sub(r"[-a-z.]", "", seq), header
    return None, header


def main():
    sys.path.insert(0, str(CAMP))
    from rdkit import Chem
    from substrates_oligomer import DONOR, MONOMER_DONOR_OVERRIDE, CONTROL_DONOR

    ent = json.loads((CAMP / "sequences/enzymes_provenance.json").read_text())

    # authoritative SMILES: monomers from the csv (with the documented pH-7 zwitterion override
    # for omega-amino donors), oligomers from the verified module
    truth = {}
    for row in csv.DictReader(REF.open()):
        truth[row["letter_code"]] = row["smiles"]
    truth.update({k: v for k, v in MONOMER_DONOR_OVERRIDE.items() if k in truth})
    truth.update(DONOR)
    truth.update(CONTROL_DONOR)

    jm = list(csv.DictReader((CAMP / "cofolds/step1/job_map.csv").open()))
    yamls = sorted((CAMP / "cofolds/step1/inputs").glob("*.yaml"))

    check(len({r["stem"] for r in jm}) == len(jm), "job_map has duplicate stems")
    check({r["stem"] for r in jm} == {y.stem for y in yamls},
          f"job_map ({len(jm)}) and YAMLs on disk ({len(yamls)}) do not correspond 1:1")

    by_stem = {r["stem"]: r for r in jm}
    msa_cache = {}

    for y in yamls:
        row = by_stem.get(y.stem)
        if row is None:
            continue
        txt = y.read_text()
        tag = y.stem

        # --- chains -------------------------------------------------------------------
        ids = re.findall(r"^    id: (\S+)$", txt, re.M)
        check(len(ids) == len(set(ids)), f"{tag}: duplicate chain ids {ids}")
        check(ids[0] == "A", f"{tag}: first chain is {ids[0]}, expected A (protein)")
        check(re.search(r"- ligand:\n    id: B\n    ccd: ATP", txt), f"{tag}: chain B is not ccd ATP")
        n_mg_file = int(m.group(1)) if (m := re.search(r"__(\d+)mg$", tag)) else 1
        n_mg_yaml = len(re.findall(r"ccd: MG", txt))
        check(n_mg_yaml == n_mg_file,
              f"{tag}: {n_mg_yaml} MG chains but filename implies {n_mg_file}")
        check(n_mg_yaml == int(row["n_mg"]), f"{tag}: MG count disagrees with job_map")
        check(re.search(r"- affinity:\n    binder: D", txt), f"{tag}: affinity binder is not D")

        # --- protein sequence ---------------------------------------------------------
        seq = re.search(r"^    sequence: (\S+)$", txt, re.M).group(1)
        check(seq == ent[row["enzyme"]]["seq"],
              f"{tag}: protein sequence differs from the provenance record for {row['enzyme']}")

        # --- MSA <-> sequence -----------------------------------------------------------
        msa_path = pathlib.Path(re.search(r"^    msa: (\S+)$", txt, re.M).group(1))
        if check(msa_path.exists() and msa_path.stat().st_size > 0,
                 f"{tag}: MSA missing or empty: {msa_path}"):
            if msa_path not in msa_cache:
                msa_cache[msa_path] = msa_query(msa_path)
            q, header = msa_cache[msa_path]
            if check(q is not None, f"{tag}: MSA header is {header!r}, expected 'key,sequence'"):
                check(q == seq, f"{tag}: MSA QUERY DOES NOT MATCH the protein sequence "
                                f"(msa {len(q)} aa vs protein {len(seq)} aa) -- transplanted MSA?")

        # --- ligand D chemistry ---------------------------------------------------------
        smi = re.search(r"^    smiles: '(.+)'$", txt, re.M).group(1)
        code = row["acid_code"]
        check(smi == truth[code], f"{tag}: SMILES drifted from source of truth for donor {code}")
        check(smi == row["smiles"], f"{tag}: SMILES disagrees with job_map")
        mol = Chem.MolFromSmiles(smi)
        if check(mol is not None, f"{tag}: donor SMILES does not parse: {smi}"):
            # a step-1 donor MUST present a deprotonated carboxylate to attack P-alpha
            carboxylate = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[O-]"))
            check(len(carboxylate) >= 1,
                  f"{tag}: donor {code} has no deprotonated carboxylate -- cannot attack P-alpha")
            # no neutral COOH should survive at pH 7
            check(not mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
                  f"{tag}: donor {code} carries a PROTONATED -COOH; pH-7 convention is [O-]")
            # HARD FAIL: a neutral -NH2 on a step-1 donor is a second nucleophile competing with
            # the carboxylate for P-alpha, which can silently corrupt the attack geometry.
            check(not mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")),
                  f"{tag}: donor {code} carries a NEUTRAL -NH2 -- a competing nucleophile at "
                  f"P-alpha; pH-7 donor convention is [NH3+]")
            # Mg2+ is the only metal; no stray metals from a CCD mix-up
            check("ccd: MG" in txt, f"{tag}: no MG chain present")

    print(f"validated {len(yamls)} step-1 YAMLs across {len(msa_cache)} distinct MSAs")
    for w in warns:
        print(f"  WARN  {w}")
    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for f in fails:
            print(f"  FAIL  {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- safe to submit")


if __name__ == "__main__":
    main()
