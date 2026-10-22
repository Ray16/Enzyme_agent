#!/usr/bin/env python3
"""Build the STEP-1 (acid-side) Boltz-2 cofold inputs for the regio paper, all figures.

See PREREGISTRATION.md for the test and MECHANISM.md for why the setup is what it is:
  * attack at P-alpha (acyl-adenylate), leaving group PPi via O3A -- validated for DesD via PDB 7TGM
  * ONE Mg2+ per active site -- counted in 7TGM and 2W02, not assumed
    (+ a pre-registered 2-Mg sensitivity set for DesD)

Chain order is held IDENTICAL to the existing literature_cofolding campaign so that `A` is
comparable across campaigns:
  A = protein (+ precomputed batched MSA)   B = ATP   C = MG (1, or 2 for the sensitivity set)
  D = acid donor, deprotonated at pH 7      properties: affinity binder D

Folds are a SHARED DEDUPED POOL keyed by (enzyme, acid, n_mg): several figures reuse the same
(enzyme, acid) pair -- acid A appears in Fig 2A, 5A and 5C; H in 2C, 2D and 4C -- and refolding
byte-identical inputs per figure would be duplicate GPU. Per-figure membership is recorded in
cofolds/step1/job_map.csv instead.
"""
import csv, json, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
REF = ROOT / "exp_data/amide_synthetase/literature_activity/substrate_reference.csv"
SHARED_MSA = ROOT / "exp_data/amide_synthetase/literature_cofolding/_shared_msa"

ALL6 = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA", "DesD"]

# Every acid donor is folded against ALL SIX enzymes, even where the paper assayed fewer
# (Figs 4/5 assayed only DdaG/SfaB/AcsA/DesD; Fig 7 only DesD). The extra cells cost nothing
# conceptually and yield PREDICTIONS for untested enzyme x donor combinations. They carry no
# experimental label, so analyze_step1.py must exclude them from the confirmatory tests and
# report them separately -- see the `has_label` column in job_map.csv.
#
# (acid donor code, the figure panel(s) that use it, enzymes the paper actually assayed there)
STEP1_JOBS = [
    ("A",   "2A (X+A->XA); 5A (MAM+A->MAMA); 5C", ALL6),
    ("Z",   "2B (M+Z->MZ)", ALL6),
    ("H",   "2C (H+H->HH); 2D (M+H->MH); 4C (MA+H->HMA)", ALL6),
    ("S",   "6 (cascade: S+M->MS, DdaG)", ["DdaG", "DesD"]),
    ("G",   "4A (MA+G->GMA)", ["DdaG", "SfaB", "AcsA", "DesD"]),
    ("MA",  "4B (MA+X->MAX); 4C (MA+H->MAH); 5C (MA+MA->MAMA)", ["DdaG", "SfaB", "AcsA", "DesD"]),
    ("MS",  "6 (cascade: MS+P->MSP, DesD)", ["DesD"]),
    ("AMA", "5B (AMA+M->MAMA)", ["DdaG", "SfaB", "AcsA", "DesD"]),
    ("XA",  "7 (XA+MA->XAMA / MAXA)", ["DesD"]),
]
# pre-registered Mg sensitivity: DesD only, 2 Mg, on the three Fig-2 donors
MG_SENSITIVITY = [("DesD", code, 2) for code in ("A", "Z", "H")]

PRIOR_MSA = {"DdaG": SHARED_MSA / "DdaG_E2JA32.csv", "SfaB": SHARED_MSA / "SfaB_A0A2H4T912.csv"}

HEADER = """version: 1
# STEP-1 acid-side | {enz} ({acc}) + ATP + {nmg} MG + {aname} ({code}) | affinity on acid donor (D)
# attack: carboxylate O -> ATP P-alpha, leaving group PPi via O3A (see MECHANISM.md)
# Fig panel(s): {panel}
sequences:
- protein:
    id: A
    sequence: {seq}
    msa: {msa}
- ligand:
    id: B
    ccd: ATP
"""
MG_CHAIN = """- ligand:
    id: {cid}
    ccd: MG
"""
TAIL = """- ligand:
    id: D
    smiles: '{smiles}'
properties:
- affinity:
    binder: D
"""


def acid_smiles():
    """Monomers from the canonical reference; oligomers from the verified builder."""
    sys.path.insert(0, str(CAMP))
    from substrates_oligomer import DONOR, MONOMER_DONOR_OVERRIDE
    smi, name = {}, {}
    for row in csv.DictReader(REF.open()):
        if "regio" in row["paper"] or row["letter_code"] == "G":
            smi[row["letter_code"]] = row["smiles"]
            name[row["letter_code"]] = row["name"]
    # omega-amino acids: the reference leaves the amine NEUTRAL, which at pH 7 is wrong and gives
    # the donor a second nucleophile competing for P-alpha. Use the zwitterion. See
    # substrates_oligomer.MONOMER_DONOR_OVERRIDE for the full reasoning.
    for code, s in MONOMER_DONOR_OVERRIDE.items():
        if code in smi:
            smi[code] = s
    for code, s in DONOR.items():
        smi[code] = s
        name[code] = f"{code} oligomer (donor form)"
    return smi, name


def main():
    ent = json.loads((CAMP / "sequences/enzymes_provenance.json").read_text())
    smi, aname = acid_smiles()

    msa = dict(PRIOR_MSA)
    index = CAMP / "msa/index.tsv"
    if index.exists():
        for line in index.read_text().splitlines()[1:]:
            if line.strip():
                name, _h, path = line.split("\t")[:3]
                msa[name] = pathlib.Path(path)
    absent = [e for e in ALL6 if e not in msa or not msa[e].exists()]
    if absent:
        sys.exit(f"no MSA for {absent} -- wait for the batched MSA job")

    # fold every donor against all six enzymes; `assayed` records which cells the paper labelled
    jobs = [(e, code, 1, panel, e in assayed)
            for code, panel, assayed in STEP1_JOBS for e in ALL6]
    jobs += [(e, c, n, "Mg sensitivity (pre-registered; MECHANISM.md sec 2)", False)
             for e, c, n in MG_SENSITIVITY]

    outdir = CAMP / "cofolds/step1/inputs"
    outdir.mkdir(parents=True, exist_ok=True)
    rows, items, seen = [], [], set()
    for enz, code, nmg, panel, has_label in jobs:
        if code not in smi:
            sys.exit(f"no SMILES for acid donor {code}")
        stem = f"step1_{enz}__{code}" + (f"__{nmg}mg" if nmg != 1 else "")
        if stem in seen:          # dedupe: a donor reused across panels folds once
            continue
        seen.add(stem)
        body = HEADER.format(enz=enz, acc=ent[enz]["acc"], nmg=nmg, aname=aname[code],
                             code=code, panel=panel, seq=ent[enz]["seq"], msa=msa[enz])
        for i in range(nmg):
            body += MG_CHAIN.format(cid="C" if i == 0 else chr(ord("E") + i - 1))
        body += TAIL.format(smiles=smi[code])
        (outdir / f"{stem}.yaml").write_text(body)
        items.append(f"cofolds/step1/inputs/{stem}.yaml")
        rows.append(dict(stem=stem, enzyme=enz, acid_code=code, acid_name=aname[code],
                         n_mg=nmg, smiles=smi[code], has_label=has_label, panels=panel))

    with (CAMP / "cofolds/step1/job_map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (CAMP / "fleet").mkdir(exist_ok=True)
    (CAMP / "fleet/items_step1.txt").write_text("\n".join(items) + "\n")

    n_lab = sum(1 for r in rows if r["has_label"])
    print(f"wrote {len(rows)} step-1 YAMLs -> {outdir}")
    print(f"  {n_lab} labelled (enter the confirmatory tests), "
          f"{len(rows) - n_lab} unlabelled predictions / controls")
    for code, _panel, assayed in STEP1_JOBS:
        print(f"  acid {code:4s} x 6 enzymes   (paper assayed {len(assayed)}: {','.join(assayed)})")
    print(f"  + {len(MG_SENSITIVITY)} DesD 2-Mg sensitivity folds")


if __name__ == "__main__":
    main()
