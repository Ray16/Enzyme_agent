#!/usr/bin/env python3
"""Build STEP-2 (aminolysis) Boltz-2 inputs: acyl-adenylate + free amine + Mg, pocket-restrained.

Mechanism encoded here (see MECHANISM.md sec 3):
  * electrophile = the acyl-adenylate CARBONYL CARBON (not a phosphorus)
  * nucleophile  = the acceptor amine N, modelled NEUTRAL because [NH3+] has no lone pair and
                   cannot attack; every spectator group keeps its dominant pH-7 state
  * leaving group = AMP
  * ideal geometry = Burgi-Dunitz ~105 deg, NOT step 1's ~180 deg in-line -- so step-2 scores are
    reported separately and are never averaged with step-1 `A`

Chains:
  A = protein (+MSA)   B = acyl-adenylate (SMILES, mass-balance verified)
  C = amine acceptor (SMILES)   D = ligand ccd MG
  constraints: pocket restraint on the amine (binder C), residues within 8 A of the step-1
               reactive-centre midpoint (donor carboxyl C <-> ATP PA), max_distance 10, force

POCKET PROVENANCE: derived from a FIXED step-1 fold per enzyme -- acid donor A, model_0 -- not from
whichever fold docked best. Choosing the best-docked fold would select the restraint on an outcome
and quietly bias every step-2 pose that depends on it.
"""
import csv, json, math, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
REF = ROOT / "exp_data/amide_synthetase/literature_activity/substrate_reference.csv"
SHARED_MSA = ROOT / "exp_data/amide_synthetase/literature_cofolding/_shared_msa"
S1 = CAMP / "cofolds/step1/results"
POCKET_R = 8.0
POCKET_MAXD = 10.0
ALL6 = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA", "DesD"]

# acceptor forms: the ATTACKING amine is neutral; spectators keep their pH-7 charge
ACCEPTOR_MONOMER = {
    "X": "NCc1ccc(C[NH3+])cc1",
    "M": "NCCCCCC[NH3+]",
    "H": "NCCCCCC(=O)[O-]",
    "P": "NCCCCCCC(=O)[O-]",
}

# (acid donor, amine acceptor, figure panel) -- every pair is folded for all six enzymes
STEP2_PAIRS = [
    ("A",   "X",   "2A (X+A->XA)"),
    ("Z",   "M",   "2B (M+Z->MZ)"),
    ("H",   "H",   "2C (H+H->HH)"),
    ("H",   "M",   "2D (M+H->MH)"),
    ("G",   "MA",  "4A (MA+G->GMA)  [clean step-2 test: G is activated by all]"),
    ("MA",  "X",   "4B (MA+X->MAX)"),
    ("MA",  "H",   "4C (MA+H->MAH)"),
    ("H",   "MA",  "4C (H+MA->HMA, the other regiochemistry)"),
    ("A",   "MAM", "5A (MAM+A->MAMA)  [clean step-2 test: A is activated by all]"),
    ("AMA", "M",   "5B (AMA+M->MAMA)"),
    ("MA",  "MA",  "5C (MA+MA->MAMA)"),
    ("MA",  "XA",  "7 (MA+XA->MAXA)"),
    ("XA",  "MA",  "7 (XA+MA->XAMA)"),
    ("XA",  "XA",  "7 (XA+XA->XAXA)"),
    ("MS",  "P",   "6 (cascade MS+P->MSP)"),
]

YAML = """version: 1
# STEP-2 aminolysis | {enz} ({acc}) | acyl-AMP({donor}) + amine({acceptor}) + 1 MG
# electrophile = acyl-adenylate carbonyl C; nucleophile = neutral amine N; leaving group = AMP
# ideal attack = Burgi-Dunitz ~105 deg (NOT step-1's ~180 in-line) -- scored separately
# Fig panel: {panel}
# pocket from FIXED step-1 fold {pocket_src} (not the best-docked one), {npocket} residues within {r} A
sequences:
- protein:
    id: A
    sequence: {seq}
    msa: {msa}
- ligand:
    id: B
    smiles: '{acyl}'
- ligand:
    id: C
    smiles: '{amine}'
- ligand:
    id: D
    ccd: MG
constraints:
- pocket:
    binder: C
    contacts: {contacts}
    max_distance: {maxd}
    force: true
properties:
- affinity:
    binder: C
"""


def pocket_residues(enz):
    """Residues with any heavy atom within POCKET_R of ATP's P-alpha in the fixed step-1 fold.

    ANCHOR CHOICE -- this matters. The first version anchored the pocket on the MIDPOINT between
    the donor carboxyl carbon and PA. That coupled the restraint to docking quality in the wrong
    direction: a well-docked donor (DesD, D_h 3.98 A) put the midpoint in open space and yielded
    only 2 pocket residues, while a badly-docked one (SfaB, 12.6 A) put it inside the protein and
    yielded 11. Restraint strength would then have been inversely related to how well step 1
    docked, confounding every step-2 comparison exactly as the fold bias confounds step 1.

    Anchoring on PA instead is uniform across the panel (15-22 residues at 8 A for all six
    enzymes), is independent of whether the DONOR docked, and is mechanistically right: the
    acyl-adenylate carbonyl that the amine must attack sits adjacent to PA by construction. ATP is
    reliably placed in every fold -- validate_outputs.py checks its 31 heavy atoms and that an Mg
    coordinates a phosphate oxygen.
    """
    stem = f"step1_{enz}__A"
    pdb = S1 / stem / f"boltz_results_{stem}" / "predictions" / stem / f"{stem}_model_0.pdb"
    if not pdb.exists():
        return None, None
    prot, pa = {}, None
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        try:
            xyz = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        except ValueError:
            continue
        if line.startswith("ATOM  ") and line[21] == "A":
            prot.setdefault(int(line[22:26]), []).append(xyz)
        elif line[21] == "B" and line[12:16].strip() == "PA":
            pa = xyz
    if pa is None or not prot:
        return None, None
    res = sorted(ri for ri, xs in prot.items()
                 if any(math.dist(x, pa) <= POCKET_R for x in xs))
    return res, stem


def main():
    sys.path.insert(0, str(CAMP))
    from substrates_oligomer import DONOR, MONOMER_DONOR_OVERRIDE, ACCEPTOR
    from acyl_adenylate import build as build_acyl

    ent = json.loads((CAMP / "sequences/enzymes_provenance.json").read_text())
    donors = {r["letter_code"]: r["smiles"] for r in csv.DictReader(REF.open())
              if r["letter_code"] in ("A", "Z", "S", "G")}
    donors.update(MONOMER_DONOR_OVERRIDE)
    donors.update(DONOR)
    acceptors = {**ACCEPTOR_MONOMER, **ACCEPTOR}

    msa = {"DdaG": SHARED_MSA / "DdaG_E2JA32.csv", "SfaB": SHARED_MSA / "SfaB_A0A2H4T912.csv"}
    for line in (CAMP / "msa/index.tsv").read_text().splitlines()[1:]:
        if line.strip():
            n, _h, p = line.split("\t")[:3]
            msa[n] = pathlib.Path(p)

    # acyl-adenylates, each mass-balance checked at construction
    acyl = {}
    for code in {d for d, _a, _p in STEP2_PAIRS}:
        acyl[code], _rep = build_acyl(donors[code], donor_code=code)

    outdir = CAMP / "cofolds/step2/inputs"
    outdir.mkdir(parents=True, exist_ok=True)
    rows, items, missing = [], [], []
    for enz in ALL6:
        res, src = pocket_residues(enz)
        if res is None:
            missing.append(enz)
            continue
        contacts = "[" + ", ".join(f'["A", {i}]' for i in res) + "]"
        for donor, acceptor, panel in STEP2_PAIRS:
            if acceptor not in acceptors:
                sys.exit(f"no acceptor SMILES for {acceptor}")
            stem = f"step2_{enz}__{donor}_{acceptor}"
            (outdir / f"{stem}.yaml").write_text(YAML.format(
                enz=enz, acc=ent[enz]["acc"], donor=donor, acceptor=acceptor, panel=panel,
                pocket_src=src, npocket=len(res), r=POCKET_R, seq=ent[enz]["seq"], msa=msa[enz],
                acyl=acyl[donor], amine=acceptors[acceptor], contacts=contacts, maxd=POCKET_MAXD))
            items.append(f"cofolds/step2/inputs/{stem}.yaml")
            rows.append(dict(stem=stem, enzyme=enz, donor_code=donor, acceptor_code=acceptor,
                             acyl_smiles=acyl[donor], amine_smiles=acceptors[acceptor],
                             n_pocket=len(res), pocket_src=src, panels=panel))

    if rows:
        with (CAMP / "cofolds/step2/job_map.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        (CAMP / "fleet/items_step2.txt").write_text("\n".join(items) + "\n")
    print(f"wrote {len(rows)} step-2 YAMLs ({len(STEP2_PAIRS)} pairs x "
          f"{len(ALL6) - len(missing)} enzymes)")
    for enz in ALL6:
        r, s = pocket_residues(enz)
        print(f"  {enz:6s} pocket {'-- step-1 fold not ready' if r is None else f'{len(r)} residues from {s}'}")
    if missing:
        print(f"  WAITING on step-1 acid-A folds for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
