#!/usr/bin/env python3
"""Oligomeric substrates (diads / triads) for the regio paper -- built and RDKit-verified.

`substrate_reference.csv` is monomers only. The regio paper's Figs 4/5/7 use amide diads and
triads as donors AND acceptors, so they are constructed here from their monomer parts and checked
against an expected formula/charge. Naming follows the paper: letters read N->C along the chain,
M = hexamethylenediamine, A = adipic acid, X = p-xylylenediamine, S = succinic acid,
H = 6-aminohexanoic acid.

Protonation follows the campaign convention (pH 7):
  * a carboxylate that is NOT the reacting group -> [O-]
  * an amine that is NOT the reacting nucleophile -> [NH3+]
  * the reacting nucleophilic amine (step 2 only)  -> neutral -NH2
So a given oligomer has a DONOR form (step 1 / acyl-AMP precursor) and an ACCEPTOR form (step 2).
"""

# --- MONOMER donor overrides -------------------------------------------------------------------
# substrate_reference.csv applies its "deprotonated at pH 7" convention to carboxylates but NOT to
# amines, so the omega-amino acids come out with a NEUTRAL -NH2 and net charge -1. At pH 7 those
# are zwitterions (COOH pKa ~4.4, NH3+ pKa ~10.7), net 0.
#
# For a STEP-1 DONOR this is not cosmetic: a neutral -NH2 is itself nucleophilic and competes with
# the carboxylate for P-alpha, so the cofold can present the wrong group to the electrophile and
# corrupt the measurement. Overridden here for donor use only.
#
# The reference's neutral-amine form remains CORRECT for the same molecule as a STEP-2 ACCEPTOR,
# where the amine IS the attacking nucleophile. Role decides protonation; do not share one string.
# Upstream substrate_reference.csv is deliberately left untouched -- it is shared with the
# literature_cofolding campaign and editing it would silently rewrite that campaign's provenance.
MONOMER_DONOR_OVERRIDE = {
    "H": "[NH3+]CCCCCC(=O)[O-]",   # 6-aminohexanoic acid, zwitterion (was NCCCCCC(=O)[O-], -1)
    "B": "[NH3+]CCCC(=O)[O-]",     # 4-aminobutyrate
    "V": "[NH3+]CCCCC(=O)[O-]",    # 5-aminovalerate
    "P": "[NH3+]CCCCCCC(=O)[O-]",  # 7-aminoheptanoic acid (regio paper letter P)
}

# --- donor forms: the reacting carboxylate is deprotonated, distal amines are [NH3+] -----------
DONOR = {
    # diads
    "MA":  "[NH3+]CCCCCCNC(=O)CCCCC(=O)[O-]",
    "XA":  "[NH3+]Cc1ccc(CNC(=O)CCCCC(=O)[O-])cc1",
    "MS":  "[NH3+]CCCCCCNC(=O)CCC(=O)[O-]",
    # triads
    "AMA": "[O-]C(=O)CCCCC(=O)NCCCCCCNC(=O)CCCCC(=O)[O-]",
}

# --- acceptor forms: ONE amine is the neutral nucleophile, everything else stays charged --------
ACCEPTOR = {
    # diads (the free amine attacks; the chain's own COOH stays [O-])
    "MA":  "NCCCCCCNC(=O)CCCCC(=O)[O-]",
    "XA":  "NCc1ccc(CNC(=O)CCCCC(=O)[O-])cc1",
    # triads (one terminal amine neutral, the distal one [NH3+])
    "MAM": "NCCCCCCNC(=O)CCCCC(=O)NCCCCCC[NH3+]",
}

# expected (formula, formal charge) -- independent of the SMILES above, so a typo cannot pass
EXPECT_MONOMER_OVERRIDE = {
    "H": ("C6H13NO2", 0), "B": ("C4H9NO2", 0),
    "V": ("C5H11NO2", 0), "P": ("C7H15NO2", 0),
}

EXPECT_DONOR = {
    "MA":  ("C12H24N2O3", 0),
    "XA":  ("C14H20N2O3", 0),
    "MS":  ("C10H20N2O3", 0),
    "AMA": ("C18H30N2O6", -2),
}
EXPECT_ACCEPTOR = {
    # acceptor diads carry a NEUTRAL nucleophilic -NH2 plus a deprotonated distal COOH -> net -1.
    # (The donor form of the same diad is neutral: [NH3+] balances the [O-]. Do not reuse it here.)
    "MA":  ("C12H23N2O3", -1),
    "XA":  ("C14H19N2O3", -1),
    "MAM": ("C18H39N4O2", 1),
}


def verify():
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    bad = []
    for label, table, expect in (("donor", DONOR, EXPECT_DONOR),
                                 ("acceptor", ACCEPTOR, EXPECT_ACCEPTOR),
                                 ("mono-ovr", MONOMER_DONOR_OVERRIDE, EXPECT_MONOMER_OVERRIDE)):
        for code, smi in table.items():
            m = Chem.MolFromSmiles(smi)
            if m is None:
                bad.append(f"{label}/{code}: unparseable SMILES"); continue
            formula = CalcMolFormula(m)
            charge = Chem.GetFormalCharge(m)
            exp_f, exp_c = expect[code]
            # CalcMolFormula appends +/- for charged species; compare the neutral stem
            stem = formula.split("+")[0].split("-")[0]
            ok = stem == exp_f and charge == exp_c
            print(f"  {label:9s} {code:4s} {formula:16s} charge={charge:+d}  "
                  f"{'OK' if ok else 'MISMATCH expected ' + exp_f + f' {exp_c:+d}'}")
            if not ok:
                bad.append(f"{label}/{code}")
    if bad:
        raise SystemExit("FAILED verification: " + ", ".join(bad))
    print("all oligomeric substrates verified")


if __name__ == "__main__":
    verify()
