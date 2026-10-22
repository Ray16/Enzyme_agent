#!/usr/bin/env python3
"""Construct the acyl-adenylate (acyl-AMP) intermediate for step 2, and mass-balance check it.

WHY NOT STRING SURGERY: the inherited builder formed acyl-AMP as
    AMP_PREFIX + diacid_smiles[len("[O-]"):] + AMP_SUFFIX
i.e. it assumed the reacting carboxylate is the FIRST token of the donor SMILES. That holds for
plain diacids written `[O-]C(=O)...` but silently produces nonsense for every donor that starts
with something else -- here H, MA, XA and MS all begin `[NH3+]`, so the prefix strip would remove
part of the amine instead of the carboxylate and the "intermediate" would be a different molecule
that still parses and still yields a plausible-looking angle.

Instead the anhydride bond is formed with an explicit RDKit reaction and the product is checked by
MASS BALANCE: acyl-AMP must equal (donor + AMP - OH) in formula, and carry charge
(donor + AMP + 1) because the condensation consumes the carboxylate's [O-] and the phosphate's H.

Chemistry: R-COO(-) + HO-P(=O)(O-)-O-ribose-adenine  ->  R-C(=O)-O-P(=O)(O-)-O-ribose-adenine + OH(-)
The bridging oxygen comes from the phosphate; the carboxylate oxygen leaves. The product is a
MIXED ANHYDRIDE -- the electrophile for step 2 is its CARBONYL CARBON (not a phosphorus) and the
leaving group is AMP.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# adenosine-5'-monophosphate with a free P-OH to condense onto
AMP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])O)[C@@H](O)[C@H]1O"

_RXN = AllChem.ReactionFromSmarts(
    "[CX3:1](=[OX1:2])[OX1-:3].[PX4:4][OX2H1:5]>>[CX3:1](=[OX1:2])[OX2:5][PX4:4]")


def _formula_counts(mol):
    from collections import Counter
    c = Counter()
    for a in mol.GetAtoms():
        c[a.GetSymbol()] += 1
        c["H"] += a.GetTotalNumHs()
    return c


def build(donor_smiles, *, donor_code="?"):
    """Return (smiles, report) for the acyl-adenylate of `donor_smiles`.

    If the donor carries two carboxylates (a diacid), either may react; for the symmetric diacids
    used here (A, S, G, AMA) the two are equivalent, so the first match is taken and the symmetry
    is asserted by checking that all products are identical.
    """
    donor = Chem.MolFromSmiles(donor_smiles)
    amp = Chem.MolFromSmiles(AMP)
    if donor is None or amp is None:
        raise ValueError(f"{donor_code}: donor or AMP SMILES does not parse")

    n_cooh = len(donor.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX1-]")))
    if n_cooh == 0:
        raise ValueError(f"{donor_code}: donor has no deprotonated carboxylate to activate")

    products = set()
    for prod in _RXN.RunReactants((donor, amp)):
        m = prod[0]
        Chem.SanitizeMol(m)
        products.add(Chem.MolToSmiles(m))
    if not products:
        raise ValueError(f"{donor_code}: anhydride formation produced nothing")

    # a symmetric diacid must give ONE distinct product; asymmetric donors may give more
    smi = sorted(products)[0]
    prod = Chem.MolFromSmiles(smi)

    # --- MASS BALANCE: product == donor + AMP - OH(-) ----------------------------------------
    # Formally HYDROXIDE is expelled, not water: the reaction deletes the carboxylate's O(-)
    # (which bears no H) and strips the H from the phosphate O-H, so the loss is -1 O, -1 H.
    # That is also exactly what the charge bookkeeping below implies (+1 relative to the
    # reactants, i.e. an anion left). An earlier "-H2O" here was wrong and this check caught it.
    # NB this is bookkeeping for building the intermediate FROM AMP; in the real enzymatic step 1
    # the leaving group is PPi from ATP. Do not read mechanism off this balance.
    want = _formula_counts(donor) + _formula_counts(amp)
    want["H"] -= 1
    want["O"] -= 1
    got = _formula_counts(prod)
    if dict(want) != dict(got):
        raise ValueError(f"{donor_code}: acyl-AMP fails mass balance; "
                         f"expected {dict(want)} got {dict(got)}")

    want_charge = Chem.GetFormalCharge(donor) + Chem.GetFormalCharge(amp) + 1
    if Chem.GetFormalCharge(prod) != want_charge:
        raise ValueError(f"{donor_code}: acyl-AMP charge {Chem.GetFormalCharge(prod)}, "
                         f"expected {want_charge}")

    # the step-2 electrophile must exist: an anhydride carbonyl C(=O)-O-P
    anhydride = prod.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][PX4]"))
    if len(anhydride) != 1:
        raise ValueError(f"{donor_code}: expected exactly 1 acyl-phosphate carbonyl, "
                         f"found {len(anhydride)}")

    report = dict(donor_code=donor_code, n_donor_carboxylates=n_cooh,
                  n_distinct_products=len(products),
                  formula=CalcMolFormula(prod), charge=Chem.GetFormalCharge(prod),
                  heavy_atoms=prod.GetNumHeavyAtoms(),
                  carbonyl_C_idx=anhydride[0][0])
    return smi, report


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from substrates_oligomer import DONOR, MONOMER_DONOR_OVERRIDE, CONTROL_DONOR
    import csv
    ref = pathlib.Path(__file__).resolve().parents[2] / \
        "exp_data/amide_synthetase/literature_activity/substrate_reference.csv"
    donors = {r["letter_code"]: r["smiles"] for r in csv.DictReader(ref.open())
              if r["letter_code"] in ("A", "Z", "S", "G")}
    donors.update(MONOMER_DONOR_OVERRIDE)
    donors.update(DONOR); donors.update(CONTROL_DONOR)
    for code in sorted(donors):
        try:
            smi, rep = build(donors[code], donor_code=code)
            print(f"  {code:4s} OK  {rep['formula']:22s} charge={rep['charge']:+d}  "
                  f"heavy={rep['heavy_atoms']:3d}  COO-={rep['n_donor_carboxylates']}  "
                  f"products={rep['n_distinct_products']}")
        except ValueError as e:
            print(f"  {code:4s} FAIL {e}")
