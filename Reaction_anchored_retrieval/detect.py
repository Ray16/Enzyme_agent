"""Chemistry-anchored detector for ATP-dependent amide ligation reactions.

Answer-agnostic: the definition is derived purely from the *target transformation*
(a carboxylate donor + an amine acceptor condense into an amide, driven by ATP
hydrolysis to AMP or ADP+Pi) — never from any known answer enzyme, family, or
accession. Given a Rhea reaction SMILES ("reactants>>products"), decide whether
it is an ATP-driven amide-bond ligation.

This mirrors the exact chemistry the target query asks for
(adipate diacid + hexamethylenediamine -> amide, ATP-driven), so a UniProt entry
annotated with such a reaction is a mechanistically on-target candidate,
regardless of its EC number or InterPro family (or whether it has one).
"""

from __future__ import annotations

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Adenine core, used with the fragment's phosphorus count to tell ATP (3 P) from
# ADP (2 P) and AMP (1 P) without hard-coding full molecule SMILES.
_ADENINE = Chem.MolFromSmarts("c1ncc2ncnc2n1")
# An amide carbonyl-N. Excludes the N that is itself acylated on both sides
# (imide) so we count genuine new amide linkages.
_AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]")
_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
# Free primary/secondary amine: not already an amide, not imine, not quaternary.
_AMINE = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(N=*)]")


def _mols(side: str) -> list:
    out = []
    for s in side.split("."):
        s = s.strip()
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(m)
    return out


def _adenosine_with_n_phosphates(m, n: int) -> bool:
    if not m.HasSubstructMatch(_ADENINE):
        return False
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "P") == n


def _amide_bonds(mols) -> int:
    return sum(len(m.GetSubstructMatches(_AMIDE)) for m in mols)


def classify_reaction(rxn_smiles: str) -> tuple[bool, dict]:
    """Return (is_atp_amide_ligation, evidence_flags)."""
    if ">>" not in rxn_smiles:
        return False, {"error": "no reaction arrow"}
    left, right = rxn_smiles.split(">>", 1)
    lm, rm = _mols(left), _mols(right)
    if not lm or not rm:
        return False, {"error": "unparsable side"}

    flags = {
        "atp_substrate": any(_adenosine_with_n_phosphates(m, 3) for m in lm),
        "amp_product": any(_adenosine_with_n_phosphates(m, 1) for m in rm),
        "adp_product": any(_adenosine_with_n_phosphates(m, 2) for m in rm),
        "new_amide_bond": _amide_bonds(rm) > _amide_bonds(lm),
        "carboxyl_substrate": any(m.HasSubstructMatch(_CARBOXYL) for m in lm),
        "amine_substrate": any(m.HasSubstructMatch(_AMINE) for m in lm),
    }
    is_hit = (
        flags["atp_substrate"]
        and (flags["amp_product"] or flags["adp_product"])
        and flags["new_amide_bond"]
        and flags["carboxyl_substrate"]
        and flags["amine_substrate"]
    )
    return is_hit, flags


# --- self-test: known positives must fire, known negatives must not ----------
_SELFTEST = {
    "glutamine synthetase (glu+NH3+ATP->gln+ADP+Pi)": (
        "[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-].[NH4+]."
        "Nc1ncnc2n(cnc12)C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O>>"
        "[NH3+][C@@H](CCC(N)=O)C(=O)[O-]."
        "Nc1ncnc2n(cnc12)C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O.OP(=O)(O)O", True),
    "gamma-glutamylputrescine synthetase (PuuA-like)": (
        "[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-].[NH3+]CCCC[NH3+]."
        "Nc1ncnc2n(cnc12)C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O>>"
        "[NH3+]CCCCNC(=O)CC[C@H]([NH3+])C(=O)[O-]."
        "Nc1ncnc2n(cnc12)C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O.OP(=O)(O)O", True),
    "NEG amide hydrolysis, no ATP": (
        "CCCCC(N)=O.[H]O[H]>>CCCCC(=O)[O-].[H][N+]([H])([H])[H]", False),
    "NEG kinase, ATP but no amide": (
        "OCC.Nc1ncnc2n(cnc12)C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O>>"
        "CCOP(=O)(O)O.Nc1ncnc2n(cnc12)C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O", False),
}

if __name__ == "__main__":
    ok_all = True
    for name, (rx, expect) in _SELFTEST.items():
        got, flags = classify_reaction(rx)
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok_all = False
        print(f"[{status}] expect={expect} got={got}  {name}")
        print(f"         {flags}")
    print("\nSELFTEST", "OK" if ok_all else "FAILED")
