#!/usr/bin/env python
"""Build step-2 (amidation) Boltz inputs for the RFD4 AS9 diad panel.

Reuses the review_20261027 MSAs byte-for-byte (identical sequences) so MSA depth cannot
differ between step 1 and step 2. Ligands are the adipoyl-AMP mixed anhydride plus
mono-protonated HMDA; see PREREGISTRATION.md for the modelled-microstate caveats.
"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
REVIEW = BASE.parent / "review_20261027"

ADIPOYL_AMP = "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCCC(=O)O)[C@@H](O)[C@H]1O"
HMDA = "NCCCCCC[NH3+]"

# (job, source step-1 yaml, seed, role)
PANEL = [
    ("WT_AS1", "WT_AS1", 20261027, "second WT reference"),
    ("WT_AS9", "WT_AS9", 20261027, "parent / primary control"),
    ("WT_AS9_rep2", "WT_AS9", 20261028, "step-2 noise floor replicate"),
    ("AS9_diad_pose0_fixed_cap_r2", "AS9_diad_pose0_fixed_cap_r2", 20261027, "design"),
    ("AS9_diad_pose0_internal_cap_r2", "AS9_diad_pose0_internal_cap_r2", 20261027, "design"),
    ("AS9_diad_pose3_internal_cap_r2", "AS9_diad_pose3_internal_cap_r2", 20261027, "design"),
]


def validate_ligands():
    """Fail loudly if the ligand graphs are not what the pre-registration declares."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdmd

    expect = {
        ADIPOYL_AMP: ("C16H22N5O10P", 0, 32),
        HMDA: ("C6H17N2+", 1, 8),
    }
    for smi, (formula, charge, heavy) in expect.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise SystemExit(f"unparseable SMILES: {smi}")
        got = (rdmd.CalcMolFormula(m), Chem.GetFormalCharge(m), m.GetNumHeavyAtoms())
        if got != (formula, charge, heavy):
            raise SystemExit(f"ligand mismatch for {smi}: {got} != {(formula, charge, heavy)}")

    # the scorer depends on there being exactly one P-O-C(=O) carbon
    m = Chem.MolFromSmiles(ADIPOYL_AMP)
    p = next(a for a in m.GetAtoms() if a.GetSymbol() == "P")
    acyl = [
        c.GetIdx()
        for o in p.GetNeighbors() if o.GetSymbol() == "O"
        for c in o.GetNeighbors()
        if c.GetIdx() != p.GetIdx() and c.GetSymbol() == "C"
        and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(c).GetSymbol() == "O" for b in c.GetBonds())
    ]
    if len(acyl) != 1:
        raise SystemExit(f"expected exactly one mixed-anhydride carbon, found {acyl}")
    print(f"ligand validation OK (unique acyl carbon at index {acyl[0]})")


def main():
    validate_ligands()
    out = BASE / "cofold_inputs"
    out.mkdir(exist_ok=True)
    manifest = []

    for job, src, seed, role in PANEL:
        src_yaml = REVIEW / "cofold_inputs" / f"{src}.yaml"
        src_cfg = json.loads(src_yaml.read_text())
        protein = src_cfg["sequences"][0]["protein"]
        seq = protein["sequence"]
        msa = Path(protein["msa"])
        if not msa.exists():
            raise SystemExit(f"{job}: MSA missing: {msa}")

        cfg = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": seq, "msa": str(msa)}},
                {"ligand": {"id": "B", "smiles": ADIPOYL_AMP}},
                {"ligand": {"id": "C", "ccd": "MG"}},
                {"ligand": {"id": "D", "smiles": HMDA}},
            ],
        }
        dst = out / f"{job}.yaml"
        dst.write_text(json.dumps(cfg, indent=2) + "\n")

        manifest.append({
            "job": job,
            "role": role,
            "seed": seed,
            "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
            "sequence_length": len(seq),
            "msa": str(msa),
            "msa_sha256": hashlib.sha256(msa.read_bytes()).hexdigest(),
            "msa_reused_from": str(src_yaml),
            "yaml": str(dst),
            "yaml_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            "ligands": {"adipoyl_AMP": ADIPOYL_AMP, "MG": "ccd", "HMDA_monoprotonated": HMDA},
        })
        print(f"{job:34s} seed={seed} len={len(seq)} msa={msa.name}")

    (BASE / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # the two fixed-cap designs share a sequence; confirm the panel's distinct genes
    uniq = {m["sequence_sha256"] for m in manifest}
    print(f"\n{len(manifest)} jobs, {len(uniq)} distinct sequences "
          f"(WT_AS9 and WT_AS9_rep2 intentionally share one)")


if __name__ == "__main__":
    main()
