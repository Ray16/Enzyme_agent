#!/usr/bin/env python3
"""Structural validation of folded ensembles -- run BEFORE scoring, gate on the result.

The frozen metric (attack_angle.py) is deliberately not modified. This module checks that the
structures it will be fed are actually what we asked for, because several failure modes are silent:

  * Boltz renames every CCD ligand to `LIG`, so residue name identifies nothing
  * Boltz RENUMBERS the substrate's atoms (adipate came back as O11-O14), so atom names do not
    identify the reactive group either -- a hard-coded selector circuit-broke a previous campaign
  * Boltz exits 0 on OOM, so an ensemble can be short or empty without an error
  * a truncated / wrong ligand still parses fine and still yields a plausible-looking angle

Checks per model, per ensemble:
  1  PDB has a populated element column (cols 77-78) -- the frozen extractor relies on it
  2  chain B is ATP: carries PA, O3A, PG, O3B and 31 heavy atoms
  3  Mg chains carry exactly one MG atom each, and the count matches the input YAML
  4  chain D heavy-atom count == heavy-atom count of the input SMILES   <-- catches wrong/truncated
  5  number of carboxylate groups in chain D == number in the input SMILES
  6  at least one Mg is within 3.5 A of an ATP phosphate oxygen (cofactor actually placed)

Exit nonzero if any ensemble fails. `--quiet` prints only failures.
"""
import csv, math, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
RESULTS = CAMP / "cofolds/step1/results"
ATP_REQUIRED = {"PA", "O3A", "PG", "O3B"}
ATP_HEAVY = 31


def atoms(path):
    """Parse HETATM/ATOM into (chain, name, element, xyz); abstain loudly on a blank element."""
    out, no_elem = [], 0
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        element = line[76:78].strip().upper()
        if not element:
            no_elem += 1
            continue
        out.append((line[21], line[12:16].strip(), element,
                    tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))))
    return out, no_elem


def n_carboxylates(ats):
    """Count C bonded to >=2 O -- the same rule the frozen extractor uses."""
    oxy = [xyz for _, _, el, xyz in ats if el == "O"]
    return sum(1 for _, _, el, xyz in ats if el == "C"
               and sum(1 for o in oxy if math.dist(o, xyz) < 1.6) >= 2)


def smiles_expectations(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return (m.GetNumHeavyAtoms(),
            len(m.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX1]"))))


def claim_held(stem):
    """True if a fleet worker still owns this job (atomic mkdir claim not yet released)."""
    return (CAMP / "fleet/claims" / f"cofolds_step1_inputs_{stem}.yaml").is_dir()


def check_ensemble(stem, smi, n_mg_expected, quiet=False):
    pred = RESULTS / stem / f"boltz_results_{stem}" / "predictions" / stem
    pdbs = sorted(pred.glob(f"{stem}_model_*.pdb"))
    fails = []
    if len(pdbs) < 50:
        # A short ensemble means one of two very different things, and conflating them makes the
        # gate abort on every in-flight job. Boltz writes poses incrementally, so a worker that
        # still HOLDS its claim is simply mid-write -> not yet scorable, not a failure. Only a
        # released claim (the worker gave up, incl. the exit-0 OOM case) is a real failure.
        if claim_held(stem):
            return []
        return [f"{stem}: only {len(pdbs)} poses and the fleet claim was RELEASED "
                f"(Boltz exits 0 on OOM -- treat as FAILED run)"]

    exp = smiles_expectations(smi)
    if exp is None:
        return [f"{stem}: input SMILES does not parse: {smi}"]
    exp_heavy, exp_cooh = exp

    for p in pdbs:
        ats, no_elem = atoms(p)
        tag = f"{stem}/{p.stem.rsplit('_', 1)[-1]}"
        if no_elem:
            fails.append(f"{tag}: {no_elem} atoms have a BLANK element column")
        by_chain = {}
        for ch, name, el, xyz in ats:
            by_chain.setdefault(ch, []).append((name, el, xyz))

        b = by_chain.get("B", [])
        names_b = {n for n, _, _ in b}
        if not ATP_REQUIRED <= names_b:
            fails.append(f"{tag}: chain B missing ATP atoms {sorted(ATP_REQUIRED - names_b)}")
        if len(b) != ATP_HEAVY:
            fails.append(f"{tag}: chain B has {len(b)} heavy atoms, expected {ATP_HEAVY} (ATP)")

        mg_chains = [c for c, v in by_chain.items()
                     if len(v) == 1 and v[0][1] == "MG"]
        if len(mg_chains) != n_mg_expected:
            fails.append(f"{tag}: {len(mg_chains)} Mg chains, YAML declares {n_mg_expected}")

        d = by_chain.get("D", [])
        if not d:
            fails.append(f"{tag}: chain D (acid donor) absent")
        else:
            if len(d) != exp_heavy:
                fails.append(f"{tag}: chain D has {len(d)} heavy atoms, SMILES implies "
                             f"{exp_heavy} -- WRONG OR TRUNCATED LIGAND")
            got_cooh = n_carboxylates([("D", n, e, x) for n, e, x in d])
            if got_cooh != exp_cooh:
                fails.append(f"{tag}: chain D has {got_cooh} carboxyl groups, SMILES implies "
                             f"{exp_cooh}")

        # cofactor actually placed at the nucleotide
        phos_o = [xyz for n, el, xyz in b if el == "O" and n.startswith("O")]
        mg_xyz = [by_chain[c][0][2] for c in mg_chains]
        if mg_xyz and phos_o and not any(
                math.dist(m, o) < 3.5 for m in mg_xyz for o in phos_o):
            fails.append(f"{tag}: no Mg within 3.5 A of an ATP phosphate oxygen")

    if not fails and not quiet:
        print(f"  OK  {stem}: 50 poses, ATP+{n_mg_expected}Mg+donor all well-formed "
              f"({exp_heavy} heavy atoms, {exp_cooh} COO-)")
    return fails


def main():
    quiet = "--quiet" in sys.argv
    jm = list(csv.DictReader((CAMP / "cofolds/step1/job_map.csv").open()))
    all_fails, checked = [], 0
    for r in jm:
        pred = RESULTS / r["stem"] / f"boltz_results_{r['stem']}" / "predictions" / r["stem"]
        if not pred.exists():
            continue
        checked += 1
        all_fails += check_ensemble(r["stem"], r["smiles"], int(r["n_mg"]), quiet)
    print(f"\nchecked {checked} ensembles present on disk")
    if all_fails:
        # collapse per-pose repeats so one systematic fault is not 50 lines
        print(f"{len(all_fails)} FAILURES:")
        for f in all_fails[:25]:
            print(f"  FAIL {f}")
        if len(all_fails) > 25:
            print(f"  ... and {len(all_fails) - 25} more")
        sys.exit(1)
    print("ALL STRUCTURAL CHECKS PASSED")


if __name__ == "__main__":
    main()
