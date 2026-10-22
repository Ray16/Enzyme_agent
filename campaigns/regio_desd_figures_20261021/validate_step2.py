#!/usr/bin/env python3
"""GATE on the step-2 inputs. Nonzero exit = do not submit.

Re-derives every expectation independently of the builder:
  1  chains: A protein(+MSA), B acyl-AMP(smiles), C amine(smiles), D ccd MG
  2  protein sequence byte-identical to the provenance record
  3  MSA query row == the protein sequence (catches a transplanted MSA)
  4  chain B re-derived from scratch with acyl_adenylate.build() and compared -- the intermediate
     must match what an independent construction gives, including its mass balance
  5  chain B carries EXACTLY ONE acyl-phosphate carbonyl (the step-2 electrophile)
  6  chain C carries EXACTLY ONE neutral -NH2 (the nucleophile; [NH3+] cannot attack) and no
     free -COOH
  7  pocket constraint: binder C, every contact a valid residue index in range, count matches
     job_map, and the per-enzyme counts are UNIFORM (a pocket coupled to docking quality would
     confound the whole comparison -- this is why the anchor was moved to ATP's PA)
  8  affinity binder is C
"""
import csv, json, pathlib, re, sys

CAMP = pathlib.Path(__file__).resolve().parent
ROOT = CAMP.parents[1]
REF = ROOT / "exp_data/amide_synthetase/literature_activity/substrate_reference.csv"
fails = []


def check(c, m):
    if not c:
        fails.append(m)
    return c


def msa_query(p):
    with open(p) as fh:
        if not fh.readline().startswith("key,sequence"):
            return None
        for line in fh:
            if line.strip():
                return re.sub(r"[-a-z.]", "", line.split(",", 1)[1].strip())
    return None


def main():
    sys.path.insert(0, str(CAMP))
    from rdkit import Chem
    from substrates_oligomer import DONOR, MONOMER_DONOR_OVERRIDE, ACCEPTOR
    from acyl_adenylate import build as build_acyl
    from build_step2_inputs import ACCEPTOR_MONOMER

    ent = json.loads((CAMP / "sequences/enzymes_provenance.json").read_text())
    donors = {r["letter_code"]: r["smiles"] for r in csv.DictReader(REF.open())
              if r["letter_code"] in ("A", "Z", "S", "G")}
    donors.update(MONOMER_DONOR_OVERRIDE); donors.update(DONOR)
    acceptors = {**ACCEPTOR_MONOMER, **ACCEPTOR}

    jm = {r["stem"]: r for r in csv.DictReader((CAMP / "cofolds/step2/job_map.csv").open())}
    yamls = sorted((CAMP / "cofolds/step2/inputs").glob("*.yaml"))
    check(len(yamls) == len(jm), f"{len(yamls)} YAMLs vs {len(jm)} job_map rows")

    acyl_cache, msa_cache, pockets = {}, {}, {}
    for y in yamls:
        row, tag, txt = jm.get(y.stem), y.stem, y.read_text()
        if row is None:
            fails.append(f"{tag}: not in job_map"); continue
        enz = row["enzyme"]

        ids = re.findall(r"^    id: (\S+)$", txt, re.M)
        check(ids == ["A", "B", "C", "D"], f"{tag}: chain ids {ids}, expected A,B,C,D")
        check(re.search(r"- ligand:\n    id: D\n    ccd: MG", txt), f"{tag}: chain D is not ccd MG")
        check(re.search(r"- affinity:\n    binder: C", txt), f"{tag}: affinity binder is not C")

        seq = re.search(r"^    sequence: (\S+)$", txt, re.M).group(1)
        check(seq == ent[enz]["seq"], f"{tag}: protein sequence differs from provenance")
        mp = pathlib.Path(re.search(r"^    msa: (\S+)$", txt, re.M).group(1))
        if check(mp.exists(), f"{tag}: MSA missing {mp}"):
            if mp not in msa_cache:
                msa_cache[mp] = msa_query(mp)
            check(msa_cache[mp] == seq, f"{tag}: MSA query != protein sequence (transplanted MSA?)")

        smis = re.findall(r"^    smiles: '(.+)'$", txt, re.M)
        check(len(smis) == 2, f"{tag}: expected 2 smiles ligands, found {len(smis)}")
        if len(smis) != 2:
            continue
        acyl_s, amine_s = smis

        # (4) independent reconstruction of the intermediate
        dcode = row["donor_code"]
        if dcode not in acyl_cache:
            acyl_cache[dcode] = build_acyl(donors[dcode], donor_code=dcode)[0]
        check(acyl_s == acyl_cache[dcode],
              f"{tag}: chain B differs from an independent acyl-AMP construction for {dcode}")
        check(acyl_s == row["acyl_smiles"], f"{tag}: chain B disagrees with job_map")

        b = Chem.MolFromSmiles(acyl_s)
        if check(b is not None, f"{tag}: acyl-AMP does not parse"):
            n_anh = len(b.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][PX4]")))
            check(n_anh == 1, f"{tag}: {n_anh} acyl-phosphate carbonyls, expected exactly 1")

        # (6) the nucleophile must be a NEUTRAL primary amine
        acode = row["acceptor_code"]
        check(amine_s == acceptors[acode], f"{tag}: amine differs from the acceptor table")
        c = Chem.MolFromSmiles(amine_s)
        if check(c is not None, f"{tag}: amine does not parse"):
            nh2 = len(c.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$([N+])]")))
            check(nh2 == 1, f"{tag}: {nh2} neutral -NH2 on the acceptor, expected exactly 1 "
                            f"(the nucleophile; [NH3+] has no lone pair and cannot attack)")
            check(not c.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2H1]")),
                  f"{tag}: acceptor carries a protonated -COOH; pH-7 convention is [O-]")

        # (7) pocket restraint
        m = re.search(r"    contacts: \[(.*?)\]\n    max_distance: ([\d.]+)\n    force: true",
                      txt, re.S)
        if check(m is not None, f"{tag}: pocket constraint absent or malformed"):
            idxs = [int(i) for i in re.findall(r'\["A", (\d+)\]', m.group(1))]
            check(bool(idxs), f"{tag}: pocket has no contacts")
            check(all(1 <= i <= len(seq) for i in idxs),
                  f"{tag}: pocket contact out of sequence range 1..{len(seq)}")
            check(len(idxs) == int(row["n_pocket"]), f"{tag}: pocket size disagrees with job_map")
            check(re.search(r"    binder: C", txt), f"{tag}: pocket binder is not C")
            pockets.setdefault(enz, set()).add(len(idxs))

    # (7b) uniformity across enzymes -- the reason the anchor was moved off the donor midpoint
    sizes = {e: next(iter(v)) for e, v in pockets.items() if len(v) == 1}
    for e, v in pockets.items():
        check(len(v) == 1, f"{e}: inconsistent pocket size across its own jobs: {sorted(v)}")
    if sizes:
        lo, hi = min(sizes.values()), max(sizes.values())
        check(hi <= 2.2 * lo, f"pocket sizes vary too much across enzymes ({sizes}) -- a restraint "
                              f"coupled to docking quality would confound the comparison")
        print(f"pocket sizes per enzyme: {sizes}  (spread {lo}-{hi}, uniform enough)")

    print(f"validated {len(yamls)} step-2 YAMLs, {len(acyl_cache)} distinct acyl-adenylates, "
          f"{len(msa_cache)} MSAs")
    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for f in fails[:25]:
            print(f"  FAIL {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED -- safe to submit")


if __name__ == "__main__":
    main()
