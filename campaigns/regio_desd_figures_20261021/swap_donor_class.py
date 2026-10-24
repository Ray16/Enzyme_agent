#!/usr/bin/env python3
"""Amendment 8 -- does donor FUNCTIONAL CLASS change step-1 registration on ATP P-alpha?

Amendment 7 found the activity matrix is perfectly nested, so there is no enzyme x substrate
interaction for any descriptor to predict. What survives is a margin-level observation: the five
DesD-exclusive panels are exactly the five whose adenylated donor carries a free ammonium. That
observation was made on these labels and is not evidence. This script tests a prediction it makes
on an independent axis -- ground-state geometry -- using folds already in hand.

The contrast is a single functional group at constant size:

    G  glutarate         [O-]C(=O)CCCC(=O)[O-]      dicarboxylate     9 heavy atoms
    H  6-aminohexanoate  [NH3+]CCCCCC(=O)[O-]       omega-amino acid  9 heavy atoms

and the specificity control is a size change WITHOUT a class change:

    S  succinate         [O-]C(=O)CCC(=O)[O-]       dicarboxylate     7 heavy atoms

CARBOXYLATE-COUNT CONFOUND, controlled by construction: a diacid presents two carboxylate groups
and a mono-acid one, so a nearest-oxygen statistic favours the diacid by chance alone. The PRIMARY
statistic is the mean over carboxylate GROUPS of each group's own closest-oxygen distance, which is
unbiased with respect to group count. Nearest-over-all-groups is reported as secondary only.

Design, descriptors, prediction, the sign test and the S<->G control were written into
PREREGISTRATION.md amendment 8 before this file was run. Nothing here may be edited in response
to a score.
"""
import math, os, pathlib, sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "1")          # courtesy on a shared node
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

CAMP = pathlib.Path(__file__).resolve().parent
RESULTS = CAMP / "cofolds/step1/results"

ENZYMES = ["DdaG", "SfaB", "AsbA", "AsbB", "AcsA"]      # the five dicarboxylate specialists
REFERENCE = "DesD"                                      # out-of-sample directional check, NOT tested
DONOR_CHAIN, NUC_CHAIN, PHOS, BRIDGE = "D", "B", "PA", "O3A"
GATE_A = 4.0
BOND_A = 1.6                                            # same covalent cutoff as the frozen scorer
AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def parse(path):
    """-> {chain: [(atom_name, element, xyz, resname, resseq)]}, altloc-filtered like the frozen code."""
    chains = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        el = line[76:78].strip().upper()
        if el in ("H", "D", ""):
            continue
        try:
            xyz = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        except ValueError:
            continue
        chains.setdefault(line[21], []).append(
            (line[12:16].strip(), el, xyz, line[17:20].strip(), line[22:27].strip()))
    return chains


def donor_groups(atoms):
    """Carboxylate groups and basic nitrogens of the donor ligand.

    Carboxylate group = a carbon carrying >= 2 oxygens within BOND_A; the group is that carbon's
    oxygen set. Returned as a LIST OF GROUPS so a per-group statistic can be formed -- a diacid
    yields two, a mono-acid one.

    Basic N = a nitrogen none of whose bonded carbons bears an oxygen within BOND_A. That excludes
    amide nitrogens (the C=O carbon is bonded to an O) and keeps the free ammonium of an
    omega-amino acid. Selection is by element and bonding only, never by atom name, because Boltz
    renames every CCD ligand to LIG.
    """
    oxys = [xyz for _n, el, xyz, _r, _s in atoms if el == "O"]
    carbons = [xyz for _n, el, xyz, _r, _s in atoms if el == "C"]
    groups = []
    for c in carbons:
        bonded = [o for o in oxys if math.dist(o, c) < BOND_A]
        if len(bonded) >= 2:
            groups.append(bonded)
    carbonyl_c = {c for c in carbons if any(math.dist(o, c) < BOND_A for o in oxys)}
    basic_n = []
    for _n, el, xyz, _r, _s in atoms:
        if el != "N":
            continue
        nb = [c for c in carbons if math.dist(c, xyz) < BOND_A]
        if nb and not any(c in carbonyl_c for c in nb):
            basic_n.append(xyz)
    return groups, basic_n


def pose_geometry(path):
    """-> (d_grpmean, d_nearest, inverted, n_xyz) for one pose, or None if unparseable."""
    ch = parse(path)
    if DONOR_CHAIN not in ch or NUC_CHAIN not in ch:
        return None
    nuc = {name: xyz for name, _el, xyz, _r, _s in ch[NUC_CHAIN]}
    if PHOS not in nuc or BRIDGE not in nuc:
        return None
    pa = nuc[PHOS]
    groups, basic_n = donor_groups(ch[DONOR_CHAIN])
    if not groups:
        return None
    per_group = [min(math.dist(o, pa) for o in g) for g in groups]
    d_nearest = min(per_group)
    inverted = bool(basic_n) and min(math.dist(n, pa) for n in basic_n) < d_nearest
    return (sum(per_group) / len(per_group), d_nearest, inverted,
            basic_n[0] if basic_n else None, ch)


def nearest_residue(ch, xyz):
    """Closest protein residue to a point -> ('ASP123', distance)."""
    best, bd = None, 1e9
    for name, _el, p, resn, resseq in ch.get("A", []):
        if resn not in AA3:
            continue
        d = math.dist(p, xyz)
        if d < bd:
            bd, best = d, f"{resn}{resseq}"
    return best, bd


def ensemble(enz, donor):
    """Summarise one 50-pose step-1 ensemble. Short ensembles are reported, never padded."""
    stem = f"step1_{enz}__{donor}"
    pred = RESULTS / stem / f"boltz_results_{stem}" / "predictions" / stem
    gm, nr, inv, ncon = [], [], [], Counter()
    for p in sorted(pred.glob(f"{stem}_model_*.pdb")):
        g = pose_geometry(p)
        if g is None:
            continue
        d_grpmean, d_nearest, invert, n_xyz, ch = g
        gm.append(d_grpmean); nr.append(d_nearest); inv.append(invert)
        if n_xyz is not None:
            res, dd = nearest_residue(ch, n_xyz)
            if res is not None and dd < 4.5:
                ncon[res] += 1
    if not gm:
        return None
    return dict(n=len(gm), d_grpmean=sum(gm) / len(gm), d_nearest=sum(nr) / len(nr),
                min_d=min(nr), n_gated=sum(1 for d in nr if d < GATE_A),
                f_inverted=sum(inv) / len(inv), n_contacts=ncon)


def sign_test(deltas):
    """One-sided sign test, H1: delta > 0. Ties are counted against H1 (conservative)."""
    pos = sum(1 for d in deltas if d > 0)
    n = len(deltas)
    p = sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n
    return pos, n, p


def main():
    rows = {}
    for enz in ENZYMES + [REFERENCE]:
        for donor in ("S", "G", "H"):
            r = ensemble(enz, donor)
            if r is None:
                print(f"  !! no parseable poses: {enz} {donor}")
                continue
            rows[(enz, donor)] = r

    print("=" * 78)
    print("PER-ENSEMBLE GEOMETRY  (d in A, averaged over poses; n_gated = poses with d_nearest<4)")
    print("=" * 78)
    print(f"{'enzyme':<8}{'donor':<7}{'n':<5}{'d_grpmean':<12}{'d_nearest':<12}"
          f"{'min_d':<9}{'n_gated':<9}{'f_inv'}")
    for enz in ENZYMES + [REFERENCE]:
        for donor in ("S", "G", "H"):
            r = rows.get((enz, donor))
            if r is None:
                continue
            print(f"{enz:<8}{donor:<7}{r['n']:<5}{r['d_grpmean']:<12.3f}{r['d_nearest']:<12.3f}"
                  f"{r['min_d']:<9.3f}{r['n_gated']:<9}{r['f_inverted']:.2f}")

    # ---- PRIMARY: paired delta on d_grpmean, G -> H, five specialists ----------------------
    print("\n" + "=" * 78)
    print("PRIMARY (8.5): delta = d_grpmean(H) - d_grpmean(G); predicted > 0 in specialists")
    print("=" * 78)
    deltas = []
    for enz in ENZYMES:
        a, b = rows.get((enz, "H")), rows.get((enz, "G"))
        if a is None or b is None:
            print(f"  {enz:<8} MISSING")
            continue
        d = a["d_grpmean"] - b["d_grpmean"]
        deltas.append(d)
        print(f"  {enz:<8} H {a['d_grpmean']:7.3f}  G {b['d_grpmean']:7.3f}   "
              f"delta = {d:+7.3f}   {'as predicted' if d > 0 else 'AGAINST prediction'}")
    pos, n, p = sign_test(deltas)
    print(f"\n  sign test: {pos}/{n} concordant, one-sided p = {p:.4f}"
          f"   DECISION: {'PASS' if p < 0.05 else 'FAIL'}  (alpha = 0.05, 5/5 required)")

    ref = (rows.get((REFERENCE, "H")), rows.get((REFERENCE, "G")))
    if all(ref):
        dref = ref[0]["d_grpmean"] - ref[1]["d_grpmean"]
        print(f"\n  out-of-sample check (NOT in the test) -- {REFERENCE}: delta = {dref:+.3f}"
              f"  -> predicted NOT degraded; {'consistent' if dref <= 0 else 'INCONSISTENT'}")

    # ---- MANDATORY CONTROL 8.6: S -> G, size change without class change --------------------
    print("\n" + "=" * 78)
    print("SPECIFICITY CONTROL (8.6): S -> G is a size change with NO class change")
    print("=" * 78)
    sg = []
    for enz in ENZYMES:
        a, b = rows.get((enz, "G")), rows.get((enz, "S"))
        if a is None or b is None:
            continue
        d = a["d_grpmean"] - b["d_grpmean"]
        sg.append(d)
        print(f"  {enz:<8} G {a['d_grpmean']:7.3f}  S {b['d_grpmean']:7.3f}   delta = {d:+7.3f}")
    if sg and deltas:
        mgh = sum(abs(d) for d in deltas) / len(deltas)
        msg = sum(abs(d) for d in sg) / len(sg)
        print(f"\n  mean |delta| class swap G->H = {mgh:.3f} A")
        print(f"  mean |delta| size  swap S->G = {msg:.3f} A")
        print(f"  class/size ratio = {mgh / msg:.2f}x   "
              f"-> interpretation {'SURVIVES' if mgh > msg else 'FALSIFIED (8.6)'}")

    # ---- descriptive: where does the free ammonium of H sit? --------------------------------
    print("\n" + "=" * 78)
    print("DESCRIPTIVE (8.3 items 4-5): the free ammonium of donor H")
    print("=" * 78)
    for enz in ENZYMES + [REFERENCE]:
        r = rows.get((enz, "H"))
        if r is None:
            continue
        top = ", ".join(f"{k} {v}/{r['n']}" for k, v in r["n_contacts"].most_common(3)) or "none <4.5 A"
        print(f"  {enz:<8} f_inverted = {r['f_inverted']:.2f}   nearest residues: {top}")


if __name__ == "__main__":
    main()
