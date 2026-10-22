#!/usr/bin/env python3
"""ATOM-LEVEL AUDIT: are we measuring between the right pairs of atoms?

The scoring code selects participants by geometry/element because Boltz renames ligands to LIG,
renumbers substrate atoms and writes no bond orders or hydrogens. That inference is the single
biggest correctness risk in the campaign, so this prints the ATOMS ACTUALLY CHOSEN, by name, and
independently verifies the bonding assumptions each selection rests on.

STEP 1 -- adenylation, in-line attack at P-alpha, PPi leaves:
  nucleophile  nearest donor carboxylate O to PA      (chain D)
  electrophile PA                                      (chain B)
  axis atom    O3A                                     (chain B)
  angle        O(nuc)-PA-O3A, ideal ~180 deg (in-line, nucleophile opposite the leaving group)
  ASSUMPTION CHECKED: O3A must bridge PA and PB (it is the PA->PPi bond that breaks). If O3A were
  instead terminal on PA, the angle would be measured to the wrong vector entirely.

STEP 2 -- aminolysis, Burgi-Dunitz attack at the acyl carbonyl, AMP leaves:
  nucleophile  nearest non-amide N to the carbonyl C   (chain C)
  electrophile anhydride carbonyl C                    (chain B)
  carbonyl O   the TERMINAL O on that carbon           (chain B)
  angle        N-C=O, ideal ~105 deg
  ASSUMPTIONS CHECKED: the carbonyl C carries exactly two O, exactly one of which bridges to P
  (the C(=O)-O-P anhydride); the other is terminal and is the one used for the angle.
"""
import math, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent


def atoms(path):
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        try:
            xyz = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        except ValueError:
            continue
        out.append((line[21], line[12:16].strip(), line[76:78].strip().upper(), xyz))
    return out


def audit_step1(path):
    ats = atoms(path)
    B = {n: x for ch, n, _e, x in ats if ch == "B"}
    D = [(n, e, x) for ch, n, e, x in ats if ch == "D"]
    print(f"  chain B (ATP) has {len(B)} atoms; chain D (donor) has {len(D)}")

    # --- verify O3A really bridges PA and PB ------------------------------------------
    for a, b in (("PA", "O3A"), ("PB", "O3A"), ("PB", "O3B"), ("PG", "O3B"), ("PA", "O1A")):
        if a in B and b in B:
            print(f"    d({a},{b}) = {math.dist(B[a], B[b]):.2f} A")
    bridging_ok = (math.dist(B['PA'], B['O3A']) < 1.8 and math.dist(B['PB'], B['O3A']) < 1.8)
    print(f"    -> O3A bridges PA and PB: {bridging_ok}  "
          f"{'(correct leaving-group axis)' if bridging_ok else '*** WRONG AXIS ***'}")
    terminal_oa = [n for n in B if n.startswith('O') and n.endswith('A')
                   and math.dist(B['PA'], B[n]) < 1.8]
    print(f"    PA-bonded oxygens: {sorted(terminal_oa)}  (O3A must be the one to PB)")

    # --- the donor carboxylate oxygens, exactly as the frozen extractor finds them -----
    oxy = [(n, x) for n, e, x in D if e == "O"]
    cands = []
    for n, e, x in D:
        if e != "C":
            continue
        bonded = [(on, ox) for on, ox in oxy if math.dist(ox, x) < 1.6]
        if len(bonded) >= 2:
            cands.append((n, x, bonded))
    print(f"    donor carboxyl carbons found: {[c[0] for c in cands]} "
          f"({len(cands)} -- a diacid should give 2, a mono-acid 1)")
    allowed = {on for _c, _x, bl in cands for on, _ox in bl}
    chosen = min(((on, ox) for on, ox in oxy if on in allowed),
                 key=lambda t: math.dist(t[1], B["PA"]))
    d = math.dist(chosen[1], B["PA"])
    a = [p - q for p, q in zip(chosen[1], B["PA"])]
    v = [p - q for p, q in zip(B["O3A"], B["PA"])]
    na, nv = math.sqrt(sum(t * t for t in a)), math.sqrt(sum(t * t for t in v))
    ang = math.degrees(math.acos(max(-1, min(1, sum(p * q for p, q in zip(a, v)) / (na * nv)))))
    print(f"    MEASURED: {chosen[0]}(chain D) ... PA = {d:.2f} A ; "
          f"angle {chosen[0]}-PA-O3A = {ang:.1f} deg  (ideal ~180, in-line)")
    print(f"    sanity: carboxylate O chosen is one of {sorted(allowed)}")
    return bridging_ok


def audit_step2(path):
    ats = atoms(path)
    B = [(n, e, x) for ch, n, e, x in ats if ch == "B"]
    C = [(n, e, x) for ch, n, e, x in ats if ch == "C"]
    print(f"  chain B (acyl-AMP) has {len(B)} atoms; chain C (amine) has {len(C)}")
    bO = [(n, x) for n, e, x in B if e == "O"]
    bP = [(n, x) for n, e, x in B if e == "P"]
    print(f"    phosphorus atoms in chain B: {[n for n, _ in bP]}  (acyl-AMP must have exactly 1)")

    carbonyls = []
    for n, e, x in B:
        if e != "C":
            continue
        bonded = [(on, ox) for on, ox in bO if math.dist(ox, x) < 1.6]
        if len(bonded) < 2:
            continue
        bridging = [(on, ox) for on, ox in bonded if any(math.dist(ox, px) < 1.75 for _pn, px in bP)]
        terminal = [(on, ox) for on, ox in bonded if (on, ox) not in bridging]
        if bridging and terminal:
            carbonyls.append((n, x, bridging, terminal))
    print(f"    anhydride carbonyl carbons: {[c[0] for c in carbonyls]} "
          f"({len(carbonyls)} -- must be exactly 1)")
    if not carbonyls:
        print("    *** no C(=O)-O-P found -- step-2 electrophile NOT identified ***")
        return False
    cn, cx, brg, term = carbonyls[0]
    print(f"      carbon {cn}: bridging O to P = {[n for n, _ in brg]}, "
          f"terminal (carbonyl) O = {[n for n, _ in term]}")

    cOx = [x for _n, e, x in C if e == "O"]
    cCx = [x for _n, e, x in C if e == "C"]
    nitro = []
    for n, e, x in C:
        if e != "N":
            continue
        amide = any(math.dist(c, x) < 1.6 and any(math.dist(o, c) < 1.35 for o in cOx)
                    for c in cCx)
        nitro.append((n, x, amide))
    print(f"    chain C nitrogens: " +
          ", ".join(f"{n}{'(amide,excluded)' if am else '(free amine)'}" for n, _x, am in nitro))
    free = [(n, x) for n, x, am in nitro if not am]
    if not free:
        print("    *** no free amine N -- step-2 nucleophile NOT identified ***")
        return False
    nn, nx = min(free, key=lambda t: math.dist(t[1], cx))
    d = math.dist(nx, cx)
    a = [p - q for p, q in zip(nx, cx)]
    v = [p - q for p, q in zip(term[0][1], cx)]
    na, nv = math.sqrt(sum(t * t for t in a)), math.sqrt(sum(t * t for t in v))
    ang = math.degrees(math.acos(max(-1, min(1, sum(p * q for p, q in zip(a, v)) / (na * nv)))))
    print(f"    MEASURED: {nn}(chain C) ... {cn}(chain B) = {d:.2f} A ; "
          f"angle {nn}-{cn}-{term[0][0]} = {ang:.1f} deg  (ideal ~105, Burgi-Dunitz)")
    return True


def main():
    ok = True
    print("=" * 78)
    print("STEP 1 -- in-line attack at ATP P-alpha")
    print("=" * 78)
    for stem in ("step1_DesD__A", "step1_DdaG__A", "step1_DesD__AMA"):
        p = (CAMP / f"cofolds/step1/results/{stem}/boltz_results_{stem}/predictions/{stem}"
             / f"{stem}_model_0.pdb")
        if p.exists():
            print(f"\n{stem}:")
            ok &= audit_step1(p)
    print("\n" + "=" * 78)
    print("STEP 2 -- Burgi-Dunitz attack at the acyl-adenylate carbonyl")
    print("=" * 78)
    found = 0
    for d in sorted((CAMP / "cofolds/step2/results").glob("step2_*")):
        p = next((d / f"boltz_results_{d.name}" / "predictions" / d.name).glob("*_model_0.pdb"), None) \
            if (d / f"boltz_results_{d.name}" / "predictions" / d.name).exists() else None
        if p:
            print(f"\n{d.name}:")
            ok &= audit_step2(p)
            found += 1
            if found >= 3:
                break
    if not found:
        print("\n  (no step-2 pose on disk yet)")
    print("\n" + ("ALL BONDING ASSUMPTIONS VERIFIED" if ok else "*** AUDIT FOUND A PROBLEM ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
