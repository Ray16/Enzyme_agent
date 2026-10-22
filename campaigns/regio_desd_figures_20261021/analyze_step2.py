#!/usr/bin/env python3
"""Score the step-2 (aminolysis) ensembles -- Burgi-Dunitz geometry, NOT step-1's in-line geometry.

Step 2 differs from step 1 in every participant (MECHANISM.md sec 3):
    nucleophile   acceptor amine N        (step 1: donor carboxylate O)
    electrophile  acyl-adenylate CARBONYL CARBON   (step 1: ATP P-alpha)
    leaving group AMP                     (step 1: PPi)
    ideal angle   ~105 deg Burgi-Dunitz   (step 1: ~180 deg in-line)
So a step-2 score is NEVER comparable with, or averaged into, step-1 `A`.

The same FROZEN estimator is reused (weighted_attack_angle: inverse-distance weighting over all 50
poses), applied to the Burgi-Dunitz angle N...C=O. Reusing the estimator keeps the statistical form
identical while the geometry is correct for this half-reaction.

ATOM IDENTIFICATION -- the PDB has no bond orders and no hydrogens, so both are inferred:
  * electrophile: in chain B, the carbon bonded to >=2 oxygens where one of those oxygens is also
    bonded to phosphorus -- i.e. the C of the C(=O)-O-P mixed anhydride. The remaining oxygen is
    the carbonyl O used for the angle.
  * nucleophile: in chain C, the amine nitrogen NEAREST the electrophile. Several acceptors carry
    two nitrogens (M, X, MAM) and protonation state is invisible without hydrogens, so the attacking
    one cannot be read off the file. Taking the nearest is the same convention step 1 uses for the
    carboxylate oxygen, and it is stated rather than hidden. Amide nitrogens are EXCLUDED (an N
    bonded to a carbon that itself bears a carbonyl O is a backbone/linkage amide, not a nucleophile).
"""
import csv, math, pathlib, sys

CAMP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CAMP.parents[1]))
from structural_modeling_agent.proxies.attack_angle import (  # noqa: E402
    METRIC_ID, weighted_attack_angle)

RESULTS = CAMP / "cofolds/step2/results"
GATE_A = 4.0          # validity precondition only, never a ranking input
BD_IDEAL = 105.0


def atoms_by_chain(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        try:
            xyz = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        except ValueError:
            continue
        el = line[76:78].strip().upper()
        if el:
            out.setdefault(line[21], []).append((line[12:16].strip(), el, xyz))
    return out


def geometry(path):
    """-> (d(N...C), Burgi-Dunitz angle N-C=O) or None if the participants are not unambiguous."""
    ch = atoms_by_chain(path)
    B, C = ch.get("B"), ch.get("C")
    if not B or not C:
        return None
    bO = [x for _n, el, x in B if el == "O"]
    bP = [x for _n, el, x in B if el == "P"]
    # the anhydride carbonyl: C with >=2 bonded O, one of which bridges to P
    carbonyl = None
    for _n, el, x in B:
        if el != "C":
            continue
        bonded = [o for o in bO if math.dist(o, x) < 1.6]
        if len(bonded) < 2:
            continue
        bridging = [o for o in bonded if any(math.dist(o, p) < 1.75 for p in bP)]
        terminal = [o for o in bonded if o not in bridging]
        if bridging and terminal:
            carbonyl = (x, terminal[0])
            break
    if carbonyl is None:
        return None
    cc, co = carbonyl

    cO = [x for _n, el, x in C if el == "O"]
    cCs = [x for _n, el, x in C if el == "C"]
    nitrogens = []
    for _n, el, x in C:
        if el != "N":
            continue
        # exclude amide N: bonded to a carbon that itself carries a carbonyl oxygen
        amide = False
        for c in cCs:
            if math.dist(c, x) < 1.6 and any(math.dist(o, c) < 1.35 for o in cO):
                amide = True
                break
        if not amide:
            nitrogens.append(x)
    if not nitrogens:
        return None
    nn = min(nitrogens, key=lambda x: math.dist(x, cc))

    d = math.dist(nn, cc)
    a = [p - q for p, q in zip(nn, cc)]
    b = [p - q for p, q in zip(co, cc)]
    na = math.sqrt(sum(v * v for v in a)); nb = math.sqrt(sum(v * v for v in b))
    if d <= 0 or na <= 0 or nb <= 0:
        return None
    cos = sum(p * q for p, q in zip(a, b)) / (na * nb)
    return d, math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def main():
    jm = {r["stem"]: r for r in csv.DictReader((CAMP / "cofolds/step2/job_map.csv").open())}
    rows, pending = [], 0
    for stem, meta in sorted(jm.items()):
        pred = RESULTS / stem / f"boltz_results_{stem}" / "predictions" / stem
        pdbs = sorted(pred.glob(f"{stem}_model_*.pdb")) if pred.exists() else []
        if len(pdbs) < 50:
            pending += 1
            continue
        ds, angs, bad = [], [], 0
        for p in pdbs:
            g = geometry(p)
            if g is None:
                bad += 1
                continue
            ds.append(g[0]); angs.append(g[1])
        if not ds:
            continue
        res = weighted_attack_angle(ds, angs)
        n_gated = sum(1 for d in ds if d < GATE_A)
        rows.append(dict(
            stem=stem, enzyme=meta["enzyme"], donor_code=meta["donor_code"],
            acceptor_code=meta["acceptor_code"],
            BD_deg=round(res["weighted_attack_angle_deg"], 2),
            dev_from_ideal=round(abs(res["weighted_attack_angle_deg"] - BD_IDEAL), 2),
            D_h_A=round(res["harmonic_distance_A"], 3),
            n_gated=n_gated, gated_frac=round(n_gated / len(ds), 3),
            min_d=round(min(ds), 3), median_d=round(sorted(ds)[len(ds) // 2], 3),
            n_poses=len(ds), n_extract_fail=bad, docked=n_gated > 0,
            metric_id=METRIC_ID + "|burgi_dunitz_N_C_O", panels=meta["panels"]))

    if rows:
        out = CAMP / "analysis/step2_scores.csv"
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"scored {len(rows)} step-2 ensembles -> {out}   ({pending} still folding)")
        print(f"{'stem':34s} {'BD':>7s} {'|dev|':>6s} {'D_h':>6s} {'n_gated':>8s}  docked")
        for r in sorted(rows, key=lambda r: r["dev_from_ideal"]):
            print(f"{r['stem']:34s} {r['BD_deg']:7.1f} {r['dev_from_ideal']:6.1f} "
                  f"{r['D_h_A']:6.2f} {r['n_gated']:8d}  {r['docked']}")
    else:
        print(f"no step-2 ensemble complete yet ({pending} folding)")


if __name__ == "__main__":
    main()
