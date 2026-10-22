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
        return
    tally(rows)


def tally(rows):
    """DESCRIPTIVE step-2 comparison against activity.

    Labelled DESCRIPTIVE deliberately. PREREGISTRATION.md requires step 2 to be reported
    SEPARATELY from step 1, but it does not fix step-2 tests with the specificity it fixes T1-T5
    for step 1 -- no ideal-angle tolerance, no stratification scheme and no alpha were declared for
    the Burgi-Dunitz deviation before folding. These numbers are therefore reported as description,
    never as a confirmatory test, and no p-value is attached to them.
    """
    import sys as _sys
    _sys.path.insert(0, str(CAMP))
    from plot_panels import SOURCES, load  # noqa: E402

    labels = {}
    for _n, (path, _panels, _o) in SOURCES.items():
        if path.exists():
            labels.update(load(path))
    if not labels:
        print("\nno digitised activity labels on disk -- step-2 tally skipped")
        return

    # Map each step-2 cell to its figure panel from the AUTHORITATIVE source -- the
    # (acid donor, amine acceptor, panel) tuples in build_step2_inputs.STEP2_PAIRS. The `+` order
    # inside the panel LABEL is prose and is not consistent (2A reads "X+A" amine-first, 4C reads
    # "MA+H" acid-first), so the label text must never be parsed for donor/acceptor identity.
    #
    # Panel 4C carries TWO tuples: the measured ("MA","H") and the alternative regiochemistry
    # ("H","MA"), flagged as such in its own label. The alternative makes a different product that
    # panel 4C does not measure, so admitting it would place two cells of the same enzyme in one
    # panel and double-count a single experimental label. It is excluded and reported.
    from build_step2_inputs import STEP2_PAIRS  # noqa: E402

    pair2panel, alt_pairs = {}, set()
    for donor, acceptor, label in STEP2_PAIRS:
        code = label.split()[0]
        if "other regiochemistry" in label:
            alt_pairs.add((donor, acceptor))
            continue
        pair2panel[(donor, acceptor)] = f"Figure {code}"

    by_panel, skipped = {}, []
    for r in rows:
        key = (r["donor_code"], r["acceptor_code"])
        if key in alt_pairs:
            skipped.append(key)
        elif key in pair2panel:
            by_panel.setdefault(pair2panel[key], []).append(r)

    if skipped:
        print(f"\n  {len(skipped)} alternative-regiochemistry cell(s) excluded (H+MA->HMA is a\n"
              f"  different product than panel 4C measures)")
    print("\n" + "=" * 78)
    print("STEP-2 (aminolysis) vs ACTIVITY -- DESCRIPTIVE, not a pre-registered test")
    print("=" * 78)
    print("ideal Burgi-Dunitz angle = 105 deg; `best BD` = smallest |BD - 105|, `occ` = n_gated/50")

    best_occ = best_bd = n_contrast = 0
    pooled = {}
    for panel in sorted(by_panel):
        lab = labels.get(panel)
        if not lab:
            continue
        cells = [r for r in by_panel[panel] if r["enzyme"] in lab]
        if len(cells) < 2:
            continue
        for r in cells:
            pooled[(r["enzyme"], r["donor_code"], r["acceptor_code"])] = (
                r, not lab[r["enzyme"]]["below_LOD"])
        det = [r for r in cells if not lab[r["enzyme"]]["below_LOD"]]
        und = [r for r in cells if lab[r["enzyme"]]["below_LOD"]]
        if not det or not und:
            print(f"\n  {panel}: no contrast ({len(det)}/{len(cells)} detected) -- not counted")
            continue
        n_contrast += 1
        top_occ = max(cells, key=lambda r: r["n_gated"])
        top_bd = min(cells, key=lambda r: r["dev_from_ideal"])
        hit_occ = top_occ in det
        hit_bd = top_bd in det
        best_occ += hit_occ
        best_bd += hit_bd
        print(f"\n  {panel}: {len(det)} active / {len(und)} below LOD")
        print(f"    highest occupancy : {top_occ['enzyme']:5s} "
              f"({top_occ['n_gated']}/50)  -> active? {hit_occ}")
        print(f"    best BD angle     : {top_bd['enzyme']:5s} "
              f"({top_bd['BD_deg']:.1f} deg, dev {top_bd['dev_from_ideal']:.1f})"
              f"  -> active? {hit_bd}")

    def auroc(vals, pos):
        p = [v for v, y in zip(vals, pos) if y]
        n = [v for v, y in zip(vals, pos) if not y]
        if not p or not n:
            return None
        s = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in p for b in n)
        return s / (len(p) * len(n))

    cells = list(pooled.values())
    ys = [y for _r, y in cells]
    a_dev = auroc([r["dev_from_ideal"] for r, _y in cells], ys)
    a_occ = auroc([r["n_gated"] for r, _y in cells], ys)
    print(f"\n  --- pooled over unique (enzyme, donor, acceptor) cells ---")
    print(f"  n = {len(cells)} cells, {sum(ys)} detected, {len(ys)-sum(ys)} below LOD")
    print(f"  panels with both classes                 : {n_contrast}")
    print(f"  active enzyme has the HIGHEST occupancy  : {best_occ}/{n_contrast}"
          f"   (chance ~ 1/n_enzymes per panel)")
    print(f"  active enzyme has the BEST BD angle      : {best_bd}/{n_contrast}")
    if a_dev is not None:
        print(f"  AUROC(BD deviation)  = {a_dev:.3f}   "
              f"[lower deviation should favour active -> inverted: {1 - a_dev:.3f}]")
        print(f"  AUROC(occupancy)     = {a_occ:.3f}")
    print("\n  Reading: occupancy is at chance and the Burgi-Dunitz angle is no better, mirroring")
    print("  step 1. The phantom-angle pathology recurs on the amine side -- cells with a textbook")
    print("  BD angle measured at a D_h far outside bonding range:")
    for r, y in sorted(cells, key=lambda t: t[0]["dev_from_ideal"])[:40]:
        if y and r["D_h_A"] > 10:
            print(f"    {r['enzyme']:5s} {r['donor_code']:>4s}+{r['acceptor_code']:<4s} "
                  f"ACTIVE  BD {r['BD_deg']:6.1f} (dev {r['dev_from_ideal']:4.1f})  "
                  f"D_h {r['D_h_A']:5.2f} A  n_gated {r['n_gated']}/50")


if __name__ == "__main__":
    main()
