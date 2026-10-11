"""Near-attack geometry over the 49x50 stock-ESMFold2 panel.

Same selection rule as exp_data/amide_synthetase/code/synthetase_geometry.py:
element-based carboxylate detection, nucleotide found by atom names, never a
hard-coded donor atom id (that selector broke once on renumbering).
"""
import csv, math, pathlib, sys

ROOT = pathlib.Path("/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/"
                    "esmfold2/aurora_xpu/results/out_panel")

def parse(path):
    lig = {}
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            c = line.split()
            if c[0] != "HETATM":
                continue
            comp, asym = c[4], c[5]
            lig.setdefault((asym, comp), []).append((c[2], c[1].upper(),
                                                     (float(c[15]), float(c[16]), float(c[17]))))
    return lig

def geom(path, phosphate="PA"):
    lig = parse(path)
    nts = [v for v in lig.values() if {"PA", "PG", "O3A", "O3B"} <= {a[0] for a in v}]
    assert len(nts) == 1, (path, "ambiguous nucleotide")
    nt = {n: x for n, _, x in nts[0]}
    p, leave = nt[phosphate], nt["O3A" if phosphate == "PA" else "O3B"]
    donors = []
    for ats in lig.values():
        if any(a[0] == "PA" for a in ats):
            continue
        ox = [x for _, e, x in ats if e == "O"]
        carb = set()
        for _, e, x in ats:
            if e == "C":
                b = [o for o in ox if math.dist(o, x) < 1.6]
                if len(b) >= 2:
                    carb.update(b)
        if carb:
            donors.append(carb)
    assert len(donors) == 1, (path, "ambiguous acid")
    o = min(donors[0], key=lambda x: math.dist(x, p))
    a = [i - j for i, j in zip(o, p)]
    b = [i - j for i, j in zip(leave, p)]
    cs = sum(i * j for i, j in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
    return math.dist(o, p), math.degrees(math.acos(max(-1, min(1, cs))))

rows = []
for tgt in sorted(ROOT.iterdir()):
    if not tgt.is_dir():
        continue
    for cif in sorted(tgt.glob("sample_*.cif")):
        d, ang = geom(cif)
        rows.append({"target": tgt.name.replace("step1_", ""), "sample": cif.stem,
                     "d_A": round(d, 4), "angle_deg": round(ang, 3)})

out = pathlib.Path(sys.argv[1])
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "poses ->", out)
