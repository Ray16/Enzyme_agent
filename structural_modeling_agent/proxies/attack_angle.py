"""Measure the established proximity-weighted attack angle without fitting."""

import math

METRIC_ID = "inverse_distance_attack_angle_v1"


def weighted_attack_angle(distances, angles):
    """Return inverse-distance mean angle and harmonic mean distance."""
    distances, angles = list(distances), list(angles)
    if not distances or len(distances) != len(angles):
        raise ValueError("A nonempty paired ensemble is required")
    if any(not math.isfinite(d) or d <= 0 for d in distances):
        raise ValueError("Distances must be finite and positive")
    if any(not math.isfinite(a) or not 0 <= a <= 180 for a in angles):
        raise ValueError("Angles must be finite and in [0, 180]")
    total = sum(1 / d for d in distances)
    return {"weighted_attack_angle_deg": sum(a / d for a, d in zip(angles, distances)) / total,
            "harmonic_distance_A": len(distances) / total, "n_poses": len(distances)}


def pdb_geometry(path, *, substrate_chain, nucleotide_chain, phosphate, bridge):
    """Find the nearest substrate carboxyl oxygen and its nucleotide attack angle."""
    chains = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        chains.setdefault(line[21], []).append((line[12:16].strip(), line[76:78].strip().upper(),
            tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))))
    nucleotide = {name: xyz for name, _, xyz in chains[nucleotide_chain]}
    center, leaving = nucleotide[phosphate], nucleotide[bridge]
    substrate = chains[substrate_chain]
    oxygens = [xyz for _, element, xyz in substrate if element == "O"]
    donors = set()
    for _, element, xyz in substrate:
        if element == "C":
            bonded = [oxygen for oxygen in oxygens if math.dist(oxygen, xyz) < 1.6]
            if len(bonded) >= 2:
                donors.update(bonded)
    if not donors:
        raise ValueError("No substrate carboxyl group identified")
    oxygen = min(donors, key=lambda xyz: math.dist(xyz, center))
    a = [x - y for x, y in zip(oxygen, center)]
    b = [x - y for x, y in zip(leaving, center)]
    distance = math.sqrt(sum(x*x for x in a))
    cosine = sum(x*y for x, y in zip(a, b)) / (distance * math.sqrt(sum(x*x for x in b)))
    return distance, math.degrees(math.acos(max(-1., min(1., cosine))))


def geometry_from_pdb(path, phosphate):
    """Resolve one nucleotide and one carboxyl-containing ligand, or abstain.

    This compatibility extractor rejects multiple possible participants. New
    modeling tasks should use the explicit entity/atom map in their plan.
    """
    from Bio.PDB.Polypeptide import is_aa
    bridge = {"PA": "O3A", "PG": "O3B"}.get(phosphate)
    if bridge is None:
        raise ValueError("unsupported reactive phosphate")
    ligands = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        if is_aa(line[17:20].strip(), standard=False):
            continue
        key = (line[21], line[22:27])
        ligands.setdefault(key, []).append((line[12:16].strip(), line[76:78].strip().upper(),
            tuple(float(line[a:b]) for a, b in ((30,38), (38,46), (46,54)))))
    nucleotides = [ats for ats in ligands.values() if {phosphate, bridge} <= {a[0] for a in ats}]
    if len(nucleotides) != 1:
        raise ValueError("one unambiguous nucleotide is required")
    nucleotide = {n: xyz for n, _, xyz in nucleotides[0]}
    donors = []
    for atoms in ligands.values():
        if atoms is nucleotides[0]:
            continue
        oxygens = [xyz for _, element, xyz in atoms if element == "O"]
        carboxyl = set()
        for _, element, xyz in atoms:
            if element == "C":
                bonded = [o for o in oxygens if math.dist(o, xyz) < 1.6]
                if len(bonded) >= 2:
                    carboxyl.update(bonded)
        if carboxyl:
            donors.append(carboxyl)
    if len(donors) != 1:
        raise ValueError("one unambiguous carboxyl-containing substrate is required")
    center, leaving = nucleotide[phosphate], nucleotide[bridge]
    oxygen = min(donors[0], key=lambda xyz: math.dist(xyz, center))
    distance = math.dist(oxygen, center)
    axis = math.dist(leaving, center)
    if distance <= 0 or axis <= 0:
        raise ValueError("degenerate reaction geometry")
    cosine = sum((x-y)*(z-y) for x,y,z in zip(oxygen, center, leaving)) / (distance*axis)
    return distance, math.degrees(math.acos(max(-1., min(1., cosine))))
