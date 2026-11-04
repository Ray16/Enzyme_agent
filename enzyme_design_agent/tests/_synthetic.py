"""A tiny synthetic holo structure, so the design tests need no real data.

Geometry is laid out deliberately: residues on a line 4 A apart, a two-atom
ligand beside residue 2, a bridging metal that reaches residue 3 only through
the ligand, and a second metal parked far away. That is the exact shape of the
mistake the pocket code exists to avoid -- a cofactor-only position that a
substrate-centred scan would miss, and a stray ion whose shell would be noise.
"""
from __future__ import annotations

from pathlib import Path

RESIDUES = ["ALA", "SER", "GLY", "VAL", "THR"]
SEQUENCE = "ASGVT"


def _atom(serial, name, resname, chain, resnum, xyz, element, het=False):
    rec = "HETATM" if het else "ATOM  "
    x, y, z = xyz
    return (f"{rec}{serial:5d} {name:^4s}{resname:>4s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")


def write_structure(path: str | Path, stray_metal: bool = True,
                    bridging_metal: bool = True) -> Path:
    """Write the synthetic holo PDB and return its path."""
    lines, serial = [], 1
    # 8 A apart, so a residue's shell membership is unambiguous: at 4 A spacing
    # every ligand near one residue is also within 5 A of its neighbours, and
    # the cofactor-only case the pocket code exists to catch cannot be posed.
    for resnum, resname in enumerate(RESIDUES, start=1):
        for name, element, off in (("N", "N", 0.0), ("CA", "C", 1.0), ("CB", "C", 1.5)):
            lines.append(_atom(serial, name, resname, "A", resnum,
                               (resnum * 8.0 + off, 0.0, 0.0), element))
            serial += 1
    # Substrate: beside residue 2 (x = 16.0-17.5), out of reach of residue 3.
    for n, xyz in enumerate(((16.5, 2.0, 0.0), (17.0, 3.0, 0.0)), start=1):
        lines.append(_atom(serial, f"C{n}", "LIG", "L", 1, xyz, "C", het=True))
        serial += 1
    if bridging_metal:
        # 4.1 A from the substrate, so it joins the site; 3.6 A from residue 3,
        # which therefore lines the cofactor ONLY.
        lines.append(_atom(serial, "MG", "MG", "T", 1, (20.5, 1.0, 0.0), "MG", het=True))
        serial += 1
    if stray_metal:
        lines.append(_atom(serial, "MG", "MG", "N", 1, (100.0, 0.0, 0.0), "MG", het=True))
        serial += 1
    lines.append(_atom(serial, "O", "HOH", "W", 1, (16.5, 4.0, 0.0), "O", het=True))

    p = Path(path)
    p.write_text("\n".join(lines) + "\nEND\n")
    return p
