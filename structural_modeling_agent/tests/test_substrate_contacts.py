"""Tests for the generic substrate-contact DETECTOR (structural_modeling_agent.analysis.substrate_contacts).

Two layers:
  1. deterministic unit tests on tiny synthetic structures -- exercise the chemistry rules
     (complementary-charge salt bridges only; generic metals; functional-group detection) with
     known geometry, independent of any campaign.
  2. regression tests against real Boltz cofold poses with a known MD answer (GlnA3 holder,
     PubC1 drifter) -- auto-skipped if those NFS poses are absent.

Run: <processivity-md python> -m pytest structural_modeling_agent/tests/test_substrate_contacts.py -q
"""
import os, json, tempfile
import numpy as np
import pytest

mda = pytest.importorskip("MDAnalysis")
from structural_modeling_agent.analysis import substrate_contacts as sc


# ---- synthetic structure helpers -------------------------------------------
def _pdb(atoms):
    """atoms: list of (record, name, resname, chain, resid, x, y, z, element)."""
    lines = []
    for i, (rec, name, rn, ch, ri, x, y, z, el) in enumerate(atoms, 1):
        nm = name if len(name) == 4 else f" {name:<3}"
        lines.append(f"{rec:<6}{i:>5} {nm}{rn:>4} {ch}{ri:>4}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")
    lines.append("END")
    f = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    f.write("\n".join(lines)); f.close()
    return f.name


def _u(atoms):
    return mda.Universe(_pdb(atoms))


# minimal ARG (cationic) and GLU (anionic) side chains at a known geometry
def _arg(cx=0.0):
    return [("ATOM", "N", "ARG", "A", 1, cx, 5, 0, "N"),
            ("ATOM", "CA", "ARG", "A", 1, cx, 6, 0, "C"),
            ("ATOM", "C", "ARG", "A", 1, cx + 1, 6, 0, "C"),
            ("ATOM", "O", "ARG", "A", 1, cx + 1, 7, 0, "O"),
            ("ATOM", "CB", "ARG", "A", 1, cx, 4, 0, "C"),
            ("ATOM", "CG", "ARG", "A", 1, cx, 3, 0, "C"),
            ("ATOM", "CD", "ARG", "A", 1, cx, 2, 0, "C"),
            ("ATOM", "NE", "ARG", "A", 1, cx, 1, 0, "N"),
            ("ATOM", "CZ", "ARG", "A", 1, cx, 0.5, 0, "C"),
            ("ATOM", "NH1", "ARG", "A", 1, cx, 0, 0, "N"),
            ("ATOM", "NH2", "ARG", "A", 1, cx + 0.5, 0.5, 0, "N")]


def _glu():
    return [("ATOM", "N", "GLU", "A", 1, 0, 5, 0, "N"),
            ("ATOM", "CA", "GLU", "A", 1, 0, 6, 0, "C"),
            ("ATOM", "C", "GLU", "A", 1, 1, 6, 0, "C"),
            ("ATOM", "O", "GLU", "A", 1, 1, 7, 0, "O"),
            ("ATOM", "CB", "GLU", "A", 1, 0, 4, 0, "C"),
            ("ATOM", "CG", "GLU", "A", 1, 0, 3, 0, "C"),
            ("ATOM", "CD", "GLU", "A", 1, 0, 1, 0, "C"),
            ("ATOM", "OE1", "GLU", "A", 1, 0, 0, 0, "O"),
            ("ATOM", "OE2", "GLU", "A", 1, 0.8, 0.6, 0, "O")]


# ---- 1. chemistry-rule unit tests ------------------------------------------
def test_cation_salt_bridges_carboxylate():
    u = _u(_arg())
    res = u.select_atoms("resname ARG").residues[0]
    target = np.array([[0.0, -2.9, 0.0]])          # ~2.9 A below NH1
    cat, d = sc._classify_residue(res.atoms, target, "ARG", target_charge=-1)
    assert cat == "salt_bridge" and d < sc.SALT_CUT


def test_cation_does_not_salt_bridge_an_amine():
    u = _u(_arg())
    res = u.select_atoms("resname ARG").residues[0]
    target = np.array([[0.0, -2.9, 0.0]])
    cat, _ = sc._classify_residue(res.atoms, target, "ARG", target_charge=+1)  # like charge
    assert cat != "salt_bridge"


def test_acid_does_not_salt_bridge_carboxylate_without_metal():
    u = _u(_glu())
    res = u.select_atoms("resname GLU").residues[0]
    target = np.array([[0.0, -2.9, 0.0]])          # carboxylate near GLU OE1 (like charge)
    cat, _ = sc._classify_residue(res.atoms, target, "GLU", target_charge=-1, mg_pos=None)
    assert cat != "salt_bridge"                     # anion-anion is never a salt bridge


def test_acid_near_carboxylate_is_not_reported():
    # an acidic residue near a carboxylate (even with a metal between) is the metal-binding
    # scaffold, NOT a substrate anchor -> not reported as a contact
    u = _u(_glu())
    res = u.select_atoms("resname GLU").residues[0]
    target = np.array([[0.0, -3.0, 0.0]])
    mg = np.array([[0.0, -1.3, 0.0]])
    cat, _ = sc._classify_residue(res.atoms, target, "GLU", target_charge=-1, mg_pos=mg)
    assert cat is None


def test_acid_salt_bridges_an_amine():
    u = _u(_glu())
    res = u.select_atoms("resname GLU").residues[0]
    target = np.array([[0.0, -2.9, 0.0]])
    cat, _ = sc._classify_residue(res.atoms, target, "GLU", target_charge=+1)  # ammonium
    assert cat == "salt_bridge"


# ---- 2. functional-group detection -----------------------------------------
def test_detects_two_carboxylates_glutarate():
    # glutarate skeleton C1..C5 with terminal carboxylate O's
    sub = [("HETATM", "C1", "LIG", "D", 1, 0, 0, 0, "C"),
           ("HETATM", "O1", "LIG", "D", 1, -1.0, 0.6, 0, "O"),
           ("HETATM", "O2", "LIG", "D", 1, 0.2, -1.2, 0, "O"),
           ("HETATM", "C2", "LIG", "D", 1, 1.3, 0.6, 0, "C"),
           ("HETATM", "C3", "LIG", "D", 1, 2.6, 0.0, 0, "C"),
           ("HETATM", "C4", "LIG", "D", 1, 3.9, 0.6, 0, "C"),
           ("HETATM", "C5", "LIG", "D", 1, 5.2, 0.0, 0, "C"),
           ("HETATM", "O3", "LIG", "D", 1, 6.2, 0.7, 0, "O"),
           ("HETATM", "O4", "LIG", "D", 1, 5.3, -1.2, 0, "O")]
    u = _u(sub)
    fgs = sc.detect_functional_groups(u.select_atoms("resname LIG"))
    kinds = [g["kind"] for g in fgs]
    assert kinds.count("carboxylate") == 2
    assert "amine" not in kinds


def test_detects_two_amines_cadaverine():
    sub = [("HETATM", "N1", "LIG", "D", 1, 0, 0, 0, "N"),
           ("HETATM", "C1", "LIG", "D", 1, 1.4, 0, 0, "C"),
           ("HETATM", "C2", "LIG", "D", 1, 2.1, 1.3, 0, "C"),
           ("HETATM", "C3", "LIG", "D", 1, 3.6, 1.3, 0, "C"),
           ("HETATM", "C4", "LIG", "D", 1, 4.3, 2.6, 0, "C"),
           ("HETATM", "C5", "LIG", "D", 1, 5.8, 2.6, 0, "C"),
           ("HETATM", "N2", "LIG", "D", 1, 6.5, 3.9, 0, "N")]
    u = _u(sub)
    fgs = sc.detect_functional_groups(u.select_atoms("resname LIG"))
    kinds = [g["kind"] for g in fgs]
    assert kinds.count("amine") == 2
    assert all(g["charge"] == 1 for g in fgs if g["kind"] == "amine")


# ---- 3. generic metal detection (not Mg-specific) --------------------------
def test_split_ligands_detects_zinc_as_metal():
    atoms = (_arg()
             + [("HETATM", "C1", "LIG", "D", 2, 10, 0, 0, "C"),
                ("HETATM", "O1", "LIG", "D", 2, 11, 0.6, 0, "O"),
                ("HETATM", "O2", "LIG", "D", 2, 10.2, -1.2, 0, "O"),
                ("HETATM", "ZN", "ZN", "E", 3, 9, 0, 0, "ZN")])
    u = _u(atoms)
    cof, metals, subg = sc.split_ligands(u)
    assert len(metals) == 1 and sc._element(metals[0]) == "ZN"
    assert len(subg) >= 3            # the carboxylate ligand


# ---- 4. regression against real poses (skip if absent) ---------------------
RUN = ("enzyme_discovery_agent/runs/20261106_000850_Find_enzymes_that_ligate_glutarate_and"
       "_cadaverine_Production_objective_deliver_a/family_cofold/family_results.csv")


def _best_pose(acc):
    import csv
    if not os.path.exists(RUN):
        return None
    row = {r["accession"]: r for r in csv.DictReader(open(RUN))}.get(acc)
    if not row or not os.path.exists(row["result"]):
        return None
    cj = json.load(open(row["result"]))
    poses = cj["analysis"]["poses"]
    near = [p for p in poses if p["distance_A"] < 4 and p["angle_deg"] > 150]
    p = min(near, key=lambda q: q["distance_A"]) if near else max(poses, key=lambda q: q["angle_deg"])
    return p["path"] if os.path.exists(p["path"]) else None


@pytest.mark.skipif(_best_pose("Q9KZC7") is None, reason="GlnA3 cofold pose not on disk")
def test_glna3_reactive_metal_coordinated_and_distal_clamp():
    u = mda.Universe(_best_pose("Q9KZC7"))
    res = sc.analyze_structure(u)
    assert res["metals_present"] == 2
    reactive = next(g for g in res["groups"] if g["role"] == "reactive")
    secondary = next(g for g in res["groups"] if g["role"] == "secondary")
    assert reactive["metal_coordinated"] is True
    distal_sb = {c["res"] for c in secondary["contacts"] if c["category"] == "salt_bridge"}
    assert {"HIS263", "ARG316"} <= distal_sb      # the MD-confirmed clamp, seen in the static pose
    # no like-charge false positive: GLU must never be a salt bridge to a carboxylate
    for g in res["groups"]:
        for c in g["contacts"]:
            if c["category"] == "salt_bridge":
                assert sc._resname_of(c) in ("ARG", "LYS", "HIS")
