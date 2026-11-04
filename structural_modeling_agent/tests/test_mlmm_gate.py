"""The gate must CATCH each failure mode that actually cost a campaign.

Every test here is a regression test against a real recorded failure, not a
smoke test. A gate that passes a bad structure is worse than no gate, so each
case asserts the specific check fires and that the report blocks.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from structural_modeling_agent.mlmm import (  # noqa: E402
    AtomRef, BondChange, LigandSpec, MLMMSpec, MetalSite, ReactionStep,
    check_freeze_consistency, check_structure, read_pdb,
)
from structural_modeling_agent.mlmm.structure import StructureError  # noqa: E402

# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

def _pdb_line(serial, name, resname, chain, resseq, xyz, element):
    x, y, z = xyz
    rec = "HETATM" if resname not in ("ASP", "GLU", "HIS", "ARG") else "ATOM  "
    return (
        f"{rec}{serial:>5d} {name:<4s}{'':1s}{resname:>3s} {chain}{resseq:>4d}"
        f"{'':4s}{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}"
        f"{'':10s}{element:>2s}\n"
    )


def write_pdb(tmp_path, atoms, name="system.pdb"):
    """atoms: list of (name, resname, chain, resseq, (x,y,z), element)."""
    path = tmp_path / name
    with open(path, "w") as fh:
        for i, (nm, rn, ch, rs, xyz, el) in enumerate(atoms, 1):
            fh.write(_pdb_line(i, nm, rn, ch, rs, xyz, el))
        fh.write("END\n")
    return str(path)


def good_atoms():
    """An in-line phosphoryl-transfer-like active site that should clear the gate.

    Nucleophile O at 3.0 A from P, leaving O on the far side (angle ~180), one Mg
    with a complete 6-donor shell, hydrogens present.
    """
    atoms = [
        ("O1", "LIG", "E", 1, (0.0, 0.0, 0.0), "O"),       # nucleophile
        ("C1", "LIG", "E", 1, (-1.3, 0.3, 0.0), "C"),
        ("O2", "LIG", "E", 1, (-2.0, -0.7, 0.0), "O"),
        ("H1", "LIG", "E", 1, (-1.6, 1.3, 0.0), "H"),
        ("PA", "ATP", "B", 2, (3.0, 0.0, 0.0), "P"),       # electrophile
        ("O1A", "ATP", "B", 2, (3.4, 1.4, 0.3), "O"),
        ("O2A", "ATP", "B", 2, (3.4, -0.8, 1.2), "O"),
        ("O3A", "ATP", "B", 2, (4.7, -0.3, -0.6), "O"),    # leaving group, ~180
        ("C5'", "ATP", "B", 2, (2.6, -0.4, -1.8), "C"),
        ("H5'", "ATP", "B", 2, (2.2, -1.4, -1.9), "H"),
    ]
    # Mg with six oxygen donors, all from distinct residues, ~2.1 A.
    mg = (12.0, 0.0, 0.0)
    atoms.append(("MG", "MG", "C", 3, mg, "MG"))
    offs = [(2.1, 0, 0), (-2.1, 0, 0), (0, 2.1, 0), (0, -2.1, 0), (0, 0, 2.1), (0, 0, -2.1)]
    for k, (dx, dy, dz) in enumerate(offs):
        pos = (mg[0] + dx, mg[1] + dy, mg[2] + dz)
        atoms.append(("O", "HOH", "W", 100 + k, pos, "O"))
        atoms.append(("H1", "HOH", "W", 100 + k,
                      (pos[0] + 0.5, pos[1] + 0.8, pos[2]), "H"))
        atoms.append(("H2", "HOH", "W", 100 + k,
                      (pos[0] - 0.5, pos[1] + 0.8, pos[2]), "H"))
    return atoms


def base_spec(**over):
    step = ReactionStep(
        step_id="adenylation",
        nucleophile=AtomRef("O1", resname="LIG"),
        electrophile=AtomRef("PA", resname="ATP"),
        leaving_group=AtomRef("O3A", resname="ATP"),
        bond_changes=(
            BondChange("form", AtomRef("O1", resname="LIG"),
                       AtomRef("PA", resname="ATP"), 1.8),
            BondChange("break", AtomRef("PA", resname="ATP"),
                       AtomRef("O3A", resname="ATP"), 2.6),
        ),
        ideal_attack_angle=180.0,
        proton_coupled=False,
    )
    kw = dict(
        system_id="test",
        step=step,
        net_charge=-1,
        ligands=(LigandSpec("ATP", -4), LigandSpec("LIG", 0)),
        metals=(MetalSite(AtomRef("MG", resname="MG"), expected_coordination=6),),
    )
    kw.update(over)
    return MLMMSpec(**kw)


def _named(report, prefix):
    return [c for c in report.checks if c.name.startswith(prefix)]


# --------------------------------------------------------------------------- #
# the happy path must actually pass, or every negative test below is vacuous
# --------------------------------------------------------------------------- #

def test_good_structure_clears_the_gate(tmp_path):
    path = write_pdb(tmp_path, good_atoms())
    report = check_structure(read_pdb(path), base_spec())
    assert report.ok, report.render()
    assert not report.failures
    assert not report.abstentions, report.render()


# --------------------------------------------------------------------------- #
# failure mode 1: not a near-attack conformation (the decarboxylase case)
# --------------------------------------------------------------------------- #

def test_distant_nucleophile_is_rejected(tmp_path):
    """Nu 5.4 A from the electrophile -- the recorded cofold-pose failure."""
    atoms = good_atoms()
    atoms[0] = ("O1", "LIG", "E", 1, (-2.4, 0.0, 0.0), "O")  # 5.4 A from P
    path = write_pdb(tmp_path, atoms)
    report = check_structure(read_pdb(path), base_spec())
    assert not report.ok
    d = _named(report, "attack_distance")[0]
    assert d.status == "FAIL"
    assert d.value == pytest.approx(5.4, abs=0.05)


def test_misaligned_attack_angle_is_rejected(tmp_path):
    """Right distance, wrong trajectory -- passes a distance-only screen."""
    atoms = good_atoms()
    # Move the leaving group perpendicular, collapsing the angle to ~90 deg.
    atoms[7] = ("O3A", "ATP", "B", 2, (3.0, 1.6, 0.0), "O")
    path = write_pdb(tmp_path, atoms)
    report = check_structure(read_pdb(path), base_spec())
    assert not report.ok
    ang = _named(report, "attack_angle")[0]
    assert ang.status == "FAIL"
    assert ang.value < 120.0
    # the distance check must still pass, proving the angle is what caught it
    assert _named(report, "attack_distance")[0].status == "PASS"


# --------------------------------------------------------------------------- #
# failure mode 2: metal shell (the 11 kcal/mol artifact)
# --------------------------------------------------------------------------- #

def test_incomplete_metal_shell_is_rejected(tmp_path):
    atoms = [a for a in good_atoms() if not (a[1] == "HOH" and a[3] == 105)]
    path = write_pdb(tmp_path, atoms)
    report = check_structure(read_pdb(path), base_spec())
    assert not report.ok
    m = [c for c in report.checks if c.name.startswith("metal:") and ":" not in c.name[6:]]
    assert m and m[0].status == "FAIL"
    assert m[0].value == 5


def test_undeclared_metal_coordination_abstains_not_passes(tmp_path):
    """Fail closed: an unchecked precondition must not read as satisfied."""
    path = write_pdb(tmp_path, good_atoms())
    spec = base_spec(metals=(MetalSite(AtomRef("MG", resname="MG"),
                                       expected_coordination=None),))
    report = check_structure(read_pdb(path), spec)
    assert not report.ok
    assert report.abstentions
    # and non-strict triage lets it through
    assert check_structure(read_pdb(path), spec, strict=False).ok


def test_metal_sequestered_nucleophile_is_flagged(tmp_path):
    """Nucleophile coordinating the metal: the recorded 40+ kcal/mol inflation."""
    atoms = good_atoms()
    mg = next(a for a in atoms if a[1] == "MG")
    atoms[0] = ("O1", "LIG", "E", 1, (mg[4][0] - 2.1, mg[4][1], mg[4][2]), "O")
    # keep the attack geometry satisfiable by moving P next to the new Nu
    atoms[4] = ("PA", "ATP", "B", 2, (mg[4][0] - 5.0, mg[4][1], mg[4][2]), "P")
    atoms[7] = ("O3A", "ATP", "B", 2, (mg[4][0] - 6.6, mg[4][1], mg[4][2]), "O")
    path = write_pdb(tmp_path, atoms)
    spec = base_spec(metals=(MetalSite(AtomRef("MG", resname="MG"),
                                       expected_coordination=6,
                                       nucleophile_must_be_free=True),))
    report = check_structure(read_pdb(path), spec)
    assert not report.ok
    free = _named(report, "metal:MG/MG:nucleophile_free")
    assert free and free[0].status == "FAIL"


# --------------------------------------------------------------------------- #
# failure mode 3: charge / electron bookkeeping
# --------------------------------------------------------------------------- #

def test_wrong_net_charge_breaks_parity(tmp_path):
    """The acyl-adenylate -1-vs--2 class of error."""
    path = write_pdb(tmp_path, good_atoms())
    report = check_structure(read_pdb(path), base_spec(net_charge=-2))
    assert not report.ok
    par = _named(report, "electron_parity")[0]
    assert par.status == "FAIL"


def test_parity_accepts_the_correct_charge(tmp_path):
    path = write_pdb(tmp_path, good_atoms())
    assert _named(check_structure(read_pdb(path), base_spec()),
                  "electron_parity")[0].status == "PASS"


def test_odd_electron_count_requires_a_doublet(tmp_path):
    path = write_pdb(tmp_path, good_atoms())
    ok = check_structure(read_pdb(path), base_spec(net_charge=-2, multiplicity=2))
    assert _named(ok, "electron_parity")[0].status == "PASS"


def test_virtual_sites_excluded_from_electron_count(tmp_path):
    """OPC extra points are MM charge sites, not atoms (the net +380 bug)."""
    atoms = good_atoms()
    atoms.append(("EPW", "HOH", "W", 100, (2.1, 0.3, 0.3), "O"))
    path = write_pdb(tmp_path, atoms)
    report = check_structure(read_pdb(path), base_spec())
    assert _named(report, "electron_parity")[0].status == "PASS", report.render()
    assert _named(report, "virtual_sites_excluded")[0].value == 1


# --------------------------------------------------------------------------- #
# failure mode 4: structure not actually prepared
# --------------------------------------------------------------------------- #

def test_heavy_atom_only_structure_is_rejected(tmp_path):
    """Raw cofold output: no hydrogens at all."""
    atoms = [a for a in good_atoms() if a[5] != "H"]
    path = write_pdb(tmp_path, atoms)
    report = check_structure(read_pdb(path), base_spec())
    assert not report.ok
    h = _named(report, "hydrogens_present")[0]
    assert h.status == "FAIL" and h.value == 0


def test_blank_element_column_fails_closed(tmp_path):
    path = tmp_path / "noelem.pdb"
    with open(path, "w") as fh:
        fh.write("HETATM    1 PA   ATP B   2       3.000   0.000   0.000  1.00  0.00\n")
        fh.write("END\n")
    with pytest.raises(StructureError, match="blank element column"):
        read_pdb(str(path))


def test_missing_declared_ligand_is_caught(tmp_path):
    path = write_pdb(tmp_path, good_atoms())
    spec = base_spec(ligands=(LigandSpec("ATP", -4), LigandSpec("NAD", 0)))
    report = check_structure(read_pdb(path), spec)
    assert not report.ok
    assert _named(report, "ligand:NAD")[0].status == "FAIL"


def test_wrong_ligand_atom_count_is_caught(tmp_path):
    """The species built is not the species declared."""
    path = write_pdb(tmp_path, good_atoms())
    spec = base_spec(ligands=(LigandSpec("ATP", -4, expected_heavy_atoms=31),
                              LigandSpec("LIG", 0)))
    report = check_structure(read_pdb(path), spec)
    assert not report.ok
    assert _named(report, "ligand:ATP")[0].status == "FAIL"


def test_steric_clash_is_caught(tmp_path):
    atoms = good_atoms()
    atoms.append(("OD1", "ASP", "A", 50, (3.4, 1.4, 0.9), "O"))  # ~0.6 A from O1A
    path = write_pdb(tmp_path, atoms)
    report = check_structure(read_pdb(path), base_spec())
    assert not report.ok
    assert _named(report, "steric_clashes")[0].status == "FAIL"


# --------------------------------------------------------------------------- #
# failure mode 5: the reaction coordinate itself
# --------------------------------------------------------------------------- #

def test_backwards_bond_change_target_is_caught(tmp_path):
    """A 'form' coordinate whose target is LONGER than the current distance."""
    path = write_pdb(tmp_path, good_atoms())
    step = base_spec().step
    bad = ReactionStep(
        step_id=step.step_id, nucleophile=step.nucleophile,
        electrophile=step.electrophile, leaving_group=step.leaving_group,
        bond_changes=(BondChange("form", AtomRef("O1", resname="LIG"),
                                 AtomRef("PA", resname="ATP"), 4.5),),
        ideal_attack_angle=180.0, proton_coupled=False,
    )
    report = check_structure(read_pdb(path), base_spec(step=bad))
    assert not report.ok
    assert _named(report, "bond1:form")[0].status == "FAIL"


def test_undeclared_bond_changes_abstain(tmp_path):
    path = write_pdb(tmp_path, good_atoms())
    step = base_spec().step
    bare = ReactionStep(
        step_id=step.step_id, nucleophile=step.nucleophile,
        electrophile=step.electrophile, leaving_group=step.leaving_group,
        proton_coupled=False,
    )
    report = check_structure(read_pdb(path), base_spec(step=bare))
    assert not report.ok
    assert _named(report, "bond_changes")[0].status == "ABSTAIN"


def test_multi_coordinate_step_warns(tmp_path):
    """Three simultaneous coordinates: the recorded hard case. Warn, don't block."""
    atoms = good_atoms()
    atoms.append(("HO", "LIG", "E", 1, (0.5, 0.9, 0.0), "H"))
    path = write_pdb(tmp_path, atoms)
    step = base_spec().step
    conc = ReactionStep(
        step_id="concerted", nucleophile=step.nucleophile,
        electrophile=step.electrophile, leaving_group=step.leaving_group,
        bond_changes=step.bond_changes, ideal_attack_angle=180.0,
        proton_coupled=True, proton=AtomRef("HO", resname="LIG"),
        proton_acceptor=AtomRef("O1A", resname="ATP"),
    )
    spec = base_spec(step=conc, net_charge=-2)
    report = check_structure(read_pdb(path), spec)
    assert conc.n_coordinates == 3
    assert _named(report, "coordinate_count")[0].status == "WARN"


def test_ambiguous_atom_reference_raises(tmp_path):
    """Two ligand copies, unqualified reference: must raise, not pick one."""
    atoms = good_atoms()
    atoms.append(("O1", "LIG", "F", 9, (20.0, 0.0, 0.0), "O"))
    path = write_pdb(tmp_path, atoms)
    structure = read_pdb(path)
    with pytest.raises(StructureError, match="ambiguous"):
        structure.select("O1", resname="LIG")
    # qualifying by chain resolves it
    assert structure.select("O1", resname="LIG", chain="E").chain == "E"


# --------------------------------------------------------------------------- #
# failure mode 6: cross-variant freeze scheme (the largest recorded artifact)
# --------------------------------------------------------------------------- #

def _variant_with_protein_donor(tmp_path, label, freeze_the_donor):
    """One Mg coordinated by a protein Asp carboxylate + 5 waters."""
    atoms = good_atoms()
    # drop one water, replace with an Asp OD1 donor
    atoms = [a for a in atoms if not (a[1] == "HOH" and a[3] == 105)]
    mg = next(a for a in atoms if a[1] == "MG")
    atoms.append(("OD1", "ASP", "A", 480, (mg[4][0], mg[4][1], mg[4][2] - 2.1), "O"))
    path = write_pdb(tmp_path, atoms, name=f"{label}.pdb")
    structure = read_pdb(path)
    donor = structure.select("OD1", resname="ASP")
    frozen = {donor.serial} if freeze_the_donor else set()
    return structure, base_spec(), frozen


def test_freeze_consistency_catches_a_pinned_metal_ligand(tmp_path):
    a = _variant_with_protein_donor(tmp_path, "free", freeze_the_donor=False)
    b = _variant_with_protein_donor(tmp_path, "pinned", freeze_the_donor=True)
    report = check_freeze_consistency({"MA": a, "XA": b})
    assert not report.ok
    bad = [c for c in report.failures if c.name.startswith("XA:")]
    assert bad, report.render()
    assert "cannot relax" in bad[0].detail
    assert not [c for c in report.failures if c.name.startswith("MA:")]


def test_freeze_consistency_passes_when_both_are_free(tmp_path):
    a = _variant_with_protein_donor(tmp_path, "a", freeze_the_donor=False)
    b = _variant_with_protein_donor(tmp_path, "b", freeze_the_donor=False)
    report = check_freeze_consistency({"MA": a, "XA": b})
    assert report.ok, report.render()


def test_freeze_consistency_abstains_on_a_single_variant(tmp_path):
    a = _variant_with_protein_donor(tmp_path, "solo", freeze_the_donor=False)
    report = check_freeze_consistency({"only": a})
    assert not report.ok
    assert report.abstentions


# --------------------------------------------------------------------------- #
# reporting contract
# --------------------------------------------------------------------------- #

def test_report_is_serialisable_and_renders(tmp_path):
    path = write_pdb(tmp_path, good_atoms())
    report = check_structure(read_pdb(path), base_spec())
    d = report.to_dict()
    assert d["ok"] is True and d["checks"]
    assert all({"name", "status", "detail"} <= set(c) for c in d["checks"])
    assert "CLEAR" in report.render()


def test_spec_describe_records_the_claim():
    text = base_spec().describe()
    assert "q = -1" in text and "multiplicity = 1" in text
    assert "LIG/O1" in text and "ATP/PA" in text
