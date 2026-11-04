"""Tests for the handoff -> ReactiveSite adapter (the setup/scoring seam).

Structures are written inline as minimal PDB text so these run with no campaign
data: the point is the wiring, not any particular enzyme.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from reaction import load_atom_selectors  # noqa: E402
from reactive_site import (  # noqa: E402
    RoleResolutionError, attach_smarts, resolver_from_handoff,
)
from structural_modeling_agent.proxies import check_resolver, parse_pose  # noqa: E402

# Acetyl phosphate, CC(=O)OP(=O)(O)O -- 8 heavy atoms, laid out with plausible
# bond lengths, plus ethylamine as the attacking amine 3.0 A from the acyl carbon.
LIGANDS = """\
HETATM    1  C1  LIG B   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG B   1       1.510   0.000   0.000  1.00  0.00           C
HETATM    3  O1  LIG B   1       2.130   1.060   0.000  1.00  0.00           O
HETATM    4  O2  LIG B   1       2.180  -1.190   0.000  1.00  0.00           O
HETATM    5  P1  LIG B   1       3.790  -1.400   0.000  1.00  0.00           P
HETATM    6  O3  LIG B   1       4.420  -0.080   0.000  1.00  0.00           O
HETATM    7  O4  LIG B   1       4.200  -2.200   1.200  1.00  0.00           O
HETATM    8  O5  LIG B   1       4.100  -2.200  -1.200  1.00  0.00           O
HETATM    9  N1  LIG C   1       1.510   0.000   3.000  1.00  0.00           N
HETATM   10  C3  LIG C   1       1.510   1.300   3.700  1.00  0.00           C
HETATM   11  C4  LIG C   1       1.510   1.300   5.200  1.00  0.00           C
"""

PROTEIN = "".join(
    f"ATOM  {i + 100:5d}  CA  ALA A{i + 1:4d}    "
    f"{10.0 + i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    for i in range(8)
)


def write_pose(tmpdir, name="m_model_0.pdb", body=None):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as fh:
        fh.write(PROTEIN + (LIGANDS if body is None else body) + "END\n")
    return path


def record(acyl="CC(=O)OP(=O)(O)O", amine="CCN", ideal=107.0, **over):
    geometry = {
        "kind": "near_attack", "step": 2, "ideal_angle": ideal,
        "nucleophile": {"kind": "ligand", "role": "amine_substrate",
                        "component_index": 2, "atom_selector": "amine_nitrogen"},
        "electrophile": {"kind": "ligand", "role": "acyl_adenylate_intermediate",
                         "component_index": 1,
                         "atom_selector": "phosphate_activated_acyl_carbon"},
        "leaving_group": {"kind": "ligand", "role": "acyl_adenylate_intermediate",
                          "component_index": 1,
                          "atom_selector": "phosphate_activated_acyl_leaving_oxygen"},
        "charge_centre": {"kind": "ligand", "role": "acyl_adenylate_intermediate",
                          "component_index": 1,
                          "atom_selector": "phosphate_activated_acyl_carbonyl_oxygen"},
    }
    geometry.update(over)
    return {"geometry": geometry, "target_substrates": [
        {"component_index": 1, "role": "acyl_adenylate_intermediate", "modeling_smiles": acyl},
        {"component_index": 2, "role": "amine_substrate", "modeling_smiles": amine}]}


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.selectors = load_atom_selectors()
        self.tmp = tempfile.TemporaryDirectory()
        self.pose = parse_pose(write_pose(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_pose_records_its_source(self):
        """The adapter re-reads the file for bond perception, so this must survive."""
        self.assertTrue(self.pose.source_path.endswith("m_model_0.pdb"))

    def test_attach_smarts_fills_every_ligand_role(self):
        rec = attach_smarts(record(), self.selectors)
        for role in ("nucleophile", "electrophile", "leaving_group", "charge_centre"):
            self.assertIn("smarts", rec["geometry"][role], role)

    def test_resolves_all_four_roles_onto_real_coordinates(self):
        site = resolver_from_handoff(attach_smarts(record(), self.selectors))(self.pose)
        self.assertIsNotNone(site)
        for role in ("nucleophile", "electrophile", "leaving", "charge_centre"):
            self.assertIsNotNone(getattr(site, role), role)
        self.assertAlmostEqual(site.attack_distance, 3.0, places=2)
        self.assertEqual(site.ideal_angle, 107.0)

    def test_ideal_angle_comes_from_the_mechanism_record(self):
        """Scoring must not re-decide the geometry the mechanism already declared."""
        site = resolver_from_handoff(attach_smarts(record(ideal=180.0), self.selectors))(self.pose)
        self.assertEqual(site.ideal_angle, 180.0)

    def test_chain_matching_is_by_composition_not_order(self):
        """Swapping the chain labels must not change the answer."""
        swapped = LIGANDS.replace(" B   1", " @@@@@").replace(" C   1", " B   1").replace(" @@@@@", " C   1")
        pose = parse_pose(write_pose(self.tmp.name, "swapped_model_0.pdb", swapped))
        site = resolver_from_handoff(attach_smarts(record(), self.selectors))(pose)
        self.assertIsNotNone(site)
        self.assertAlmostEqual(site.attack_distance, 3.0, places=2)

    def test_refuses_when_chemistry_does_not_match_the_structure(self):
        """A handoff describing a different molecule must resolve nothing, not guess."""
        rec = attach_smarts(record(acyl="CC(=O)OP(=O)(O)OCCCCCCCCCC"), self.selectors)
        self.assertIsNone(resolver_from_handoff(rec)(self.pose))

    def test_missing_geometry_block_is_an_error(self):
        with self.assertRaises(RoleResolutionError):
            resolver_from_handoff({"target_substrates": []})

    def test_check_resolver_accepts_the_adapter(self):
        resolver = resolver_from_handoff(attach_smarts(record(), self.selectors))
        report = check_resolver(resolver, [self.pose] * 6)
        self.assertTrue(report.ok, report)
        self.assertEqual(report.coverage, 1.0)


class SelectorVocabularyTest(unittest.TestCase):
    """The selectors the reaction coordinates reference must exist and be unambiguous."""

    def setUp(self):
        self.selectors = load_atom_selectors()

    def test_every_referenced_selector_exists(self):
        import json
        from reaction import DEFAULT_COORDINATES
        known = (set(self.selectors["ligand_smarts"])
                 | set(self.selectors["cofactor_ccd_atoms"])
                 | set(self.selectors.get("protein_selectors") or ())
                 | set(self.selectors.get("solvent_selectors") or ()))
        with open(DEFAULT_COORDINATES) as fh:
            doc = json.load(fh)
        for key, mech in doc["mechanisms"].items():
            for step in mech["steps"]:
                for role, spec in (step.get("attack") or {}).items():
                    if isinstance(spec, dict) and "atom_selector" in spec:
                        self.assertIn(spec["atom_selector"], known,
                                      f"{key} step {step['index']} {role}")

    def test_every_step_declares_a_full_attack_geometry(self):
        import json
        from reaction import DEFAULT_COORDINATES
        with open(DEFAULT_COORDINATES) as fh:
            doc = json.load(fh)
        for key, mech in doc["mechanisms"].items():
            for step in mech["steps"]:
                attack = step.get("attack") or {}
                if not attack:
                    continue
                for role in ("nucleophile", "electrophile", "leaving_group",
                             "charge_centre", "ideal_angle"):
                    self.assertIn(role, attack, f"{key} step {step['index']} missing {role}")

    def test_activated_acyl_selector_is_unambiguous_on_a_diacid_adenylate(self):
        """The regression that motivated the rename: the deprecated selector matched
        both the reactive acyl carbon and the substrate's distal free carboxylate."""
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        adenylate = ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OC(=O)CCCCC(=O)[O-])"
                     "[C@@H](O)[C@H]1O")
        mol = Chem.MolFromSmiles(adenylate)
        smarts = self.selectors["ligand_smarts"]
        good = mol.GetSubstructMatches(
            Chem.MolFromSmarts(smarts["phosphate_activated_acyl_carbon"]))
        bad = mol.GetSubstructMatches(
            Chem.MolFromSmarts(smarts["acyl_carbonyl_carbon"]))
        self.assertEqual(len(good), 1)
        self.assertGreater(len(bad), 1)


if __name__ == "__main__":
    unittest.main()
