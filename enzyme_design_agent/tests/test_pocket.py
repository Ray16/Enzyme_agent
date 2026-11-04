"""Pocket definition: the complex is the centre, and strays are excluded."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from enzyme_design_agent.saturation.pocket import admit_complex, define_pocket
from enzyme_design_agent.structure import load_structure
from enzyme_design_agent.tests._synthetic import SEQUENCE, write_structure


class StructureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdb = write_structure(Path(self.tmp.name) / "holo.pdb")
        self.st = load_structure(self.pdb)

    def tearDown(self):
        self.tmp.cleanup()

    def test_water_is_not_a_component(self):
        self.assertNotIn("W", [c.id for c in self.st.components])

    def test_one_het_residue_per_chain_gets_the_chain_id(self):
        self.assertEqual({"L", "T", "N"}, {c.id for c in self.st.components})

    def test_sequence_is_one_based_and_gap_free(self):
        self.assertEqual(SEQUENCE, self.st.sequence())

    def test_hydrogens_are_dropped(self):
        extra = self.pdb.read_text().replace(
            "END\n",
            "ATOM    999  HA   ALA A   1       4.000   0.000   0.000  1.00  0.00           H\nEND\n")
        p2 = Path(self.tmp.name) / "h.pdb"
        p2.write_text(extra)
        before = load_structure(self.pdb).residues[0].xyz.shape[0]
        self.assertEqual(before, load_structure(p2).residues[0].xyz.shape[0])

    def test_gapped_numbering_is_refused(self):
        text = "\n".join(l for l in self.pdb.read_text().splitlines()
                         if not (l.startswith("ATOM") and l[22:26].strip() == "3"))
        p2 = Path(self.tmp.name) / "gap.pdb"
        p2.write_text(text + "\n")
        with self.assertRaisesRegex(ValueError, "gap in residue numbering"):
            load_structure(p2).sequence()


class PocketTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdb = write_structure(Path(self.tmp.name) / "holo.pdb")
        self.st = load_structure(self.pdb)

    def tearDown(self):
        self.tmp.cleanup()

    def test_anchor_defaults_to_the_largest_component(self):
        pk = define_pocket(self.st)
        self.assertEqual("L", pk.anchor)

    def test_bridging_metal_is_admitted_stray_metal_is_not(self):
        pk = define_pocket(self.st)
        self.assertEqual(("L", "T"), pk.admitted)
        self.assertEqual(["N"], [e.component for e in pk.excluded])

    def test_excluded_component_reports_its_distance(self):
        pk = define_pocket(self.st)
        self.assertGreater(pk.excluded[0].d_to_site, 40.0)
        self.assertIn("link_cutoff", pk.excluded[0].reason)

    def test_cofactor_only_position_is_found(self):
        """The whole point: residue 3 lines the metal, not the substrate.

        A substrate-centred scan cannot return it, which is why defining the
        pocket around one ligand decides the result in advance.
        """
        whole = define_pocket(self.st)
        self.assertIn(3, whole.resnums())
        substrate_only = define_pocket(self.st, components=["L"])
        self.assertNotIn(3, substrate_only.resnums())

    def test_membership_names_the_components_touched(self):
        pk = define_pocket(self.st)
        m = {p.resnum: p.membership for p in pk.positions}
        self.assertEqual({2, 3}, set(m))
        self.assertEqual("L+T", m[2])       # residue 2 lines substrate and cofactor
        self.assertEqual("T", m[3])         # residue 3 lines the cofactor alone

    def test_cutoff_is_respected(self):
        tight = define_pocket(self.st, cutoff=2.5)
        wide = define_pocket(self.st, cutoff=8.0)
        self.assertLess(len(tight.positions), len(wide.positions))
        self.assertTrue(all(p.d_complex <= 2.5 for p in tight.positions))

    def test_require_connected_false_admits_everything(self):
        pk = define_pocket(self.st, require_connected=False)
        self.assertEqual(set(), {e.component for e in pk.excluded})
        self.assertIn("N", pk.admitted)

    def test_explicit_anchor_is_honoured(self):
        pk = define_pocket(self.st, anchor="T")
        self.assertEqual("T", pk.anchor)

    def test_unknown_component_raises_with_the_available_list(self):
        with self.assertRaisesRegex(ValueError, "no such component"):
            define_pocket(self.st, components=["ZZ"])

    def test_unknown_anchor_raises(self):
        with self.assertRaisesRegex(ValueError, "anchor"):
            define_pocket(self.st, anchor="ZZ")

    def test_structure_with_no_components_raises(self):
        apo = write_structure(Path(self.tmp.name) / "apo.pdb",
                              stray_metal=False, bridging_metal=False)
        text = "\n".join(l for l in apo.read_text().splitlines()
                         if not l.startswith("HETATM"))
        apo.write_text(text + "\n")
        with self.assertRaisesRegex(ValueError, "no non-polymer components"):
            define_pocket(load_structure(apo))

    def test_admit_complex_grows_transitively(self):
        """A metal linked to the substrate only through another metal still joins."""
        st = load_structure(self.pdb)
        comps = [c for c in st.components]
        admitted, excluded, anchor = admit_complex(comps, link_cutoff=100.0)
        self.assertEqual(3, len(admitted))
        self.assertEqual([], excluded)
        self.assertEqual("L", anchor)

    def test_rows_carry_one_column_per_admitted_component(self):
        pk = define_pocket(self.st)
        row = pk.to_rows()[0]
        self.assertIn("d_L", row)
        self.assertIn("d_T", row)
        self.assertNotIn("d_N", row)


if __name__ == "__main__":
    unittest.main()
