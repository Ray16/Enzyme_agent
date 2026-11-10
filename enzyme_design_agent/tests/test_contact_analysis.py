import unittest

from enzyme_design_agent.contact_analysis import (
    Atom, ContactPolicy, TargetGroup, collapse_for_display, distinct_residue_count,
    evaluate_contacts,
)


def atom(res, name, xyz, ri="1", chain="A"):
    return Atom(chain, res, ri, name, name[0], xyz)


class ContactAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.target = [atom("LIG", "O1", (0, 0, 0), chain="D"),
                       atom("LIG", "O2", (0, 2, 0), chain="D")]
        self.group = TargetGroup("carboxylate", "D", "1", ("O1", "O2"), -1, "test chemical graph")

    def evaluate(self, *atoms):
        return evaluate_contacts([*self.target, *atoms], [self.group])

    def test_chemistry_not_all_nitrogens_or_oxygens(self):
        contacts = self.evaluate(atom("ASN", "OD1", (2.8, 0, 0)),
                                 atom("ASN", "ND2", (2.8, 0, 0), ri="2"))
        self.assertFalse(contacts[0]["accepted"])
        self.assertEqual(distinct_residue_count(contacts), 1)

    def test_separate_class_boundaries_and_ambiguous_histidine(self):
        contacts = self.evaluate(atom("LYS", "NZ", (3.9, 0, 0)),
                                 atom("SER", "OG", (3.9, 0, 0), ri="2"),
                                 atom("HIS", "NE2", (2.9, 0, 0), ri="3"))
        self.assertTrue(next(c for c in contacts if c['res'] == 'LYS')['accepted'])
        self.assertFalse(next(c for c in contacts if c['res'] == 'SER')['accepted'])
        self.assertTrue(next(c for c in contacts if c['res'] == 'HIS')['conditional'])
        self.assertEqual(distinct_residue_count(contacts), 1)

    def test_duplicate_lines_removed_but_two_point_grip_preserved(self):
        contacts = self.evaluate(atom("ARG", "NH1", (2.8, 0, 0)),
                                 atom("ARG", "NH2", (2.8, 2, 0)))
        group = collapse_for_display(contacts)[0]
        self.assertEqual(group['display_line_count'], 1)
        self.assertEqual(len(group['independent_pairs']), 2)
        self.assertEqual(group['collapsed_atom_pairs'], len(contacts) - 1)
        self.assertEqual(distinct_residue_count(contacts), 1)

    def test_one_donor_two_oxygens_is_not_two_point_grip(self):
        group = collapse_for_display(self.evaluate(atom("LYS", "NZ", (2.8, 1, 0))))[0]
        self.assertEqual(len(group['independent_pairs']), 1)

    def test_one_residue_can_contact_two_groups_without_double_counting(self):
        groups = [TargetGroup("end1", "D", "1", ("O1",), -1, "graph"),
                  TargetGroup("end2", "D", "1", ("O2",), -1, "graph")]
        contacts = evaluate_contacts([*self.target, atom("LYS", "NZ", (2.8, 1, 0))], groups)
        self.assertEqual(len(collapse_for_display(contacts)), 2)
        self.assertEqual(distinct_residue_count(contacts), 1)

    def test_severe_clash_untyped_and_missing_atoms_fail_closed(self):
        self.assertEqual(self.evaluate(atom("LYS", "NZ", (.8, 0, 0)))[0]['reason'], 'severe_clash')
        self.assertEqual(self.evaluate(atom("UNK", "N1", (2.8, 0, 0)))[0]['reason'], 'untyped_residue')
        with self.assertRaises(KeyError):
            evaluate_contacts([], [self.group])
        with self.assertRaises(ValueError):
            ContactPolicy(scope="guess_everything")

    def test_invalid_coordinates_and_atom_aliases_fail_closed(self):
        with self.assertRaises(ValueError):
            atom('LYS', 'NZ', (float('nan'), 0, 0))
        with self.assertRaises(ValueError):
            evaluate_contacts([*self.target, self.target[0]], [self.group])

    def test_frame_and_atom_order_do_not_change_classification(self):
        atoms = [*self.target, atom('SER', 'OG', (2.8, 0, 0))]
        from dataclasses import replace
        moved = [replace(a, xyz=(-a.xyz[1]+17, a.xyz[0]-31, a.xyz[2]+9)) for a in reversed(atoms)]
        key = lambda c: (c['ri'], c['atom'], c['O'], c['accepted'], c['conditional'])
        self.assertEqual(sorted(map(key, evaluate_contacts(atoms, [self.group]))),
                         sorted(map(key, evaluate_contacts(moved, [self.group]))))

    def test_identically_named_groups_on_different_ligands_remain_distinct(self):
        from dataclasses import replace
        other = [replace(a, chain='E') for a in self.target]
        groups = [self.group, replace(self.group, chain='E')]
        contacts = evaluate_contacts([*self.target, *other, atom('LYS', 'NZ', (2.8, 1, 0))], groups)
        self.assertEqual(len(collapse_for_display(contacts)), 2)
        self.assertEqual(distinct_residue_count(contacts), 1)

    def test_short_contact_cannot_hide_behind_other_oxygen(self):
        contacts = self.evaluate(atom('SER', 'OG', (1.96, 0, 0)))
        self.assertEqual(distinct_residue_count(contacts), 0)
        self.assertTrue(all(c['conditional'] for c in contacts))
        self.assertIn('anomalously_short_contact', collapse_for_display(contacts)[0]['uncertainty_reasons'])

    def test_exact_distance_boundaries_are_inclusive(self):
        g = TargetGroup('one', 'D', '1', ('O1',), -1, 'graph')
        for res, name, distance in [('SER', 'OG', 2.4), ('SER', 'OG', 3.5), ('LYS', 'NZ', 4.0)]:
            contacts = evaluate_contacts([*self.target, atom(res, name, (distance, 0, 0))], [g])
            self.assertTrue(contacts[0]['accepted'])

    def test_looser_sensitivity_cannot_restore_quarantined_group(self):
        from enzyme_design_agent.tools.audit_as9_contacts import count_at_thresholds
        target = [atom('LIG', 'O1', (0, 0, 0), chain='D'),
                  atom('LIG', 'O2', (0, 3, 0), chain='D')]
        contacts = evaluate_contacts([*target, atom('SER', 'OG', (1.96, 0, 0))], [self.group])
        self.assertEqual(count_at_thresholds(contacts, 3.8, 4.5), 0)


if __name__ == '__main__':
    unittest.main()
