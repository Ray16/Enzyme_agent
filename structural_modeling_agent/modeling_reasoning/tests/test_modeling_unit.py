"""Tests for the oligomeric-unit reasoner's robustness fixes:

* gemmi is a hard dependency imported at module top -- a missing gemmi fails LOUDLY
  at import, never degrading silently into the annotation-only default;
* among bound ligands, the MOST-ENGAGED one is chosen as the evidence, not simply
  the first qualifying het in chain order.

These mock the network/structure seams so they need no campaign data or GPU.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import modeling_unit as mu  # noqa: E402


class TestLoudGemmi(unittest.TestCase):
    def test_gemmi_imported_at_module_top(self):
        # The module imports gemmi at the top, so a missing gemmi would raise at
        # import time (loud) rather than being swallowed per-candidate at runtime.
        self.assertTrue(hasattr(mu, "gemmi"))
        self.assertTrue(hasattr(mu.gemmi, "read_structure"))


class TestMostEngagedLigand(unittest.TestCase):
    def _facts(self):
        return {"subunit": "", "active_sites": [], "pdb_ids": ["1ABC"],
                "sequence": "MKAAA", "interpro": []}

    def test_picks_most_engaged_over_first(self):
        weak = {"comp_id": "WEK", "chain": "A", "contact_residues": 3,
                "other_chain_residues": 0, "other_chain_fraction": 0.0,
                "by_chain": {"A": 3}}
        strong = {"comp_id": "SUB", "chain": "A", "contact_residues": 12,
                  "other_chain_residues": 6, "other_chain_fraction": 0.5,
                  "by_chain": {"A": 6, "B": 6}}
        with mock.patch.object(mu, "uniprot_facts", return_value=self._facts()), \
                mock.patch.object(mu, "largest_assembly", return_value=("1", 4)), \
                mock.patch.object(mu, "measure_ligand_contacts",
                                  return_value=[weak, strong]):   # weak listed FIRST
            ev, facts = mu.gather_evidence("P12345")
        self.assertEqual(ev.liganded_comp_id, "SUB")             # best, not first
        self.assertEqual(ev.liganded_other_chain_fraction, 0.5)
        # The contested choice is recorded for audit.
        self.assertTrue(any("most-engaged" in n for n in ev.notes))

    def test_single_ligand_no_contest_note(self):
        only = {"comp_id": "SUB", "chain": "A", "contact_residues": 9,
                "other_chain_residues": 0, "other_chain_fraction": 0.0,
                "by_chain": {"A": 9}}
        with mock.patch.object(mu, "uniprot_facts", return_value=self._facts()), \
                mock.patch.object(mu, "largest_assembly", return_value=("1", 1)), \
                mock.patch.object(mu, "measure_ligand_contacts", return_value=[only]):
            ev, facts = mu.gather_evidence("P12345")
        self.assertEqual(ev.liganded_comp_id, "SUB")
        self.assertFalse(any("most-engaged" in n for n in ev.notes))


if __name__ == "__main__":
    unittest.main()
