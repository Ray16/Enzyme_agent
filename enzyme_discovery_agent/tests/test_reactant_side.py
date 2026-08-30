"""Reactant-side substrate extraction: products must not stand in for substrates.

An amide-forming enzyme's product carries an amine that would spuriously match an
amine target and hide that the true amine partner is cofactor-scale (ammonia).
``_reactant_side_chebi_ids`` must drop the clear product while keeping the
reactants — including one whose name cannot be side-matched (kept as a fallback).
"""

from __future__ import annotations

import unittest

from enzyme_discovery_agent.tools.family_reactions import _reactant_side_chebi_ids


def _entry(reaction_name, participants):
    # participants: list of ChEBI ids (cofactors excluded by the caller's set)
    xrefs = [{"database": "ChEBI", "id": cid} for cid in participants]
    return {"comments": [{
        "commentType": "CATALYTIC ACTIVITY",
        "reaction": {"name": reaction_name, "reactionCrossReferences": xrefs},
    }]}


# Generic amide-forming reaction: acid + amine -> amide.  Names mirror how a
# reaction string and ChEBI labels relate (product has a distinctive name;
# one reactant's label does not match the text token, as with ammonium/NH4+).
_NAMES = {
    "CHEBI:1": "an organic acid",
    "CHEBI:2": "ammonium",        # label differs from the text token "NH4(+)"
    "CHEBI:3": "an amide",         # the product
}


class TestReactantSideSplit(unittest.TestCase):
    def _name_of(self, cid):
        return _NAMES.get(cid)

    def test_product_is_dropped_reactants_kept(self):
        entry = _entry("an organic acid + NH4(+) = an amide",
                       ["CHEBI:1", "CHEBI:2", "CHEBI:3"])
        ids = _reactant_side_chebi_ids(entry, self._name_of)
        self.assertIn("CHEBI:1", ids)          # reactant matched on the left
        self.assertIn("CHEBI:2", ids)          # unmatched name -> kept (fallback)
        self.assertNotIn("CHEBI:3", ids)       # product matched on the right -> dropped

    def test_no_equals_keeps_all(self):
        entry = _entry("some catalytic activity", ["CHEBI:1", "CHEBI:3"])
        ids = _reactant_side_chebi_ids(entry, self._name_of)
        self.assertEqual(set(ids), {"CHEBI:1", "CHEBI:3"})

    def test_name_on_both_sides_is_kept(self):
        # A participant appearing on both sides (e.g. a shared scaffold) is a
        # reactant too — must not be dropped.
        entry = _entry("an organic acid + an amide = an amide + water",
                       ["CHEBI:1", "CHEBI:3"])
        ids = _reactant_side_chebi_ids(entry, self._name_of)
        self.assertIn("CHEBI:3", ids)

    def test_unresolvable_name_is_kept(self):
        entry = _entry("an organic acid + X = an amide", ["CHEBI:1", "CHEBI:9"])
        ids = _reactant_side_chebi_ids(entry, lambda cid: None)
        self.assertIn("CHEBI:9", ids)          # no name -> cannot classify -> keep


if __name__ == "__main__":
    unittest.main()
